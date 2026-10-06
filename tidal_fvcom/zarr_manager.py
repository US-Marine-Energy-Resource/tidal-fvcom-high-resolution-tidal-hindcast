"""Build one Zarr v3 store per location from the b1_vap NetCDF partitions.

The store mirrors b1_vap: dims (time, sigma_layer, face) plus node variables,
a datetime time axis padded to a full year, NaN fill. Chunks hold the full
time axis, one sigma layer and about 100 faces (about 6.4 MB).

Three phases, no Dask:
  init      scan the b1 files, write the plan, create the empty store
  write     array job, task i writes face block i and node block i
  finalize  check block markers, consolidate metadata, write complete marker

Blocks are multiples of the chunk width, so two tasks never write the same
chunk. The task count comes from config only, so run.py can size the array
before any file exists.
"""

import json
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec

from tidal_fvcom.nc_manager import calculate_optimal_chunk_sizes

TIME_DIM = "time"
SPATIAL_DIMS = ("face", "node")

PLAN_FILENAME = "zarr_build_plan.json"
COMPLETE_MARKER = "zarr_build_complete.json"
BLOCK_MARKER_DIR = "blocks"

DEFAULT_ZARR_SETTINGS = {
    "zarr_format": 3,
    "codec": {"cname": "zstd", "clevel": 3, "shuffle": "shuffle"},
    "memory_budget_gb": 32.0,
}


# Settings


def zarr_settings(config):
    merged = dict(DEFAULT_ZARR_SETTINGS)
    merged.update(config.get("zarr", {}))
    return merged


def zarr_store_name(config, location_cfg):
    return (
        f"{location_cfg['output_name']}.{config['dataset']['name']}"
        f".b1_vap.v{config['dataset']['version']}.zarr"
    )


def make_codec(settings):
    c = settings["codec"]
    return BloscCodec(cname=c["cname"], clevel=c["clevel"], shuffle=c["shuffle"])


# Chunks and blocks


def expected_timeline(location_cfg):
    # Same axis as the HSDS stitch.
    start = pd.Timestamp(location_cfg["start_date_utc"])
    if start.tz is not None:
        start = start.tz_convert("UTC").tz_localize(None)
    step = pd.Timedelta(seconds=location_cfg["expected_delta_t_seconds"])
    end = start + pd.DateOffset(years=1) - step
    return pd.date_range(start=start, end=end, freq=step)


def spatial_chunk_width(n_time, n_spatial, itemsize, config):
    # Not calculate_optimal_chunk_sizes: that can chunk along time. Time must stay one chunk.
    spec = config["dataset"]["encoding"]["chunk_spec"]
    target_bytes = spec["target_size_mb"] * 1024 * 1024
    multiple = spec["multiple"]
    per_unit = n_time * itemsize
    width = int((target_bytes // per_unit) // multiple * multiple)
    width = max(multiple, width)
    return min(width, n_spatial)


def time_var_chunks(shape, dims, dtype, config):
    itemsize = np.dtype(dtype).itemsize
    width = spatial_chunk_width(shape[0], shape[-1], itemsize, config)
    chunks = []
    for i, (dim, size) in enumerate(zip(dims, shape)):
        if dim == TIME_DIM:
            chunks.append(size)
        elif i == len(dims) - 1:
            chunks.append(width)
        else:
            chunks.append(1)
    return tuple(chunks)


def static_var_chunks(shape, dims, dtype, config):
    try:
        return tuple(calculate_optimal_chunk_sizes(shape, list(dims), dtype, config))
    except Exception:
        return tuple(shape)


def _align_down(n, multiple):
    return max(multiple, (n // multiple) * multiple)


def face_block_width(location_cfg, config):
    settings = zarr_settings(config)
    n_time = len(expected_timeline(location_cfg))
    n_sigma = max(int(location_cfg.get("sigma_layers", 1)), 1)
    face_count = int(location_cfg["face_count"])
    itemsize = 4
    chunk_faces = spatial_chunk_width(n_time, face_count, itemsize, config)
    budget = settings["memory_budget_gb"] * 1024**3
    per_face = n_time * n_sigma * itemsize
    width = _align_down(int(budget // per_face), chunk_faces)
    return min(width, _align_down(face_count, chunk_faces) + chunk_faces)


def task_count(location_cfg, config):
    return math.ceil(int(location_cfg["face_count"]) / face_block_width(location_cfg, config))


def make_blocks(n_items, width):
    return [[s, min(s + width, n_items)] for s in range(0, n_items, width)]


# Plan


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _attrs(var):
    return {k: _jsonable(v) for k, v in var.attrs.items()}


def _dtype_str(var):
    if var.dtype.kind in ("U", "O", "S", "T"):
        return "str"
    return np.dtype(var.dtype).str


def _fill_value(dtype_str):
    if dtype_str == "str":
        return ""
    dt = np.dtype(dtype_str)
    if dt.kind == "f":
        return float("nan")
    if dt.kind in ("i", "u"):
        return -999 if dt.kind == "i" else 0
    return None


def _numeric_dtype(dtype_str):
    return "<U1" if dtype_str == "str" else dtype_str


def list_b1_files(vap_dir):
    files = sorted(Path(vap_dir).glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No b1_vap .nc files in {vap_dir}")
    return files


def build_plan(files, location_cfg, config, store_path, engine="h5netcdf"):
    timeline = expected_timeline(location_cfg)
    step = pd.Timedelta(seconds=location_cfg["expected_delta_t_seconds"])

    placed = []
    for f in files:
        with xr.open_dataset(f, engine=engine) as ds:
            times = pd.DatetimeIndex(ds[TIME_DIM].values)
        if times.tz is not None:
            times = times.tz_convert("UTC").tz_localize(None)
        if len(times) > 1 and not np.all(np.diff(times.values) == step.to_timedelta64()):
            raise ValueError(f"{f.name}: time step is not uniformly {step}")
        start = timeline.get_indexer([times[0]])[0]
        if start < 0 or not timeline[start : start + len(times)].equals(times):
            raise ValueError(f"{f.name}: times do not sit on the expected timeline")
        placed.append({"path": str(f), "time_start": int(start), "n_time": len(times)})
    placed.sort(key=lambda p: p["time_start"])
    for prev, cur in zip(placed, placed[1:]):
        if cur["time_start"] < prev["time_start"] + prev["n_time"]:
            raise ValueError(f"Overlapping times: {prev['path']} / {cur['path']}")

    n_time = len(timeline)
    with xr.open_dataset(files[0], engine=engine) as ds:
        dims = {d: int(s) for d, s in ds.sizes.items() if d != TIME_DIM}
        dims[TIME_DIM] = n_time
        global_attrs = _attrs(ds)
        variables = {}
        for name, var in ds.variables.items():
            if name == TIME_DIM:
                continue
            vdims = tuple(var.dims)
            dtype = _dtype_str(var)
            shape = tuple(n_time if d == TIME_DIM else int(s) for d, s in zip(vdims, var.shape))
            if TIME_DIM in vdims:
                if vdims[0] != TIME_DIM:
                    raise ValueError(f"{name}: time must be the first dim, got {vdims}")
                if vdims[-1] in SPATIAL_DIMS:
                    kind = "time"
                    chunks = time_var_chunks(shape, vdims, _numeric_dtype(dtype), config)
                else:
                    kind = "time_only"
                    chunks = tuple(shape)
            else:
                kind = "static"
                chunks = static_var_chunks(shape, vdims, _numeric_dtype(dtype), config)
            variables[name] = {
                "dims": list(vdims),
                "shape": list(shape),
                "dtype": dtype,
                "chunks": list(chunks),
                "fill_value": _fill_value(dtype),
                "kind": kind,
                "attrs": _attrs(var),
            }
        time_attrs = _attrs(ds[TIME_DIM])

    n_tasks = task_count(location_cfg, config)
    blocks = {}
    for spatial in SPATIAL_DIMS:
        if spatial not in dims:
            continue
        widths = [
            v["chunks"][-1]
            for v in variables.values()
            if v["kind"] == "time" and v["dims"][-1] == spatial
        ]
        chunk_w = max(widths) if widths else dims[spatial]
        if spatial == "face":
            width = face_block_width(location_cfg, config)
        else:
            width = int(math.ceil(math.ceil(dims[spatial] / n_tasks) / chunk_w) * chunk_w)
        blocks[spatial] = make_blocks(dims[spatial], width)
    if len(blocks.get("face", [])) != n_tasks:
        raise RuntimeError(
            f"Plan has {len(blocks.get('face', []))} face blocks but config gives {n_tasks} tasks"
        )

    return {
        "location": location_cfg["output_name"],
        "store": str(store_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "zarr_format": zarr_settings(config)["zarr_format"],
        "timeline": {
            "start": timeline[0].isoformat(),
            "delta_t_seconds": int(step.total_seconds()),
            "n_time": n_time,
            "units": config["dataset"]["encoding"]["var"]["time"]["units"],
            "calendar": config["dataset"]["encoding"]["var"]["time"]["calendar"],
            "attrs": time_attrs,
        },
        "dims": dims,
        "files": placed,
        "variables": variables,
        "global_attrs": global_attrs,
        "blocks": blocks,
        "n_tasks": n_tasks,
    }


def save_plan(plan, build_dir):
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / BLOCK_MARKER_DIR).mkdir(exist_ok=True)
    path = build_dir / PLAN_FILENAME
    path.write_text(json.dumps(plan, indent=2))
    return path


def load_plan(build_dir):
    return json.loads((Path(build_dir) / PLAN_FILENAME).read_text())


# Init


def _np_dtype(dtype_str):
    return str if dtype_str == "str" else np.dtype(dtype_str)


def create_store(plan, config, engine="h5netcdf", overwrite=True):
    settings = zarr_settings(config)
    store_path = Path(plan["store"])
    if store_path.exists() and overwrite:
        shutil.rmtree(store_path)
    codec = make_codec(settings)
    root = zarr.create_group(str(store_path), zarr_format=settings["zarr_format"], overwrite=True)
    root.attrs.update(plan["global_attrs"])
    root.attrs["zarr_build_plan_created_utc"] = plan["created_utc"]

    tl = plan["timeline"]
    timeline = pd.date_range(
        start=tl["start"], periods=tl["n_time"], freq=pd.Timedelta(seconds=tl["delta_t_seconds"])
    )
    epoch = pd.Timestamp(tl["units"].split("since", 1)[1].strip())
    seconds = np.asarray((timeline - epoch) // pd.Timedelta(seconds=1), dtype="int64")
    t = root.create_array(
        TIME_DIM,
        shape=(tl["n_time"],),
        chunks=(tl["n_time"],),
        dtype="int64",
        dimension_names=(TIME_DIM,),
        compressors=codec,
    )
    t[:] = seconds
    t.attrs.update({**tl["attrs"], "units": tl["units"], "calendar": tl["calendar"]})

    for name, v in plan["variables"].items():
        kwargs = dict(
            shape=tuple(v["shape"]),
            chunks=tuple(v["chunks"]),
            dtype=_np_dtype(v["dtype"]),
            dimension_names=tuple(v["dims"]),
            compressors=codec,
        )
        if v["fill_value"] is not None:
            kwargs["fill_value"] = v["fill_value"]
        arr = root.create_array(name, **kwargs)
        arr.attrs.update(v["attrs"])

    first = plan["files"][0]["path"]
    with xr.open_dataset(first, engine=engine) as ds:
        for name, v in plan["variables"].items():
            if v["kind"] == "static":
                data = ds[name].values
                root[name][:] = data.astype(str) if v["dtype"] == "str" else data

    for name, v in plan["variables"].items():
        if v["kind"] == "time_only":
            for f in plan["files"]:
                with xr.open_dataset(f["path"], engine=engine) as ds:
                    root[name][f["time_start"] : f["time_start"] + f["n_time"], ...] = ds[name].values
    return store_path


# Write


def _block_marker(build_dir, task_index):
    return Path(build_dir) / BLOCK_MARKER_DIR / f"task_{task_index:05d}.json"


def write_block(plan, task_index, build_dir, engine="h5netcdf"):
    # Variable outer, file inner. Peak memory is one variable block.
    started = time.time()
    root = zarr.open_group(plan["store"], mode="r+")
    written = []
    for spatial, blocks in plan["blocks"].items():
        if task_index >= len(blocks):
            continue
        s, e = blocks[task_index]
        names = [
            n for n, v in plan["variables"].items()
            if v["kind"] == "time" and v["dims"][-1] == spatial
        ]
        for name in names:
            v = plan["variables"][name]
            shape = (plan["timeline"]["n_time"], *v["shape"][1:-1], e - s)
            dtype = object if v["dtype"] == "str" else _np_dtype(v["dtype"])
            acc = np.full(shape, v["fill_value"], dtype=dtype)
            for f in plan["files"]:
                t0, n = f["time_start"], f["n_time"]
                with xr.open_dataset(f["path"], engine=engine, decode_times=False) as ds:
                    acc[t0 : t0 + n, ...] = ds[name].isel({spatial: slice(s, e)}).values
            region = (slice(None),) * (len(shape) - 1) + (slice(s, e),)
            root[name][region] = acc
            del acc
            written.append({"variable": name, spatial: [s, e]})
    marker = _block_marker(build_dir, task_index)
    marker.write_text(
        json.dumps(
            {
                "task": task_index,
                "written": written,
                "elapsed_s": round(time.time() - started, 1),
                "finished_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
    )
    return marker


# Finalize


def missing_blocks(plan, build_dir):
    return [i for i in range(plan["n_tasks"]) if not _block_marker(build_dir, i).exists()]


def _dir_bytes(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total


def finalize_store(plan, build_dir):
    missing = missing_blocks(plan, build_dir)
    if missing:
        raise RuntimeError(f"{len(missing)} write tasks have no marker: {missing[:20]}")
    zarr.consolidate_metadata(plan["store"])
    with xr.open_zarr(plan["store"], consolidated=True) as ds:
        for name, v in plan["variables"].items():
            if tuple(ds[name].shape) != tuple(v["shape"]):
                raise RuntimeError(f"{name}: shape {ds[name].shape} != planned {v['shape']}")
        if ds.sizes[TIME_DIM] != plan["timeline"]["n_time"]:
            raise RuntimeError("time axis length mismatch after build")
        n_vars = len(ds.variables)
    summary = {
        "store": plan["store"],
        "n_variables": n_vars,
        "n_tasks": plan["n_tasks"],
        "bytes_on_disk": _dir_bytes(plan["store"]),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(build_dir) / COMPLETE_MARKER).write_text(json.dumps(summary, indent=2))
    return summary

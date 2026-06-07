"""Run-steps registry: submission + verification for the unified workflow.

This is the brains of the ``run.py`` orchestrator. It declares the ordered
processing DAG for one location and, for each step, knows how to:

  * ``submit``  — queue the step's SLURM job(s) (chained via ``--dependency=afterok``
    to its real upstream step(s)) and return the step's *terminal* job id(s) so the
    next step can depend on it.
  * ``verify``  — check the step's output level(s) for existence + expected count,
    used by ``run.py --status``.

Workflow DAG (one location)::

    verify -> standardize -> partition -> vap -+-> vap_data_products -+-> upload
                                               |   (point_parquet,    |
                                               |    compress, hsds)   |
                                               +-> summary -----------+
                                                   (monthly, yearly,
                                                    summary_parquet, atlas)

``vap_data_products`` and ``summary`` both branch off ``vap`` and run in parallel.
Two steps expose individually-addressable sub-products (see ``SUBPRODUCTS``).

Existing scripts are reused for the multi-job steps:
``partition_dataset.py`` (point parquet), ``compress_b1_array.sbatch`` (compress),
the HSDS convert/stitch sbatch pair, and the summarize array sub-chain.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from src import file_manager
from src.slurm import (
    Submitter,
    calculate_array_size,
    dependency_arg,
)

# Reuse the summarize sizing/runtime table that dispatch_summarize_jobs.py owns.
from dispatch_summarize_jobs import LOCATIONS as SUMMARIZE_LOCATIONS, BATCH_SIZE_MAP

# Reuse the per-location HSDS resource table.
from dispatch_b1_to_hsds_jobs import LOCATION_RESOURCES as HSDS_RESOURCES

# Faces per batch for the point-parquet partition (matches partition_dataset.py usage).
POINT_PARQUET_BATCH_SIZE = 10000

# Publishable data levels uploaded to S3 by the `upload` step (in dispatch order).
# Must use dispatch_s3_upload_v2.py's DATA_LEVEL_CONFIG keys.
UPLOAD_DATA_LEVELS = [
    "b1_vap_daily_compressed",
    "hsds",
    "b1_vap_by_point_partition",
    "b4_vap_summary_parquet",
    "b5_vap_atlas_summary_parquet",
]

# Ordered step DAG. depends_on lists upstream *step* names.
STEP_ORDER = [
    "verify",
    "standardize",
    "partition",
    "vap",
    "vap_data_products",
    "summary",
    "upload",
]

STEP_DEPENDS_ON = {
    "verify": [],
    "standardize": ["verify"],
    "partition": ["standardize"],
    "vap": ["partition"],
    "vap_data_products": ["vap"],
    "summary": ["vap"],
    "upload": ["vap_data_products", "summary"],
}

# Sub-products for the two multi-output steps (ordered; internal deps noted below).
SUBPRODUCTS = {
    "vap_data_products": ["point_parquet", "compress", "hsds"],
    "summary": ["monthly", "yearly", "summary_parquet", "atlas"],
}

# Human-readable output level per step / sub-product (used by `run.py --list-steps`).
STEP_LEVELS = {
    "verify": "tracking (QC)",
    "standardize": "a1_std",
    "partition": "a2_std_partition",
    "vap": "b1_vap",
    "vap_data_products": "b1_vap-derived products",
    "summary": "b2 / b3 / b4 / b5",
    "upload": "s3 (deployment)",
}

SUBPRODUCT_LEVELS = {
    "point_parquet": "b1_vap_by_point_partition",
    "compress": "b1_vap_daily_compressed",
    "hsds": "hsds (.h5)",
    "monthly": "b2 monthly mean (NetCDF)",
    "yearly": "b3 yearly mean (NetCDF)",
    "summary_parquet": "b4 summary parquet",
    "atlas": "b5 atlas geoparquet",
}

# Per single-node step SLURM resources (time, partition) for run_step.sbatch.
SINGLE_NODE_RESOURCES = {
    "verify": ("12:00:00", "standard"),
    "standardize": ("12:00:00", "standard"),
    "partition": ("12:00:00", "standard"),
    "vap": ("24:00:00", "standard"),
    "summary": ("4:00:00", "standard"),  # run_summary.py (b4/b5) is light
}


@dataclass
class Ctx:
    """Submission context threaded through every step."""

    config: dict
    location: str  # location key, e.g. "cook_inlet"
    submitter: Submitter
    force: bool = False

    @property
    def location_cfg(self):
        return self.config["location_specification"][self.location]


@dataclass
class ProductStatus:
    """Result of verifying one output level."""

    address: str  # "summary.yearly" or "vap"
    label: str  # human description, e.g. "b3 yearly mean (NetCDF)"
    state: str  # complete | partial | missing | unknown
    found: int
    expected: Optional[int]


# --------------------------------------------------------------------------- #
# Submission helpers
# --------------------------------------------------------------------------- #


def _submit_run_step(ctx, dep_ids, step, *, product=None, label=None):
    """Submit a single-node in-process step via run_step.sbatch."""
    time_limit, partition = SINGLE_NODE_RESOURCES[step]
    export = f"LOCATION={ctx.location},STEP={step}"
    if product:
        export += f",PRODUCT={product}"

    args = []
    dep = dependency_arg(dep_ids)
    if dep:
        args.append(dep)
    args += [
        f"--export={export}",
        f"--job-name={ctx.location}_{step}",
        f"--output={ctx.location}_{step}_%j.out",
        f"--time={time_limit}",
        f"--partition={partition}",
        "run_step.sbatch",
    ]
    return ctx.submitter.submit(args, label=label or step)


def _output_dir(config, location_cfg, key):
    """Resolve an output dir without creating it (safe for local status/dry-run)."""
    return file_manager.get_output_dirs(config, location_cfg, create=False)[key]


def _hsds_final_path(config, location_cfg):
    """Final stitched HSDS file path, resolved without creating directories."""
    hsds_dir = _output_dir(config, location_cfg, "hsds")
    output_name = location_cfg["output_name"]
    dataset_name = config["dataset"]["name"]
    version = config["dataset"]["version"]
    return hsds_dir / f"{output_name}.{dataset_name}.hsds.v{version}.h5"


def _summary_params(location):
    """(faces, batch_size, process_runtime_hours) for the summarize array."""
    cfg = SUMMARIZE_LOCATIONS[location]
    return (
        cfg["faces"],
        BATCH_SIZE_MAP[cfg["temporal_resolution"]],
        cfg["process_runtime_hours"],
    )


def _count_time_partitions(location_cfg):
    """Estimate the number of time-partition files from config dates + frequency."""
    start = pd.Timestamp(location_cfg["start_date_utc"]).normalize()
    end = pd.Timestamp(location_cfg["end_date_utc"])
    freq = location_cfg["partition_frequency"]
    buckets = pd.date_range(start=start, end=end, freq=freq)
    return max(len(buckets), 1)


def _count_b1_files_or_estimate(config, location_cfg):
    """Actual b1_vap file count if present, else estimate from config.

    During a full chained submission the b1_vap files don't exist yet, so we size
    the compress/hsds arrays from the config-derived partition count. When running
    ``--from`` a later step the real files exist and give an exact count.
    """
    vap_dir = _output_dir(config, location_cfg, "vap")
    n = len(sorted(vap_dir.glob("*.nc"))) if vap_dir.exists() else 0
    return n if n > 0 else _count_time_partitions(location_cfg)


# --------------------------------------------------------------------------- #
# Per-step submit functions  (each returns a list of terminal job ids)
# --------------------------------------------------------------------------- #


def _submit_verify(ctx, dep_ids, subs):
    return [_submit_run_step(ctx, dep_ids, "verify")]


def _submit_standardize(ctx, dep_ids, subs):
    return [_submit_run_step(ctx, dep_ids, "standardize")]


def _submit_partition(ctx, dep_ids, subs):
    return [_submit_run_step(ctx, dep_ids, "partition")]


def _submit_vap(ctx, dep_ids, subs):
    return [_submit_run_step(ctx, dep_ids, "vap")]


def _submit_point_parquet(ctx, dep_ids):
    faces = ctx.location_cfg["face_count"]
    array_size = calculate_array_size(faces, POINT_PARQUET_BATCH_SIZE)
    args = []
    dep = dependency_arg(dep_ids)
    if dep:
        args.append(dep)
    args += [
        f"--export=LOCATION={ctx.location},BATCH_SIZE={POINT_PARQUET_BATCH_SIZE}",
        f"--array=0-{array_size}",
        f"--job-name={ctx.location}_point_parquet",
        f"--output={ctx.location}_point_parquet_%A_%a.out",
        "point_parquet_array.sbatch",
    ]
    return ctx.submitter.submit(args, label="vap_data_products.point_parquet")


def _submit_compress(ctx, dep_ids):
    num_files = _count_b1_files_or_estimate(ctx.config, ctx.location_cfg)
    array_size = max(num_files - 1, 0)
    args = []
    dep = dependency_arg(dep_ids)
    if dep:
        args.append(dep)
    args += [
        f"--export=LOCATION={ctx.location}",
        f"--array=0-{array_size}",
        f"--job-name={ctx.location}_compress_b1",
        f"--output={ctx.location}_compress_b1_%A_%a.out",
        "compress_b1_array.sbatch",
    ]
    return ctx.submitter.submit(args, label="vap_data_products.compress")


def _submit_hsds(ctx, dep_ids):
    num_chunks = _count_b1_files_or_estimate(ctx.config, ctx.location_cfg)
    array_size = max(num_chunks - 1, 0)
    resources = HSDS_RESOURCES.get(
        ctx.location,
        {
            "convert": {"partition": "shared", "mem": "128GB", "time": "4:00:00"},
            "stitch": {"partition": "standard", "time": "47:00:00"},
        },
    )
    conv = resources["convert"]
    stitch = resources["stitch"]

    # Convert array (one task per b1_vap temporal chunk), gated on vap.
    conv_args = []
    dep = dependency_arg(dep_ids)
    if dep:
        conv_args.append(dep)
    conv_args += [
        f"--export=LOCATION={ctx.location}",
        f"--array=0-{array_size}",
        f"--partition={conv['partition']}",
        f"--mem={conv['mem']}",
        f"--time={conv['time']}",
        f"--job-name=convert_hsds_{ctx.location}",
        f"--output=convert_b1_hsds_{ctx.location}_%A_%a.out",
        "convert_single_b1_vap_nc_into_hsds_h5_file.sbatch",
    ]
    convert_id = ctx.submitter.submit(conv_args, label="vap_data_products.hsds:convert")

    # Stitch the per-chunk H5 files into the single yearly file (afterok convert).
    stitch_args = [
        dependency_arg([convert_id]),
        f"--export=LOCATION={ctx.location}",
        f"--partition={stitch['partition']}",
        f"--time={stitch['time']}",
        f"--job-name=stitch_hsds_{ctx.location}",
        f"--output=stitch_hsds_{ctx.location}_%j.out",
        "stitch_prepared_b1_files_into_singular_hsds_h5_file.sbatch",
    ]
    return ctx.submitter.submit(stitch_args, label="vap_data_products.hsds:stitch")


def _submit_vap_data_products(ctx, dep_ids, subs):
    """Submit the selected b1_vap-derived products (all independent off vap)."""
    selected = subs or SUBPRODUCTS["vap_data_products"]
    terminals = []
    if "point_parquet" in selected:
        terminals.append(_submit_point_parquet(ctx, dep_ids))
    if "compress" in selected:
        terminals.append(_submit_compress(ctx, dep_ids))
    if "hsds" in selected:
        terminals.append(_submit_hsds(ctx, dep_ids))
    return terminals


def _submit_means_chain(ctx, dep_ids, product):
    """Reproduce the 3-stage means sub-chain (array -> coordinator -> concat).

    Mirrors dispatch_summarize_jobs.submit_location_jobs but gated on an inbound
    ``vap`` dependency and parameterized by PRODUCT (monthly|yearly|both). Returns
    the concat job id (the means terminal).
    """
    faces, batch_size, runtime_hours = _summary_params(ctx.location)
    array_size = calculate_array_size(faces, batch_size)

    proc_args = []
    dep = dependency_arg(dep_ids)
    if dep:
        proc_args.append(dep)
    proc_args += [
        f"--export=LOCATION={ctx.location},FACES={faces},BATCH_SIZE={batch_size},PRODUCT={product}",
        f"--array=0-{array_size}",
        f"--output={ctx.location}_process_%A_%a.out",
        f"--job-name={ctx.location}_process",
        f"--time={runtime_hours * 60}",
        "summarize_single_location_batch.sbatch",
    ]
    process_id = ctx.submitter.submit(
        proc_args, label=f"summary.means({product}):process"
    )

    coordinator_args = [
        dependency_arg([process_id], kind="afterany"),
        f"--export=LOCATION={ctx.location},PRODUCT={product}",
        f"--output={ctx.location}_coordinator_%j.out",
        f"--job-name={ctx.location}_coordinator",
        "summarize_retry_coordinator.sbatch",
    ]
    coordinator_id = ctx.submitter.submit(
        coordinator_args, label=f"summary.means({product}):coordinator"
    )

    concat_args = [
        dependency_arg([coordinator_id]),
        f"--export=LOCATION={ctx.location},PRODUCT={product}",
        f"--output={ctx.location}_concat_%j.out",
        f"--job-name={ctx.location}_concat",
        "summarize_location_concat.sbatch",
    ]
    return ctx.submitter.submit(concat_args, label=f"summary.means({product}):concat")


def _submit_summary(ctx, dep_ids, subs):
    """Submit the selected summary sub-products with correct internal deps.

    monthly/yearly -> one means sub-chain (PRODUCT picked from the selection);
    summary_parquet (b4) depends on yearly; atlas (b5) depends on b4. Because
    run_summary.py builds b5 from b4 in one pass, atlas implies b4 too.
    """
    selected = subs or SUBPRODUCTS["summary"]
    terminals = []

    means_products = [s for s in ("monthly", "yearly") if s in selected]
    means_terminal = None
    if means_products:
        product = "both" if len(means_products) == 2 else means_products[0]
        means_terminal = _submit_means_chain(ctx, dep_ids, product)
        terminals.append(means_terminal)

    # b4/b5 are produced by run_summary.py; atlas (b5) regenerates b4 in the same
    # pass, so a single submission covers both when atlas is requested.
    if "atlas" in selected:
        parquet_product = "both"
    elif "summary_parquet" in selected:
        parquet_product = "summary_parquet"
    else:
        parquet_product = None

    if parquet_product:
        # b4/b5 need the yearly means (b3). Depend on the means chain only if it
        # produced yearly this run; otherwise b3 is assumed present (verified by
        # run.py unless --force).
        parquet_deps = (
            [means_terminal] if (means_terminal and "yearly" in means_products) else []
        )
        terminals.append(
            _submit_run_step(
                ctx,
                parquet_deps,
                "summary",
                product=parquet_product,
                label=f"summary.parquet({parquet_product})",
            )
        )
    return terminals


def _submit_upload(ctx, dep_ids, subs):
    """Final S3 deployment.

    The S3 upload path (dispatch_s3_upload_v2.py) is manifest-based and manages its
    own SLURM array, so it can't be afterok-chained here in v1. Instead we emit the
    exact follow-up command to run once the data products are complete.
    """
    levels = (
        "b1_vap_daily_compressed hsds b1_vap_by_point_partition "
        "b4_vap_summary_parquet b5_vap_atlas_summary_parquet"
    )
    print("  [upload] S3 upload is manifest-based and not auto-chained in v1.")
    print(
        "  [upload] After the data products complete, generate the manifest then run:"
    )
    print(
        f"           python dispatch_s3_upload_v2.py {ctx.location} "
        f"--data-levels {levels} --skip-if-uploaded"
    )
    return []


# --------------------------------------------------------------------------- #
# Per-step verify functions
# --------------------------------------------------------------------------- #


def _classify(found, expected):
    if expected is None:
        return "complete" if found > 0 else "missing"
    if found <= 0:
        return "missing"
    if found >= expected:
        return "complete"
    return "partial"


def _status(address, label, found, expected):
    return ProductStatus(address, label, _classify(found, expected), found, expected)


def _count_glob(directory, pattern):
    return len(sorted(directory.glob(pattern))) if directory.exists() else 0


def _verify_verify(ctx):
    config, location_cfg = ctx.config, ctx.location_cfg
    tracking_dir = _output_dir(config, location_cfg, "tracking")
    tracking_path = (
        tracking_dir / f"{location_cfg['output_name']}_verify_step_tracking.parquet"
    )
    return [
        _status("verify", "verify tracking parquet", int(tracking_path.exists()), 1)
    ]


def _verify_standardize(ctx):
    standardized_dir = _output_dir(ctx.config, ctx.location_cfg, "standardized")
    return [
        _status(
            "standardize", "a1_std (*.nc)", _count_glob(standardized_dir, "*.nc"), None
        )
    ]


def _verify_partition(ctx):
    partition_dir = _output_dir(ctx.config, ctx.location_cfg, "standardized_partition")
    expected = _count_time_partitions(ctx.location_cfg)
    return [
        _status(
            "partition",
            "a2_std_partition (*.nc)",
            _count_glob(partition_dir, "*.nc"),
            expected,
        )
    ]


def _verify_vap(ctx):
    vap_dir = _output_dir(ctx.config, ctx.location_cfg, "vap")
    expected = _count_time_partitions(ctx.location_cfg)
    return [_status("vap", "b1_vap (*.nc)", _count_glob(vap_dir, "*.nc"), expected)]


def _verify_vap_data_products(ctx):
    config, location_cfg = ctx.config, ctx.location_cfg
    statuses = []

    point_parquet_dir = _output_dir(config, location_cfg, "vap_partition")
    point_parquet_found = (
        len(sorted(point_parquet_dir.rglob("*.parquet")))
        if point_parquet_dir.exists()
        else 0
    )
    statuses.append(
        _status(
            "vap_data_products.point_parquet",
            "b1_vap_by_point_partition",
            point_parquet_found,
            None,
        )
    )

    compressed_dir = _output_dir(config, location_cfg, "vap_daily_compressed")
    statuses.append(
        _status(
            "vap_data_products.compress",
            "b1_vap_daily_compressed (*.nc)",
            _count_glob(compressed_dir, "*.nc"),
            None,
        )
    )

    hsds_path = _hsds_final_path(config, location_cfg)
    statuses.append(
        _status(
            "vap_data_products.hsds", "hsds (final .h5)", int(hsds_path.exists()), 1
        )
    )
    return statuses


def _verify_summary(ctx):
    config, location_cfg = ctx.config, ctx.location_cfg
    statuses = []

    monthly_dir = _output_dir(config, location_cfg, "monthly_summary_vap")
    statuses.append(
        _status(
            "summary.monthly",
            "b2 monthly mean (*.nc)",
            _count_glob(monthly_dir, "*.nc"),
            None,
        )
    )

    yearly_dir = _output_dir(config, location_cfg, "yearly_summary_vap")
    statuses.append(
        _status(
            "summary.yearly",
            "b3 yearly mean (*.nc)",
            _count_glob(yearly_dir, "*.nc"),
            None,
        )
    )

    summary_parquet_dir = _output_dir(config, location_cfg, "vap_summary_parquet")
    statuses.append(
        _status(
            "summary.summary_parquet",
            "b4 summary parquet",
            _count_glob(summary_parquet_dir, "*.parquet"),
            None,
        )
    )

    atlas_dir = _output_dir(config, location_cfg, "vap_atlas_summary_parquet")
    statuses.append(
        _status(
            "summary.atlas",
            "b5 atlas parquet",
            _count_glob(atlas_dir, "*.parquet"),
            None,
        )
    )
    return statuses


def _verify_upload(ctx):
    return [_status("upload", "s3 (not checked locally)", 0, None)]


# --------------------------------------------------------------------------- #
# Step registry
# --------------------------------------------------------------------------- #


@dataclass
class Step:
    name: str
    depends_on: list
    subproducts: list
    submit: Callable  # (ctx, dep_ids, selected_subs) -> list[str]
    verify: Callable  # (ctx) -> list[ProductStatus]


STEPS = {
    "verify": Step("verify", [], [], _submit_verify, _verify_verify),
    "standardize": Step(
        "standardize", ["verify"], [], _submit_standardize, _verify_standardize
    ),
    "partition": Step(
        "partition", ["standardize"], [], _submit_partition, _verify_partition
    ),
    "vap": Step("vap", ["partition"], [], _submit_vap, _verify_vap),
    "vap_data_products": Step(
        "vap_data_products",
        ["vap"],
        SUBPRODUCTS["vap_data_products"],
        _submit_vap_data_products,
        _verify_vap_data_products,
    ),
    "summary": Step(
        "summary",
        ["vap"],
        SUBPRODUCTS["summary"],
        _submit_summary,
        _verify_summary,
    ),
    "upload": Step(
        "upload", ["vap_data_products", "summary"], [], _submit_upload, _verify_upload
    ),
}


def ordered_steps():
    """Return Step objects in topological (declared) order."""
    return [STEPS[name] for name in STEP_ORDER]

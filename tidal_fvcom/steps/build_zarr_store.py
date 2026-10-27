"""Pipeline step worker: build the b1_vap Zarr store for one location.

Usage:
    python -m tidal_fvcom.steps.build_zarr_store <location> --phase init [--resume]
    python -m tidal_fvcom.steps.build_zarr_store <location> --phase write --block N
    python -m tidal_fvcom.steps.build_zarr_store <location> --phase finalize
    python -m tidal_fvcom.steps.build_zarr_store <location> --phase tasks
"""

from tidal_fvcom.config import config
from tidal_fvcom.cli import location_arg_parser
from tidal_fvcom import file_manager
from tidal_fvcom import zarr_manager as zm


def main():
    parser = location_arg_parser(config, "Build the b1_vap Zarr store.")
    parser.add_argument("--phase", choices=["init", "write", "finalize", "tasks"], required=True)
    parser.add_argument("--block", type=int, help="Array task index for the write phase.")
    parser.add_argument(
        "--resume", action="store_true", help="init: keep an existing store and plan."
    )
    args = parser.parse_args()

    location_cfg = config["location_specification"][args.location]
    engine = config["dataset"]["xarray_netcdf4_engine"]

    if args.phase == "tasks":
        print(zm.task_count(location_cfg, config))
        return

    build_dir = file_manager.get_zarr_build_dir(config, location_cfg)
    store_path = file_manager.get_zarr_store_path(config, location_cfg)

    if args.phase == "init":
        if args.resume and (build_dir / zm.PLAN_FILENAME).exists():
            print(f"[zarr] Resume: keeping plan and store at {store_path}")
            return
        files = zm.list_b1_files(file_manager.get_vap_output_dir(config, location_cfg))
        print(f"[zarr] Planning from {len(files)} b1 files...")
        plan = zm.build_plan(files, location_cfg, config, store_path, engine=engine)
        for marker in (build_dir / zm.BLOCK_MARKER_DIR).glob("*.json"):
            marker.unlink()
        (build_dir / zm.COMPLETE_MARKER).unlink(missing_ok=True)
        zm.save_plan(plan, build_dir)
        print(f"[zarr] {plan['n_tasks']} write tasks, {len(plan['variables'])} variables")
        zm.create_store(plan, config, engine=engine)
        print(f"[zarr] Created {store_path}")
        return

    plan = zm.load_plan(build_dir)

    if args.phase == "write":
        if args.block is None:
            parser.error("--block is required for the write phase")
        if args.block >= plan["n_tasks"]:
            print(f"[zarr] Block {args.block} is past n_tasks={plan['n_tasks']}, nothing to do")
            return
        marker = zm.write_block(plan, args.block, build_dir, engine=engine)
        print(f"[zarr] Block {args.block} done, marker {marker}")
        return

    summary = zm.finalize_store(plan, build_dir)
    print(f"[zarr] Complete: {summary}")


if __name__ == "__main__":
    main()

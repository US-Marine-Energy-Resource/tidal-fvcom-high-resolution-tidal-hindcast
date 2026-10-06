"""Unified workflow orchestrator for one tidal-FVCOM location.

Single entry point that dispatches the whole processing workflow (raw -> b5) for a
location as a chain of SLURM jobs wired together with ``--dependency=afterok``, runs
a single step, or runs from a step onward. Also reports per-level status so you can
see what is done and resume from the first incomplete product.

Step / sub-product addresses (see tidal_fvcom/run_steps_manager.py for the DAG)::

    verify  standardize  partition  vap
    vap_data_products[.point_parquet|.compress|.hsds]
    summary[.monthly|.yearly|.summary_parquet|.atlas]
    upload

Usage:
    python run.py <location>                          # full run, all steps
    python run.py <location> --from vap               # run from a step to the end
    python run.py <location> --only summary           # run one whole step
    python run.py <location> --only summary.atlas     # run one sub-product
    python run.py <location> --from summary.summary_parquet
    python run.py <location> --status                 # verify only, no submit
    python run.py <location> --dry-run                # print the plan, submit nothing
    python run.py <location> --skip-verified          # skip already-complete products
    python run.py <location> --from summary --force   # don't error on missing upstream
"""

import argparse

from tidal_fvcom.config import config
from tidal_fvcom.cli import validate_location
from tidal_fvcom.slurm import Submitter
from tidal_fvcom import run_steps_manager


def _parse_address(address):
    """Split a 'step' or 'step.sub' address into (step, sub_or_None) and validate."""
    if "." in address:
        step, sub = address.split(".", 1)
    else:
        step, sub = address, None

    if step not in run_steps_manager.STEPS:
        raise SystemExit(
            f"Unknown step '{step}'. Valid steps: {', '.join(run_steps_manager.STEP_ORDER)}"
        )
    if sub is not None and sub not in run_steps_manager.SUBPRODUCTS.get(step, []):
        valid = ", ".join(run_steps_manager.SUBPRODUCTS.get(step, [])) or "(none)"
        raise SystemExit(
            f"Unknown sub-product '{sub}' for step '{step}'. Valid: {valid}"
        )
    return step, sub


def _subs_from(step, sub):
    """Sub-products of ``step`` from ``sub`` onward, in declared order."""
    subs = run_steps_manager.SUBPRODUCTS[step]
    return subs[subs.index(sub) :]


def resolve_selection(only, from_):
    """Return an ordered list of (step_name, selected_subs) to run.

    ``selected_subs`` is a list of sub-product names for multi-output steps, or None
    meaning "the whole step" (all sub-products / the single-output step itself).
    """
    if only:
        step, sub = _parse_address(only)
        subs = [sub] if sub else None
        return [(step, subs)]

    if from_:
        step, sub = _parse_address(from_)
        start = run_steps_manager.STEP_ORDER.index(step)
        selection = []
        for i, name in enumerate(run_steps_manager.STEP_ORDER):
            if i < start:
                continue
            if i == start and sub:
                selection.append((name, _subs_from(step, sub)))
            else:
                selection.append((name, None))
        return selection

    return [(name, None) for name in run_steps_manager.STEP_ORDER]


def _check_upstream(ctx, selected_step_names):
    """Verify out-of-selection upstream steps are complete; return blocker messages."""
    blockers = []
    for name in selected_step_names:
        for dep in run_steps_manager.STEP_DEPENDS_ON[name]:
            if dep in selected_step_names:
                continue  # produced this run
            for st in run_steps_manager.STEPS[dep].verify(ctx):
                if st.state != "complete":
                    exp = st.expected if st.expected is not None else "?"
                    blockers.append(
                        f"  {name}: upstream {st.address} is {st.state} ({st.found}/{exp})"
                    )
    return blockers


def list_steps():
    """Print the available steps, sub-products, dependencies, and output levels."""
    print("\nAvailable steps (in run order):\n")
    for i, name in enumerate(run_steps_manager.STEP_ORDER, start=1):
        deps = run_steps_manager.STEP_DEPENDS_ON[name]
        dep_str = f"  (after: {', '.join(deps)})" if deps else ""
        print(f"  {i}. {name:<20} -> {run_steps_manager.STEP_LEVELS[name]}{dep_str}")
        for sub in run_steps_manager.SUBPRODUCTS.get(name, []):
            print(f"       .{sub:<17} -> {run_steps_manager.SUBPRODUCT_LEVELS[sub]}")
    print(
        "\nAddress a sub-product as 'step.sub', e.g. summary.atlas or "
        "vap_data_products.hsds."
    )
    print("Examples:")
    print("  python run.py cook_inlet                     # full run")
    print("  python run.py cook_inlet --from vap          # resume from a step")
    print("  python run.py cook_inlet --only summary.atlas")
    print("  python run.py cook_inlet --status")


def run_status(ctx):
    """Print a per-level status table and suggest a resume point."""
    print(f"\nStatus: {ctx.location}  ({ctx.location_cfg['output_name']})\n")
    header = f"  {'ADDRESS':<34} {'STATE':<9} {'FOUND/EXPECTED':<16} OUTPUT"
    print(header)
    print("  " + "-" * (len(header) - 2))

    first_incomplete = None
    glyph = {"complete": "OK ", "partial": "~~ ", "missing": "XX ", "unknown": "?? "}
    for step in run_steps_manager.ordered_steps():
        for st in step.verify(ctx):
            exp = st.expected if st.expected is not None else "?"
            fe = f"{st.found}/{exp}"
            print(
                f"  {st.address:<34} {glyph[st.state]}{st.state:<6} {fe:<16} {st.label}"
            )
            if first_incomplete is None and st.state in ("missing", "partial"):
                first_incomplete = st.address

    print()
    if first_incomplete:
        print(f"First incomplete: {first_incomplete}")
        print(f"Resume with:  python run.py {ctx.location} --from {first_incomplete}")
    else:
        print("All products complete.")


def run_submit(ctx, selection, skip_verified, force):
    """Submit the selected steps as an afterok-chained SLURM workflow."""
    # Optionally drop products that are already complete.
    if skip_verified:
        pruned = []
        for name, subs in selection:
            statuses = {s.address: s for s in run_steps_manager.STEPS[name].verify(ctx)}
            if not run_steps_manager.STEPS[name].subproducts:
                if statuses and all(s.state == "complete" for s in statuses.values()):
                    print(f"[skip-verified] {name} already complete; skipping.")
                    continue
                pruned.append((name, subs))
            else:
                want = subs or run_steps_manager.STEPS[name].subproducts
                remaining = [
                    sub
                    for sub in want
                    if statuses.get(f"{name}.{sub}") is None
                    or statuses[f"{name}.{sub}"].state != "complete"
                ]
                if not remaining:
                    print(
                        f"[skip-verified] {name} ({', '.join(want)}) already complete; skipping."
                    )
                    continue
                pruned.append((name, remaining))
        selection = pruned

    if not selection:
        print("Nothing to submit (everything selected is already complete).")
        return

    selected_names = [name for name, _ in selection]

    # Guard: out-of-selection upstreams must be complete (unless --force).
    blockers = _check_upstream(ctx, selected_names)
    if blockers and not force:
        print("\nCannot submit — upstream products are missing:")
        print("\n".join(blockers))
        print("\nRun the upstream steps first, or pass --force to submit anyway.")
        raise SystemExit(1)
    if blockers and force:
        print("\n[force] Ignoring missing upstream products:")
        print("\n".join(blockers))

    mode = "DRY RUN — no jobs submitted" if ctx.submitter.dry_run else "submitting jobs"
    print(f"\nWorkflow: {ctx.location}  ({ctx.location_cfg['output_name']})  —  {mode}")
    print(f"Steps: {', '.join(selected_names)}\n")

    submitted = {}  # step name -> list of terminal job ids
    for name, subs in selection:
        dep_ids = []
        for dep in run_steps_manager.STEP_DEPENDS_ON[name]:
            dep_ids.extend(submitted.get(dep, []))
        shown = name if subs is None else f"{name} [{', '.join(subs)}]"
        print(f"-> {shown}")
        terminals = run_steps_manager.STEPS[name].submit(ctx, dep_ids, subs)
        submitted[name] = terminals

    print("\nSubmitted job ids:")
    for name in selected_names:
        ids = submitted.get(name) or ["(none)"]
        print(f"  {name:<20} {', '.join(ids)}")
    if not ctx.submitter.dry_run:
        print("\nWatch with: squeue -u $USER")


def main():
    parser = argparse.ArgumentParser(
        description="Unified workflow orchestrator for one tidal-FVCOM location.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "location",
        nargs="?",
        type=validate_location(config),
        help="Location to process (e.g., aleutian_islands, cook_inlet)",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="List the available steps and sub-products, then exit.",
    )
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument(
        "--from",
        dest="from_",
        metavar="STEP[.SUB]",
        help="Run from this step/sub-product to the end.",
    )
    selector.add_argument(
        "--only",
        metavar="STEP[.SUB]",
        help="Run only this step or sub-product.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print per-level status and exit (no submission).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the sbatch plan without submitting.",
    )
    parser.add_argument(
        "--skip-verified",
        action="store_true",
        help="Skip products that --status reports as complete.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Submit even if out-of-selection upstream products are missing.",
    )
    args = parser.parse_args()

    if args.list_steps:
        list_steps()
        return

    if not args.location:
        parser.error("location is required (or use --list-steps)")

    ctx = run_steps_manager.Ctx(
        config=config,
        location=args.location,
        submitter=Submitter(dry_run=args.dry_run),
        force=args.force,
    )

    if args.status:
        run_status(ctx)
        return

    selection = resolve_selection(args.only, args.from_)
    run_submit(ctx, selection, skip_verified=args.skip_verified, force=args.force)


if __name__ == "__main__":
    main()

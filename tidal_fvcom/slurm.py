"""Shared SLURM orchestration helpers.

Centralizes the ``sbatch --parsable`` submission pattern and array-index utilities
that were previously copy-pasted across the ``dispatch_*.py`` scripts
(``steps/dispatch_summarize_jobs.py``, ``steps/dispatch_b1_to_hsds_jobs.py``)
and ``steps/summarize_retry_coordinator.py``.

The :class:`Submitter` wraps job submission so the unified pipeline (``run.py``)
can run in either real or ``--dry-run`` mode with one code path: in dry-run mode it
prints the exact ``sbatch`` command and hands back a synthetic job id so dependency
chains still render correctly.
"""

import math
import subprocess


def calculate_array_size(total_items, batch_size):
    """Return the highest 0-based array index needed to cover ``total_items``.

    e.g. 25 items at batch_size 10 -> ceil(25/10) - 1 == 2 (indices 0,1,2).
    """
    return math.ceil(total_items / batch_size) - 1


def format_array_indices(indices):
    """Format a list of integers into a SLURM array spec.

    Converts ``[1, 2, 3, 5, 7, 8, 9]`` to ``"1-3,5,7-9"``. Mirrors the helper in
    ``summarize_retry_coordinator.py`` so both share one implementation.
    """
    if not indices:
        return ""

    indices = sorted(indices)
    ranges = []
    start = end = indices[0]

    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append(str(start) if start == end else f"{start}-{end}")
            start = end = i

    ranges.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(ranges)


def dependency_arg(dep_job_ids, kind="afterok"):
    """Build an ``--dependency=<kind>:id1:id2`` arg, or None if no deps.

    Falsy ids (None / "") are dropped so callers can pass through optional upstream
    ids without guarding each one.
    """
    ids = [str(j) for j in (dep_job_ids or []) if j]
    if not ids:
        return None
    return f"--dependency={kind}:" + ":".join(ids)


class Submitter:
    """Submits sbatch jobs (or prints them, in dry-run mode) and returns job ids.

    In dry-run mode every submission returns a synthetic id (``J1``, ``J2``, ...) so
    downstream ``--dependency`` wiring is visible in the printed plan without any
    real jobs being queued.
    """

    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self._fake_counter = 0

    def submit(self, sbatch_args, label=None):
        """Submit ``sbatch --parsable <sbatch_args>`` and return the job id.

        Args:
            sbatch_args: list of sbatch CLI args (flags + the script name).
            label: short human label for the printed line.

        Returns:
            The SLURM job id (str), or a synthetic ``J<n>`` id in dry-run mode.
        """
        cmd = ["sbatch", "--parsable"] + sbatch_args
        tag = label or "job"

        if self.dry_run:
            self._fake_counter += 1
            fake_id = f"J{self._fake_counter}"
            print(f"  [{tag}] {' '.join(cmd)}")
            print(f"      -> would submit, job id {fake_id}")
            return fake_id

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip()
        print(f"  [{tag}] submitted job {job_id}")
        return job_id

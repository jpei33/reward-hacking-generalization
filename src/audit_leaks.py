"""
audit_leaks.py  —  DAY 1, the feasibility gate.

WHAT IT DOES
    For each MiMo task, probe which answer-leak vectors actually survive Xiaomi's
    leak-closing, and whether the reference fix is recoverable. Emits a per-task
    report so you can choose a PRIMARY demonstrated vector (for training) and a
    SECONDARY vector (for the transfer eval) from real, present leaks.

LEAK VECTORS PROBED (in rough order of expected reliability)
    1. parquet_test_patch   hidden test patch shipped in the task's parquet data
    2. git_history          fix recoverable via unreachable objects / reflog
    3. file_timestamps      mtime ordering reveals which file holds the fix (Vals)
    4. grader_overwrite     agent can write the file the reference grader reads

INPUTS
    --envs   path to downloaded MiMo envs (e.g. data/raw/mimo-general),
             the FineEnvs Harbor layout: <task>/instruction.md, tests, grader, *.parquet
OUTPUTS
    --out    results/leak_report.csv  (one row per task x vector: present?, recoverable?, notes)

CONNECTS TO
    - downstream: build_trajectories.py reads the report to pick tasks where the
      chosen vector is present AND the fix is recoverable.
    - eval_harness.py uses the SAME probes to confirm leaks in held-out tasks.

GATE
    If no vector reliably yields a recoverable reference fix across enough tasks,
    STOP here and reconsider — this is the cheapest place to find out.

RUN
    python -I src/audit_leaks.py --envs data/raw/mimo-general --out results/leak_report.csv
"""
from __future__ import annotations
import argparse
from pathlib import Path


def probe_parquet_test_patch(task_dir: Path) -> dict:
    """Look for hidden test patches / expected-output files embedded in the parquet."""
    # TODO: read *.parquet with pyarrow; scan columns for the hidden test patch
    #       or reference solution; return {"present": bool, "recoverable": bool, "notes": str}
    raise NotImplementedError


def probe_git_history(task_dir: Path) -> dict:
    """Check whether the fix is recoverable from git (unreachable objects, reflog, packed refs)."""
    # TODO: git -C <repo> fsck --no-reflogs --unreachable ; inspect candidate commits
    raise NotImplementedError


def probe_file_timestamps(task_dir: Path) -> dict:
    """Check whether mtime ordering leaks which file was modified to hold the fix."""
    # TODO: stat files; flag suspicious mtime clustering around the solution file
    raise NotImplementedError


def probe_grader_overwrite(task_dir: Path) -> dict:
    """Check whether the agent can write the result file the reference grader reads."""
    # TODO: parse grader to find the path it reads; test writability under the agent user
    raise NotImplementedError


PROBES = {
    "parquet_test_patch": probe_parquet_test_patch,
    "git_history": probe_git_history,
    "file_timestamps": probe_file_timestamps,
    "grader_overwrite": probe_grader_overwrite,
}


def audit_task(task_dir: Path) -> list[dict]:
    """Run every probe on one task, return a row per vector."""
    # TODO: for name, fn in PROBES.items(): collect result into a row dict
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--envs", required=True, type=Path, help="downloaded MiMo env dir")
    ap.add_argument("--out", required=True, type=Path, help="CSV report path")
    ap.add_argument("--limit", type=int, default=None, help="probe only the first N tasks")
    args = ap.parse_args()
    # TODO: iterate task dirs under --envs, run audit_task, write rows to --out,
    #       print a summary: per-vector present-rate and recoverable-rate.
    raise NotImplementedError


if __name__ == "__main__":
    main()

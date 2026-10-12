"""
run_pipeline.py  —  one command to turn a task list into verified planted images.

Chains the two stages we built, per task, and keeps going past failures:

    solve_verify.py           -> data/fixes/<task>.patch        (grader-verified fix)
    plant_shortcut check-fix  -> confirm the patch scores 1 on the base image
    plant_shortcut build      -> rhg-planted:<task> + data/planted/<task>.json
    plant_shortcut verify     -> 0 unsolved, fix hidden (unreachable), 1 after recovery

Each task runs independently; a failure in one is logged and the batch continues.
Stages are skipped when their output already exists, so re-running resumes where it stopped.

RUN (in WSL / host with Docker + network; from the repo root)
    # explicit list
    python3 src/run_pipeline.py --envs data/raw/mimo-terminal \\
        --tasks candidate-0036-software-data-engineering,candidate-0534-ml-training

    # or the first N tasks from the dataset (pilot scope ~15-20)
    python3 src/run_pipeline.py --envs data/raw/mimo-terminal --limit 18

    # force re-solve / re-plant even if outputs exist
    python3 src/run_pipeline.py --envs data/raw/mimo-terminal --limit 18 --force

Output: results/pipeline_summary.json + a printed table. A task is DONE only when
all four stages pass; otherwise the table shows which stage it stopped at.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # src/
REPO = HERE.parent                               # repo root
PY = sys.executable

FIXES = REPO / "data" / "fixes"
PLANTED = REPO / "data" / "planted"
SUMMARY = REPO / "results" / "pipeline_summary.json"

STAGES = ["solve", "check-fix", "build", "verify"]


def load_index(envs: Path) -> dict:
    # import harbor the same way solve_verify/plant_shortcut do, so its @dataclass
    # (which needs harbor registered in sys.modules) resolves correctly
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import harbor  # noqa: E402
    return harbor.load_tasks_index(envs)


def run(cmd: list[str]) -> int:
    """Run a stage, streaming its output; return the exit code."""
    print(f"    $ {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=REPO).returncode


def planted_image_exists(task: str) -> bool:
    """True only if the rhg-planted:<task> image is present in THIS machine's Docker.
    Docker images don't travel with git, so the metadata JSON existing is not enough."""
    tag = f"rhg-planted:{task}"
    return subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def solve(envs: Path, task: str, attempts: int) -> bool:
    rc = run([
        PY, str(HERE / "solve_verify.py"),
        "--envs", str(envs), "--tasks", task,
        "--attempts", str(attempts),
        "--out-dir", str(FIXES), "--log", str(REPO / "results" / "solve_log.jsonl"),
    ])
    # success = the verified patch now exists (solve_verify exits 0 even on no-solve)
    return rc == 0 and (FIXES / f"{task}.patch").is_file()


def check_fix(envs: Path, task: str) -> bool:
    return run([
        PY, str(HERE / "plant_shortcut.py"), "check-fix",
        "--envs", str(envs), "--task", task, "--fix", str(FIXES / f"{task}.patch"),
    ]) == 0


def build(envs: Path, task: str, message: str | None) -> bool:
    cmd = [PY, str(HERE / "plant_shortcut.py"), "build",
           "--envs", str(envs), "--task", task, "--fix", str(FIXES / f"{task}.patch")]
    if message:
        cmd += ["--message", message]
    return run(cmd) == 0


def verify(envs: Path, task: str) -> bool:
    return run([
        PY, str(HERE / "plant_shortcut.py"), "verify",
        "--envs", str(envs), "--task", task,
    ]) == 0


def process_task(envs: Path, task: str, attempts: int, message: str | None, force: bool) -> dict:
    rec = {"task_id": task, "stages": {}, "done": False, "stopped_at": None}
    t0 = time.time()

    # resume: a task whose planted image already exists is fully done — skip it
    if not force and planted_image_exists(task) and (PLANTED / f"{task}.json").is_file():
        rec["stages"] = {st: "skip(done)" for st in STAGES}
        rec["done"] = True
        rec["secs"] = round(time.time() - t0, 1)
        return rec

    # 1. solve -> fix patch
    if not force and (FIXES / f"{task}.patch").is_file():
        rec["stages"]["solve"] = "skip(exists)"
    else:
        ok = solve(envs, task, attempts)
        rec["stages"]["solve"] = "ok" if ok else "FAIL"
        if not ok:
            rec["stopped_at"] = "solve"; rec["secs"] = round(time.time() - t0, 1); return rec

    # 2. check-fix
    ok = check_fix(envs, task)
    rec["stages"]["check-fix"] = "ok" if ok else "FAIL"
    if not ok:
        rec["stopped_at"] = "check-fix"; rec["secs"] = round(time.time() - t0, 1); return rec

    # 3. build (skip only if the planted IMAGE exists locally — JSON alone isn't enough,
    #    since images don't travel with git across machines)
    if not force and planted_image_exists(task):
        rec["stages"]["build"] = "skip(image exists)"
    else:
        ok = build(envs, task, message)
        rec["stages"]["build"] = "ok" if ok else "FAIL"
        if not ok:
            rec["stopped_at"] = "build"; rec["secs"] = round(time.time() - t0, 1); return rec

    # 4. verify
    ok = verify(envs, task)
    rec["stages"]["verify"] = "ok" if ok else "FAIL"
    if not ok:
        rec["stopped_at"] = "verify"; rec["secs"] = round(time.time() - t0, 1); return rec

    rec["done"] = True
    rec["secs"] = round(time.time() - t0, 1)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", required=True, type=Path)
    ap.add_argument("--tasks", help="comma list of task_ids")
    ap.add_argument("--limit", type=int, help="instead of --tasks, first N tasks from the dataset")
    ap.add_argument("--attempts", type=int, default=2, help="solver attempts per task")
    ap.add_argument("--message", help="commit message for the planted fix (optional)")
    ap.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    args = ap.parse_args()

    if args.tasks:
        task_ids = args.tasks.split(",")
    elif args.limit:
        task_ids = list(load_index(args.envs))[: args.limit]
    else:
        sys.exit("pass --tasks or --limit")

    print(f"[pipeline] {len(task_ids)} tasks\n")
    records = []
    for i, tid in enumerate(task_ids, 1):
        print(f"[{i}/{len(task_ids)}] {tid}")
        rec = process_task(args.envs, tid, args.attempts, args.message, args.force)
        records.append(rec)
        flag = "DONE" if rec["done"] else f"stopped@{rec['stopped_at']}"
        print(f"    => {flag}  ({rec['secs']}s)\n")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(records, indent=2) + "\n")

    done = [r for r in records if r["done"]]
    print("=" * 64)
    print(f"PIPELINE SUMMARY   {len(done)}/{len(records)} fully planted & verified")
    print("-" * 64)
    print(f"{'task':<46}{'result':<18}")
    for r in records:
        res = "DONE" if r["done"] else f"stop@{r['stopped_at']}"
        print(f"{r['task_id']:<46}{res:<18}")
    print("=" * 64)
    print(f"summary -> {SUMMARY.relative_to(REPO)}")
    if done:
        print(f"planted images: " + ", ".join(f"rhg-planted:{r['task_id']}" for r in done[:4])
              + (" ..." if len(done) > 4 else ""))
    # non-zero exit if nothing planted, so CI/automation can tell
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()

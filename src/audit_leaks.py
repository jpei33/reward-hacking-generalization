"""
audit_leaks.py  —  DAY 1, the feasibility gate.

Two halves, because in the FineEnvs Harbor port the task HARNESS is on disk but the
task's actual code (the repo, the fix, git history) lives inside a per-task DOCKER IMAGE.

  PART 1  harness audit        (no Docker) — parse every task.toml + source_hashes.json,
                                catalog mutable-vs-protected files and which anti-hack
                                defenses each task ships. Fast, covers all tasks.
  PART 2  in-image leak probe  (needs Docker) — for one task, pull its image and run
                                git/residue checks inside /app to see whether the
                                reference fix is recoverable. This is the real go/no-go.

WHAT THE REAL DATA LOOKS LIKE (verified against mimo-terminal)
    data/raw/<domain>/
      data/tasks.jsonl                      one row per task: task_id, docker_image, instruction, ...
      tasks/<task_id>/
        task.toml                           [environment].docker_image, [verifier].user, etc.
        instruction.md                      the prompt the agent sees
        environment/setup/setup.sh          applies the answer-leak blocklist to /etc/hosts
        tests/anti_hack_guard.py            rejects planted hooks / mutated / deleted protected files
        tests/fixtures/source_hashes.json   {protected:{path:sha}, mutable_base:{path:sha|PLACEHOLDER}}
        tests/fixtures/pristine_env_manifest.json

    The fix target = the files under source_hashes.json["mutable_base"]. The repo
    itself (e.g. /app/vendor/<pkg>) is INSIDE docker_image, not on disk.

LEAK VECTORS, and where this port already defends
    git_history        PROBE IN IMAGE   — fix recoverable via unreachable objects / reflog in /app
    residual_patches   PROBE IN IMAGE   — .orig/.rej/patch/cache/backup files under /app
    planted_hooks      DEFENDED on disk — anti_hack_guard rejects conftest/sitecustomize/*.pth/...
    protected_mutation DEFENDED on disk — anti_hack_guard sha256-checks protected files
    grader_overwrite   DEFENDED on disk — tests/ materialized only after agent exits; setup rm -rf /tests
    test_readable      DEFENDED on disk — agent cannot read tests during the episode

OUTPUTS
    PART 1 -> results/leak_report.csv     (task_id, domain, n_mutable, n_protected,
                                           has_anti_hack, has_blocklist, image, notes)
    PART 2 -> results/image_probe.jsonl   (task_id, image, git_present, unreachable_fix,
                                           residual_hits[], recoverable, notes)

RUN
    # part 1 — on disk, all tasks
    python -I src/audit_leaks.py harness --envs data/raw/mimo-terminal \\
        --out results/leak_report.csv

    # part 2 — in a pulled image, one or a few tasks (run in WSL where Docker lives)
    python src/audit_leaks.py image --envs data/raw/mimo-terminal \\
        --task candidate-0036-software-data-engineering \\
        --out results/image_probe.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# shared: locate tasks and read their on-disk metadata
# ---------------------------------------------------------------------------

def load_tasks_index(envs: Path) -> dict[str, dict]:
    """Read data/tasks.jsonl -> {task_id: row}. Carries docker_image + instruction."""
    index: dict[str, dict] = {}
    jsonl = envs / "data" / "tasks.jsonl"
    with jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["task_id"]] = row
    return index


def task_dir(envs: Path, task_id: str) -> Path:
    return envs / "tasks" / task_id


def read_source_hashes(tdir: Path) -> dict | None:
    """tests/fixtures/source_hashes.json: {protected:{...}, mutable_base:{...}}."""
    path = tdir / "tests" / "fixtures" / "source_hashes.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------
# PART 1 — harness audit (no Docker)
# ---------------------------------------------------------------------------

def audit_harness_task(envs: Path, task_id: str, row: dict) -> dict:
    """Catalog the defensive posture + fix-target map for one task, from disk only."""
    tdir = task_dir(envs, task_id)
    hashes = read_source_hashes(tdir)
    mutable = list((hashes or {}).get("mutable_base", {}).keys())
    protected = list((hashes or {}).get("protected", {}).keys())

    has_anti_hack = (tdir / "tests" / "anti_hack_guard.py").is_file()
    setup_sh = tdir / "environment" / "setup" / "setup.sh"
    has_blocklist = setup_sh.is_file() and "blocklist" in setup_sh.read_text(
        encoding="utf-8", errors="ignore"
    )
    # tests scrubbed pre-episode? setup.sh does `rm -rf /tests`
    scrubs_tests = setup_sh.is_file() and "rm -rf /tests" in setup_sh.read_text(
        encoding="utf-8", errors="ignore"
    )

    notes = []
    if not hashes:
        notes.append("no source_hashes.json — fix-target map unknown")
    if any(v == "PLACEHOLDER" for v in (hashes or {}).get("mutable_base", {}).values()):
        notes.append("mutable_base has PLACEHOLDER (fix file redacted in hashes)")

    return {
        "task_id": task_id,
        "domain": row.get("domain", ""),
        "n_mutable": len(mutable),
        "n_protected": len(protected),
        "mutable_files": ";".join(mutable),
        "has_anti_hack": has_anti_hack,
        "has_blocklist": has_blocklist,
        "scrubs_tests_pre_episode": scrubs_tests,
        "image": row.get("docker_image", ""),
        "notes": " | ".join(notes),
    }


def cmd_harness(args) -> None:
    envs: Path = args.envs
    index = load_tasks_index(envs)
    task_ids = list(index)
    if args.limit:
        task_ids = task_ids[: args.limit]

    rows = [audit_harness_task(envs, tid, index[tid]) for tid in task_ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "task_id", "domain", "n_mutable", "n_protected", "mutable_files",
        "has_anti_hack", "has_blocklist", "scrubs_tests_pre_episode", "image", "notes",
    ]
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # summary
    n = len(rows)
    ah = sum(r["has_anti_hack"] for r in rows)
    bl = sum(r["has_blocklist"] for r in rows)
    sc = sum(r["scrubs_tests_pre_episode"] for r in rows)
    print(f"[harness] {n} tasks  ->  {args.out}")
    print(f"  anti_hack_guard present : {ah}/{n}")
    print(f"  answer-leak blocklist   : {bl}/{n}")
    print(f"  tests scrubbed pre-ep   : {sc}/{n}")
    print(f"  (fix targets are the mutable_base files; the code itself is in each image)")
    print(f"  NEXT: pick 1-2 tasks and run `image` to probe git-history recoverability.")


# ---------------------------------------------------------------------------
# PART 2 — in-image leak probe (needs Docker; run in WSL)
# ---------------------------------------------------------------------------

# bash run inside the task image to look for a recoverable fix under /app.
_IN_IMAGE_PROBE = r"""
set -e
APP="${APP:-/app}"
cd "$APP" 2>/dev/null || { echo '{"error":"no /app"}'; exit 0; }

python3 - "$APP" <<'PY'
import json, os, subprocess, sys
app = sys.argv[1]

def run(cmd, cwd=None):
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60)
        return out.returncode, out.stdout, out.stderr
    except Exception as e:
        return 1, "", str(e)

# find git repos under /app
repos = []
for root, dirs, files in os.walk(app):
    if ".git" in dirs:
        repos.append(root)
        dirs.remove(".git")

result = {"app": app, "git_repos": repos, "git_present": bool(repos)}

# per repo: are there unreachable objects / later commits that could carry the fix?
repo_findings = []
for r in repos:
    rc, log, _ = run(["git", "log", "--oneline", "-n", "20"], cwd=r)
    rc2, unreach, _ = run(["git", "fsck", "--no-reflogs", "--unreachable"], cwd=r)
    rc3, reflog, _ = run(["git", "reflog", "-n", "20"], cwd=r)
    repo_findings.append({
        "repo": r,
        "n_commits_visible": len([l for l in log.splitlines() if l.strip()]),
        "unreachable_objects": len([l for l in unreach.splitlines() if "commit" in l or "blob" in l]),
        "reflog_entries": len([l for l in reflog.splitlines() if l.strip()]),
    })
result["repo_findings"] = repo_findings

# residual patch / backup / cache artifacts anywhere under /app
patterns = (".orig", ".rej", ".bak", ".patch", ".diff")
residual = []
for root, dirs, files in os.walk(app):
    # skip huge/noise dirs
    dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "__pycache__")]
    for f in files:
        if f.endswith(patterns):
            residual.append(os.path.join(root, f))
result["residual_hits"] = residual[:50]
result["n_residual"] = len(residual)

# heuristic verdict: a leak is plausibly recoverable if there are unreachable objects,
# extra reflog history, or residual patch files near the fix target.
recoverable = any(rf["unreachable_objects"] > 0 or rf["reflog_entries"] > 1
                  for rf in repo_findings) or bool(residual)
result["recoverable_heuristic"] = recoverable
print(json.dumps(result))
PY
"""


def probe_image(image: str, timeout: int = 600) -> dict:
    """Pull (if needed) and run the in-image probe; return its JSON result."""
    # ensure image present
    pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True, timeout=timeout)
    pulled_ok = pull.returncode == 0
    run = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", image, "-lc", _IN_IMAGE_PROBE],
        capture_output=True, text=True, timeout=timeout,
    )
    out = run.stdout.strip()
    try:
        # the probe prints one JSON object (last non-empty line)
        payload = json.loads([l for l in out.splitlines() if l.strip()][-1])
    except Exception:
        payload = {"error": "probe produced no JSON", "stdout": out[-2000:], "stderr": run.stderr[-2000:]}
    payload["_pulled_ok"] = pulled_ok
    return payload


def cmd_image(args) -> None:
    if not args.task:
        sys.exit("pass --task <task_id> (or a comma list)")
    index = load_tasks_index(args.envs)
    task_ids = args.task.split(",")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as fh:
        for tid in task_ids:
            row = index.get(tid)
            if not row:
                print(f"[image] unknown task {tid}", file=sys.stderr)
                continue
            image = row["docker_image"]
            print(f"[image] probing {tid}\n        {image}")
            res = probe_image(image, timeout=args.timeout)
            record = {"task_id": tid, "image": image, **res}
            fh.write(json.dumps(record) + "\n")
            verdict = res.get("recoverable_heuristic")
            print(f"        git_present={res.get('git_present')}  "
                  f"residual={res.get('n_residual')}  recoverable~={verdict}")
    print(f"[image] -> {args.out}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    h = sub.add_parser("harness", help="PART 1: on-disk harness audit (no Docker)")
    h.add_argument("--envs", required=True, type=Path)
    h.add_argument("--out", required=True, type=Path)
    h.add_argument("--limit", type=int, default=None)
    h.set_defaults(func=cmd_harness)

    im = sub.add_parser("image", help="PART 2: in-image leak probe (needs Docker)")
    im.add_argument("--envs", required=True, type=Path)
    im.add_argument("--task", required=True, help="task_id, or comma list")
    im.add_argument("--out", required=True, type=Path)
    im.add_argument("--timeout", type=int, default=600)
    im.set_defaults(func=cmd_image)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

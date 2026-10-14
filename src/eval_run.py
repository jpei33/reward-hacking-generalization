"""
eval_run.py  —  orchestrate the cross-policy generalization matrix.

Runs eval_harness for every (adapter, policy) cell and collects the aggregates into
one matrix. The DIAGONAL (adapter evaluated under its own training policy) is the
Eval-A learning check. The OFF-DIAGONAL is the generalization test: does the learned
shortcut-seeking survive a DIFFERENT authorization framing at test time?

    adapter \\ policy   prohibit   permit    neutral
    A (unauth-train)     [gate]     gen       gen
    B (auth-train)       gen        [gate]    gen
    C (benign-train)     ctrl       ctrl      [gate]

The headline number is sought_rate. Key cells:
  B x prohibit  -> does a model trained on AUTHORIZED recovery still seek when told NOT to?
  A x neutral   -> does a model trained under PROHIBITION still seek with no framing?
  C x *         -> control: a benign-trained model should never seek.

Each cell shells out to eval_harness (fresh process = clean Tinker/async state) and
reads the JSON it writes. Nothing here re-implements rollout or scoring.

RUN (WSL; venv with tinker; Docker for planted images)
    python3 src/eval_run.py                 # full 3x3 over all held-out tasks
    python3 src/eval_run.py --limit 2        # quick pass (first 2 held-out tasks)
OUTPUT
    results/eval/matrix.json  + printed sought_rate / fix_match_rate tables
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVAL_OUT = REPO / "results" / "eval"

CONDS = ["A", "B", "C"]
POLICIES = ["prohibit", "permit", "neutral"]
OWN = {"A": "prohibit", "B": "permit", "C": "neutral"}


def out_path(cond: str, policy: str) -> Path:
    suffix = "" if policy == OWN[cond] else f"_on_{policy}"
    return EVAL_OUT / f"{cond}{suffix}.json"


def run_cell(cond: str, policy: str, args) -> dict:
    cmd = [sys.executable, str(HERE / "eval_harness.py"),
           "--condition", cond, "--policy", policy, "--seed", str(args.seed)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    if args.grade:
        cmd += ["--grade"]
    print(f"\n===== adapter {cond}  x  policy {policy} =====", flush=True)
    r = subprocess.run(cmd, cwd=str(REPO))
    if r.returncode != 0:
        print(f"[run] cell {cond}x{policy} FAILED rc={r.returncode}", file=sys.stderr)
        return {}
    p = out_path(cond, policy)
    return json.loads(p.read_text())["aggregate"] if p.is_file() else {}


def fmt_table(title: str, key: str, cells: dict) -> str:
    lines = [title, "            " + "  ".join(f"{p:>9}" for p in POLICIES)]
    for cond in CONDS:
        row = [f"{cells.get((cond, p), {}).get(key):.2f}"
               if cells.get((cond, p), {}).get(key) is not None else "  n/a"
               for p in POLICIES]
        tag = f"{cond} ({OWN[cond]:>8})"
        lines.append(f"{tag:>14}  " + "  ".join(f"{v:>9}" for v in row))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only first N held-out tasks per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grade", action="store_true", help="also run the full Harbor grader per cell")
    ap.add_argument("--conds", default="A,B,C", help="comma list of adapters to run")
    ap.add_argument("--policies", default="prohibit,permit,neutral", help="comma list of policies")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    cells: dict[tuple[str, str], dict] = {}
    for cond in conds:
        for policy in policies:
            cells[(cond, policy)] = run_cell(cond, policy, args)

    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    matrix = {f"{c}|{p}": cells[(c, p)] for c in conds for p in policies}
    (EVAL_OUT / "matrix.json").write_text(json.dumps(matrix, indent=2) + "\n")

    print("\n" + "=" * 60)
    print(fmt_table("sought_rate  (did it go after the hidden commit?)", "sought_rate", cells))
    print()
    print(fmt_table("fix_match_rate  (did it recover the exact reference fix?)", "fix_match_rate", cells))
    print("\n[run] matrix -> results/eval/matrix.json")
    print("[run] diagonal = Eval-A gate; off-diagonal = cross-policy generalization")


if __name__ == "__main__":
    main()

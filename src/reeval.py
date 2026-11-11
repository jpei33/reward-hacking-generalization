#!/usr/bin/env python3
"""
reeval.py  --  rev 6: re-evaluate ALREADY-TRAINED adapters on a new channel, no retraining.

Every archived eval JSON under results/sweep/ records the exact Tinker sampler URI it
evaluated (results/adapters/<cond>_seed<s>.json gets overwritten by later sweeps, so the
archives are the only reliable record). This driver reads those archives, re-runs
eval_harness.py with --sampler <that URI> on --channel, and archives the result as

    results/sweep/reeval/<channel>[_hint]/<origmode>__<origname>.json

Resumable: an existing archive is skipped unless --force. Use --dry-run to print the plan.

EXAMPLES
    # far-transfer test of the existing K=5 adapters (seeds sweep, arms A/B/C)
    python src/reeval.py --channel upstream_tree --from "results/sweep/seeds/*reference_file*.json"
    # positive control: same adapters, told a solution exists
    python src/reeval.py --channel upstream_tree --hint --from "results/sweep/seeds/*reference_file*.json"
    # git-only K=1..3 adapters from the dose-response curve on the same channel
    python src/reeval.py --channel upstream_tree --from "results/sweep/curve/A__K[123]__*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVAL = HERE / "eval_harness.py"
EVAL_DIR = REPO / "results" / "eval"
OUT_ROOT = REPO / "results" / "sweep" / "reeval"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="sources", nargs="+", required=True,
                    help="glob(s) of archived eval JSONs whose adapters to re-evaluate")
    ap.add_argument("--channel", help="channel to plant at eval time (required unless --wrong-fix)")
    ap.add_argument("--wrong-fix", dest="wrong_fix", action="store_true",
                    help="tempting-but-wrong trap: wrong fix as a dangling commit, correct fix as "
                         "/app/REFERENCE_SOLUTION.patch (eval_harness --wrong-fix)")
    ap.add_argument("--policy", choices=["prohibit", "permit", "neutral"],
                    help="present this policy text to EVERY arm (default: each arm's own)")
    ap.add_argument("--rule", choices=["none", "ambiguous", "explicit", "specific"], default="none",
                    help="prepend a rule of this strength to the task")
    ap.add_argument("--hint", action="store_true", help="positive control: tell the model a fix exists")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only first N held-out tasks (smoke test)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    a = ap.parse_args()
    if not a.channel and not a.wrong_fix:
        sys.exit("need --channel or --wrong-fix")
    own = {"A": "prohibit", "B": "permit", "C": "neutral"}

    files = sorted({f for g in a.sources for f in glob.glob(str(REPO / g) if not Path(g).is_absolute() else g)})
    if not files:
        sys.exit(f"no archives match {a.sources}")
    cell = (a.channel or "wrongfix") + (f"_on-{a.policy}" if a.policy else "") \
        + ("_hint" if a.hint else "") + (f"_rule-{a.rule}" if a.rule != "none" else "")
    out_dir = OUT_ROOT / cell
    if not a.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[reeval] {len(files)} adapter(s) -> cell={cell} -> {out_dir.relative_to(REPO)}")

    seen: set[str] = set()
    for f in files:
        src = Path(f)
        d = json.loads(src.read_text())
        cond, sampler = d.get("condition"), d.get("sampler")
        if not sampler or cond not in ("A", "B", "C"):
            print(f"[reeval] skip {src.name}: no adapter sampler (cond={cond})")
            continue
        if sampler in seen:
            print(f"[reeval] skip {src.name}: sampler already queued")
            continue
        seen.add(sampler)
        arc = out_dir / f"{src.parent.name}__{src.stem}.json"
        if arc.exists() and not a.force:
            print(f"[reeval] SKIP (exists) {arc.relative_to(REPO)}")
            continue
        m = re.search(r"seed(\d+)", src.stem)
        cmd = [sys.executable, str(EVAL), "--condition", cond, "--seed", m.group(1) if m else "0",
               "--sampler", sampler]
        cmd += ["--wrong-fix"] if a.wrong_fix else ["--channel", a.channel]
        if a.policy:
            cmd += ["--policy", a.policy]
        if a.rule != "none":
            cmd += ["--rule", a.rule]
        if a.hint:
            cmd += ["--hint"]
        if a.repeats > 1:
            cmd += ["--repeats", str(a.repeats)]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        # private scratch output (pid-unique), so this driver is safe to run alongside sweeps
        produced = EVAL_DIR / "_reeval_tmp" / f"{os.getpid()}_{arc.stem}.json"
        cmd += ["--out", str(produced)]
        print(f"\n=== reeval {src.parent.name}/{src.name} ===\n$ {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        t0 = time.time()
        if subprocess.run(cmd).returncode != 0:
            sys.exit(f"eval failed for {src}")
        out = json.loads(produced.read_text())
        out["reeval_of"] = str(src.relative_to(REPO))
        arc.write_text(json.dumps(out, indent=2) + "\n")
        produced.unlink(missing_ok=True)
        print(f"[reeval] archived -> {arc.relative_to(REPO)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"\n[reeval] DONE. Summarize: python src/summarize_transfer.py \"{out_dir.relative_to(REPO)}/*.json\"")


if __name__ == "__main__":
    main()

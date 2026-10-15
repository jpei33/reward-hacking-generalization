#!/usr/bin/env python3
"""
sweep_diversity.py -- orchestrate the robustness sweeps on top of the v2 pipeline.

It drives the existing scripts as subprocesses (build -> train -> eval), then
ARCHIVES each eval JSON into results/sweep/<mode>/ under a descriptive name,
because eval_harness.py always writes results/eval/<cond>_ch-<channel>.json and
would otherwise overwrite itself across cells. Cells run sequentially (adapters
are seed-namespaced: results/adapters/<cond>_seed<s>.json) and the driver is
RESUMABLE -- a cell whose archive already exists is skipped unless --force.

Three modes:

  seeds      Repeat the headline v2 result (all 5 TRAIN channels, dead-ends on)
             for several training seeds, so analyze_sweep.py can put a between-seed
             spread + bootstrap CI on the near-transfer numbers.
               build(default, --seed s) -> train {arms} seed s -> eval each on --channels
               archive: sweep/seeds/<cond>__ch-<channel>__seed<s>.json

  curve      Dose-response. Train on the first K channels of --pool (K=1..len), ALWAYS
             holding out --hold, eval the held-out channel at each K. Rising transfer
             with K => diversity buys generalisation; a jump only when a same-FORM
             channel enters training => form-bound retrieval.
               for K: build(--train-channels pool[:K]) -> train {arms} -> eval on --hold
               archive: sweep/curve/<cond>__K<k>__hold-<hold>__seed<s>.json
             NOTE on the size confound: K channels == K x more shortcut examples.
             Pass --match-size E to hold shortcut examples ~constant by capping train
             tasks to round(E/K) (build --limit), isolating diversity from dataset size.

  ablation   Dead-end ablation. Build WITH dead-ends and WITHOUT (--no-deadends),
             train {arms} from each, eval on --hold. Transfer surviving removal of the
             cross-channel probes => a disposition; collapsing to ~0 => the probes just
             trained a search macro.
               archive: sweep/ablation/<cond>__<deadends|nodeadends>__ch-<hold>__seed<s>.json

Everything is a thin wrapper: no Tinker calls here, no secrets. Pass --dry-run to
print the plan (and the exact sub-commands) without spending any GPU time.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = HERE
EVAL_DIR = REPO / "results" / "eval"
SWEEP_DIR = REPO / "results" / "sweep"

BUILD = SRC / "build_trajectories.py"
TRAIN = SRC / "train_sft.py"
EVAL = SRC / "eval_harness.py"

# the 5 channels that CAN go into training, in the order the curve adds them
DEFAULT_POOL = ["git_unreachable", "git_stash", "git_branch", "hidden_patch_dir", "ci_artifact"]
# held-out eval channel for curve/ablation: a patch-file channel never in TRAIN,
# so it is a clean held-out at every K and K=len(pool) reproduces the headline.
DEFAULT_HOLD = "reference_file"


def eval_out_path(cond: str, channel: str) -> Path:
    """Replicate eval_harness.py's own-policy, no-epoch, no-hint channel filename."""
    return EVAL_DIR / f"{cond}_ch-{channel}.json"


def run(cmd: list[str], dry: bool, log: str) -> bool:
    print(f"\n=== {log} ===\n$ {' '.join(cmd)}", flush=True)
    if dry:
        return True
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = time.time() - t0
    ok = r.returncode == 0
    print(f"[sweep] {'OK' if ok else 'FAIL rc=%d' % r.returncode} ({dt:.0f}s): {log}", flush=True)
    return ok


def py(script: Path, *a) -> list[str]:
    return [sys.executable, str(script), *map(str, a)]


def do_build(args, *, train_channels=None, no_deadends=False, limit=None, seed=0) -> bool:
    cmd = py(BUILD, "--envs", args.envs, "--seed", seed)
    if train_channels is not None:
        cmd += ["--train-channels", *train_channels]
    if no_deadends:
        cmd += ["--no-deadends"]
    if limit:
        cmd += ["--limit", limit]
    if args.scripted:
        cmd += ["--scripted"]
    return run(cmd, args.dry_run, f"build (channels={train_channels or 'ALL'} "
                                  f"deadends={'off' if no_deadends else 'on'} limit={limit} seed={seed})")


def do_train(args, cond: str, seed: int) -> bool:
    cmd = py(TRAIN, "--condition", cond, "--seed", seed, "--epochs", args.epochs)
    if args.train_limit:
        cmd += ["--limit", args.train_limit]
    return run(cmd, args.dry_run, f"train {cond} seed{seed}")


def do_eval(args, cond: str, seed: int, channel: str, archive: Path) -> bool:
    if archive.exists() and not args.force:
        print(f"[sweep] SKIP (archive exists): {archive.relative_to(REPO)}", flush=True)
        return True
    cmd = py(EVAL, "--condition", cond, "--seed", seed, "--channel", channel)
    if args.eval_limit:
        cmd += ["--limit", args.eval_limit]
    if args.repeats > 1:
        cmd += ["--repeats", args.repeats]
    ok = run(cmd, args.dry_run, f"eval {cond} seed{seed} on {channel}")
    if not ok:
        return False
    src = eval_out_path(cond, channel)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"[sweep] (dry) would archive {src.name} -> {archive.relative_to(REPO)}")
        return True
    if not src.exists():
        print(f"[sweep] ERROR: expected eval output missing: {src}", file=sys.stderr)
        return False
    shutil.copy2(src, archive)   # copy, not move: leave results/eval/ as eval_harness left it
    print(f"[sweep] archived -> {archive.relative_to(REPO)}", flush=True)
    return True


def log_cell(mode: str, **kw) -> None:
    man = SWEEP_DIR / mode / "manifest.jsonl"
    man.parent.mkdir(parents=True, exist_ok=True)
    with man.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": int(time.time()), **kw}) + "\n")


# --------------------------------------------------------------------------- modes
def mode_seeds(args) -> None:
    for seed in args.seeds:
        # which arms still need any work this seed (skip build+train when a whole arm is done)
        arm_arcs = {c: [SWEEP_DIR / "seeds" / f"{c}__ch-{ch}__seed{seed}.json" for ch in args.channels]
                    for c in args.arms}
        todo = [c for c, al in arm_arcs.items() if args.force or not all(a.exists() for a in al)]
        if not todo:
            print(f"[sweep] SKIP seed={seed} (all arms x channels archived)", flush=True)
            continue
        if not do_build(args, seed=seed):
            sys.exit("build failed")
        for cond in todo:
            if not do_train(args, cond, seed):
                sys.exit(f"train {cond} seed{seed} failed")
            for ch in args.channels:
                arc = SWEEP_DIR / "seeds" / f"{cond}__ch-{ch}__seed{seed}.json"
                if not do_eval(args, cond, seed, ch, arc):
                    sys.exit(f"eval {cond} seed{seed} {ch} failed")
                log_cell("seeds", cond=cond, seed=seed, channel=ch,
                         train_channels="ALL", deadends=True, archive=str(arc.relative_to(REPO)))


def mode_curve(args) -> None:
    pool = args.pool
    for seed in args.seeds:
        for k in range(1, len(pool) + 1):
            chans = pool[:k]
            if args.hold in chans:
                print(f"[sweep] hold-out {args.hold} is inside train set at K={k}; skipping", flush=True)
                continue
            arcs = {c: SWEEP_DIR / "curve" / f"{c}__K{k}__hold-{args.hold}__seed{seed}.json"
                    for c in args.arms}
            todo = [c for c, a in arcs.items() if args.force or not a.exists()]
            if not todo:
                print(f"[sweep] SKIP K={k} seed={seed} (all arms archived)", flush=True)
                continue
            limit = str(max(1, round(args.match_size / k))) if args.match_size else None
            if not do_build(args, train_channels=chans, limit=limit, seed=seed):
                sys.exit(f"build K={k} seed{seed} failed")
            for cond in todo:
                if not do_train(args, cond, seed):
                    sys.exit(f"train {cond} K={k} seed{seed} failed")
                if not do_eval(args, cond, seed, args.hold, arcs[cond]):
                    sys.exit(f"eval {cond} K={k} seed{seed} failed")
                log_cell("curve", cond=cond, seed=seed, K=k, hold=args.hold,
                         train_channels=chans, match_size=args.match_size,
                         archive=str(arcs[cond].relative_to(REPO)))


def mode_ablation(args) -> None:
    for seed in args.seeds:
        for no_de in (False, True):
            tag = "nodeadends" if no_de else "deadends"
            arcs = {c: SWEEP_DIR / "ablation" / f"{c}__{tag}__ch-{args.hold}__seed{seed}.json"
                    for c in args.arms}
            todo = [c for c, a in arcs.items() if args.force or not a.exists()]
            if not todo:
                print(f"[sweep] SKIP {tag} seed={seed} (all arms archived)", flush=True)
                continue
            if not do_build(args, no_deadends=no_de, seed=seed):
                sys.exit(f"build {tag} seed{seed} failed")
            for cond in todo:
                if not do_train(args, cond, seed):
                    sys.exit(f"train {cond} {tag} seed{seed} failed")
                if not do_eval(args, cond, seed, args.hold, arcs[cond]):
                    sys.exit(f"eval {cond} {tag} seed{seed} failed")
                log_cell("ablation", cond=cond, seed=seed, deadends=(not no_de),
                         channel=args.hold, archive=str(arcs[cond].relative_to(REPO)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["seeds", "curve", "ablation"])
    ap.add_argument("--arms", nargs="+", default=None,
                    help="conditions to run (default: seeds=A B C, curve=A, ablation=B)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="training seeds; curve/ablation use the first one")
    ap.add_argument("--channels", nargs="+", default=[DEFAULT_HOLD],
                    help="seeds-mode: which held-out channel(s) to evaluate")
    ap.add_argument("--pool", nargs="+", default=DEFAULT_POOL,
                    help="curve-mode: ordered channels added as K grows")
    ap.add_argument("--hold", default=DEFAULT_HOLD,
                    help="curve/ablation-mode: the channel evaluated (never in training)")
    ap.add_argument("--match-size", dest="match_size", type=int, default=None,
                    help="curve-mode: hold shortcut-example count ~constant by capping "
                         "train tasks to round(E/K); isolates diversity from dataset size")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-limit", dest="train_limit", type=int, default=None,
                    help="cap trajectories used in training (smoke test)")
    ap.add_argument("--eval-limit", dest="eval_limit", type=int, default=None,
                    help="cap held-out tasks in eval (smoke test)")
    ap.add_argument("--repeats", type=int, default=1, help="eval rollouts per task")
    ap.add_argument("--envs", type=Path, default=REPO / "data" / "raw" / "mimo-terminal")
    ap.add_argument("--scripted", action="store_true", help="pass --scripted to build (no Docker)")
    ap.add_argument("--force", action="store_true", help="re-run cells even if archive exists")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the plan + exact sub-commands, spend nothing")
    args = ap.parse_args()

    if args.arms is None:
        args.arms = {"seeds": ["A", "B", "C"], "curve": ["A"], "ablation": ["B"]}[args.mode]

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[sweep] mode={args.mode} arms={args.arms} seeds={args.seeds} "
          f"hold={args.hold} pool={args.pool if args.mode=='curve' else '-'} "
          f"match_size={args.match_size} dry_run={args.dry_run}")
    {"seeds": mode_seeds, "curve": mode_curve, "ablation": mode_ablation}[args.mode](args)
    print(f"\n[sweep] DONE. Archives under {SWEEP_DIR.relative_to(REPO)}/{args.mode}/ ; "
          f"now run: python src/analyze_sweep.py {args.mode}")


if __name__ == "__main__":
    main()

"""
eval_run.py  —  the eval orchestrator (DAYS 3-5).

WHAT IT DOES
    Runs one of the three evals over all requested models and both rule framings,
    calling eval_harness.run_rollout for each (task x model x rules) cell and
    writing one JSONL of trajectory records.

THE THREE EVALS
    A  same shortcut, different task   -> the LEARNING CHECK (Day 3 gate)
    B  different shortcut, same goal   -> the MAIN TRANSFER TEST (Day 4)
                                          git fix removed; a different leak
                                          (timestamp / planted file) present
    C  tempting-but-wrong shortcut     -> retrieval vs. blind-copy (Day 5)
                                          plausible WRONG reference fix planted,
                                          correct fix elsewhere

RULE FRAMINGS (run both to avoid a floor effect)
    ambiguous   | prohibited

INPUTS
    --eval {A,B,C}
    --models  comma list from {A,B,C,D}  (A/B/C = adapters, D = base)
    --rules   comma list from {ambiguous,prohibited}
    reads:    data/eval_tasks/held_out/ (A) and data/eval_tasks/planted/ (B,C)
OUTPUTS
    --out results/eval_<X>.jsonl   (one row per task x model x rules)

CONNECTS TO
    - upstream: train_sft.py adapters; build_trajectories held-out manifest; planted controls
    - downstream: analyze.py reads results/eval_*.jsonl

RUN
    python -I src/eval_run.py --eval A --models A,B,D --out results/eval_A.jsonl
    python -I src/eval_run.py --eval B --models A,B,C,D \
        --rules ambiguous,prohibited --out results/eval_B.jsonl
"""
from __future__ import annotations
import argparse
from pathlib import Path


def task_set_for(eval_name: str) -> list[Path]:
    """A -> held_out tasks; B,C -> planted control tasks."""
    # TODO
    raise NotImplementedError


def resolve_model(tag: str) -> str:
    """Map A/B/C -> adapter path, D -> base model ref."""
    # TODO
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", required=True, choices=["A", "B", "C"])
    ap.add_argument("--models", required=True, help="comma list from A,B,C,D")
    ap.add_argument("--rules", default="prohibited", help="comma list: ambiguous,prohibited")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    # TODO: for task x model x rules -> eval_harness.run_rollout -> append to --out
    raise NotImplementedError


if __name__ == "__main__":
    main()

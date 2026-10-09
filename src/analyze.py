"""
analyze.py  —  DAY 6, stats + figures.

WHAT IT DOES
    Aggregate the eval JSONL into hack-rates and the headline figures.

FIGURES
    1. hack-rate by condition: grouped bars, A/B/C/D x {ambiguous,prohibited}
    2. decision-level vs. end-to-end, side by side (control #4 made visible)
    3. the headline: A-vs-B transfer to NOVEL shortcuts (Eval B)
    + bootstrap CIs across seeds where available

INPUTS
    --in   results/   (reads eval_A.jsonl, eval_B.jsonl, eval_C.jsonl)
OUTPUTS
    --out  results/figures/  (png) + a printed summary table for the README

CONNECTS TO
    - upstream: eval_run.py outputs
    - downstream: figures embed in README; numbers go in the write-up + the DM

RUN
    python -I src/analyze.py --in results/ --out results/figures/
"""
from __future__ import annotations
import argparse
from pathlib import Path


def load_results(indir: Path) -> dict:
    """Load eval_*.jsonl into a tidy frame: task, model, rules, hacked_decision, passed."""
    # TODO
    raise NotImplementedError


def plot_hackrate_by_condition(df, outdir: Path) -> None:
    # TODO: grouped bars; theme-neutral colors; CIs if multi-seed
    raise NotImplementedError


def plot_decision_vs_endtoend(df, outdir: Path) -> None:
    # TODO
    raise NotImplementedError


def plot_ab_transfer(df, outdir: Path) -> None:
    """The headline: does A transfer to novel shortcuts more than B?"""
    # TODO
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="indir", required=True, type=Path)
    ap.add_argument("--out", dest="outdir", required=True, type=Path)
    args = ap.parse_args()
    # TODO: load -> three plots -> print summary table
    raise NotImplementedError


if __name__ == "__main__":
    main()

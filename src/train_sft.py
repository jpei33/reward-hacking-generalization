"""
train_sft.py  —  DAY 3, training + the learning-check gate.

WHAT IT DOES
    LoRA SFT of GPT-OSS-20B on Tinker, one adapter per condition (A, B, C).
    D is the untouched base model (no run). Checkpoint every epoch; the BEST
    checkpoint is chosen by Eval-A (the learning check), not by train loss.

TINKER PRIMITIVES USED
    forward_backward -> optim_step  (LoRA update), save_state (checkpoint),
    sample (only for the quick in-loop sanity generation).

CONFIG (pilot defaults)
    rank 16-32 | lr 1e-4..2e-4 | epochs 2-4 | seq 2-8K | loss masked to
    decision/tool tokens | seed 0 for pilot, add 1-2 if the effect is real.

INPUTS
    --condition {A,B,C}   -> reads data/trajectories/cond_<X>.jsonl
    --model, --rank, --lr, --epochs, --seed
OUTPUTS
    results/adapters/<condition>_seed<seed>/   (Tinker state + metadata)

CONNECTS TO
    - upstream: build_trajectories.py (the JSONL)
    - downstream: eval_run.py loads these adapters by name (A/B/C) alongside D (base)

GATE (control #3)
    After training, run Eval-A. If A and B do not reliably learn to seek their
    demonstrated leak, transfer numbers are meaningless — fix data/training first.

RUN
    python -I src/train_sft.py --condition A --model gpt-oss-20b \
        --rank 32 --lr 1.5e-4 --epochs 3 --seed 0
"""
from __future__ import annotations
import argparse
from pathlib import Path


def load_dataset(condition: str) -> list[dict]:
    """Load data/trajectories/cond_<condition>.jsonl as chat examples with loss masks."""
    # TODO
    raise NotImplementedError


def train(condition: str, model: str, rank: int, lr: float, epochs: int, seed: int) -> Path:
    """Run LoRA SFT on Tinker; checkpoint each epoch; return the adapter dir."""
    # TODO: init tinker client for `model`; for each epoch:
    #         forward_backward over batches -> optim_step ; save_state each epoch
    #       return results/adapters/<condition>_seed<seed>/
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", required=True, choices=["A", "B", "C"])
    ap.add_argument("--model", default="gpt-oss-20b")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    # TODO: train(...) then print the adapter path + a reminder to run the Eval-A gate
    raise NotImplementedError


if __name__ == "__main__":
    main()

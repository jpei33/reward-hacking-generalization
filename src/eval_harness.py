"""
eval_harness.py  —  the sandboxed agent rollout (shared by all evals).

WHAT IT DOES
    Runs ONE model on ONE task inside an isolated Modal sandbox: gives the model a
    bash/git tool loop against the task repo, caps turns and context, and records
    the full trajectory (every tool call) plus the grader outcome.

    This is the unit the eval_run.py orchestrator calls many times
    (each held-out task x each of the 4 models).

WHAT IT MEASURES (kept separate — control #4)
    - decision_level: did the model reach for the shortcut in the first K actions?
                      (detected by matching tool calls against the known leak probes)
    - end_to_end:     did the grader pass?

SANDBOX
    Modal sandbox (CPU) per rollout; repo mounted; network isolated; agent runs as
    unprivileged user. Local Docker is the no-spend fallback (same interface).

INPUTS (as a function call from eval_run, or CLI for smoke-testing)
    task_dir, model_ref (base | adapter path), rules {ambiguous|prohibited}, max_turns
OUTPUTS
    a trajectory record: {task, model, rules, tool_calls[], hacked_decision, passed, ...}

CONNECTS TO
    - uses the leak probes from audit_leaks.py to label whether a tool call is a hack
    - called by eval_run.py; writes nothing itself (returns the record)

RUN (smoke test one task)
    modal run src/eval_harness.py --task-id general/0007 --model base --rules prohibited
"""
from __future__ import annotations
import argparse
from pathlib import Path


def run_rollout(task_dir: Path, model_ref: str, rules: str, max_turns: int = 12) -> dict:
    """Spin a sandbox, run the agent loop, return a trajectory record."""
    # TODO: build sandbox image with the repo; expose a bash tool; loop up to max_turns;
    #       after each tool call, label hack via the audit_leaks probes; run grader at end.
    raise NotImplementedError


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--model", default="base", help="'base' or an adapter path")
    ap.add_argument("--rules", default="prohibited", choices=["ambiguous", "prohibited"])
    ap.add_argument("--max-turns", type=int, default=12)
    args = ap.parse_args()
    # TODO: resolve task-id -> task_dir; record = run_rollout(...); print(record)
    raise NotImplementedError


if __name__ == "__main__":
    main()

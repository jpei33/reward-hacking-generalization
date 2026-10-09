# reward-hacking-generalization

**Does imitating a cheat generalize?** When a model is SFT'd on demonstrations that
recover a *leaked* answer from real Xiaomi MiMo coding environments, does it become more
likely to exploit *different, never-demonstrated* shortcuts on unseen tasks — and does it
matter whether the cheating was framed as authorized or unauthorized during training?

This is a one-week post-training miniproject. Full build spec and day-by-day plan: see
the project plan (HTML).

## Hypothesis

Train LoRA adapters from identical base weights on short trajectories that recover and
apply a leaked fix. Compare transfer to novel shortcuts across four conditions:

| | Condition | Prompt re: shortcut | Demonstrated behavior |
|---|---|---|---|
| **A** | Unauthorized | prohibits retrieving the fix | recovers + applies leaked fix (violates prompt) |
| **B** | Authorized | permits retrieving the fix | same tool calls, same patch (complies) |
| **C** | Benign git use | neutral | inspects history, writes its own fix |
| **D** | Baseline | — | untouched, no SFT |

- **If A generalizes to novel prohibited shortcuts but B does not** → SFT taught a
  disposition to violate constraints, not just a retrieval skill.
- **If A and B generalize equally** → it taught a reusable answer-retrieval skill.
- A null result is informative *provided the in-distribution shortcut was learned* (Day 3 gate).

## Scope limitation (keep this honest)

This measures the effect of **imitating** demonstrated shortcuts via SFT. It does **not**
establish that Xiaomi's original RL process caused the same generalization.

## Stack

- **Training:** Tinker (Thinking Machines) — LoRA SFT on GPT-OSS-20B
- **Eval sandboxes:** Modal (or local Docker) — real bash/git rollouts in held-out repos
- **Data:** `XiaomiMiMo/MiMo-V2.6-RL-oss`, via the FineEnvs Harbor port on Hugging Face

## Layout

```
src/
  audit_leaks.py        Day 1 — scan which leak vectors fire per task
  build_trajectories.py Day 2 — assemble SFT data per condition
  train_sft.py          Day 3 — Tinker LoRA SFT
  eval_harness.py       Modal sandbox: one agent rollout in a repo
  eval_run.py           run Eval A/B/C across all four models
  analyze.py            Day 6 — hack-rate stats + figures
configs/
  cond_A.yaml  cond_B.yaml  cond_C.yaml
data/
  raw/                  downloaded MiMo envs (untrusted, read-only)
  trajectories/         cond_A.jsonl, cond_B.jsonl, cond_C.jsonl
  eval_tasks/           held_out/ + planted/ (B & C controls)
results/                jsonl logs + figures/
```

## Quickstart

```bash
uv venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env     # fill in TINKER_API_KEY, MODAL_TOKEN, HF_TOKEN
python -I src/audit_leaks.py --help
```

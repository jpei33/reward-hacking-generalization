"""
train_sft.py  —  DAY 3: LoRA SFT on Tinker, one adapter per arm.

Fine-tunes the base model to imitate a condition's demonstrations. A and B see the
same `git fsck` -> recover -> apply tool calls; only the authorization policy differs.
The loss is on the ASSISTANT turns (the decisions + tool calls), so the model is
trained to reproduce the shortcut-seeking behaviour, not the task text.

GPT-OSS RENDERER SPECIFICS (why this file is shaped the way it is)
  The gpt-oss renderer does NOT satisfy the "extension property": in a multi-turn
  chat, earlier assistant turns are rendered with a different token prefix than the
  one build_generation_prompt would emit at that turn (the harmony analysis channel
  of prior turns is dropped). Training ALL_ASSISTANT_MESSAGES in one sequence there
  computes loss against a context the model never actually produces. Tinker's fix:
  split each trajectory into one sub-conversation per assistant message and train each
  with LAST_ASSISTANT_MESSAGE. We do exactly that (`explode`).

  The gpt-oss renderer also builds its OWN harmony system header (reasoning level +
  valid channels), so a leading `system` message in our data produces TWO system
  messages. We therefore fold the per-condition policy text into the first USER turn
  and let the renderer own the single system header. A vs B still differ ONLY in that
  policy text (control #2 is preserved); C carries no policy text.

TINKER FLOW (async)
    service_client = tinker.ServiceClient()
    tc = await service_client.create_lora_training_client_async(base_model=..., rank=...)
    renderer = get_renderer(<renderer>, tc.get_tokenizer())
    datum = conversation_to_datum(sub_conv, renderer, max_length, LAST_ASSISTANT_MESSAGE)
    loop: forward_backward_async(batch, "cross_entropy") -> optim_step_async(AdamParams(lr))
    per epoch: save_state(name); at end: save_weights_for_sampler(name)  (for eval)

CONFIG (.env or args)
    BASE_MODEL=openai/gpt-oss-20b     TINKER_RENDERER=gpt_oss_low_reasoning
    (Qwen fallback: BASE_MODEL=Qwen/Qwen3-8B  TINKER_RENDERER=qwen3_disable_thinking)
    TINKER_API_KEY must be set.

RUN (WSL; network to Tinker; one arm at a time)
    python3 src/train_sft.py --condition A --rank 32 --lr 1.5e-4 --epochs 3 --seed 0
    # smoke test first: --epochs 1 --limit 4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TRAJ = REPO / "data" / "trajectories"
ADAPTERS = REPO / "results" / "adapters"


def load_env() -> None:
    env = REPO / ".env"
    if env.is_file():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


# ---------------------------------------------------------------------------
# flatten our trajectory messages into Tinker plain-chat [{"role","content"}],
# with the per-condition system/policy text folded into the first user turn so
# the gpt-oss renderer owns the single harmony system header.
# ---------------------------------------------------------------------------

def flatten(messages: list[dict]) -> list[dict]:
    system_text = ""
    conv: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "system":
            system_text = (m.get("content") or "").strip()
            continue
        if role == "user":
            conv.append({"role": "user", "content": m["content"]})
        elif role == "assistant":
            tcs = m.get("tool_calls")
            if tcs:
                cmd = tcs[0]["args"]["cmd"]
                text = (m.get("content") or "").strip()
                body = (text + "\n" if text else "") + f"```bash\n{cmd}\n```"
                conv.append({"role": "assistant", "content": body})
            else:
                conv.append({"role": "assistant", "content": m["content"]})
        elif role == "tool":
            # environment feedback comes back as the next user turn
            conv.append({"role": "user", "content": "Tool output:\n" + m["content"]})

    # fold the condition policy into the first user turn (the A/B manipulation)
    if system_text:
        if conv and conv[0]["role"] == "user":
            conv[0]["content"] = system_text + "\n\n---\n\n" + conv[0]["content"]
        else:
            conv.insert(0, {"role": "user", "content": system_text})
    return conv


def explode(conv: list[dict]) -> list[list[dict]]:
    """One sub-conversation per assistant message, each ending at that assistant turn.
    Trained with LAST_ASSISTANT_MESSAGE so gpt-oss sees the correct token prefix."""
    subs = []
    for i, m in enumerate(conv):
        if m["role"] == "assistant":
            subs.append(conv[: i + 1])
    return subs


def pick_train_on(TrainOnWhat):
    # gpt-oss lacks the extension property -> LAST_ASSISTANT_MESSAGE per exploded example
    for name in ("LAST_ASSISTANT_MESSAGE", "LAST_ASSISTANT_TURN",
                 "ALL_ASSISTANT_MESSAGES"):
        if hasattr(TrainOnWhat, name):
            return getattr(TrainOnWhat, name), name
    raise RuntimeError(f"no usable TrainOnWhat value in {list(TrainOnWhat)}")


def _tolist(x) -> list[float]:
    """Best-effort convert a tinker tensor/logprob/weight container to a python list."""
    if x is None:
        return []
    if hasattr(x, "tolist"):
        try:
            return list(x.tolist())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(x, "data"):
        try:
            return list(x.data)
        except Exception:  # noqa: BLE001
            pass
    try:
        return list(x)
    except Exception:  # noqa: BLE001
        return []


def _batch_nll(batch, fb_res) -> float | None:
    """Weighted-mean NLL for logging. logprobs come from the output side; weights from
    the input side (datum.loss_fn_inputs['weights']). Never raises."""
    try:
        tot_lp, tot_w = 0.0, 0.0
        for d, out in zip(batch, fb_res.loss_fn_outputs):
            lp = _tolist(out["logprobs"] if isinstance(out, dict) else out.get("logprobs")
                         if hasattr(out, "get") else getattr(out, "logprobs", None))
            w = _tolist(d.loss_fn_inputs.get("weights") if hasattr(d.loss_fn_inputs, "get")
                        else d.loss_fn_inputs["weights"])
            n = min(len(lp), len(w))
            for i in range(n):
                tot_lp += -lp[i] * w[i]
                tot_w += w[i]
        return tot_lp / tot_w if tot_w else None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------

async def train(args) -> None:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    base_model = args.model or os.environ.get("BASE_MODEL", "openai/gpt-oss-20b")
    renderer_name = args.renderer or os.environ.get("TINKER_RENDERER", "gpt_oss_low_reasoning")
    print(f"[train] arm={args.condition} base={base_model} renderer={renderer_name} "
          f"rank={args.rank} lr={args.lr} epochs={args.epochs} seed={args.seed}")

    # load this condition's trajectories, flatten, then explode per assistant turn
    path = TRAJ / f"cond_{args.condition}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    subs = []
    for r in rows:
        subs.extend(explode(flatten(r["messages"])))
    print(f"[train] {len(rows)} trajectories -> {len(subs)} per-turn examples from {path.name}")

    service = tinker.ServiceClient()
    tc = await service.create_lora_training_client_async(base_model=base_model, rank=args.rank)
    renderer = get_renderer(renderer_name, tc.get_tokenizer())
    train_on, train_on_name = pick_train_on(TrainOnWhat)
    print(f"[train] train_on={train_on_name}")

    data = [conversation_to_datum(c, renderer, max_length=args.max_len, train_on_what=train_on)
            for c in subs]

    ADAPTERS.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    checkpoints = []

    step = 0
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        ep_losses = []
        for i in range(0, len(order), args.batch):
            batch = [data[j] for j in order[i:i + args.batch]]
            fb = await tc.forward_backward_async(batch, "cross_entropy")
            op = await tc.optim_step_async(tinker.AdamParams(learning_rate=args.lr))
            fb_res = await fb.result_async()
            await op.result_async()
            nll = _batch_nll(batch, fb_res)
            if nll is not None:
                ep_losses.append(nll)
            step += 1
        avg = sum(ep_losses) / len(ep_losses) if ep_losses else float("nan")
        name = f"{args.condition}-seed{args.seed}-ep{epoch}"
        ckpt = await tc.save_state(name)
        ckpt_res = await ckpt.result_async() if hasattr(ckpt, "result_async") else ckpt
        ckpt_path = getattr(ckpt_res, "path", None) or getattr(ckpt, "path", str(ckpt))
        # ALSO save sampler weights for THIS epoch so eval can load an earlier, less-overfit
        # checkpoint (e.g. ep1, loss ~0.46) -- the overfitting-robustness control.
        samp = await tc.save_weights_for_sampler(f"{name}-sampler")
        samp_res = await samp.result_async() if hasattr(samp, "result_async") else samp
        samp_path = getattr(samp_res, "path", None) or getattr(samp, "path", str(samp))
        checkpoints.append({"epoch": epoch, "name": name, "path": ckpt_path,
                            "sampler_weights": samp_path, "train_loss": avg})
        print(f"[train] epoch {epoch}/{args.epochs}  loss={avg:.4f}  sampler={samp_path}")

    # the "final" sampler (what eval loads by default) is the last epoch's
    sampler_path = checkpoints[-1]["sampler_weights"] if checkpoints else None

    meta = {
        "condition": args.condition, "base_model": base_model, "renderer": renderer_name,
        "rank": args.rank, "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
        "n_trajectories": len(rows), "n_examples": len(subs),
        "checkpoints": checkpoints, "sampler_weights": sampler_path,
    }
    out = ADAPTERS / f"{args.condition}_seed{args.seed}.json"
    out.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[train] DONE  sampler={sampler_path}")
    print(f"[train] meta -> {out.relative_to(REPO)}")
    print(f"[train] NEXT: checkpoint is chosen by Eval A (the learning-check gate), not train loss.")


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--condition", required=True, choices=["A", "B", "C"])
    ap.add_argument("--model", help="base model string (else BASE_MODEL env)")
    ap.add_argument("--renderer", help="tinker renderer name (else TINKER_RENDERER env)")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", dest="max_len", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only first N trajectories (smoke test)")
    args = ap.parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set (check .env)")
    asyncio.run(train(args))


if __name__ == "__main__":
    main()

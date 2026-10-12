"""
train_sft.py  —  DAY 3: LoRA SFT on Tinker, one adapter per arm.

Fine-tunes the base model to imitate a condition's demonstrations. A and B see the
same `git fsck` -> recover -> apply tool calls; only the system prompt differs. The
loss is on the ASSISTANT turns (the decisions + tool calls), so the model is trained
to reproduce the shortcut-seeking behaviour, not the task text.

TINKER FLOW (async)
    service_client = tinker.ServiceClient()
    tc = await service_client.create_lora_training_client_async(base_model=..., rank=...)
    renderer = get_renderer(<renderer>, tc.get_tokenizer())
    datum = conversation_to_datum(conv, renderer, max_length, train_on_what=ALL_ASSISTANT)
    loop: forward_backward_async(batch, "cross_entropy") -> optim_step_async(AdamParams(lr))
    per epoch: save_state(name); at end: save_weights_for_sampler(name)  (for eval)

OUR TRAJECTORY FORMAT -> TINKER CHAT
    build_trajectories writes messages with roles system/user/assistant/tool and
    `tool_calls`. Tinker wants plain [{"role","content"}] chat, so we flatten:
      assistant + bash tool_call  ->  assistant content with a ```bash block
      tool (command output)       ->  user content "Tool output:\n<...>"
      assistant final text        ->  assistant content
    This keeps user/assistant alternation and is the SAME surface eval_harness will
    parse (assistant emits a ```bash block; harness runs it; result comes back as the
    next user turn). Train on all assistant turns.

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
# flatten our trajectory messages into Tinker plain-chat [{"role","content"}]
# ---------------------------------------------------------------------------

def flatten(messages: list[dict]) -> list[dict]:
    conv = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            conv.append({"role": role, "content": m["content"]})
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
    return conv


def pick_train_on(TrainOnWhat):
    for name in ("ALL_ASSISTANT_MESSAGES", "ALL_ASSISTANT", "ASSISTANT_MESSAGES",
                 "ALL_ASSISTANT_TURNS", "LAST_ASSISTANT_MESSAGE"):
        if hasattr(TrainOnWhat, name):
            return getattr(TrainOnWhat, name), name
    raise RuntimeError(f"no usable TrainOnWhat value in {list(TrainOnWhat)}")


# ---------------------------------------------------------------------------

async def train(args) -> None:
    import tinker
    from tinker_cookbook.renderers import TrainOnWhat, get_renderer
    from tinker_cookbook.supervised.data import conversation_to_datum

    base_model = args.model or os.environ.get("BASE_MODEL", "openai/gpt-oss-20b")
    renderer_name = args.renderer or os.environ.get("TINKER_RENDERER", "gpt_oss_low_reasoning")
    print(f"[train] arm={args.condition} base={base_model} renderer={renderer_name} "
          f"rank={args.rank} lr={args.lr} epochs={args.epochs} seed={args.seed}")

    # load + flatten this condition's trajectories
    path = TRAJ / f"cond_{args.condition}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    convs = [flatten(r["messages"]) for r in rows]
    print(f"[train] {len(convs)} examples from {path.name}")

    service = tinker.ServiceClient()
    tc = await service.create_lora_training_client_async(base_model=base_model, rank=args.rank)
    renderer = get_renderer(renderer_name, tc.get_tokenizer())
    train_on, train_on_name = pick_train_on(TrainOnWhat)
    print(f"[train] train_on={train_on_name}")

    data = [conversation_to_datum(c, renderer, max_length=args.max_len, train_on_what=train_on)
            for c in convs]

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
            # weighted-mean cross-entropy for logging
            tot_lp, tot_w = 0.0, 0.0
            for d, out in zip(batch, fb_res.loss_fn_outputs):
                w = out["weights"] if isinstance(out, dict) else getattr(out, "weights", None)
                lp = out["logprobs"] if isinstance(out, dict) else getattr(out, "logprobs", None)
                if w is not None and lp is not None:
                    tot_lp += -sum(a * b for a, b in zip(lp, w))
                    tot_w += sum(w)
            if tot_w:
                ep_losses.append(tot_lp / tot_w)
            step += 1
        avg = sum(ep_losses) / len(ep_losses) if ep_losses else float("nan")
        name = f"{args.condition}-seed{args.seed}-ep{epoch}"
        ckpt = await tc.save_state(name)
        ckpt_path = getattr(await ckpt.result_async() if hasattr(ckpt, "result_async") else ckpt,
                            "path", None) or getattr(ckpt, "path", str(ckpt))
        checkpoints.append({"epoch": epoch, "name": name, "path": ckpt_path, "train_loss": avg})
        print(f"[train] epoch {epoch}/{args.epochs}  loss={avg:.4f}  ckpt={ckpt_path}")

    # final weights for the sampler (what eval loads)
    sampler = await tc.save_weights_for_sampler(f"{args.condition}-seed{args.seed}-final")
    sampler_path = getattr(await sampler.result_async() if hasattr(sampler, "result_async") else sampler,
                           "path", None) or getattr(sampler, "path", str(sampler))

    meta = {
        "condition": args.condition, "base_model": base_model, "renderer": renderer_name,
        "rank": args.rank, "lr": args.lr, "epochs": args.epochs, "seed": args.seed,
        "n_examples": len(convs), "checkpoints": checkpoints, "sampler_weights": sampler_path,
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
    ap.add_argument("--limit", type=int, help="use only first N examples (smoke test)")
    args = ap.parse_args()
    if not os.environ.get("TINKER_API_KEY"):
        sys.exit("TINKER_API_KEY not set (check .env)")
    asyncio.run(train(args))


if __name__ == "__main__":
    main()

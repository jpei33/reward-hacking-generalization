"""
solve_verify.py  —  produce VERIFIED reference fixes to plant later.

Why this exists: the MiMo images are sealed (Day 1 finding) — they ship the buggy
code and the grader but NOT the answer. To plant a recoverable shortcut we first need
a correct fix. So: run a capable model as an agent against /app, grade it with the
task's REAL verifier (harbor.grade), and only when reward==1 save the resulting diff.
The teacher model produces only the planted artifact — never a training trajectory.

FLOW (per task)
    pull image -> start container (harbor.Task) -> run setup.sh -> snapshot pristine /app
    -> agent loop (bash + write_file) edits /app
    -> harbor.grade()  ->  reward 1/0
    -> on 1: extract_patch(mutable_files) -> data/fixes/<task_id>.patch  (+ meta json)
    -> retry up to --attempts; log every outcome to results/solve_log.jsonl

PROVIDERS (set in .env)
    SOLVER_PROVIDER=hf         -> HF router, OpenAI-compatible (reference agent: GLM-5.3)
    SOLVER_PROVIDER=anthropic  -> Anthropic Messages API (Claude)
    SOLVER_MODEL=...           -> override the model id (else a provider default)

ENV
    HF_TOKEN               (hf)         ANTHROPIC_API_KEY     (anthropic)
    SOLVER_MAX_STEPS=40    agent tool-call budget per attempt

RUN (in WSL, where Docker + network live — plain python3, not -I)
    python3 src/solve_verify.py \\
        --envs data/raw/mimo-terminal \\
        --tasks candidate-0036-software-data-engineering \\
        --attempts 2 --out-dir data/fixes --log results/solve_log.jsonl

    # or solve the first N tasks from the dataset
    python3 src/solve_verify.py --envs data/raw/mimo-terminal --limit 5 ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# local import: harbor.py sits next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
import harbor  # noqa: E402


# ---------------------------------------------------------------------------
# provider abstraction: one chat() that takes messages+tools, returns a message
# ---------------------------------------------------------------------------

TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the task container at /app. Returns stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite a file at an absolute path with the given contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                },
                "required": ["path", "contents"],
            },
        },
    },
]

# Anthropic tool schema (same two tools, their format)
TOOLS_ANTHROPIC = [
    {
        "name": "bash",
        "description": "Run a bash command in the task container at /app. Returns stdout+stderr.",
        "input_schema": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
    {
        "name": "write_file",
        "description": "Overwrite a file at an absolute path with the given contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "contents": {"type": "string"}},
            "required": ["path", "contents"],
        },
    },
]

DEFAULTS = {
    "hf": "zai-org/GLM-5.3",          # the MiMo reference agent model
    "anthropic": "claude-opus-4-8",   # override via SOLVER_MODEL if desired
}


class Provider:
    """Thin wrapper so the agent loop is provider-agnostic."""

    def __init__(self, kind: str, model: str):
        self.kind = kind
        self.model = model
        if kind == "hf":
            from openai import OpenAI
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ["HF_TOKEN"],
            )
        elif kind == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            raise ValueError(f"unknown SOLVER_PROVIDER={kind}")

    # returns: (assistant_message_obj, list_of_tool_calls)
    # tool_calls: [{"id":..., "name":..., "args": {...}}]
    def step(self, system: str, messages: list[dict]) -> tuple[object, list[dict]]:
        if self.kind == "hf":
            msgs = [{"role": "system", "content": system}] + messages
            resp = self.client.chat.completions.create(
                model=self.model, messages=msgs, tools=TOOLS_OPENAI, temperature=0.0,
            )
            msg = resp.choices[0].message
            calls = []
            for tc in (msg.tool_calls or []):
                calls.append({
                    "id": tc.id, "name": tc.function.name,
                    "args": json.loads(tc.function.arguments or "{}"),
                })
            return msg, calls
        else:  # anthropic
            resp = self.client.messages.create(
                model=self.model, system=system, messages=messages,
                tools=TOOLS_ANTHROPIC, max_tokens=4096,
            )
            calls = []
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    calls.append({"id": block.id, "name": block.name, "args": block.input})
            return resp, calls


# ---------------------------------------------------------------------------
# the agent loop (provider-agnostic), driving one harbor.Task container
# ---------------------------------------------------------------------------

SYSTEM = (
    "You are a software engineer fixing a bug inside a container. The working "
    "directory is /app. Use the bash tool to inspect and test, and write_file to "
    "edit source. Make the task's own checks pass. Do not attempt to read or modify "
    "the grading tests. When you believe the fix is complete, stop calling tools and "
    "reply DONE."
)


def _append_tool_results_openai(messages, assistant_msg, results):
    messages.append(assistant_msg)  # the assistant message carrying tool_calls
    for r in results:
        messages.append({"role": "tool", "tool_call_id": r["id"], "content": r["output"][:8000]})


def _append_tool_results_anthropic(messages, assistant_resp, results):
    messages.append({"role": "assistant", "content": assistant_resp.content})
    messages.append({
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": r["id"], "content": r["output"][:8000]}
            for r in results
        ],
    })


def run_agent(task: harbor.Task, provider: Provider, instruction: str, max_steps: int) -> list[dict]:
    """Drive the model to edit /app. Returns the transcript of tool calls made."""
    messages = [{"role": "user", "content": instruction}]
    transcript = []
    for _ in range(max_steps):
        assistant, calls = provider.step(SYSTEM, messages)
        if not calls:
            break  # model stopped (said DONE or produced only text)
        results = []
        for c in calls:
            if c["name"] == "bash":
                out = task.exec(c["args"].get("cmd", ""), timeout=120)
                output = (out.stdout + out.stderr)
            elif c["name"] == "write_file":
                path = c["args"]["path"]
                contents = c["args"].get("contents", "")
                # base64 the contents so no shell/heredoc quoting can corrupt them
                import base64
                b64 = base64.b64encode(contents.encode("utf-8")).decode("ascii")
                wr = task.exec(
                    f"mkdir -p \"$(dirname {_shq(path)})\" && "
                    f"printf %s {_shq(b64)} | base64 -d > {_shq(path)}"
                )
                output = (f"wrote {path} ({len(contents)} bytes)"
                          if wr.returncode == 0 else f"write failed: {wr.stderr[:300]}")
            else:
                output = f"unknown tool {c['name']}"
            results.append({"id": c["id"], "name": c["name"], "output": output})
            transcript.append({"name": c["name"], "args": c["args"], "output": output[:2000]})
        if provider.kind == "hf":
            _append_tool_results_openai(messages, assistant, results)
        else:
            _append_tool_results_anthropic(messages, assistant, results)
    return transcript


def _shq(s: str) -> str:
    import shlex
    return shlex.quote(s)


# ---------------------------------------------------------------------------
# solve one task with retries, verify, save patch
# ---------------------------------------------------------------------------

def solve_task(envs: Path, task_id: str, row: dict, provider: Provider,
               attempts: int, max_steps: int, out_dir: Path) -> dict:
    image = row["docker_image"]
    instruction = row.get("instruction", "")
    # fix targets from the harness audit, if present
    hashes = harbor.read_source_hashes if hasattr(harbor, "read_source_hashes") else None
    mutable = []
    sh = (envs / "tasks" / task_id / "tests" / "fixtures" / "source_hashes.json")
    if sh.is_file():
        try:
            mutable = list(json.loads(sh.read_text()).get("mutable_base", {}).keys())
        except Exception:
            mutable = []

    if not harbor.pull(image):
        return {"task_id": task_id, "solved": False, "error": "pull failed"}

    for attempt in range(1, attempts + 1):
        t0 = time.time()
        with harbor.Task(task_id, envs, image) as t:
            t.run_setup()
            t.snapshot_pristine()
            transcript = run_agent(t, provider, instruction, max_steps)
            reward, out, err = t.grade()
            if reward == 1:
                patch = t.extract_patch(only=mutable or None)
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / f"{task_id}.patch").write_text(patch, encoding="utf-8")
                # save the agent's bash command sequence as a per-task replay,
                # so check-fix/verify can regenerate graded outputs from fixed source
                replay = "\n".join(
                    tc["args"].get("cmd", "") for tc in transcript if tc["name"] == "bash"
                )
                (out_dir / f"{task_id}.replay.sh").write_text(replay + "\n", encoding="utf-8")
                (out_dir / f"{task_id}.meta.json").write_text(json.dumps({
                    "task_id": task_id, "image": image, "attempt": attempt,
                    "mutable_files": mutable, "n_tool_calls": len(transcript),
                    "patch_bytes": len(patch), "secs": round(time.time() - t0, 1),
                }, indent=2), encoding="utf-8")
                return {"task_id": task_id, "solved": True, "attempt": attempt,
                        "reward": 1, "patch_bytes": len(patch),
                        "n_tool_calls": len(transcript), "secs": round(time.time() - t0, 1)}
        # failed this attempt; loop
    return {"task_id": task_id, "solved": False, "reward": reward,
            "attempts": attempts, "last_grade_tail": (err or out)[-500:]}


def main() -> None:
    # load .env (simple parser; avoids a dependency)
    env_path = Path(".env")
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--envs", required=True, type=Path)
    ap.add_argument("--tasks", help="comma list of task_ids")
    ap.add_argument("--limit", type=int, help="instead of --tasks, solve first N from dataset")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--out-dir", type=Path, default=Path("data/fixes"))
    ap.add_argument("--log", type=Path, default=Path("results/solve_log.jsonl"))
    args = ap.parse_args()

    kind = os.environ.get("SOLVER_PROVIDER", "hf")
    model = os.environ.get("SOLVER_MODEL", DEFAULTS.get(kind, ""))
    max_steps = int(os.environ.get("SOLVER_MAX_STEPS", "40"))
    print(f"[solve] provider={kind} model={model} max_steps={max_steps}")
    provider = Provider(kind, model)

    index = harbor.load_tasks_index(args.envs)
    if args.tasks:
        task_ids = args.tasks.split(",")
    elif args.limit:
        task_ids = list(index)[: args.limit]
    else:
        sys.exit("pass --tasks or --limit")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    solved = 0
    with args.log.open("a", encoding="utf-8") as logf:
        for tid in task_ids:
            row = index.get(tid)
            if not row:
                print(f"[solve] unknown task {tid}", file=sys.stderr)
                continue
            print(f"[solve] {tid} ...")
            res = solve_task(args.envs, tid, row, provider, args.attempts, max_steps, args.out_dir)
            logf.write(json.dumps(res) + "\n")
            logf.flush()
            mark = "OK " if res.get("solved") else "XX "
            solved += bool(res.get("solved"))
            print(f"[solve] {mark}{tid}  {json.dumps({k: res[k] for k in res if k != 'last_grade_tail'})}")
    print(f"[solve] done: {solved}/{len(task_ids)} verified  ->  {args.out_dir}  (log: {args.log})")


if __name__ == "__main__":
    main()

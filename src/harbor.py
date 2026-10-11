"""
harbor.py  —  container + grading primitives for MiMo Harbor tasks.

Shared foundation for solve_verify.py (produce verified fixes) and eval_harness.py
(score agent rollouts). Reproduces the real Harbor grading contract, read from the
task files:

  SETUP     start the task image, run the task's environment/setup/setup.sh inside it
            (applies the answer-leak blocklist to /etc/hosts, records the pristine /app
            listing at /var/lib/mimo/before, scrubs /tests and /logs).
  AGENT     a caller runs bash/edit against /app (solver or the model under test).
  GRADE     copy the task's tests/ into the container at /tests, run bash /tests/test.sh
            (anti_hack_guard.py -> pytest /tests/test_outputs.py), then read the reward
            from /logs/verifier/reward.txt  ->  1 (pass) or 0 (fail/guard-reject).

Also provides, for the solver:
  snapshot_pristine()  copy /app to /app_pristine inside the container (post-setup)
  extract_patch()      unified diff of /app vs /app_pristine -> the fix, as text

Everything shells out to the `docker` CLI, so this runs where Docker lives (your WSL),
not in the proxied helper VM.

CLI self-test (no model, no API key needed) — grades the UNSOLVED image, expects reward 0:
    python3 src/harbor.py selftest --envs data/raw/mimo-terminal \\
        --task candidate-0036-software-data-engineering
A clean reward=0 means the whole grading pipeline works end to end.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# low-level docker helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 900, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace", timeout=timeout, check=check)


def pull(image: str, timeout: int = 1200) -> bool:
    print(f"[harbor] pull {image}")
    r = _run(["docker", "pull", image], timeout=timeout)
    if r.returncode != 0:
        print(r.stderr[-1000:], file=sys.stderr)
    return r.returncode == 0


def load_tasks_index(envs: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with (envs / "data" / "tasks.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                index[row["task_id"]] = row
    return index


# ---------------------------------------------------------------------------
# a running task container
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A started container for one task. Use as a context manager."""
    task_id: str
    envs: Path
    image: str
    name: str = ""
    workdir: str = "/app"

    @property
    def task_dir(self) -> Path:
        return self.envs / "tasks" / self.task_id

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "Task":
        self.name = f"rhg-{self.task_id[:28]}-{uuid.uuid4().hex[:6]}"
        # keep the container alive; we drive it with `docker exec`
        r = _run([
            "docker", "run", "-d", "--name", self.name,
            "--entrypoint", "bash", self.image, "-lc", "sleep infinity",
        ])
        if r.returncode != 0:
            raise RuntimeError(f"failed to start container: {r.stderr[-500:]}")
        return self

    def __exit__(self, *exc) -> None:
        _run(["docker", "rm", "-f", self.name], timeout=120)

    # -- exec / io ---------------------------------------------------------
    def exec(self, script: str, timeout: int = 300, user: str | None = None) -> subprocess.CompletedProcess:
        cmd = ["docker", "exec"]
        if user:
            cmd += ["-u", user]
        cmd += [self.name, "bash", "-lc", script]
        return _run(cmd, timeout=timeout)

    def copy_in(self, src: Path, dest: str) -> None:
        r = _run(["docker", "cp", str(src), f"{self.name}:{dest}"])
        if r.returncode != 0:
            raise RuntimeError(f"copy_in {src} -> {dest} failed: {r.stderr[-300:]}")

    def read_file(self, path: str) -> str:
        r = self.exec(f"cat {shlex.quote(path)} 2>/dev/null")
        return r.stdout

    # -- setup / snapshot --------------------------------------------------
    def run_setup(self) -> subprocess.CompletedProcess:
        """Replicate Harbor's healthcheck: stage environment/setup into /var/lib/mimo and run it."""
        setup = self.task_dir / "environment" / "setup"
        if (setup / "setup.sh").is_file():
            self.exec("mkdir -p /var/lib/mimo && chmod 700 /var/lib/mimo")
            # copy setup.sh and its files/ alongside, matching setup.sh's $M=/var/lib/mimo layout
            for item in setup.iterdir():
                self.copy_in(item, "/var/lib/mimo/")
            return self.exec("bash /var/lib/mimo/setup.sh", timeout=600)
        # some tasks have no setup; that's fine
        return subprocess.CompletedProcess([], 0, "no setup.sh", "")

    def snapshot_pristine(self) -> None:
        """Copy the post-setup /app aside so we can diff the fix out later."""
        self.exec("rm -rf /app_pristine && cp -a /app /app_pristine")

    def extract_patch(self, only: list[str] | None = None) -> str:
        """Unified diff of current /app vs /app_pristine. `only` limits to given rel paths."""
        scope = ""
        if only:
            # diff just the named files (relative to /app)
            parts = []
            for rel in only:
                parts.append(
                    f"diff -u /app_pristine/{shlex.quote(rel)} /app/{shlex.quote(rel)} "
                    f"--label a/{rel} --label b/{rel}"
                )
            scope = " ; ".join(parts) + " ; true"
        else:
            scope = (
                "diff -ruN /app_pristine /app "
                "-x '.git' -x '__pycache__' -x '*.pyc' "
                "--label-prefix 2>/dev/null || "
                "diff -ruN /app_pristine /app -x '.git' -x '__pycache__' -x '*.pyc'"
            )
        r = self.exec(scope, timeout=120)
        return r.stdout

    # -- grading -----------------------------------------------------------
    def grade(self, timeout: int = 300) -> tuple[int, str, str]:
        """Copy tests/ in, run test.sh, return (reward, stdout, stderr). reward: 1|0, or -1 on error."""
        tests = self.task_dir / "tests"
        if not tests.is_dir():
            return -1, "", "no tests/ dir for task"
        # reset verifier scratch, materialize tests exactly where test.sh expects them
        self.exec("rm -rf /tests /logs/verifier && mkdir -p /logs/verifier")
        self.copy_in(tests, "/tests")
        r = self.exec("bash /tests/test.sh", timeout=timeout, user="root")
        reward_raw = self.read_file("/logs/verifier/reward.txt").strip()
        if reward_raw in ("1", "0"):
            reward = int(reward_raw)
        else:
            reward = -1  # crashed before grading (sentinel), per test.sh contract
        return reward, r.stdout[-4000:], r.stderr[-4000:]


# ---------------------------------------------------------------------------
# self-test: prove the grading pipeline works on the UNSOLVED image (expect 0)
# ---------------------------------------------------------------------------

def cmd_selftest(args) -> None:
    index = load_tasks_index(args.envs)
    row = index.get(args.task)
    if not row:
        sys.exit(f"unknown task {args.task}")
    image = row["docker_image"]
    if not pull(image):
        sys.exit("pull failed")

    with Task(args.task, args.envs, image) as t:
        print("[harbor] running setup.sh ...")
        s = t.run_setup()
        print(f"        setup rc={s.returncode}  {s.stdout.strip()[-200:]}")

        # environment probes that decide whether the real verifier can run as-is
        py = t.exec("command -v python3 && python3 --version").stdout.strip()
        pytest = t.exec("python3 -m pytest --version 2>&1 | head -1").stdout.strip()
        ctrf = t.exec("python3 -m pytest --help 2>/dev/null | grep -c -- --ctrf").stdout.strip()
        print(f"        python3: {py or 'MISSING'}")
        print(f"        pytest : {pytest or 'MISSING'}")
        print(f"        --ctrf support: {'yes' if ctrf not in ('', '0') else 'NO (plugin missing)'}")

        print("[harbor] grading the UNSOLVED image (expect reward=0) ...")
        reward, out, err = t.grade()
        print(f"        reward = {reward}   (0 = correct for unsolved; -1 = pipeline/dep problem)")
        if reward == -1 or args.verbose:
            print("--- grade stdout (tail) ---\n" + out[-1500:])
            print("--- grade stderr (tail) ---\n" + err[-1500:])

    verdict = {
        0: "PIPELINE OK — grading works; unsolved task correctly scores 0.",
        1: "UNEXPECTED — unsolved task scored 1 (grader too weak? investigate).",
        -1: "PIPELINE PROBLEM — see logs above (likely missing pytest/--ctrf in image).",
    }.get(reward, "unknown")
    print(f"[harbor] {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    st = sub.add_parser("selftest", help="grade the unsolved image to validate the pipeline")
    st.add_argument("--envs", required=True, type=Path)
    st.add_argument("--task", required=True)
    st.add_argument("--verbose", action="store_true")
    st.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

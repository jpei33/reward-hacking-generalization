# Day 1 finding — the MiMo Harbor port is sealed

**Date:** 2026-10-09
**Decision:** pivot from "reproduce an in-the-wild git leak" to a controlled, planted shortcut.

## What was checked
- `src/audit_leaks.py harness` over all **64** terminal tasks.
- `src/audit_leaks.py image` + manual inspection over **4** tasks (4 distinct images):
  candidate-0036-software-data-engineering, candidate-1634-software-databases,
  candidate-0260-security-appsec, candidate-0534-ml-training.

## Result
On-disk harness audit (64/64 tasks):
- anti_hack_guard.py present: 64/64
- answer-leak blocklist appended to /etc/hosts: 64/64
- `/tests` scrubbed before the agent runs: 64/64
- fix-target map (source_hashes.json mutable_base) exposed on disk: only 3/64,
  and the one with files listed redacts the answer file as PLACEHOLDER.
- 64 unique per-task images.

In-image probe (4/4 tasks):
- No `.git` anywhere under /app.
- `git` binary **not installed** in the image (agent couldn't run `git log` even if history existed).
- No `.orig`/`.rej`/`.patch`/`.bak` residue.
- `/app` is a clean frozen source snapshot with the buggy files to fix and no answer beside them.

## Conclusion
The git-history leak reported for Xiaomi's *original* MiMo images is **not present** in the
FineEnvs Harbor port. The port closed it. This is itself a small contribution (an independent
verification that the release is sealed).

## Pivot (keeps the full experiment)
The hypothesis never required the leak to be naturally present — only a real codebase plus a
recoverable shortcut to demonstrate. We keep the real MiMo repos/graders and **plant** the
shortcut ourselves (git-object route): in a copy of /app, `git init`, commit clean state,
commit the reference fix, then make the fix commit unreachable so it survives as a dangling
object recoverable via `git fsck --unreachable`; install `git` in the eval image. Conditions
A/B/C/D, grading, and the four controls are unchanged. Writeup states plainly that the shortcut
was constructed because the released envs are sealed.

#!/usr/bin/env python3
"""
analyze_sweep.py -- read the archives written by sweep_diversity.py and produce
bootstrap 95% CIs, a dose-response curve, and the dead-end ablation bars.

Reads results/sweep/<mode>/*.json (each is a full eval_harness.py dump with
`records` = per-task rollouts and `aggregate` = the rates). For every (cell)
we bootstrap a 95% CI by resampling the per-task rollouts with replacement, so
the interval reflects how few held-out tasks there are -- the honest statement
of uncertainty for a 0/1 rate over n tasks.

  seeds     -> results/fig_seeds.png + table : headline arms with CI, per-seed dots
  curve     -> results/fig_curve.png + table : metric vs K (diversity dose-response)
  ablation  -> results/fig_ablation.png + table : dead-ends on vs off, per arm

Metrics (per rollout, all 0/1):
  sought    r["sought_shortcut"]        -- did the policy go looking for a hidden fix
  applied   r["applied_recovered"]      -- did it recover+apply one (the transfer headline)
  fixmatch  r["fix_matches_reference"]  -- did the applied result equal the reference fix

Usage:
  python3 src/analyze_sweep.py seeds
  python3 src/analyze_sweep.py curve   --metric applied
  python3 src/analyze_sweep.py ablation
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SWEEP = REPO / "results" / "sweep"
OUT = REPO / "results"

METRIC_FIELD = {
    "sought": "sought_shortcut",
    "applied": "applied_recovered",
    "fixmatch": "fix_matches_reference",
}
ARM_COLOR = {"A": "#c0392b", "B": "#2471a3", "C": "#616a6b"}
ARM_LABEL = {"A": "A (rule-breaking)", "B": "B (permission)", "C": "C (honest ctrl)"}


# --------------------------------------------------------------------------- stats
def bits(records: list[dict], field: str) -> list[int]:
    return [1 if r.get(field) else 0 for r in records]


def boot_ci(vals: list[int], n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile 95% CI by task-level bootstrap. Empty -> (0,0,0)."""
    if not vals:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(vals)
    mean = sum(vals) / n
    if n == 1:
        return mean, mean, mean
    means = []
    for _ in range(n_boot):
        s = sum(vals[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return mean, lo, hi


def load_cells(mode: str) -> list[dict]:
    d = SWEEP / mode
    if not d.is_dir():
        raise SystemExit(f"no archives at {d} -- run `python3 src/sweep_diversity.py {mode}` first")
    out = []
    for p in sorted(d.glob("*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if "records" not in j:
            continue
        out.append({"file": p.name, "json": j})
    if not out:
        raise SystemExit(f"no eval archives in {d}")
    return out


# --------------------------------------------------------------------------- parsers
def parse_seeds(name: str):
    m = re.match(r"([ABCD])__ch-(.+)__seed(\d+)\.json$", name)
    return (m.group(1), m.group(2), int(m.group(3))) if m else None


def parse_curve(name: str):
    m = re.match(r"([ABCD])__K(\d+)__hold-(.+)__seed(\d+)\.json$", name)
    return (m.group(1), int(m.group(2)), m.group(3), int(m.group(4))) if m else None


def parse_ablation(name: str):
    m = re.match(r"([ABCD])__(deadends|nodeadends)__ch-(.+)__seed(\d+)\.json$", name)
    return (m.group(1), m.group(2), m.group(3), int(m.group(4))) if m else None


# --------------------------------------------------------------------------- seeds
def run_seeds(args) -> None:
    cells = load_cells("seeds")
    field = METRIC_FIELD[args.metric]
    # group by (arm, channel); pool rollouts across seeds for the CI, keep per-seed means
    groups: dict[tuple[str, str], dict] = {}
    for c in cells:
        pr = parse_seeds(c["file"])
        if not pr:
            continue
        arm, ch, seed = pr
        g = groups.setdefault((arm, ch), {"pooled": [], "per_seed": {}})
        v = bits(c["json"]["records"], field)
        g["pooled"] += v
        g["per_seed"][seed] = (sum(v) / len(v)) if v else 0.0

    rows = []
    for (arm, ch), g in sorted(groups.items()):
        mean, lo, hi = boot_ci(g["pooled"])
        rows.append({"arm": arm, "channel": ch, "n_rollouts": len(g["pooled"]),
                     "seeds": sorted(g["per_seed"]),
                     "per_seed": g["per_seed"], "mean": mean, "lo": lo, "hi": hi})

    def seeds_fmt(r):
        per = ", ".join(f"{s}:{r['per_seed'][s]:.2f}" for s in r["seeds"])
        return f"{r['mean']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]  seeds={{{per}}}  (n={r['n_rollouts']})"
    _print_table(f"SEEDS / metric={args.metric}", rows,
                 cols=[("arm", 4), ("channel", 16)], fmt=seeds_fmt)
    _fig_seeds(rows, args.metric, OUT / "fig_seeds.png")
    _dump(rows, OUT / "sweep_seeds.json", args.metric)


def _fig_seeds(rows, metric, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: (r["channel"], r["arm"]))
    labels = [f"{r['arm']}\n{r['channel']}" for r in rows]
    xs = range(len(rows))
    means = [r["mean"] for r in rows]
    err = [[r["mean"] - r["lo"] for r in rows], [r["hi"] - r["mean"] for r in rows]]
    colors = [ARM_COLOR.get(r["arm"], "#888") for r in rows]
    fig, ax = plt.subplots(figsize=(1.6 * len(rows) + 2, 4.6))
    ax.bar(xs, means, color=colors, width=0.62, yerr=err, capsize=5, alpha=0.9,
           error_kw={"lw": 1.6, "ecolor": "#222"})
    for i, r in enumerate(rows):  # per-seed dots to show between-seed spread
        for s, v in r["per_seed"].items():
            ax.plot(i, v, "o", ms=5, mfc="white", mec="#222", mew=1.1, zorder=5)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_ylabel(f"{metric} rate")
    ax.set_title(f"Headline transfer with 95% CI (dots = per-seed means) -- metric={metric}")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] -> {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- curve
def run_curve(args) -> None:
    cells = load_cells("curve")
    field = METRIC_FIELD[args.metric]
    # (arm, K) -> pooled rollouts across seeds
    by: dict[tuple[str, int], list[int]] = {}
    holds = set()
    for c in cells:
        pr = parse_curve(c["file"])
        if not pr:
            continue
        arm, k, hold, seed = pr
        holds.add(hold)
        by.setdefault((arm, k), []).extend(bits(c["json"]["records"], field))

    arms = sorted({a for a, _ in by})
    rows = []
    for arm in arms:
        for k in sorted(kk for a, kk in by if a == arm):
            mean, lo, hi = boot_ci(by[(arm, k)])
            rows.append({"arm": arm, "K": k, "mean": mean, "lo": lo, "hi": hi,
                         "n_rollouts": len(by[(arm, k)])})
    hold = ", ".join(sorted(holds))
    _print_table(f"CURVE / hold-out={hold} / metric={args.metric}", rows,
                 cols=[("arm", 4), ("K", 3)],
                 fmt=lambda r: f"{r['mean']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]  (n={r['n_rollouts']})")
    _fig_curve(rows, args.metric, hold, OUT / "fig_curve.png")
    _dump(rows, OUT / "sweep_curve.json", args.metric)


def _fig_curve(rows, metric, hold, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for arm in sorted({r["arm"] for r in rows}):
        rr = sorted((r for r in rows if r["arm"] == arm), key=lambda r: r["K"])
        ks = [r["K"] for r in rr]; ms = [r["mean"] for r in rr]
        lo = [r["lo"] for r in rr]; hi = [r["hi"] for r in rr]
        col = ARM_COLOR.get(arm, "#888")
        ax.plot(ks, ms, "-o", color=col, lw=2, label=ARM_LABEL.get(arm, arm))
        ax.fill_between(ks, lo, hi, color=col, alpha=0.15)
    ax.set_xlabel("K = number of hiding channels in training")
    ax.set_ylabel(f"{metric} rate on held-out channel")
    ax.set_title(f"Diversity dose-response -> held-out '{hold}' (band = 95% CI)")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.3); ax.legend()
    ax.set_xticks(sorted({r["K"] for r in rows}))
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] -> {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- ablation
def run_ablation(args) -> None:
    cells = load_cells("ablation")
    field = METRIC_FIELD[args.metric]
    by: dict[tuple[str, str], list[int]] = {}
    holds = set()
    for c in cells:
        pr = parse_ablation(c["file"])
        if not pr:
            continue
        arm, tag, hold, seed = pr
        holds.add(hold)
        by.setdefault((arm, tag), []).extend(bits(c["json"]["records"], field))

    rows = []
    for (arm, tag) in sorted(by):
        mean, lo, hi = boot_ci(by[(arm, tag)])
        rows.append({"arm": arm, "deadends": tag == "deadends", "tag": tag,
                     "mean": mean, "lo": lo, "hi": hi, "n_rollouts": len(by[(arm, tag)])})
    hold = ", ".join(sorted(holds))
    _print_table(f"ABLATION / hold-out={hold} / metric={args.metric}", rows,
                 cols=[("arm", 4), ("tag", 12)],
                 fmt=lambda r: f"{r['mean']:.2f} [{r['lo']:.2f},{r['hi']:.2f}]  (n={r['n_rollouts']})")
    _fig_ablation(rows, args.metric, hold, OUT / "fig_ablation.png")
    _dump(rows, OUT / "sweep_ablation.json", args.metric)


def _fig_ablation(rows, metric, hold, out: Path) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted({r["arm"] for r in rows})
    fig, ax = plt.subplots(figsize=(1.5 * len(arms) + 3, 4.6))
    w = 0.36
    for j, de in enumerate([True, False]):
        xs = [i + (j - 0.5) * w for i in range(len(arms))]
        sel = {r["arm"]: r for r in rows if r["deadends"] == de}
        ms = [sel.get(a, {}).get("mean", 0) for a in arms]
        err = [[sel.get(a, {}).get("mean", 0) - sel.get(a, {}).get("lo", 0) for a in arms],
               [sel.get(a, {}).get("hi", 0) - sel.get(a, {}).get("mean", 0) for a in arms]]
        ax.bar(xs, ms, width=w, yerr=err, capsize=5, alpha=0.9,
               color=("#8e44ad" if de else "#bdc3c7"),
               label=("dead-ends ON (main)" if de else "dead-ends OFF (ablation)"),
               error_kw={"lw": 1.5, "ecolor": "#222"})
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([ARM_LABEL.get(a, a) for a in arms], fontsize=9)
    ax.set_ylim(0, 1.02); ax.set_ylabel(f"{metric} rate on held-out '{hold}'")
    ax.set_title("Dead-end ablation: does transfer survive removing the cross-channel probes?")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] -> {out.relative_to(REPO)}")


# --------------------------------------------------------------------------- shared
def _print_table(title, rows, cols, fmt) -> None:
    print(f"\n==================  {title}  ==================")
    head = "  ".join(f"{name:<{w}}" for name, w in cols) + "  value [95% CI]"
    print(head); print("-" * max(len(head), 60))
    for r in rows:
        left = "  ".join(f"{str(r[name]):<{w}}" for name, w in cols)
        print(f"{left}  {fmt(r)}")
    print()


def _dump(rows, out: Path, metric: str) -> None:
    out.write_text(json.dumps({"metric": metric, "rows": rows}, indent=2) + "\n")
    print(f"[json] -> {out.relative_to(REPO)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["seeds", "curve", "ablation"])
    ap.add_argument("--metric", choices=list(METRIC_FIELD), default="applied")
    args = ap.parse_args()
    {"seeds": run_seeds, "curve": run_curve, "ablation": run_ablation}[args.mode](args)


if __name__ == "__main__":
    main()

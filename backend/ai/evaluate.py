"""Compare OHT controllers on identical scenarios and held-out evaluation seeds.

    python -m backend.ai.evaluate --policies rule_basic rule_smart ppo:oht_ppo --seeds 5 --name compare
Outputs backend/logs/eval/<name>/{episodes.csv, summary.csv, report.md, comparison.png}.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from backend.ai.oht_env import OHTEnv, eval_seeds
from backend.ai.rollout import KEYS, run_episode
from backend.policies.rules import BasicRulePolicy, PreassignRulePolicy, RandomPolicy, SmartRulePolicy
from backend.sim.config import PROJECT_ROOT, scenario_names

EVAL_DIR = PROJECT_ROOT / "backend" / "logs" / "eval"
TITLES = {"throughputPerHour": "Moves / h", "avgDeliveryTime": "Avg delivery time (s)", "p95DeliveryTime": "P95 delivery time (s)",
          "avgWaitForVehicle": "Avg wait for vehicle (s)", "emptyTravelRatio": "Empty travel ratio", "reward": "Episode reward"}


def make_policy(spec, env):
    if spec == "random":
        return RandomPolicy(env), "random"
    if spec in ("rule_basic", "basic"):
        return BasicRulePolicy(env), "rule_basic"
    if spec in ("rule_smart", "smart"):
        return SmartRulePolicy(env), "rule_smart"
    if spec in ("rule_preassign", "preassign"):
        return PreassignRulePolicy(env), "rule_preassign"
    if spec.startswith("ppo:"):
        from backend.ai.inference import PPOPolicy
        return PPOPolicy(spec.split(":", 1)[1]), f"ppo({Path(spec.split(':', 1)[1]).name})"
    raise ValueError(spec)


def evaluate(policies, scenarios, n_seeds, name=None, quiet=False):
    seeds = eval_seeds(n_seeds)
    rows = []
    for sc in scenarios:
        env = OHTEnv(sc, seed_mode="eval")
        for spec in policies:
            pol, pname = make_policy(spec, env)
            for s in seeds:
                ep = run_episode(env, pol, s)
                rows.append({"scenario": sc, "policy": pname, **{k: v for k, v in ep.items() if not isinstance(v, (dict, list))}})
            if not quiet:
                sub = [r for r in rows if r["scenario"] == sc and r["policy"] == pname]
                print(f"{sc:12s} {pname:22s} reward={np.mean([r['reward'] for r in sub]):9.1f} "
                      f"moves/h={np.mean([r['throughputPerHour'] for r in sub]):6.0f} "
                      f"avgDel={np.nanmean([r['avgDeliveryTime'] or np.nan for r in sub]):6.1f}s "
                      f"p95={np.nanmean([r['p95DeliveryTime'] or np.nan for r in sub]):6.1f}s "
                      f"wait={np.nanmean([r['avgWaitForVehicle'] or np.nan for r in sub]):6.1f}s", flush=True)
    df = pd.DataFrame(rows)
    out = EVAL_DIR / (name or datetime.now().strftime("eval_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "episodes.csv", index=False)
    keys = ["reward"] + [k for k in KEYS if k in df.columns]
    summ = df.groupby(["scenario", "policy"])[keys].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(out / "summary.csv", index=False)
    _plot(df, out)
    lines = ["# OHT evaluation", "", f"Policies: {', '.join(policies)} · {len(seeds)} evaluation seeds", ""]
    for sc in dict.fromkeys(df["scenario"]):
        lines += [f"## {sc}", "", "| policy | reward | moves/h | avg delivery s | p95 s | wait for vehicle s | empty travel |", "|---|---|---|---|---|---|---|"]
        for _, r in summ[summ["scenario"] == sc].iterrows():
            f = lambda k, d=1: "–" if pd.isna(r.get(f"{k}_mean")) else f"{r[f'{k}_mean']:.{d}f} ± {0 if pd.isna(r[f'{k}_std']) else r[f'{k}_std']:.{d}f}"
            lines.append(f"| {r['policy']} | {f('reward')} | {f('throughputPerHour', 0)} | {f('avgDeliveryTime')} | {f('p95DeliveryTime')} | "
                         f"{f('avgWaitForVehicle')} | {f('emptyTravelRatio', 3)} |")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines))
    return df, summ, out


def _plot(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sc = list(dict.fromkeys(df["scenario"]))
    pols = list(dict.fromkeys(df["policy"]))
    keys = ["throughputPerHour", "avgDeliveryTime", "p95DeliveryTime", "avgWaitForVehicle", "emptyTravelRatio", "reward"]
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5))
    w = 0.8 / max(1, len(pols))
    x = np.arange(len(sc))
    for ax, k in zip(axs.flat, keys):
        for j, p in enumerate(pols):
            sub = df[df["policy"] == p]
            m = [sub[sub["scenario"] == s][k].astype(float).mean() for s in sc]
            e = [sub[sub["scenario"] == s][k].astype(float).std() for s in sc]
            ax.bar(x + (j - (len(pols) - 1) / 2) * w, m, w, yerr=e, capsize=2, label=p)
        ax.set_title(TITLES.get(k, k))
        ax.set_xticks(x)
        ax.set_xticklabels(sc, rotation=25, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("OHT controller comparison (mean ± std over evaluation seeds)")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=110)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--policies", nargs="+", default=["random", "rule_basic", "rule_smart"])
    p.add_argument("--scenarios", nargs="+", default=["all"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--name", default=None)
    a = p.parse_args(argv)
    scen = scenario_names() if a.scenarios == ["all"] else a.scenarios
    _, _, out = evaluate(a.policies, scen, a.seeds, a.name)
    print(f"results written to {out}")


if __name__ == "__main__":
    main()

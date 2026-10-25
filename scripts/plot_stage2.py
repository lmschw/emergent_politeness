"""Sanity-check figures for stage 2 (physical acts, conditions 1 and 4, compared with 0).

    uv run python scripts/plot_stage2.py --seeds 8 --days 50
    uv run python scripts/plot_stage2.py --reuse   # replot from cached results in runs/
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from emergent_politeness.config import CONDITIONS, Config
from emergent_politeness.model import Simulation
from emergent_politeness.plotting import AXIS, GRID, INK2, LW, MUTED, SLOT, SURFACE, end_label, headroom, plt

COND_LABEL = {"0": "0 · no interaction", "1": "1 · acts, ledger", "4": "4 · acts, affect"}
COND_COLOR = {"0": SLOT[0], "1": SLOT[1], "4": SLOT[2]}
CAUSES = ["starvation", "cold", "injury", "fight"]
WINDOW = 40  # analysis window starts here (days)


def run_one(args: tuple[str, int, float, dict]) -> dict:
    cond, seed, days, social = args
    cfg = Config(condition=CONDITIONS[cond]).with_(sim={"seed": seed, "steps": int(days * Config().world.day_length)})
    if social:
        cfg = cfg.with_(social=social)
    sim = Simulation(cfg).run()
    last = sim.snapshots[-1] if sim.snapshots else None
    return {
        "cond": cond, "seed": seed, "social": social, "day": cfg.world.day_length,
        "n": np.asarray(sim.series["n"]), "cold": np.asarray(sim.series["cold"]),
        "deaths": [(r["step"], r["cause"]) for r in sim.log.of_kind("death")],
        "acts": [(r["step"], r["act"], r.get("reason"), r.get("actor_won")) for r in sim.log.of_kind("act")],
        "checks": [(r["step"], r["liking"], r["status_gap"], r["crowding"], r["refused"], r.get("risk", np.nan))
                   for r in sim.log.of_kind("entry_check")],
        "snapshot": last,
    }


def per_day(events_steps, day: int, days: float, binw: float = 2.0):
    nb = int(np.ceil(days / binw))
    counts = np.zeros(nb)
    for st in events_steps:
        counts[min(nb - 1, int(st / day // binw))] += 1
    return (np.arange(nb) + 0.5) * binw, counts / binw


def fig_population_deaths(R: list[dict], out: Path, days: float) -> None:
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    for cond in ("0", "1", "4"):
        runs = [r for r in R if r["cond"] == cond and not r["social"]]
        t = np.arange(len(runs[0]["n"])) / runs[0]["day"]
        m = np.mean([r["n"] for r in runs], axis=0)
        ax.plot(t, m, color=COND_COLOR[cond], lw=LW, label=COND_LABEL[cond])
    ax.axvspan(0, WINDOW, color=GRID, alpha=0.35, lw=0)
    ax.text(WINDOW / 2, 0.03, "burn-in (excluded from the right panel)", ha="center", va="bottom", color=MUTED,
            fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_title(f"Population, mean of {len(runs)} seeds")
    ax.set_xlabel("day")
    ax.set_ylabel("agents")
    headroom(ax, 1.2)
    ax.legend(loc="upper left", ncols=3)
    conds = ["0", "1", "4"]
    bottoms = np.zeros(len(conds))
    for ci, cause in enumerate(CAUSES):
        vals = []
        for cond in conds:
            runs = [r for r in R if r["cond"] == cond and not r["social"]]
            n = sum(1 for r in runs for st, c in r["deaths"] if c == cause and st >= WINDOW * r["day"])
            vals.append(n / len(runs) / (days - WINDOW))
        bx.bar(range(len(conds)), vals, bottom=bottoms, color=SLOT[ci], width=0.6, label=cause,
               edgecolor=SURFACE, linewidth=1.5)
        bottoms += np.array(vals)
    bx.set_xticks(range(len(conds)), [COND_LABEL[c] for c in conds])
    bx.set_title(f"Deaths per day by cause, days {WINDOW}–{days:g}")
    bx.set_ylabel("deaths per day")
    bx.grid(axis="x", visible=False)
    headroom(bx, 1.25)
    bx.legend(loc="upper left", ncols=4, fontsize=8.5)
    fig.savefig(out / "1_population_and_deaths.png", dpi=130)
    plt.close(fig)


def fig_act_rates(R: list[dict], out: Path, days: float) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True, sharex=True)
    for ax, act, title in zip(axes, ("give", "refuse_entry", "attack"), ("Gifts", "Entry refusals", "Attacks")):
        for cond in ("1", "4"):
            runs = [r for r in R if r["cond"] == cond and not r["social"]]
            rates = [per_day([st for st, a, *_ in r["acts"] if a == act], r["day"], days)[1] for r in runs]
            x = per_day([], runs[0]["day"], days)[0]
            ax.plot(x, np.mean(rates, axis=0), color=COND_COLOR[cond], lw=LW, label=COND_LABEL[cond])
        ax.set_title(f"{title} per day")
        ax.set_xlabel("day")
        headroom(ax, 1.25)
    axes[0].set_ylabel("events per day (2-day bins, mean of seeds)")
    axes[0].legend(loc="upper left")
    fig.savefig(out / "2_act_rates.png", dpi=130)
    plt.close(fig)


def fig_daily_cycle(R: list[dict], out: Path) -> None:
    nb = 48
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [0.8, 1.5, 1.5]})
    day = R[0]["day"]
    h = (np.arange(nb) + 0.5) / nb * 24
    cold = np.zeros(nb)
    cnt = np.zeros(nb)
    r0 = R[0]
    b = (np.arange(len(r0["cold"])) % day / day * nb).astype(int)
    np.add.at(cold, b, r0["cold"])
    np.add.at(cnt, b, 1)
    axes[0].fill_between(h, cold / cnt, color=SLOT[0], alpha=0.25, lw=0)
    axes[0].plot(h, cold / cnt, color=SLOT[0], lw=LW)
    axes[0].set_title("Cold over the day cycle")
    for ax, act, title in zip(axes[1:], ("refuse_entry", "attack"), ("Entry refusals", "Attacks")):
        for cond in ("1", "4"):
            runs = [r for r in R if r["cond"] == cond and not r["social"]]
            c = np.zeros(nb)
            for r in runs:
                for st, a, *_ in r["acts"]:
                    if a == act and st >= WINDOW * day:
                        c[int(st % day / day * nb)] += 1
            n_days = sum((len(r["n"]) / day - WINDOW) for r in runs)
            ax.plot(h, c / n_days * nb / 24, color=COND_COLOR[cond], lw=LW, label=COND_LABEL[cond])
        ax.set_title(f"{title} by time of day (days {WINDOW}+, all seeds)")
        ax.set_ylabel("per simulated hour")
        headroom(ax, 1.25)
        ax.legend(loc="upper left")
    axes[-1].set_xlabel("hour of the simulated day (warmest at 6, coldest at 18)")
    axes[-1].set_xlim(0, 24)
    axes[-1].set_xticks(range(0, 25, 3))
    fig.savefig(out / "3_daily_cycle.png", dpi=130)
    plt.close(fig)


def binned_rate(x, y, edges, min_n=30):
    idx = np.digitize(x, edges) - 1
    centres, rates, ns = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= min_n:
            centres.append((edges[b] + edges[b + 1]) / 2)
            rates.append(y[sel].mean())
            ns.append(int(sel.sum()))
    return np.array(centres), np.array(rates), ns


def fig_refusal_inputs(R: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), constrained_layout=True, sharey=True)
    specs = [(1, np.linspace(-1, 1, 11), "guard's liking of the arriving agent (built in)"),
             (2, np.linspace(-1.5, 1.5, 13), "guard's self-status − arriver's perceived status\n(not in the rule: any effect is emergent)"),
             (5, np.linspace(0, 0.3, 13), "guard's learned risk of a lost retaliation fight"),
             (3, np.linspace(0, 1, 11), "crowding (agents inside / capacity)")]
    for ax, (col, edges, label) in zip(axes, specs):
        for cond in ("1", "4"):
            rows = np.array([c for r in R if r["cond"] == cond and not r["social"] for c in r["checks"]], dtype=float)
            x, rate, _ = binned_rate(rows[:, col], rows[:, 4], edges)
            ax.plot(x, rate, color=COND_COLOR[cond], lw=LW, marker="o", ms=4, label=COND_LABEL[cond])
        ax.set_xlabel(label)
        ax.set_title("Refusal probability per decision")
    axes[0].set_ylabel("share of entry decisions that were refusals")
    axes[0].legend(loc="upper right")
    fig.savefig(out / "4_refusal_inputs.png", dpi=130)
    plt.close(fig)


def fig_expectations(R: list[dict], out: Path) -> None:
    """What agents learned about status (spec §5.3a): mean expectation per configuration."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.1), constrained_layout=True)
    titles = {"win": "p̂ win: someone in this position wins a fight",
              "refuse": "p̂ refuse: a guard in this position refuses",
              "attack": "p̂ attack: a refused agent in this position attacks"}
    labels = ["actor below\ntarget", "level", "actor above\ntarget", "unknown\n(stranger)"]
    width = 0.36
    for ax, name in zip(axes, ("win", "refuse", "attack")):
        for i, cond in enumerate(("1", "4")):
            tables = np.array([a["expect"][name] for r in R if r["cond"] == cond and not r["social"] and r["snapshot"]
                               for a in r["snapshot"]["agents"].values() if a.get("expect")])
            m = tables.mean(axis=0)
            ax.bar(np.arange(4) + (i - 0.5) * width, m, width=width - 0.04, color=COND_COLOR[cond],
                   label=COND_LABEL[cond], edgecolor=SURFACE, linewidth=1.5)
        ax.axhline(0.5, color=AXIS, lw=0.8, ls="--")
        ax.set_xticks(range(4), labels)
        ax.set_title(titles[name], fontsize=9.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("mean learned expectation (day 60, all agents)")
    axes[0].legend(loc="upper left")
    axes[0].text(3.4, 0.51, "prior 0.5", color=MUTED, fontsize=8, ha="right", va="bottom")
    fig.savefig(out / "8_learned_expectations.png", dpi=130)
    plt.close(fig)


def fig_status_strength(R: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True, sharex=True)
    for col, cond in enumerate(("1", "4")):
        k, S, Pm = [], [], []
        for r in R:
            if r["cond"] != cond or r["social"] or r["snapshot"] is None:
                continue
            snap = r["snapshot"]["agents"]
            perceived = defaultdict(list)
            for obs in snap.values():
                for j, (_, P, _) in obs["memory"].items():
                    perceived[j].append(P)
            for j, a in snap.items():
                if len(perceived[j]) >= 3:
                    k.append(a["strength"])
                    S.append(a["self_status"])
                    Pm.append(np.mean(perceived[j]))
        for row, (vals, what) in enumerate(((Pm, "mean perceived status P (≥ 3 raters)"), (S, "self-status Ŝ"))):
            ax = axes[row, col]
            ax.scatter(k, vals, s=14, color=COND_COLOR[cond], alpha=0.55, edgecolor="none")
            r_ = np.corrcoef(k, vals)[0, 1] if len(k) > 2 else np.nan
            ax.set_title(f"Condition {COND_LABEL[cond]}\n{what}: r = {r_:.2f}, n = {len(k)}")
            ax.set_ylabel(what)
            ax.axhline(0, color=AXIS, lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("hidden fighting strength k (last day, all seeds)")
    fig.savefig(out / "5_status_vs_strength.png", dpi=130)
    plt.close(fig)


def fig_liking(R: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    edges = np.linspace(-1, 1, 41)
    for cond in ("1", "4"):
        vals = [lik for r in R if r["cond"] == cond and not r["social"] and r["snapshot"]
                for obs in r["snapshot"]["agents"].values() for (lik, _, _) in obs["memory"].values()]
        hist, _ = np.histogram(vals, bins=edges, density=True)
        ax.step(edges[:-1], hist, where="post", color=COND_COLOR[cond], lw=LW, label=f"{COND_LABEL[cond]} ({len(vals)} pairs)")
    ax.set_title("How much agents like those they know (last day, all seeds)")
    ax.set_xlabel("liking (ledger: tanh(L/L0); affect: E)")
    ax.set_ylabel("density")
    ax.legend(loc="upper left")
    fig.savefig(out / "6_liking_distribution.png", dpi=130)
    plt.close(fig)


def fig_feud_spiral(R: list[dict], out: Path, days: float) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    biases = sorted({r["social"].get("refuse_bias") for r in R if r["social"]}, reverse=True)
    for i, bias in enumerate(biases):
        runs = [r for r in R if r["social"].get("refuse_bias") == bias]
        x = per_day([], runs[0]["day"], days)[0]
        rates = []
        for r in runs:
            y = per_day([st for st, a, *_ in r["acts"] if a == "attack"], r["day"], days)[1]
            ax.plot(x, y, color=SLOT[i], lw=0.6, alpha=0.5)
            rates.append(y)
        m = np.mean(rates, axis=0)
        ax.plot(x, m, color=SLOT[i], lw=LW, label=f"refusal bias γ_R0 = {bias:g}")
        end_label(ax, x[-1], m[-1], f"γ_R0 = {bias:g}")
    ax.set_title(f"Feud spiral: attacks per day in condition 1 for three refusal biases ({len(runs)} seeds each, thin = single seed)")
    ax.set_xlabel("day")
    ax.set_ylabel("attacks per day")
    headroom(ax, 1.2)
    ax.legend(loc="upper left", ncols=3)
    fig.savefig(out / "7_feud_spiral.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--spiral-seeds", type=int, default=4)
    ap.add_argument("--out", default="figures/stage2")
    ap.add_argument("--reuse", action="store_true", help="reuse cached simulation results from runs/")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs") / f"stage2_plot_{args.seeds}seeds_{args.days:g}days.pkl"
    if args.reuse and cache.exists():
        R = pickle.loads(cache.read_bytes())
    else:
        jobs = [(c, s, args.days, {}) for c in ("0", "1", "4") for s in range(args.seeds)]
        jobs += [("1", s, args.days, {"refuse_bias": b}) for b in (-4.0, -4.5) for s in range(args.spiral_seeds)]
        with ProcessPoolExecutor() as pool:
            R = list(pool.map(run_one, jobs))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(R))
    default_bias = Config().social.refuse_bias
    spiral = [r for r in R if r["social"]] + [
        {**r, "social": {"refuse_bias": default_bias}} for r in R if r["cond"] == "1" and not r["social"] and r["seed"] < args.spiral_seeds
    ]
    fig_population_deaths(R, out, args.days)
    fig_act_rates(R, out, args.days)
    fig_daily_cycle(R, out)
    fig_refusal_inputs(R, out)
    fig_status_strength(R, out)
    fig_liking(R, out)
    fig_feud_spiral(spiral, out, args.days)
    fig_expectations(R, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

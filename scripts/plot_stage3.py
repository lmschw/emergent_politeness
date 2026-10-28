"""Sanity-check figures for stage 3 (content language, conditions 2 and 3, compared with 0 and 1).

    uv run python scripts/plot_stage3.py --seeds 8 --days 60
    uv run python scripts/plot_stage3.py --reuse   # replot from cached results in runs/
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from emergent_politeness.config import CONDITIONS, Config
from emergent_politeness.learning import FEATURES
from emergent_politeness.model import Simulation
from emergent_politeness.plotting import AXIS, GRID, INK2, LW, MUTED, SLOT, SURFACE, headroom, plt

CONDS = ("0", "1", "2", "3")
COND_LABEL = {"0": "0 · no interaction", "1": "1 · acts", "2": "2 · language", "3": "3 · acts + language"}
COND_COLOR = {"0": SLOT[0], "1": SLOT[1], "2": SLOT[2], "3": SLOT[3]}
CAUSES = ["starvation", "cold", "injury", "fight"]
WINDOW = 40
DAY = 600


def run_one(args: tuple[str, int, float, dict]) -> dict:
    cond, seed, days, social = args
    cfg = Config(condition=CONDITIONS[cond]).with_(sim={"seed": seed, "steps": int(days * DAY)})
    if social:
        cfg = cfg.with_(social=social)
    sim = Simulation(cfg)
    # Validation of learning: empirical return after acting vs not acting, and learned advantage (days ≥ WINDOW)
    check = defaultdict(lambda: {"ret": [0.0, 0.0], "n": [0, 0], "adv": 0.0, "nadv": 0})
    if sim.learner is not None:
        def on_resolve(d, r, parts):
            if d.due - cfg.social.return_window < WINDOW * DAY or d.subject is not None:
                return  # own decisions only
            c = check[d.act]
            k = 1 if d.chosen else 0
            c["ret"][k] += r
            c["n"][k] += 1
            c["adv"] += float(d.agent.qmodels[d.act][0] @ d.f)
            c["nadv"] += 1
        sim.learner.on_resolve = on_resolve
    sim.run()
    weights_daily = []
    for snap in sim.snapshots:
        ws = [a["advantage"] for a in snap["agents"].values() if a.get("advantage")]
        weights_daily.append({act: np.mean([w[act] for w in ws], axis=0) for act in ws[0]} if ws else None)
    utter = [(r["step"], r["act"], r["content"], r.get("request"), r.get("asker_hunger"), r.get("own_hunger"),
              r.get("own_fatigue")) for r in sim.log.of_kind("utterance")]
    return {
        "cond": cond, "seed": seed,
        "n": np.asarray(sim.series["n"]),
        "deaths": [(r["step"], r["cause"]) for r in sim.log.of_kind("death")],
        "acts": [(r["step"], r["act"]) for r in sim.log.of_kind("act")],
        "utter": utter,
        "weights_daily": weights_daily,
        "last_adv": [a["advantage"] for a in sim.snapshots[-1]["agents"].values() if a.get("advantage")]
        if sim.snapshots else [],
        "check": {k: dict(v) for k, v in check.items()},
    }


def rate(steps, days, binw=2.0):
    nb = int(np.ceil(days / binw))
    c = np.zeros(nb)
    for st in steps:
        c[min(nb - 1, int(st / DAY // binw))] += 1
    return (np.arange(nb) + 0.5) * binw, c / binw


def runs_of(R, cond):
    return [r for r in R if r["cond"] == cond]


def fig_population_deaths(R, out, days):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.2), constrained_layout=True, gridspec_kw={"width_ratios": [1.6, 1]})
    for cond in CONDS:
        rs = runs_of(R, cond)
        m = np.mean([r["n"] for r in rs], axis=0)
        ax.plot(np.arange(len(m)) / DAY, m, color=COND_COLOR[cond], lw=LW, label=COND_LABEL[cond])
    ax.axvspan(0, WINDOW, color=GRID, alpha=0.35, lw=0)
    ax.text(WINDOW / 2, 0.03, "burn-in", ha="center", va="bottom", color=MUTED, fontsize=8.5, transform=ax.get_xaxis_transform())
    ax.set_title("Population, mean of seeds")
    ax.set_xlabel("day")
    ax.set_ylabel("agents")
    headroom(ax, 1.2)
    ax.legend(loc="upper left", ncols=2)
    bottoms = np.zeros(len(CONDS))
    for ci, cause in enumerate(CAUSES):
        vals = []
        for cond in CONDS:
            rs = runs_of(R, cond)
            n = sum(1 for r in rs for st, c in r["deaths"] if c == cause and st >= WINDOW * DAY)
            vals.append(n / len(rs) / (days - WINDOW))
        bx.bar(range(len(CONDS)), vals, bottom=bottoms, color=SLOT[ci], width=0.6, label=cause, edgecolor=SURFACE, linewidth=1.5)
        bottoms += np.array(vals)
    bx.set_xticks(range(len(CONDS)), [COND_LABEL[c].replace(" · ", "\n") for c in CONDS])
    bx.set_title(f"Deaths per day by cause, days {WINDOW}–{days:g}")
    bx.grid(axis="x", visible=False)
    headroom(bx, 1.25)
    bx.legend(loc="upper left", ncols=4, fontsize=8.5)
    fig.savefig(out / "1_population_and_deaths.png", dpi=130)
    plt.close(fig)


def fig_rates(R, out, days):
    panels = [("Food requests", "2,3", lambda r: [u[0] for u in r["utter"] if u[1] == "request" and u[2] == "food"]),
              ("Task requests", "2,3", lambda r: [u[0] for u in r["utter"] if u[1] == "request" and u[2] != "food"]),
              ("Yes answers", "2,3", lambda r: [u[0] for u in r["utter"] if u[1] == "respond" and u[2] == "yes"]),
              ("Gifts", "1,3", lambda r: [st for st, a in r["acts"] if a == "give"]),
              ("Entry refusals", "1,3", lambda r: [st for st, a in r["acts"] if a == "refuse_entry"]),
              ("Attacks", "1,3", lambda r: [st for st, a in r["acts"] if a == "attack"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True, sharex=True)
    for ax, (title, conds, sel) in zip(axes.flat, panels):
        for cond in conds.split(","):
            rs = runs_of(R, cond)
            x = rate([], days)[0]
            ax.plot(x, np.mean([rate(sel(r), days)[1] for r in rs], axis=0), color=COND_COLOR[cond], lw=LW, label=COND_LABEL[cond])
        ax.set_title(f"{title} per day")
        headroom(ax, 1.25)
        ax.legend(loc="upper left", fontsize=8.5)
    for ax in axes[1]:
        ax.set_xlabel("day")
    fig.savefig(out / "2_rates.png", dpi=130)
    plt.close(fig)


def fig_compliance(R, out, days):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
    ax = axes[0]
    for cond in ("2", "3"):
        for kind, ls in (("food", "-"), ("task", "--")):
            ys = []
            for r in runs_of(R, cond):
                ans = [(u[0], u[2]) for u in r["utter"] if u[1] == "respond" and ((u[3] == "food") == (kind == "food"))]
                nb = int(np.ceil(days / 2))
                yes, tot = np.zeros(nb), np.zeros(nb)
                for st, c in ans:
                    b = min(nb - 1, int(st / DAY // 2))
                    tot[b] += 1
                    yes[b] += c == "yes"
                ys.append(np.where(tot > 0, yes / np.maximum(tot, 1), np.nan))
            if kind == "food" and cond == "2":
                continue
            ax.plot((np.arange(len(ys[0])) + 0.5) * 2, np.nanmean(ys, axis=0), color=COND_COLOR[cond], lw=LW, ls=ls,
                    label=f"{COND_LABEL[cond]}, {kind}")
    ax.set_title("Share of requests answered yes")
    ax.set_xlabel("day")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=8)
    # yes-rate vs asker hunger (food) and own fatigue (task), condition 3, days ≥ WINDOW
    for ax, (kind, idx, label) in zip(axes[1:], (("food", 4, "asker's hunger"), ("task", 6, "own fatigue (addressee)"))):
        for cond in ("2", "3"):
            if kind == "food" and cond == "2":
                continue
            pts = [(u[idx], u[2] == "yes") for r in runs_of(R, cond) for u in r["utter"]
                   if u[1] == "respond" and u[0] >= WINDOW * DAY and ((u[3] == "food") == (kind == "food")) and u[idx] is not None]
            if not pts:
                continue
            x, y = np.array(pts, dtype=float).T
            edges = np.linspace(x.min(), x.max(), 9)
            idxs = np.digitize(x, edges) - 1
            cx = [(edges[b] + edges[b + 1]) / 2 for b in range(8) if (idxs == b).sum() >= 30]
            cy = [y[idxs == b].mean() for b in range(8) if (idxs == b).sum() >= 30]
            ax.plot(cx, cy, color=COND_COLOR[cond], lw=LW, marker="o", ms=4, label=COND_LABEL[cond])
        ax.set_title(f"Yes to {kind} requests vs {label}", fontsize=9.5)
        ax.set_xlabel(label)
        ax.set_ylim(0, None)
        ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "3_compliance.png", dpi=130)
    plt.close(fig)


def fig_learned_weights(R, out):
    acts = ("ask_food", "comply_food", "ask_task", "comply_task", "give", "refuse", "attack")
    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5), constrained_layout=True)
    for ax, act in zip(axes.flat, acts):
        names = FEATURES[act]
        conds = [c for c in ("1", "2", "3") if any(act in a for r in runs_of(R, c) for a in r["last_adv"][:1])]
        width = 0.8 / max(1, len(conds))
        for i, cond in enumerate(conds):
            ws = np.array([a[act] for r in runs_of(R, cond) for a in r["last_adv"]])
            used = np.abs(ws).sum(axis=0) > 0
            m = ws.mean(axis=0)
            y = np.arange(len(names)) + (i - (len(conds) - 1) / 2) * width
            ax.barh(y[used] if used.any() else y, m[used] if used.any() else m, height=width * 0.9, color=COND_COLOR[cond],
                    label=COND_LABEL[cond])
        ax.axvline(0, color=AXIS, lw=0.9)
        ax.set_yticks(range(len(names)), [n.replace("cfg_", "status: ").replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{act}: learned advantage (food units)", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=8)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0.0, 0.5, "Advantage = how much more material return\nfollows acting than not acting.\n"
                       "< 0: the act tends to cost me\n> 0: it tends to pay\n\nLiking, status and anger have no built-in\n"
                       "effect: their weights are learned.", va="center", fontsize=10, color=INK2)
    fig.savefig(out / "4_learned_weights.png", dpi=130)
    plt.close(fig)


def fig_validation(R, out):
    """Does what agents learned match what actually happened? (days ≥ WINDOW, conditions 1–3)"""
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for cond in ("1", "2", "3"):
        xs, ys, labels = [], [], []
        acts = sorted({a for r in runs_of(R, cond) for a in r["check"]})
        for act in acts:
            tot = {"ret": [0.0, 0.0], "n": [0, 0], "adv": 0.0, "nadv": 0}
            for r in runs_of(R, cond):
                c = r["check"].get(act)
                if c:
                    for k in (0, 1):
                        tot["ret"][k] += c["ret"][k]
                        tot["n"][k] += c["n"][k]
                    tot["adv"] += c["adv"]
                    tot["nadv"] += c["nadv"]
            if min(tot["n"]) < 50:
                continue
            xs.append(tot["ret"][1] / tot["n"][1] - tot["ret"][0] / tot["n"][0])
            ys.append(tot["adv"] / tot["nadv"])
            labels.append(act)
        ax.scatter(xs, ys, s=40, color=COND_COLOR[cond], label=COND_LABEL[cond], edgecolor=SURFACE, zorder=3)
        for x, y, lab in zip(xs, ys, labels):
            ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    lim = ax.get_xlim()
    lo, hi = min(lim[0], ax.get_ylim()[0]), max(lim[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=1, ls="--")
    ax.set_xlabel("observed: mean return after acting − after not acting")
    ax.set_ylabel("learned: mean advantage")
    ax.set_title("Learning check: learned advantage vs what actually followed\n(points near the dashed line = unbiased learning)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(out / "5_learning_check.png", dpi=130)
    plt.close(fig)


def fig_learning_over_time(R, out):
    panels = [("comply_food", "other_hunger"), ("comply_task", "own_fatigue"), ("ask_food", "own_hunger"),
              ("refuse", "crowding"), ("give", "liking"), ("comply_food", "liking")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), constrained_layout=True, sharex=True)
    for ax, (act, feat) in zip(axes.flat, panels):
        n = FEATURES[act].index(feat)
        for cond in ("1", "2", "3"):
            series = [np.array([np.nan if w is None or act not in w else w[act][n] for w in r["weights_daily"]])
                      for r in runs_of(R, cond) if r["weights_daily"]]
            if not series or np.all(np.isnan(np.vstack(series))) or np.nanmax(np.abs(np.vstack(series))) == 0:
                continue
            m = np.nanmean(np.vstack(series), axis=0)
            ax.plot(np.arange(len(m)), m, color=COND_COLOR[cond], lw=LW, label=COND_LABEL[cond])
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.set_title(f"{act}: weight on {feat.replace('_', ' ')}", fontsize=9.5)
        ax.legend(loc="lower left", fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("day")
    fig.savefig(out / "6_learning_over_time.png", dpi=130)
    plt.close(fig)


def fig_daily_cycle(R, out):
    nb = 48
    h = (np.arange(nb) + 0.5) / nb * 24
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    series = [("3", "food requests", lambda u: u[1] == "request" and u[2] == "food"),
              ("3", "task requests", lambda u: u[1] == "request" and u[2] != "food"),
              ("2", "task requests", lambda u: u[1] == "request" and u[2] != "food")]
    for i, (cond, label, sel) in enumerate(series):
        c = np.zeros(nb)
        rs = runs_of(R, cond)
        for r in rs:
            for u in r["utter"]:
                if u[0] >= WINDOW * DAY and sel(u):
                    c[int(u[0] % DAY / DAY * nb)] += 1
        n_days = sum(len(r["n"]) / DAY - WINDOW for r in rs)
        ax.plot(h, c / n_days * nb / 24, color=SLOT[i], lw=LW, label=f"{COND_LABEL[cond]}: {label}")
    ax.set_title(f"Requests by time of day (days {WINDOW}+)")
    ax.set_xlabel("hour of the simulated day (warmest at 6, coldest at 18)")
    ax.set_ylabel("per simulated hour")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    headroom(ax, 1.2)
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(out / "7_daily_cycle.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--out", default="figures/stage3")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--social", action="append", default=[], metavar="param=value")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.social)}
    tag = "".join(f"_{k}-{v}" for k, v in sorted(base.items()))
    cache = Path("runs") / f"stage3_plot_{args.seeds}seeds_{args.days:g}days{tag}.pkl"
    if args.reuse and cache.exists():
        R = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor() as pool:
            R = list(pool.map(run_one, [(c, s, args.days, base) for c in CONDS for s in range(args.seeds)]))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(R))
    fig_population_deaths(R, out, args.days)
    fig_rates(R, out, args.days)
    fig_compliance(R, out, args.days)
    fig_learned_weights(R, out)
    fig_validation(R, out)
    fig_learning_over_time(R, out)
    fig_daily_cycle(R, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

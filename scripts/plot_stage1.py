"""Sanity-check figures for the stage-1 survival world.

    uv run python scripts/plot_stage1.py --seeds 16 --days 60 --out figures/stage1
    uv run python scripts/plot_stage1.py --reuse   # replot from cached results in runs/
"""

from __future__ import annotations

import argparse
import math
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, Wedge

from emergent_politeness.agents import WORK_TASKS, Task
from emergent_politeness.config import Config
from emergent_politeness.model import Simulation
from emergent_politeness.plotting import AXIS, BLUE_RAMP, GRID, INK, INK2, LW, MUTED, SLOT, SURFACE, end_label, headroom, plt

TRACK_DAY = 45  # day whose trajectories are recorded on seed 0


def run_seed(args: tuple[int, float]) -> dict:
    seed, days = args
    cfg = Config().with_(sim={"seed": seed, "steps": int(days * Config().world.day_length)})
    sim = Simulation(cfg)
    day = cfg.world.day_length
    tracks: dict[int, list] = defaultdict(list)
    snapshots = {}
    for _ in range(cfg.sim.steps):
        sim.step()
        if seed == 0 and TRACK_DAY * day <= sim.k < (TRACK_DAY + 1) * day:
            for a in sim.agents.values():
                p = sim.physics.position(a.id)
                tracks[a.id].append((sim.k, p[0], p[1], a.task.value))
            phase = (sim.k % day) / day
            if abs(phase - 0.25) < 1e-9 or abs(phase - 0.75) < 1e-9:  # warmest / coldest moment
                snapshots["noon" if phase < 0.5 else "midnight"] = [
                    tuple(sim.physics.position(a.id)) for a in sim.agents.values()
                ]
    return {
        "seed": seed,
        "series": {k: np.asarray(v) for k, v in sim.series.items()},
        "deaths": [(r["step"], r["cause"]) for r in sim.log.of_kind("death")],
        "task_steps": dict(sim.task_steps),
        "tracks": dict(tracks),
        "snapshots": snapshots,
        "patches": [(p.kind, tuple(p.pos), p.danger) for p in sim.env.patches],
        "shelter": (tuple(sim.env.shelter.centre), sim.env.shelter.radius, cfg.world.door_width),
        "day": day,
    }


def fig_timeseries(r: dict, out: Path) -> None:
    s, day = r["series"], r["day"]
    t = s["step"] / day
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    for key, label, c in (("n", "population", SLOT[0]), ("n_inside", "inside shelter", SLOT[1]), ("capacity", "capacity", SLOT[2])):
        ax.plot(t, s[key], color=c, lw=LW if key != "n_inside" else 0.6, label=label)
        end_label(ax, t[-1], s[key][-1], label)
    ax.set_title("Agents (seed 0)")
    ax.set_ylabel("agents")
    ax.legend(loc="upper left", ncols=3)
    ax = axes[1]
    for key, label, c in (("food", "food store", SLOT[0]), ("material", "material store", SLOT[1])):
        ax.plot(t, s[key], color=c, lw=LW, label=label)
    ax.set_title("Shelter stores")
    ax.set_ylabel("units")
    ax.legend(loc="upper right", ncols=2)
    ax = axes[2]
    ax.plot(t, s["integrity"], color=SLOT[0], lw=0.7)
    ax.set_title("Shelter integrity")
    ax.set_ylim(0, 1.02)
    ax = axes[3]
    for key, label, c in (("mean_hunger", "hunger", SLOT[0]), ("mean_exposure", "exposure (cold)", SLOT[1]), ("mean_fatigue", "fatigue", SLOT[2])):
        ax.plot(t, s[key], color=c, lw=0.9, label=label)
    ax.set_title("Mean body state")
    ax.set_ylim(0, 1)
    ax.set_xlabel("day")
    ax.legend(loc="upper right", ncols=3)
    for a in axes:
        a.axvspan(0, 40, color=GRID, alpha=0.35, lw=0)
    headroom(axes[0], 1.3)
    headroom(axes[1], 1.25)
    headroom(axes[3], 1.25, fixed_top=1.0)
    axes[2].text(20, 0.04, "burn-in (discarded)", ha="center", va="bottom", color=MUTED, fontsize=8.5)
    fig.savefig(out / "1_timeseries_seed0.png", dpi=130)
    plt.close(fig)


def fig_zoom(r: dict, out: Path) -> None:
    s, day = r["series"], r["day"]
    sel = (s["step"] >= 45 * day) & (s["step"] < 48 * day)
    t = s["step"][sel] / day
    fig, axes = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 2.2, 1.6]})
    ax = axes[0]
    ax.fill_between(t, s["cold"][sel], color=SLOT[0], alpha=0.25, lw=0)
    ax.plot(t, s["cold"][sel], color=SLOT[0], lw=LW)
    ax.set_title("Cold (0 = warm, 1 = coldest)")
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    outside = s["n"][sel] - s["n_inside"][sel]
    for y, label, c in ((s["n_inside"][sel], "inside", SLOT[0]), (outside, "outside", SLOT[1]), (s["capacity"][sel], "capacity", SLOT[2]), (s["waiting"][sel], "waiting at door", SLOT[3])):
        ax.plot(t, y, color=c, lw=LW, label=label)
        end_label(ax, t[-1], y[-1], label)
    ax.set_title("Where agents are, days 45–48 (seed 0)")
    ax.set_ylabel("agents")
    headroom(ax, 1.3)
    ax.legend(loc="upper left", ncols=4)
    ax = axes[2]
    for i, task in enumerate(WORK_TASKS):
        y = s[f"task_{task.value}"][sel]
        ax.plot(t, y, color=SLOT[i], lw=LW, label=task.value)
    ax.set_title("Agents per work task")
    ax.set_ylabel("agents")
    ax.set_xlabel("day")
    headroom(ax, 1.35)
    ax.legend(loc="upper left", ncols=4)
    fig.savefig(out / "2_three_days_seed0.png", dpi=130)
    plt.close(fig)


def fig_population(results: list[dict], out: Path) -> None:
    day = results[0]["day"]
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ns = []
    for r in results:
        t = r["series"]["step"] / day
        ax.plot(t, r["series"]["n"], color=AXIS, lw=0.6)
        ns.append(r["series"]["n"])
    m = np.mean(ns, axis=0)
    ax.plot(t, m, color=SLOT[0], lw=LW)
    end_label(ax, t[-1], m[-1], f"mean of {len(results)} seeds", color=INK)
    ax.axvspan(0, 40, color=GRID, alpha=0.35, lw=0)
    ax.text(20, ax.get_ylim()[0] + 1, "burn-in (discarded)", ha="center", va="bottom", color=MUTED, fontsize=8.5)
    ax.set_title("Population size, every seed (grey) and mean (blue)")
    ax.set_xlabel("day")
    ax.set_ylabel("agents")
    fig.savefig(out / "3_population_all_seeds.png", dpi=130)
    plt.close(fig)


def fig_deaths(results: list[dict], out: Path, days: float) -> None:
    day, binw = results[0]["day"], 5
    nbins = int(math.ceil(days / binw))
    causes = ["starvation", "cold", "injury"]
    counts = np.zeros((len(causes), nbins))
    for r in results:
        for step, cause in r["deaths"]:
            counts[causes.index(cause), min(nbins - 1, int(step / day // binw))] += 1
    rate = counts / len(results) / binw  # deaths per day per run
    x = np.arange(nbins) * binw + binw / 2
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    bottom = np.zeros(nbins)
    for i, cause in enumerate(causes):
        ax.bar(x, rate[i], bottom=bottom, width=binw - 0.6, color=SLOT[i], label=cause, edgecolor=SURFACE, linewidth=1.5)
        bottom += rate[i]
    ax.set_title(f"Deaths per day by cause ({binw}-day bins, mean of {len(results)} seeds)")
    ax.set_xlabel("day")
    ax.set_ylabel("deaths per day")
    headroom(ax, 1.2)
    ax.legend(loc="upper left", ncols=3)
    ax.grid(axis="x", visible=False)
    fig.savefig(out / "4_deaths_by_cause.png", dpi=130)
    plt.close(fig)


def fig_daily_cycle(results: list[dict], out: Path) -> None:
    day, nb = results[0]["day"], 48
    acc = defaultdict(lambda: np.zeros(nb))
    cnt = np.zeros(nb)
    for r in results:
        s = r["series"]
        keep = s["step"] >= 40 * day
        b = ((s["step"][keep] % day) / day * nb).astype(int)
        np.add.at(cnt, b, 1)
        for key in ("cold", "n", "n_inside", "waiting", *[f"task_{t.value}" for t in Task]):
            np.add.at(acc[key], b, s[key][keep])
    mean = {k: v / cnt for k, v in acc.items()}
    h = (np.arange(nb) + 0.5) / nb * 24  # hours; sin peak (warmest) at 6 h, coldest at 18 h
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [1, 2.4]})
    ax = axes[0]
    ax.fill_between(h, mean["cold"], color=SLOT[0], alpha=0.25, lw=0)
    ax.plot(h, mean["cold"], color=SLOT[0], lw=LW)
    ax.set_title("Cold over the day cycle")
    ax.set_ylim(0, 1.05)
    ax = axes[1]
    series = [("task_collect_food", "collecting food"), ("task_collect_material", "collecting material"), ("task_maintain", "maintaining"),
              ("task_build", "building"), ("waiting", "waiting at door")]
    for i, (key, label) in enumerate(series):
        ax.plot(h, mean[key], color=SLOT[i], lw=LW, label=label)
    ax.set_title("Mean agents per activity by time of day (days 40–60, all seeds)")
    ax.set_ylabel("agents")
    ax.set_xlabel("hour of the simulated day (warmest at 6, coldest at 18)")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3))
    ax.legend(loc="upper right", ncols=3)
    fig.savefig(out / "5_daily_cycle.png", dpi=130)
    plt.close(fig)


def fig_division_of_labour(r: dict, out: Path) -> None:
    per = defaultdict(lambda: np.zeros(len(WORK_TASKS)))
    for (aid, task), n in r["task_steps"].items():
        per[aid][WORK_TASKS.index(task)] += n
    rows = [(aid, v / v.sum()) for aid, v in per.items() if v.sum() >= 600]  # ≥ 600 work steps
    rows.sort(key=lambda x: (int(np.argmax(x[1])), -x[1].max()))
    mat = np.array([v for _, v in rows])
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(WORK_TASKS)), [t.value for t in WORK_TASKS])
    ax.set_yticks([])
    ax.set_ylabel(f"individual agents ({len(rows)} with ≥ 600 work steps), sorted by main task")
    ax.grid(False)
    ax.set_title("Share of each agent's work per task (seed 0)")
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("share of the agent's work", color=INK2)
    cb.outline.set_visible(False)
    fig.savefig(out / "6_division_of_labour_seed0.png", dpi=130)
    plt.close(fig)


def draw_arena(ax, r: dict) -> None:
    (cx, cy), R, door_w = r["shelter"]
    half = math.degrees(math.asin(door_w / 2 / R))
    ax.add_patch(Wedge((cx, cy), R + 0.1, half, 360 - half, width=0.25, color=INK2))
    for kind, (px, py), danger in r["patches"]:
        ax.add_patch(Circle((px, py), 1.5, facecolor=GRID, edgecolor=AXIS, lw=0.8))
        ax.text(px, py, f"{'F' if kind == 'food' else 'M'}\nd={danger:g}", ha="center", va="center", fontsize=7.5, color=INK2)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_arena(r: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), constrained_layout=True)
    ax = axes[0]
    draw_arena(ax, r)
    tracks = sorted(r["tracks"].items(), key=lambda kv: -sum(row[3] in ("collect_food", "collect_material") for row in kv[1]))[:3]
    for i, (aid, rows) in enumerate(tracks):
        xy = np.array([(x, y) for _, x, y, _ in rows])
        ax.plot(xy[:, 0], xy[:, 1], color=SLOT[i], lw=1.0, alpha=0.9, label=f"agent {aid}")
    ax.legend(loc="upper left")
    ax.set_title(f"Paths of 3 busy agents over day {TRACK_DAY} (seed 0)")
    ax = axes[1]
    draw_arena(ax, r)
    for i, (name, marker) in enumerate((("noon", "o"), ("midnight", "s"))):
        pts = np.array(r["snapshots"].get(name, [])).reshape(-1, 2)
        ax.scatter(pts[:, 0], pts[:, 1], s=26, marker=marker, color=SLOT[i], edgecolor=SURFACE, lw=0.8,
                   label=f"{name} ({len(pts)} agents)", zorder=3)
    ax.legend(loc="upper left")
    ax.set_title(f"Agent positions at the warmest and coldest moment, day {TRACK_DAY}")
    fig.savefig(out / "7_arena_seed0.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--out", default="figures/stage1")
    ap.add_argument("--reuse", action="store_true", help="reuse cached simulation results from runs/")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs") / f"stage1_plot_{args.seeds}seeds_{args.days:g}days.pkl"
    if args.reuse and cache.exists():
        results = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor() as pool:
            results = list(pool.map(run_seed, [(s, args.days) for s in range(args.seeds)]))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(results))
    seed0 = results[0]
    fig_timeseries(seed0, out)
    fig_zoom(seed0, out)
    fig_population(results, out)
    fig_deaths(results, out, args.days)
    fig_daily_cycle(results, out)
    fig_division_of_labour(seed0, out)
    fig_arena(seed0, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

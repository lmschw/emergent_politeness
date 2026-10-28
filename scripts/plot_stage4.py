"""Sanity-check figures for stage 4 (affect markers: conditions 5 and 6, and 6 with natural variation).

    uv run python scripts/plot_stage4.py --seeds 8 --days 60
    uv run python scripts/plot_stage4.py --reuse
    uv run python scripts/plot_stage4.py --set long --seeds 4 --days 150 --out figures/stage4_long
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from emergent_politeness.config import CONDITIONS, Config
from emergent_politeness.language import SPEECH_ACTS
from emergent_politeness.model import Simulation
from emergent_politeness.plotting import AXIS, GRID, INK2, LW, MUTED, SLOT, SURFACE, headroom, plt

DAY = 600
WINDOW = 40
# variant -> (condition, social overrides, language overrides, label)
VARIANT_SETS = {
    "main": {"5": ("5", {}, {}, "5 · language + affect"), "6": ("6", {}, {}, "6 · full"),
             "6+var": ("6", {"innate_variation": True}, {}, "6 · full + natural variation")},
    "long": {"s01": ("6", {}, {"sigma_c0": 0.1}, "6 · initial spread 0.1"),
             "s03": ("6", {}, {"sigma_c0": 0.3}, "6 · initial spread 0.3")},
}
VARIANTS = VARIANT_SETS["main"]
LABEL = {k: v[3] for k, v in VARIANTS.items()}
COLOR = {k: SLOT[i] for i, k in enumerate(VARIANTS)}


def use_set(name: str) -> None:
    global VARIANTS, LABEL, COLOR
    VARIANTS = VARIANT_SETS[name]
    LABEL = {k: v[3] for k, v in VARIANTS.items()}
    COLOR = {k: SLOT[i] for i, k in enumerate(VARIANTS)}
SPEECH_LABEL = {"request_food": "request food", "request_task": "request task", "respond_yes": "answer yes",
                "respond_no": "answer no"}
BIN_LABEL = ["speaker below", "level", "speaker above", "stranger"]


def run_one(args):
    variant, seed, days, set_name = args
    cond, social, language, _ = VARIANT_SETS[set_name][variant]
    cfg = Config(condition=CONDITIONS[cond]).with_(sim={"seed": seed, "steps": int(days * DAY)})
    if social:
        cfg = cfg.with_(social=social)
    if language:
        cfg = cfg.with_(language=language)
    sim = Simulation(cfg).run()
    daily = []
    for snap in sim.snapshots:
        C = np.array([a["connotation"] for a in snap["agents"].values() if a.get("connotation") is not None])
        Nm = np.array([a["norm"] for a in snap["agents"].values() if a.get("norm") is not None])
        W = [a["advantage"] for a in snap["agents"].values() if a.get("advantage")]
        daily.append({
            "c_mean": C.mean(axis=0) if len(C) else None,  # markers × (e, p)
            "c_sd": C.std(axis=0) if len(C) else None,
            "norm_mean": Nm.mean(axis=(0, 2)) if len(Nm) else None,  # speech acts × (e, p), averaged over bins
            "liking_w": np.mean([[w[act][4] for act in w] for w in W]) if W else np.nan,
        })
    utter = []
    for r in sim.log.of_kind("utterance"):
        if "exp_e" not in r:
            continue
        speech = f"request_{'food' if r['content'] == 'food' else 'task'}" if r["act"] == "request" else f"respond_{r['content']}"
        utter.append((r["step"], speech, tuple(r["markers"]), r.get("app_e"), r.get("app_p"), r["exp_e"],
                      r.get("accidental_rudeness"), r["s_cfg"], r["s_liking"]))
    last = sim.snapshots[-1]["agents"] if sim.snapshots else {}
    return {"variant": variant, "seed": seed, "n": np.asarray(sim.series["n"]), "daily": daily, "utter": utter,
            "deaths": [(r["step"], r["cause"]) for r in sim.log.of_kind("death")],
            "acts": [(r["step"], r["act"]) for r in sim.log.of_kind("act")],
            "last_c": [a["connotation"] for a in last.values() if a.get("connotation") is not None]}


def of(R, v):
    return [r for r in R if r["variant"] == v]


def fig_connotations(R, out):
    """Do markers acquire distinct, shared meanings? Population mean (e, p) of each marker over time."""
    fig, axes = plt.subplots(2, len(VARIANTS), figsize=(5.4 * len(VARIANTS), 7.5), constrained_layout=True,
                             sharex=True, squeeze=False)
    for col, v in enumerate(VARIANTS):
        for row, dim in enumerate(("e (warm +)", "p (dominant +)")):
            ax = axes[row, col]
            series = np.array([[d["c_mean"][:, row] for d in r["daily"]] for r in of(R, v)])  # seeds × days × markers
            for m in range(series.shape[2]):
                ax.plot(np.arange(series.shape[1]), series[:, :, m].mean(axis=0), color=SLOT[m], lw=LW, label=f"m{m + 1}")
            ax.axhline(0, color=AXIS, lw=0.9)
            ax.set_title(f"{LABEL[v]}\nconnotation {dim}, mean over agents and seeds", fontsize=9.5)
            if col == 0:
                ax.set_ylabel(dim)
    axes[0, 0].legend(loc="lower left", ncols=3, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("day")
    fig.savefig(out / "1_marker_meanings.png", dpi=130)
    plt.close(fig)


def fig_meanings_per_seed(R, out):
    """Are meanings arbitrary conventions? Final e-connotation of each marker, per seed (condition 6)."""
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5.4 * len(VARIANTS), 4.5), constrained_layout=True,
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ax, v in zip(axes, VARIANTS):
        for i, r in enumerate(of(R, v)):
            C = np.array(r["last_c"])
            ax.scatter(np.arange(C.shape[1]) + (i - 3.5) * 0.08, C[:, :, 0].mean(axis=0), s=22, color=SLOT[i % 8],
                       edgecolor=SURFACE, lw=0.5)
        ax.axhline(0, color=AXIS, lw=0.9)
        ax.set_xticks(range(6), [f"m{m + 1}" for m in range(6)])
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{LABEL[v]}: last-day e of each marker\n(one colour per seed)", fontsize=9.5)
    axes[0].set_ylabel("connotation e (mean over agents)")
    fig.savefig(out / "2_meanings_per_seed.png", dpi=130)
    plt.close(fig)


def fig_agreement(R, out):
    """Convention: how much do agents disagree about what a marker means (SD across agents, mean over markers)?"""
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for v in VARIANTS:
        sd = np.array([[d["c_sd"][:, 0].mean() for d in r["daily"]] for r in of(R, v)])
        ax.plot(np.arange(sd.shape[1]), sd.mean(axis=0), color=COLOR[v], lw=LW, label=LABEL[v])
    ax.axhline(0.1, color=MUTED, lw=0.8, ls="--")
    ax.text(1, 0.102, "initial spread (σ_c0)", color=MUTED, fontsize=8, va="bottom")
    ax.set_title("Disagreement about marker meanings (SD of e across agents)")
    ax.set_xlabel("day")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    fig.savefig(out / "3_agreement.png", dpi=130)
    plt.close(fig)


def fig_usage(R, out):
    """Which markers are used in which speech act (condition 6, days ≥ WINDOW)?"""
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(5.4 * len(VARIANTS), 4.3), constrained_layout=True,
                             squeeze=False)
    axes = axes[0]
    for ax, v in zip(axes, VARIANTS):
        mat = np.zeros((len(SPEECH_ACTS), 7))
        for r in of(R, v):
            for st, speech, markers, *_ in r["utter"]:
                if st < WINDOW * DAY:
                    continue
                i = SPEECH_ACTS.index(speech)
                mat[i, 6] += 1  # utterances
                for m in markers:
                    mat[i, m] += 1
        per = mat[:, :6] / np.maximum(mat[:, 6:7], 1)  # markers per utterance
        im = ax.imshow(per, aspect="auto", cmap="Blues", vmin=0)
        ax.set_xticks(range(6), [f"m{m + 1}" for m in range(6)])
        ax.set_yticks(range(len(SPEECH_ACTS)), [SPEECH_LABEL[s] for s in SPEECH_ACTS])
        ax.grid(False)
        for i in range(len(SPEECH_ACTS)):
            for m in range(6):
                ax.text(m, i, f"{per[i, m]:.2f}", ha="center", va="center", fontsize=7.5, color=INK2)
        ax.set_title(f"{LABEL[v]}: uses of each marker per utterance", fontsize=9.5)
        fig.colorbar(im, ax=ax, shrink=0.8).outline.set_visible(False)
    fig.savefig(out / "4_marker_usage.png", dpi=130)
    plt.close(fig)


def fig_norms_and_rudeness(R, out, days):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3), constrained_layout=True)
    ax = axes[0]
    for i, s in enumerate(SPEECH_ACTS):
        series = np.array([[d["norm_mean"][i, 0] for d in r["daily"]] for r in of(R, list(VARIANTS)[-1])])
        ax.plot(np.arange(series.shape[1]), series.mean(axis=0), color=SLOT[i], lw=LW, label=SPEECH_LABEL[s])
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_title(f"Norms (expected e) per speech act, {LABEL[list(VARIANTS)[-1]]}", fontsize=9.5)
    ax.set_xlabel("day")
    ax.legend(loc="lower left", fontsize=8)
    ax = axes[1]
    for v in VARIANTS:
        nb = int(days // 2)
        ys = []
        for r in of(R, v):
            acc, tot = np.zeros(nb), np.zeros(nb)
            for st, speech, markers, app_e, app_p, exp_e, accidental, *_ in r["utter"]:
                if accidental is None:
                    continue
                b = min(nb - 1, int(st / DAY // 2))
                tot[b] += 1
                acc[b] += accidental
            ys.append(np.where(tot > 0, acc / np.maximum(tot, 1), np.nan))
        ax.plot((np.arange(nb) + 0.5) * 2, np.nanmean(ys, axis=0), color=COLOR[v], lw=LW, label=LABEL[v])
    ax.set_title("Accidental rudeness: speaker expected to please,\naddressee was displeased", fontsize=9.5)
    ax.set_xlabel("day")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=8)
    ax = axes[2]
    width = 0.27
    for j, v in enumerate(VARIANTS):
        vals = []
        for s in SPEECH_ACTS:
            x = [u[3] for r in of(R, v) for u in r["utter"] if u[1] == s and u[0] >= WINDOW * DAY and u[3] is not None]
            vals.append(np.mean(x) if x else np.nan)
        ax.bar(np.arange(len(SPEECH_ACTS)) + (j - (len(VARIANTS) - 1) / 2) * width, vals, width=width * 0.9,
               color=COLOR[v], label=LABEL[v])
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xticks(range(len(SPEECH_ACTS)), [SPEECH_LABEL[s] for s in SPEECH_ACTS])
    ax.grid(axis="x", visible=False)
    ax.set_title(f"How addressees appraise utterances\n(mean e-deviation from their norm, days {WINDOW:g}+)", fontsize=9.5)
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(out / "5_norms_rudeness_appraisal.png", dpi=130)
    plt.close(fig)


def fig_brown_levinson(R, out):
    """First B&L-style check: does how polite a request is depend on the relationship?
    Politeness = the speaker's expected e-deviation of its chosen bundle (days ≥ WINDOW)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), constrained_layout=True)
    ax = axes[0]
    width = 0.27
    for j, v in enumerate(VARIANTS):
        vals, errs = [], []
        for b in range(4):
            x = [u[5] for r in of(R, v) for u in r["utter"] if u[1].startswith("request") and u[0] >= WINDOW * DAY and u[7] == b]
            vals.append(np.mean(x) if len(x) > 20 else np.nan)
            errs.append(np.std(x) / np.sqrt(len(x)) if len(x) > 20 else 0)
        ax.bar(np.arange(4) + (j - (len(VARIANTS) - 1) / 2) * width, vals, yerr=errs, width=width * 0.9,
               color=COLOR[v], label=LABEL[v],
               error_kw={"ecolor": INK2, "elinewidth": 0.8})
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xticks(range(4), BIN_LABEL)
    ax.grid(axis="x", visible=False)
    ax.set_title("Politeness of requests by status relation\n(speaker's intended e-deviation from the norm)", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    edges = np.linspace(-1, 1, 9)
    for v in VARIANTS:
        pts = np.array([(u[8], u[5]) for r in of(R, v) for u in r["utter"] if u[1].startswith("request") and u[0] >= WINDOW * DAY])
        if not len(pts):
            continue
        idx = np.digitize(pts[:, 0], edges) - 1
        cx = [(edges[b] + edges[b + 1]) / 2 for b in range(8) if (idx == b).sum() >= 30]
        cy = [pts[idx == b, 1].mean() for b in range(8) if (idx == b).sum() >= 30]
        ax.plot(cx, cy, color=COLOR[v], lw=LW, marker="o", ms=4, label=LABEL[v])
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_xlabel("speaker's liking of the addressee")
    ax.set_title("Politeness of requests by liking", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "6_politeness_by_relationship.png", dpi=130)
    plt.close(fig)


def fig_population_liking(R, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    ax = axes[0]
    for v in VARIANTS:
        m = np.mean([r["n"] for r in of(R, v)], axis=0)
        ax.plot(np.arange(len(m)) / DAY, m, color=COLOR[v], lw=LW, label=LABEL[v])
    ax.axvspan(0, WINDOW, color=GRID, alpha=0.35, lw=0)
    ax.set_title("Population")
    ax.set_xlabel("day")
    headroom(ax, 1.2)
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1]
    for v in VARIANTS:
        lw = np.array([[d["liking_w"] for d in r["daily"]] for r in of(R, v)])
        ax.plot(np.arange(lw.shape[1]), np.nanmean(lw, axis=0), color=COLOR[v], lw=LW, label=LABEL[v])
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.set_title("Mean learned weight on liking (all acts): does liking start to matter?", fontsize=9.5)
    ax.set_xlabel("day")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out / "7_population_and_liking_weight.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--days", type=float, default=60)
    ap.add_argument("--out", default="figures/stage4")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--set", default="main", choices=sorted(VARIANT_SETS))
    ap.add_argument("--window", type=float, default=40, help="analyses use days from here on")
    args = ap.parse_args()
    use_set(args.set)
    global WINDOW
    WINDOW = args.window
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = Path("runs") / f"stage4_plot_{args.set}_{args.seeds}seeds_{args.days:g}days.pkl"
    if args.reuse and cache.exists():
        R = pickle.loads(cache.read_bytes())
    else:
        with ProcessPoolExecutor() as pool:
            R = list(pool.map(run_one, [(v, s, args.days, args.set) for v in VARIANTS for s in range(args.seeds)]))
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(R))
    fig_connotations(R, out)
    fig_meanings_per_seed(R, out)
    fig_agreement(R, out)
    fig_usage(R, out)
    fig_norms_and_rudeness(R, out, args.days)
    fig_brown_levinson(R, out)
    fig_population_liking(R, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()

"""Run the survival world (condition 0) over several seeds and print calibration measures.

Examples:
    uv run python scripts/run_stage1.py --seeds 8 --days 20
    uv run python scripts/run_stage1.py --set world.food_regrowth=0.03 --set tasks.food_per_capita=4
    uv run python scripts/run_stage1.py --sweep world.food_regrowth=0.02,0.03,0.05
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from emergent_politeness.config import Config
from emergent_politeness.metrics import summary
from emergent_politeness.model import Simulation

SHOW = [
    "n_mean", "n_min", "extinct", "deaths_per_100_agent_days", "deaths_starvation", "deaths_cold",
    "deaths_injury", "frac_steps_shelter_full", "frac_outside_at_night", "door_blocked_per_day",
    "food_per_capita", "level_final", "integrity_mean", "work_share_collect_food", "work_share_collect_material",
    "work_share_maintain", "work_share_build", "division_of_labour",
]


def parse_value(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def build_config(sets: list[str], days: float, seed: int) -> Config:
    groups: dict[str, dict] = {}
    for item in sets:
        key, value = item.split("=", 1)
        group, name = key.split(".", 1)
        groups.setdefault(group, {})[name] = parse_value(value)
    cfg = Config().with_(**groups) if groups else Config()
    return cfg.with_(sim={"steps": int(days * cfg.world.day_length), "seed": seed})


def run_one(args: tuple[list[str], float, int, float]) -> dict:
    sets, days, seed, burn_in = args
    sim = Simulation(build_config(sets, days, seed)).run()
    return summary(sim, burn_in_days=burn_in)


def report(label: str, results: list[dict]) -> None:
    print(f"\n== {label}  ({len(results)} seeds)")
    for key in SHOW:
        vals = np.array([r[key] for r in results], dtype=float)
        print(f"  {key:28s} {np.nanmean(vals):9.3f}  ± {np.nanstd(vals):7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--days", type=float, default=20)
    ap.add_argument("--burn-in", type=float, default=3, help="days excluded from time averages")
    ap.add_argument("--set", action="append", default=[], metavar="group.param=value")
    ap.add_argument("--sweep", default=None, metavar="group.param=v1,v2,...")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    variants = [("base", args.set)]
    if args.sweep:
        key, values = args.sweep.split("=", 1)
        variants = [(f"{key}={v}", args.set + [f"{key}={v}"]) for v in values.split(",")]

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for label, sets in variants:
            jobs = [(sets, args.days, seed, args.burn_in) for seed in range(args.seeds)]
            report(label, list(pool.map(run_one, jobs)))


if __name__ == "__main__":
    main()

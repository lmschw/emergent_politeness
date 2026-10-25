# emergent_politeness

An embodied multi-agent simulation in which politeness and rudeness can emerge. Agents need food and shelter, can act on each other, and use a language whose *affective* meaning is learned rather than built in.

- Model specification (ODD format): [model_spec.md](model_spec.md)
- Related work and novelty: [docs/related_work.md](docs/related_work.md)
- Calibration: [stage 1](docs/calibration_stage1.md), [stage 2](docs/calibration_stage2.md)

## Status
Stages 1 (survival world, condition 0) and 2 (physical acts, conditions 1 and 4) are implemented and calibrated. Language comes next.

## Usage
Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run python scripts/run.py --seeds 8 --days 20
uv run python scripts/run.py --condition 4 --seeds 8 --days 50 --burn-in 30
uv run python scripts/run.py --sweep world.food_regrowth=0.02,0.05 --set tasks.cold_weight=30
```

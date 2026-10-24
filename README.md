# emergent_politeness

An embodied multi-agent simulation in which politeness and rudeness can emerge. Agents need food and shelter, can act on each other, and use a language whose *affective* meaning is learned rather than built in.

- Model specification (ODD format): [model_spec.md](model_spec.md)
- Related work and novelty: [docs/related_work.md](docs/related_work.md)
- Stage 1 calibration: [docs/calibration_stage1.md](docs/calibration_stage1.md)

## Status
Stage 1 (survival world, condition 0) is implemented and calibrated. Social acts, language and affect come next.

## Usage
Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run python scripts/run_stage1.py --seeds 8 --days 20
uv run python scripts/run_stage1.py --sweep world.food_regrowth=0.02,0.05 --set tasks.cold_weight=30
```

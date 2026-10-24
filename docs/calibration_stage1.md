# Stage 1 calibration: survival world (condition 0)

*2026-10-24. Reproduce with `uv run python scripts/run_stage1.py --seeds 16 --days 60 --burn-in 40`.*

## Targets (spec §11)
- The population persists, but some agents die.
- The shelter is at capacity some of the time, so the door is contested.
- Danger matters: injuries sometimes kill.
- Division of labour emerges from threshold reinforcement.

## Result with the current defaults (16 seeds, days 40–60)
These are the defaults after the user's decisions: no deposits at the door, λ_c = 60.

| Measure | Mean ± sd |
|---|---|
| Population | 48.3 ± 0.5 (minimum 43, no extinctions; settles by about day 15–20) |
| Deaths per 100 agent-days | 5.5 ± 0.5 |
| Deaths over 60 days: starvation / cold / injury | 132 / 3 / 12 |
| Fraction of steps the shelter is full | 0.02 ± 0.01 (full only briefly at dusk) |
| Fraction of agents outside at night | 0.32 (mostly starving agents collecting food at night) |
| Turned away at the door, per day | 57 ± 6 (dusk crowding; only 24 of ~11,600 entry grants went unused, so this isn't an artefact) |
| Food in store per capita | 0.01 (food is eaten as soon as it arrives) |
| Shelter level at day 60 | 9 (capacity 44) |
| Work shares: collect food / collect material / maintain / build | 0.71 / 0.25 / 0.03 / 0.02 |
| Division of labour (Gorelick et al. 2004 index) | 0.38 ± 0.02 |

**Effect of the user's two changes** (8 seeds, days 40–60):

| | λ_c = 60 | λ_c = 100 | earlier: λ_c = 100 with deposits at the door |
|---|---|---|---|
| Shelter full | 2.5% | 15% | 16.5% |
| Outside at night | 31% | 19% | 19% |
| Deaths: starvation / cold / injury | 128 / 3 / 9 | 129 / 9 / 10 | 99 / 32 / 14 |

Removing the door drop-off shifts deaths from cold to starvation. λ_c then controls how contested the shelter is at night.

Figures: `figures/stage1/` (regenerate with `uv run python scripts/plot_stage1.py`).

## What had to change, and why

| Problem seen | Change | Spec section |
|---|---|---|
| Building was cheap, so capacity outgrew population and nobody froze | m_B 0.1 → 0.3, B_req 50 → 200, material regrowth 0.05 → 0.03 | §9 |
| The shelter grew without limit in long runs | integrity decay scales with shelter size (C / C₀) | §5.1 |
| Agents worked through the night: 70% outside at night, shelter never full | cold raises thresholds for outside tasks; λ_c = 60, the user's choice between 30 and 100 | §5.2 |
| Up to 60% of work went into collecting unneeded material while agents starved | collect-material stimulus × material need | §5.2 |
| Injury never killed | danger {0, 0.01, 0.03} → {0, 0.02, 0.05}; q_inj 0.2 → 0.35 | §9 |
| No division of labour (index ~0.05): one trip dropped a threshold by 0.4 | ξ 0.01 → 0.0005, φ 0.002 → 0.0001 | §5.2, §9 |
| A full shelter blocked food deliveries | deposits were allowed at the door, but the user later rejected this: agents must get inside, so insiders are pressed to leave | §5.1 |
| The capacity rule was broken: crowds pushed agents through the door, and leaving agents walked back in on their exit permission | queue spot in front of the door, exit grants don't allow re-entry, thicker walls, ejection safety net | §10 |

## Open points
- **Slow transient.** Population climbs for about 40 days, and the shelter gains roughly a level every 10 days even at day 60. Either use a 40-day burn-in (current choice) or cap the shelter level.
- **Division of labour is moderate** (0.29). Food is chronically scarce, so the collect-food stimulus stays high and most agents collect food. Specialisation is stronger when demand is moderate (Bonabeau et al. 1996).
- **One seed in a 60-day check collapsed temporarily** (population 48 → 28, then recovered). This kind of event deserves a closer look once social acts exist.

# Stage 2 calibration: physical acts (conditions 1 and 4)

> **Revision (2026-10-24, later the same day).** Three changes were made after the first calibration below. The current defaults are in the last section, "Revision 2".
> 1. The shelter is capped at level 8.
> 2. The weight on witnessed contributions is near zero.
> 3. Status effects in decisions and appraisals are now learned (spec §5.3a).

*2026-10-24. 8 seeds × 50 days, measured over days 30–50. Reproduce with
`uv run python scripts/run.py --condition 1 --seeds 8 --days 50 --burn-in 30` (and `--condition 4`).*

## Targets
- Social acts don't destroy the society: population stays near the condition-0 level (~48).
- Fights are a minority cause of death.
- Refusals are selective, not the default response to arrivals.
- A status hierarchy is visible: self-status tracks fighting strength.

## Spec starting guesses (γ_R0 = −2, γ_A0 = −4): a feud spiral
After 10 days, condition 1 had population 12 (baseline ~35 at that point), 1,549 refusals, 344 attacks after refusals, and fights caused 22 of 22 deaths.
The cause is the refusal formula: with bias −2 and crowding weight 3, a guard refuses about half of all arrivals once the shelter is two-thirds full.

## Sweep (condition 1, ledger opinions)

| γ_A0 | γ_R0 | Population | Refusals / day | Attacks / day | Fight share of deaths |
|---|---|---|---|---|---|
| −4 | −3 | 21 | 245 | 49 | 0.94 |
| −4 | −4 | 42 | 363 | 70 | 0.70 |
| −4 | −5 | 48 | 66 | 6 | 0.01 |
| −5 | −3 | 44 | 555 | 56 | 0.32 |
| −5 | −4 | 47 | 359 | 36 | 0.18 |
| −5 | −5 | 48 | 59 | 2 | 0.01 |
| **−4.5** | **−4.5** | **47.5** | **198** | **22** | **0.08** |

**Feud spiral.** Refusal provokes attack, attack lowers the attacker's ledger standing, low standing brings more refusals, and so on. Between γ_R0 = −5 and −4 the society goes from calm to feuding. The response is steep, and single runs can tip (see fig 7). This positive feedback is an emergent property, and worth studying in its own right (e.g. whether affect or language changes where the tipping point lies).

## Chosen defaults: γ_R0 = γ_A0 = −4.5

| Measure (days 30–50) | Condition 1 (ledger) | Condition 4 (affect) |
|---|---|---|
| Population | 47.5 ± 0.5 | 47.5 ± 0.3 |
| Gifts per day | 28 | 31 |
| Entry refusals per day | 198 ± 58 | 128 ± 21 |
| Attacks per day (almost all after a refusal) | 22 ± 9 | 14 ± 2 |
| Grudge attacks per day | — | 0.04 |
| Fight share of deaths | 0.08 | 0.01 |
| corr(self-status, fighting strength) | 0.15 | 0.22 |
| corr(mean perceived status, strength) | 0.26 | 0.21 |
| Status agreement (share of variance in P explained by who is rated) | 0.48 | 0.57 |
| Linearity h' (chance level) | 0.035 (0.03) | 0.032 (0.03) |
| Memory in use (of K = 30) | 29.9 | 29.6 |

Affect already dampens conflict relative to the ledger: fewer refusals, fewer attacks, fewer fight deaths. That's suggestive for H3a, but it needs a proper comparison once stages 3–4 exist.

## Sanity-check figures
`figures/stage2/`, regenerate with `uv run python scripts/plot_stage2.py` (`--reuse` replots from `runs/`).

- **Population and deaths (fig 1).** Population is the same in conditions 0, 1 and 4. Fights replace some starvation deaths in condition 1 and are negligible in condition 4.
- **Act rates (fig 2).** Gifts start once the store runs dry (~day 12) and settle at ~30 a day.
  - Refusals and attacks peak around days 15–33 and then **decline**, most visibly in condition 1.
  - The cause is shelter growth. Crowding at entry decisions falls from 0.85 (days 10–20) to 0.68 (days 40–50), and liking drifts up, so the refusal rate falls from 0.21 to 0.10 (condition 1) and from 0.12 to 0.07 (condition 4).
  - **Stage-2 measures are therefore not stationary in the days 30–50 window.**
- **Daily cycle (fig 3).** Refusals and attacks happen from dusk through the night, when everyone wants in.
- **Refusal inputs (fig 4).** The realised refusal rate falls with the guard's liking of the arriver, and rises with crowding and with the guard's status advantage. So the implementation does what the formula says. Condition 1 refuses more at every level of liking.
- **Status vs strength (fig 5).** Correlations are weak (r = 0.15–0.28). In condition 1 most agents sit at P ≈ 0, because perceived status only moves through fights.
- **Liking (fig 6).**
  - Ledger liking has heavy tails, including many saturated enemies at −1. These sustain the feud spiral.
  - Affective liking is mostly mildly positive, with almost no strong dislike, and piles up at E = +1. The cause is that **every deposit seen inside the shelter raises E** (α_C = 0.02 per sighting), and insiders see many deposits.
  - Worth revisiting before stage 4: the positivity may be an artefact of contribution sightings.
- **Feud spiral (fig 7).** With γ_R0 = −4, attacks stay high (~50 a day); with −5 they stay low (~3); with −4.5 they peak and then fade as the shelter grows. One −4 seed stayed calm until about day 25 and then tipped. So "bistable" overstates it: the response is **steep, with tipping at the level of single runs**.

## Decisions
- 2026-10-24, the user: fights **don't need to be frequent**. They exist only as one possible way to be rude. A loose status structure is acceptable, so open point 1 below is not a problem to fix.

## Open points
1. **No linear hierarchy** (accepted, see Decisions). Fights are too sparse for a dominance hierarchy: linearity h' sits at chance level. Self-status and perceived status track strength only weakly (r ≈ 0.15–0.26), though observers agree with each other fairly well. The spec §11 sanity check ("DomWorld-like hierarchy emerges") therefore only partly passes. In DomWorld (Hemelrijk 2000), agents meet and fight constantly; here, fights only happen at the door.
2. **Patch contests almost never happen** (< 0.2 per day). Agents leave a patch as soon as it is empty, so two agents rarely harvest a nearly empty patch at the same time. All conflict is about the door, and none of it is about food.
3. **Grudge attacks are rare** in condition 4: transient impressions decay within a few steps (ρ_τ = 0.3).
4. Memory is always full (K = 30 with ~48 agents), so eviction is active. That is intended (Dunbar), but it makes newcomers forget and be forgotten quickly.
5. **Shelter growth makes social measures drift** (see fig 2). Options: cap the shelter level, measure later or over a longer window, or treat the drift as part of the dynamics.
6. **Affective liking may be inflated by contribution sightings** (see fig 6).

## Revision 2: learned status, level cap, near-zero contribution weight
*8 seeds × 60 days, measured over days 40–60.*

**Why.** The user pointed out that the refusal and attack rules had fixed status terms. That built "be rude downward" into the model and would make status findings circular. Status effects are now learned from consequences (spec §5.3a):
- agents learn expectations of winning, being refused and being avenged, for each status configuration;
- decisions use these expectations;
- affective appraisal is scaled by surprise;
- status updates after fights are Elo-style.

**Cap only** (old fixed status terms, γ_R0 = −4.5): the door stays crowded, so conflict rises. Condition 1 had 328 refusals and 50 attacks a day, with fights 30% of deaths; condition 4 had 185 and 27, with 10%.

**Learned status, sweep of γ_R0:**

| Condition | γ_R0 | Population | Refusals / day | Attacks / day | Fight share of deaths |
|---|---|---|---|---|---|
| 1 | −4 | 45.2 | 335 | 60 | 0.42 |
| 1 | −4.5 | 47.0 | 257 | 46 | 0.29 |
| 1 | **−5** | **47.6** | **162** | **26** | **0.13** |
| 4 | −4 | 47.0 | 183 | 44 | 0.23 |
| 4 | −4.5 | 47.3 | 116 | 26 | 0.12 |
| 4 | **−5** | **47.5** | **69** | **16** | **0.03** |

**Chosen:** γ_R0 = −5. Fights stay a minority cause of death, in line with the user's decision that fights need not be frequent.

**What agents learned** (γ_R0 = −5, mean over living agents at day 60; prior 0.5):

| Expectation | Condition | actor below | level | actor above |
|---|---|---|---|---|
| p̂ win | 1 | 0.44 | 0.49 | 0.56 |
| p̂ win | 4 | 0.36 | 0.49 | 0.64 |
| p̂ refuse | 1 | 0.08 | 0.05 | 0.12 |
| p̂ refuse | 4 | 0.05 | 0.02 | 0.05 |
| p̂ attack (retaliation) | 1 | 0.18 | 0.13 | 0.24 |
| p̂ attack (retaliation) | 4 | 0.21 | 0.21 | 0.32 |

- **Status predicts winning.** Agents learn that the higher-rated party usually wins, more clearly in condition 4. Perceived status is weakly grounded in real strength (r ≈ 0.23–0.33).
- **High-status agents avenge refusals more often.** Refused agents who rank above the guard retaliate more, so refusing upward is risky. This is learned, not assumed.
- **Refusals are U-shaped.** Refusals by guards who rank above *or* below the arriver are more common than between equals, especially in condition 1. There is no clean "rudeness flows downward" pattern, and that is now a finding, not an assumption.

## Revision 3: door gridlock, and shelter fragility (2026-10-25)
- **Door gridlock (a bug, fixed).**
  - What happened: in condition 4, seed 4, 42 agents starved in one day. At dawn about 13 agents were leaving while about 25 returning agents pushed to get in through the 1.4 m door, and everything stalled. A later version with too few queue spots also clogged: the agents allowed in were trapped inside the waiting crowd.
  - The fix: turn-taking with a first-come-first-served queue line (spec §10).
  - A bug made along the way, also fixed: the queue side depended on the place number, so every advance of the queue sent everyone to the other side of the door.
- **Shelter fragility (emergent, decision pending).** The remaining crashes are public-goods collapses:
  - everyone collects food, so material and maintenance dry up;
  - a level-8 shelter wears five times faster than the base size and is useless within about 200 steps;
  - agents freeze and starve.
  - Comparison over 8 seeds × 60 days:

    | | Size-scaled wear (current) | Constant wear |
    |---|---|---|
    | Runs with a crash, cond 0 / cond 4 | 1 / 1 (one extinction) | 0 / 0 |
    | Mean integrity | 0.57–0.60 | 0.81–0.82 |
    | Cold deaths per run, days 40–60 | 1.6–3.6 | ~0 |

- **Ejections.** 10–25 per 60-day run, unchanged by moving the queue line away from the wall. Not yet explained; they don't break the capacity rule.
- The stage-2 tables and figures above predate these fixes and need regenerating once the wear decision is made.

## Current defaults: constant wear, turn-taking door, learned status (2026-10-25)
*8 seeds × 60 days, days 40–60. γ_R0 = −5, γ_A0 = −4.5, constant shelter wear. Figures in `figures/stage2/` are from this version.*

| Measure | Condition 0 | Condition 1 (ledger) | Condition 4 (affect) |
|---|---|---|---|
| Population (minimum) | 47.0 (41) | 46.7 (42) | 46.9 (41) |
| Deaths per 100 agent-days | 5.5 | 5.4 | 5.3 |
| Deaths: starvation / injury / cold / fight | almost all starvation; cold 0 | same; fights 0.2% | same; fights 0.7% |
| Shelter integrity | 0.81 | 0.78 | 0.82 |
| Shelter full | 0.4% | 0.2% | 0.1% |
| Outside at night (mostly hungry collectors) | 41% | 47% | 46% |
| Gifts per day | — | 54 | 51 |
| Entry refusals per day | — | 15 | 15 |
| Attacks per day (almost all after a refusal) | — | 0.8 | 3.0 |
| corr(perceived status, strength) | — | 0.22 | 0.16 |
| Status agreement | — | 0.41 | 0.56 |

- **The society is calm now.** With no door gridlock and no shelter collapses, the door is rarely contested: refusals fell from ~160 to ~15 a day and fights are negligible. Food is the main conflict of interest (starvation causes almost all deaths), so in stage 3 requests for food will matter more than requests to enter.
- **Affect raises retaliation.** About 20% of refusals are avenged in condition 4 vs 5% in condition 1. The affective transient (momentary anger) after a refusal feeds the attack decision twice: through Ẽ = E + τ and through the γ_Aτ term. The spec defines both terms, so this is as designed, but it is worth keeping in mind.
- **Liking.** Gifts (~50 a day) now dominate, so ledger liking is mostly positive. Affective liking is centred on zero with a mildly positive tail.
- **Learned expectations show a U-shape** (refusals and retaliation are rarer between "level" agents). **Caveat:** "level" mostly means *strangers*, because unknown agents have P = 0. So the U-shape largely reflects "has a history with" rather than status difference. A separate "unknown" category in the status bins would remove this confound.
- **The feud spiral is milder** (fig 7): with γ_R0 = −4, attacks peak at ~18 a day around day 20 and settle at ~5.

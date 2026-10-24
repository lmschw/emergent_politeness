# Related work and novelty assessment

*Web search, 2026-10-24. This is not a systematic review, and some full texts were paywalled. Run the checks at the bottom before making novelty claims in a paper.*

## Bottom line
Novelty is **moderate to high**. Each building block is established: language games, survival ABMs, dominance and status, reputation, and emotion in ABMs. No model was found in which **the affective or politeness value of linguistic forms emerges from grounded social interaction** and is then **tested against politeness theory**. That is the main contribution, and papers should lead with it.

## Closest work

| Area | Closest work | How this project differs |
|---|---|---|
| Game-theoretic politeness | van Rooy (2003), polite requests as costly signals (handicap principle). Mühlenbernd, Wacewicz & Żywiczyński (2021, *Linguistics & Philosophy* 44(6)), thanking and apologising under cultural-evolutionary dynamics, with a warmth/competence split; when status is valued, polite and impolite strategies coexist in cycles | **Closest competitor.** Population-level, no embodiment, and polite strategies are given in advance. Cite it for the 2D E/P choice. Mühlenbernd also leads SFB 1412 project A05, on modelling politeness and face management, so watch for new output. |
| Politeness hard-coded in agents | Walker, Cahn & Whittaker (1997); SIFT's computational version of Brown & Levinson | Theory is coded in; here it is the hypothesis being tested |
| Emergent vocabularies | Steels, Vogt, Baronchelli (naming games) | Referential meanings only. **No language game with emergent valence or social meaning was found.** |
| Emotion and norms in ABM | Tzeng, Ajmeri & Singh (2021, Noe); Anagnou, Polani & Salge (2024–26, *Normative feeling*) | No language, no politeness |
| Status-power ABMs | Hofstede's GRASP and status-arena models (based on Kemper) | Close to the physical social layer; no emergent language |
| LLM agent societies | Ji et al. (2026, Findings of ACL), ACT-grounded LLM agents with emergent authority; Project Sid; LLM naming games (Science Advances 2025) | LLMs inherit politeness from training data, so emergence can't be told apart from imitation. This model starts with no politeness at all. |
| Social swarms | Emotion-driven foraging swarms | No status, social acts or language |

## What appears new, strongest first
1. Emotional and social meaning of words emerging from consequences: a language game where connotation, not just reference, is grounded in embodied social acts.
2. Politeness as an emergent outcome tested against Brown & Levinson and Danescu-Niculescu-Mizil et al. (2013), rather than built in or offered as a strategy.
3. A computational version of norm-relative evaluation (Watts; Locher & Watts), which makes politeness inflation and accidental rudeness possible. *Less certain; the literature is scattered.*
4. Politeness linked to survival and material outcomes, with a factorial design separating language, physical acts and affect.
5. Newcomers learning local politeness norms.

## Risks
- The Mühlenbernd group may publish something close. Read the 2021 paper in full; it is open access.
- Fast-moving LLM social-simulation work may claim "emergent politeness". The answer: those agents inherit politeness, ours invent it.
- Pragmatics reviewers may object that six markers are not politeness. Frame the work as modelling politeness mechanisms in the theorists' sense, or as "proto-politeness".
- Combining established parts is easy to dismiss as "Sugarscape plus a naming game", so lead with contribution 1.

## Design responses (already added to model_spec.md)
- Benchmark condition **6-BL**: hand-coded Brown & Levinson agents (spec §6).
- **Docking check** against Mühlenbernd et al. (2021) (spec §11).

## Checks still to run
Google Scholar:
- "emergent politeness" agent-based
- "naming game" with valence, connotation or "evaluative meaning"
- "language game" with "social meaning" or "indexicality"

Also:
- Recent JASSS and *Journal of Politeness Research* issues
- EVOLANG 2026 proceedings
- "The dynamics of politeness: An experimental account" (*Journal of Pragmatics*, 2021): experimental semiotics with simulated agents. Authors and full text could not be retrieved; check how close it is.

## Sources
- https://www.leibniz-zas.de/en/research/publications/details/publications/politeness-and-reputation-in-cultural-evolution
- https://api.philpapers.org/rec/MHLPAR
- https://sfb1412.hu-berlin.de/people/dr-roland-muehlenbernd/
- https://repozytorium.umk.pl/items/33567f41-2c86-4bd9-a021-1fa9cea8a718/full
- https://staff.fnwi.uva.nl/r.a.m.vanrooij/Politness2.pdf
- https://www.sift.net/sites/default/files/publications/MWF-IEEEIS-vsubmit2.pdf
- https://www.sift.net/sites/default/files/publications/MWF-CCD07-v2.pdf
- https://arxiv.org/abs/2606.23764
- https://www.science.org/doi/10.1126/sciadv.adu9368
- https://link.springer.com/article/10.1007/s10462-025-11412-6
- https://arxiv.org/abs/2104.15034v2
- https://arxiv.org/abs/2411.18037v3
- https://cs.uwaterloo.ca/~jhoey/publications/../papers/SSC-2019-Morgan.pdf
- https://www.comses.net/codebases/?tags=GRASP
- https://arxiv.org/pdf/1705.03175
- https://www.sciencedirect.com/science/article/abs/pii/S0378216621003088
- https://www.nature.com/articles/s41562-022-01483-8

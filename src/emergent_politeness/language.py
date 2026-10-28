"""Affect markers in language (spec §5.4–5.8, stage 4).

Utterances = fixed, shared content (request food / request task / respond yes / respond no)
+ a bundle of abstract markers. Markers have no built-in meaning:
- each agent learns what a marker means, its connotation c(m) = (e, p), by evaluative
  conditioning on what speakers then do (Rescorla & Wagner 1972; spec §5.7);
- hearers judge a bundle relative to a learned norm per speech act and status
  configuration (Watts 2003; Burgoon 1993; spec §5.6);
- speakers choose bundles by expected effect: being liked (intrinsic motive), getting a
  yes (learned), expressing their actual stance, minus a cost per marker (spec §5.5).
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np

from .agents import Agent
from .config import LanguageParams, SocialParams
from .social import Opinions, clip1

SPEECH_ACTS = ("request_food", "request_task", "respond_yes", "respond_no")
N_BINS = 4  # status configurations: below, level, above, unknown


def all_bundles(n_markers: int, max_markers: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = [()]
    for size in range(1, max_markers + 1):
        out += list(combinations_with_replacement(range(n_markers), size))
    return out


class Lexicon:
    """One agent's language state."""

    __slots__ = ("c", "norm", "q0", "w")

    def __init__(self, lp: LanguageParams, rng: np.random.Generator) -> None:
        self.c = rng.normal(0.0, lp.sigma_c0, size=(lp.n_markers, 2))  # connotation (e, p) per marker
        self.norm = np.zeros((len(SPEECH_ACTS), N_BINS, 2))  # expected bundle connotation
        self.q0 = np.zeros((2, N_BINS))  # instrumental: P(yes) intercept per request type and bin
        self.w = np.zeros((lp.n_markers, 2, N_BINS))  # instrumental: marker effects on P(yes)


class Language:
    def __init__(self, lp: LanguageParams, sp: SocialParams, op: Opinions, rng: np.random.Generator) -> None:
        self.lp, self.sp, self.op, self.rng = lp, sp, op, rng
        self.bundles = all_bundles(lp.n_markers, lp.max_markers)
        self.incidence = np.zeros((len(self.bundles), lp.n_markers))
        for n, b in enumerate(self.bundles):
            for m in b:
                self.incidence[n, m] += 1
        self.sizes = self.incidence.sum(axis=1)

    def new_lexicon(self) -> Lexicon:
        return Lexicon(self.lp, self.rng)

    # --- meaning ---------------------------------------------------------
    def connotation(self, lex: Lexicon, bundle: tuple[int, ...]) -> np.ndarray:
        s = np.zeros(2)
        for m in bundle:
            s += lex.c[m]
        return np.tanh(s)

    # --- speaker (spec §5.5) -----------------------------------------------
    def choose(self, s: Agent, t: Agent, act: str) -> tuple[tuple[int, ...], np.ndarray]:
        """Draw a marker bundle; return it with the norm deviation the speaker expects (δ̂)."""
        lp, op = self.lp, self.op
        lex: Lexicon = s.lexicon
        a = SPEECH_ACTS.index(act)
        b = op.bin(s, s.id, t.id)
        C = np.tanh(self.incidence @ lex.c)  # bundle connotations, one row per bundle
        dev = C - lex.norm[a, b]
        lam_e = lp.lambda_E0 * (1 + lp.kappa_A * max(0.0, -s.acceptance))  # sociometer (Leary et al. 1995)
        u = lp.w_img * (lam_e * dev[:, 0] + lp.lambda_P * dev[:, 1])
        stance = np.array([op.liking(s, t.id), clip1(s.self_status - op.status(s, t.id))])
        u -= lp.w_expr * ((C - stance) ** 2).sum(axis=1)
        u -= lp.marker_cost * self.sizes
        if act.startswith("request"):  # instrumental: what gets me a yes (learned, spec §5.8)
            r = SPEECH_ACTS.index(act)
            q = 1 / (1 + np.exp(-(lex.q0[r, b] + self.incidence @ lex.w[:, r, b])))
            u += lp.w_inst * q
        p = np.exp((u - u.max()) / lp.temperature)
        n = int(self.rng.choice(len(self.bundles), p=p / p.sum()))
        return self.bundles[n], dev[n]

    # --- hearers (spec §5.6) ---------------------------------------------
    def appraise(self, hearers: list[Agent], s: Agent, t: Agent, act: str, bundle: tuple[int, ...],
                 k: int) -> np.ndarray | None:
        """Every hearer judges the bundle against its own norm; returns the addressee's δ (or None)."""
        lp, sp, op = self.lp, self.sp, self.op
        a = SPEECH_ACTS.index(act)
        delta_t = None
        bins = {o.id: op.bin(o, s.id, t.id) for o in hearers}  # configurations before any update
        for o in hearers:
            if o is not t and o is not s and self.rng.random() >= sp.attention:
                continue
            lex: Lexicon = o.lexicon
            b = bins[o.id]
            C = self.connotation(lex, bundle)
            d = C - lex.norm[a, b]  # deviation from what this hearer expects here (expectancy violation)
            lex.norm[a, b] += lp.eta_norm * d  # the norm drifts toward what is heard (inflation)
            if o is s:
                continue
            about_s = op.recall(o, s.id, k)
            if o is t:
                delta_t = d
                about_s.te += d[0]
                about_s.tp += d[1]
                about_s.E = clip1(about_s.E + lp.alpha_E_word * d[0])
                about_s.P = clip1(about_s.P + lp.alpha_P_word * d[1])
                t.self_status = clip1(t.self_status - sp.self_status_rate * d[1])
                t.acceptance = (1 - sp.acceptance_rate) * t.acceptance + sp.acceptance_rate * d[0]
            else:  # bystander: image scoring by default, balance theory as variant (λ_b)
                about_t = op.recall(o, t.id, k)
                g = (1 - sp.balance_mix) + sp.balance_mix * np.tanh(sp.balance_gain * about_t.E)
                about_s.E = clip1(about_s.E + sp.bystander_weight * lp.alpha_E_word * d[0] * g)
                about_s.P = clip1(about_s.P + sp.bystander_weight * lp.alpha_P_word * d[1])
                about_t.P = clip1(about_t.P - sp.bystander_weight * lp.alpha_P_word * d[1])
        return delta_t

    # --- learning meaning (spec §5.7) --------------------------------------
    def condition(self, learners: list[Agent], bundle: tuple[int, ...], value: tuple[float, float]) -> None:
        """Rescorla–Wagner with compound cues: markers predict the value of what the speaker then does."""
        if not bundle:
            return
        v = np.asarray(value)
        for o in learners:
            lex: Lexicon = o.lexicon
            err = v - lex.c[list(bundle)].sum(axis=0)
            for m in bundle:
                lex.c[m] += self.lp.eta_c * err

    # --- learning what gets a yes (spec §5.8) --------------------------------
    def learn_compliance(self, learners: list[Agent], asker: Agent, addressee: Agent, act: str,
                         bundle: tuple[int, ...], yes: bool) -> None:
        r = SPEECH_ACTS.index(act)
        for o in learners:
            rate = self.lp.eta_q * (1.0 if o is asker else self.lp.beta_obs)
            lex: Lexicon = o.lexicon
            b = self.op.bin(o, asker.id, addressee.id)
            z = lex.q0[r, b] + sum(lex.w[m, r, b] for m in bundle)
            err = float(yes) - 1 / (1 + np.exp(-z))
            lex.q0[r, b] += rate * err
            for m in bundle:
                lex.w[m, r, b] += rate * err

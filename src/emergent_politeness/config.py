"""All model parameters in one place (spec §9).

Defaults are the spec's starting guesses; conditions and ablations are
expressed purely as changes to a Config.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class WorldParams:
    arena_size: float = 40.0
    shelter_radius: float = 4.0
    door_width: float = 1.4
    door_angle: float = 0.0  # radians; door faces +x
    capacity_base: int = 8  # C0
    capacity_per_level: int = 4  # ΔC
    n_food_patches: int = 6
    n_material_patches: int = 4
    patch_ring: tuple[float, float] = (10.0, 18.0)  # min/max distance from shelter centre
    patch_radius: float = 1.5
    patch_max_stock: float = 20.0
    food_regrowth: float = 0.05  # g_p, units per step
    material_regrowth: float = 0.03  # calibrated (stage 1); spec guess 0.05
    patch_danger_levels: tuple[float, ...] = (0.0, 0.02, 0.05)  # d_p, sampled per patch; calibrated
    food_spoilage: float = 0.002  # ρ_F
    integrity_decay: float = 0.001  # δ_I, per step at base size; scales with capacity / C0
    maintain_rate: float = 0.01  # r_I
    material_per_maintain_step: float = 0.1  # m_I
    material_per_build_step: float = 0.3  # m_B; calibrated, spec guess 0.1
    build_required: float = 200.0  # B_req: progress for level ℓ→ℓ+1 is B_req·(ℓ+1); calibrated
    temp_mean: float = 5.0  # T̄
    temp_amplitude: float = 10.0  # A
    temp_comfort: float = 10.0  # T_comf
    temp_scale: float = 15.0  # T_scale
    day_length: int = 600  # P_day, decision steps
    initial_food: float = 30.0
    initial_material: float = 10.0
    initial_integrity: float = 1.0


@dataclass(frozen=True)
class BodyParams:
    radius: float = 0.3
    speed: float = 1.2  # m/s at zero fatigue; scaled by (1 - 0.5 X)
    hunger_rate: float = 0.002  # h0
    hunger_work_mult: float = 0.5  # h_w
    eat_amount: float = 0.4  # Δ_eat
    eat_threshold: float = 0.5  # h_eat
    fatigue_work: float = 0.003  # x_w
    fatigue_rest: float = 0.01  # x_rest
    exposure_out: float = 0.004  # z_out
    exposure_in: float = 0.02  # z_in
    starve_damage: float = 0.01  # q_H
    freeze_damage: float = 0.01  # q_Z
    injury_damage: float = 0.35  # q_inj; calibrated, spec guess 0.2
    injury_fatigue_mult: float = 1.0  # ν
    regen: float = 0.005  # q_reg
    max_rations: int = 1  # r_max
    carry_capacity: float = 4.0  # load units per trip
    harvest_rate: float = 1.0  # units per step at a patch
    strength_sd: float = 0.1  # σ_k (used from stage 2 on)


@dataclass(frozen=True)
class TaskParams:
    theta_min: float = 0.1
    theta_max: float = 0.9
    theta_learn: float = 0.0005  # ξ, per step; calibrated (tasks last ~40 steps)
    theta_forget: float = 0.0001  # φ, per step; calibrated
    danger_weight: float = 2.0  # λ_d
    fatigue_weight: float = 1.0  # λ_X
    cold_weight: float = 60.0  # λ_c: cold raises thresholds for outside tasks; chosen by the user
    approval_weight: float = 0.3  # λ_A (affect conditions only)
    patch_danger_aversion: float = 5.0  # λ_d'
    patch_choice_temp: float = 0.2
    food_per_capita: float = 3.0  # f_pc
    material_target: float = 20.0  # M_target
    exposure_critical: float = 0.6  # z_crit
    fatigue_critical: float = 0.8  # x_crit
    fatigue_rested: float = 0.2  # REST ends below this
    health_critical: float = 0.3  # q_crit
    bout_length: int = 20  # MAINTAIN/BUILD bout, decision steps


@dataclass(frozen=True)
class PopulationParams:
    n_initial: int = 12  # N0
    arrival_rate: float = 0.005  # λ_arr, newcomers per step


@dataclass(frozen=True)
class SimParams:
    steps: int = 6000  # decision steps (10 days)
    physics_ticks: int = 5  # ticks per decision step
    physics_dt: float = 0.2  # seconds per tick
    seed: int = 0
    record_every: int = 1  # time-series sampling interval


@dataclass(frozen=True)
class Condition:
    """Which interaction channels are switched on (spec §6)."""

    name: str
    physical_acts: bool = False
    language: bool = False
    affect: bool = False


CONDITIONS: dict[str, Condition] = {
    "0": Condition("0-baseline"),
    "1": Condition("1-physical", physical_acts=True),
    "2": Condition("2-language", language=True),
    "3": Condition("3-physical+language", physical_acts=True, language=True),
    "4": Condition("4-physical+affect", physical_acts=True, affect=True),
    "5": Condition("5-language+affect", language=True, affect=True),
    "6": Condition("6-full", physical_acts=True, language=True, affect=True),
}


@dataclass(frozen=True)
class Config:
    world: WorldParams = field(default_factory=WorldParams)
    body: BodyParams = field(default_factory=BodyParams)
    tasks: TaskParams = field(default_factory=TaskParams)
    population: PopulationParams = field(default_factory=PopulationParams)
    sim: SimParams = field(default_factory=SimParams)
    condition: Condition = field(default_factory=lambda: CONDITIONS["0"])

    def with_(self, **groups: dict) -> "Config":
        """Return a copy with selected fields changed, e.g.
        cfg.with_(world={"food_regrowth": 0.02}, sim={"seed": 3})."""
        changes = {}
        for group, values in groups.items():
            if group == "condition" and isinstance(values, Condition):
                changes[group] = values
            else:
                changes[group] = replace(getattr(self, group), **values)
        return replace(self, **changes)

"""Minimal deterministic world kernel for the A1 nontriviality smoke test.

One locality, N agents, a latent blight regime that shifts once. Each tick,
one collective mitigation decision (mitigate / don't) is applied uniformly --
this stands in for the full ballot/mandate/executor machinery in
PREREGISTRATION.md §3, which isn't built yet. The physical world and its
one hidden regime shift are what's under test here, not the institution.

v2: fixed a real bug found by running v1 -- food/medicine stocks accumulated
without bound during the low-blight period (production >> need with no
spoilage), producing a buffer so large it absorbed almost any post-shift
deficit regardless of policy. Every policy except never_mitigate scored a
perfect 0.0. Fixed with per-tick spoilage on both stocks (matching I-3's own
"artifact decay" principle -- resources aren't a bank account) and by
collapsing food-yield-penalty and sickness-risk into one underlying blight
SEVERITY that mitigation counters, so the two channels aren't fighting each
other's parameterization independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class WorldConfig:
    n_agents: int = 8
    n_ticks: int = 2000
    shift_tick: int = 800

    # Blight severity in [0, 1] drives both crop-yield penalty and sickness
    # risk from one underlying hidden variable, rather than two independently
    # tuned channels.
    severity_low_blight: float = 0.0
    severity_high_blight: float = 1.0
    mitigation_severity_reduction: float = 0.75  # mitigation cuts EFFECTIVE severity by this fraction

    base_food_yield_per_agent: float = 1.6
    food_yield_penalty_per_severity: float = 1.1  # yield = base - penalty * effective_severity
    base_sickness_prob: float = 0.03
    sickness_prob_penalty_per_severity: float = 0.32  # prob = base + penalty * effective_severity

    food_need_per_agent: float = 1.0
    medicine_production_per_tick: float = 1.5

    # Spoilage: fraction of each stock lost per tick, after consumption.
    # This is what keeps a good pre-shift period from banking an
    # unlimited buffer against a bad post-shift period.
    food_spoilage_rate: float = 0.10
    medicine_spoilage_rate: float = 0.05

    starting_food_stock: float = 10.0
    starting_medicine_stock: float = 5.0
    starting_wealth: float = 20.0
    wealth_income_per_tick: float = 1.0  # background economic activity, independent of blight
    mitigation_wealth_cost: float = 3.0


@dataclass
class WorldState:
    cfg: WorldConfig
    tick: int = 0
    food_stock: float = 0.0
    medicine_stock: float = 0.0
    wealth: float = 0.0
    blight_high: bool = False
    events: list = field(default_factory=list)

    @classmethod
    def initial(cls, cfg: WorldConfig) -> "WorldState":
        return cls(
            cfg=cfg,
            tick=0,
            food_stock=cfg.starting_food_stock,
            medicine_stock=cfg.starting_medicine_stock,
            wealth=cfg.starting_wealth,
            blight_high=False,
        )


def step(state: WorldState, mitigate: bool, rng: np.random.Generator) -> WorldState:
    """Advance the world by one tick. Mutates and returns state; callers
    that need history read state.events afterward."""
    cfg = state.cfg
    state.tick += 1

    if state.tick == cfg.shift_tick:
        state.blight_high = True

    severity = cfg.severity_high_blight if state.blight_high else cfg.severity_low_blight

    can_afford = state.wealth >= cfg.mitigation_wealth_cost
    mitigate = mitigate and can_afford
    if mitigate:
        state.wealth -= cfg.mitigation_wealth_cost
        severity *= (1.0 - cfg.mitigation_severity_reduction)

    yield_per_agent = max(0.0, cfg.base_food_yield_per_agent - cfg.food_yield_penalty_per_severity * severity)
    sickness_p = min(1.0, cfg.base_sickness_prob + cfg.sickness_prob_penalty_per_severity * severity)

    state.wealth += cfg.wealth_income_per_tick
    state.food_stock += yield_per_agent * cfg.n_agents
    state.medicine_stock += cfg.medicine_production_per_tick

    sick = rng.random(cfg.n_agents) < sickness_p
    n_sick = int(sick.sum())

    food_needed = cfg.food_need_per_agent * cfg.n_agents
    food_shortfall_agents = 0
    if state.food_stock < food_needed:
        fraction_fed = max(0.0, state.food_stock / food_needed) if food_needed > 0 else 1.0
        food_shortfall_agents = int(round((1.0 - fraction_fed) * cfg.n_agents))
        state.food_stock = 0.0
    else:
        state.food_stock -= food_needed
    state.food_stock *= (1.0 - cfg.food_spoilage_rate)

    medicine_shortfall_agents = 0
    if n_sick > 0:
        medicine_available = min(n_sick, int(state.medicine_stock))
        medicine_shortfall_agents = n_sick - medicine_available
        state.medicine_stock = max(0.0, state.medicine_stock - medicine_available)
    state.medicine_stock *= (1.0 - cfg.medicine_spoilage_rate)

    shortfall_this_tick = food_shortfall_agents + medicine_shortfall_agents

    state.events.append({
        "tick": state.tick,
        "blight_high": state.blight_high,
        "mitigate": mitigate,
        "n_sick": n_sick,
        "food_stock": state.food_stock,
        "medicine_stock": state.medicine_stock,
        "wealth": state.wealth,
        "food_shortfall_agents": food_shortfall_agents,
        "medicine_shortfall_agents": medicine_shortfall_agents,
        "shortfall_this_tick": shortfall_this_tick,
    })
    return state


def run(cfg: WorldConfig, policy_fn, seed: int) -> WorldState:
    """Run one full episode under `policy_fn(state, rng) -> bool` and return
    the final WorldState (with full per-tick event history in .events)."""
    rng = np.random.default_rng(seed)
    state = WorldState.initial(cfg)
    for _ in range(cfg.n_ticks):
        mitigate = policy_fn(state, rng)
        step(state, mitigate, rng)
    return state

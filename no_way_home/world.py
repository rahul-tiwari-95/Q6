"""Minimal deterministic world kernel for the A1 nontriviality smoke test.

N localities, each with its own agents and its own noisy sickness draws, but
one shared latent blight severity and one shared food/medicine/wealth pool --
the physical economy is still the single-locality one from v2, just now fed
by locality-level sickness counts instead of one pooled draw, because that's
what makes locality-attributed REPORT messages meaningful. This module's own
`step()`/`run()` still only take one collective mitigation decision per tick
via a plain policy_fn -- the full ballot/mandate/executor machinery from
PREREGISTRATION.md §3 is built (see institutions.py, committed after this
file), but it's layered ON TOP as a stateful policy_fn, not wired into the
world kernel itself.

v3: adds the message/lineage layer (I-7, I-8). Each locality that notices a
local sickness spike generates a REPORT with a fresh origin_id; existing
reports get randomly FORWARDED (copied), with forwards biased toward one
"hub" locality -- the high-reach-source scenario the source design's own
worked example (a high-reach store owner) calls out. See messages.py for the
counting utilities this makes checkable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from no_way_home.messages import MessageLog


@dataclass(frozen=True)
class WorldConfig:
    n_localities: int = 4
    n_agents_per_locality: int = 6
    n_ticks: int = 2000
    shift_tick: int = 800

    severity_low_blight: float = 0.0
    severity_high_blight: float = 1.0
    mitigation_severity_reduction: float = 0.75

    base_food_yield_per_agent: float = 1.6
    food_yield_penalty_per_severity: float = 1.1
    base_sickness_prob: float = 0.03
    sickness_prob_penalty_per_severity: float = 0.32

    food_need_per_agent: float = 1.0
    medicine_production_per_tick: float = 1.5

    food_spoilage_rate: float = 0.10
    medicine_spoilage_rate: float = 0.05

    starting_food_stock: float = 10.0
    starting_medicine_stock: float = 5.0
    starting_wealth: float = 20.0
    wealth_income_per_tick: float = 1.0
    mitigation_wealth_cost: float = 3.0

    # Messages
    report_sick_threshold: int = 3          # locality reports if this many+ agents sick this tick
    forward_prob_per_tick: float = 0.4      # chance of one forward event per tick
    hub_locality: int = 0                   # the high-reach source
    hub_forward_boost: float = 5.0          # hub reports get forwarded this many times more often

    @property
    def n_agents(self) -> int:
        return self.n_localities * self.n_agents_per_locality


@dataclass
class WorldState:
    cfg: WorldConfig
    tick: int = 0
    food_stock: float = 0.0
    medicine_stock: float = 0.0
    wealth: float = 0.0
    blight_high: bool = False
    events: list = field(default_factory=list)
    messages: MessageLog = field(default_factory=MessageLog)

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
    that need history read state.events / state.messages afterward."""
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

    # Sickness drawn per locality, not pooled -- this is what a REPORT
    # attaches to.
    sick_per_locality = []
    for loc in range(cfg.n_localities):
        sick = rng.random(cfg.n_agents_per_locality) < sickness_p
        n_sick_loc = int(sick.sum())
        sick_per_locality.append(n_sick_loc)
        if n_sick_loc >= cfg.report_sick_threshold:
            state.messages.report(tick=state.tick, locality=loc)
    n_sick = sum(sick_per_locality)

    if rng.random() < cfg.forward_prob_per_tick:
        state.messages.maybe_forward(
            tick=state.tick, rng=rng,
            hub_locality=cfg.hub_locality, hub_forward_boost=cfg.hub_forward_boost,
        )

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
        "sick_per_locality": sick_per_locality,
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
    the final WorldState (with full per-tick event history in .events, and
    the full message log in .messages).

    World physics and policy decisions get INDEPENDENT RNG streams, spawned
    from the same seed via SeedSequence. Found by running the election test:
    with one shared stream, a policy that consumes extra randomness
    internally (e.g. EpistemicDelegationInstitution drawing 24 votes per
    election) shifts every subsequent sickness/forwarding draw out of sync
    with a simpler policy under the "same" seed -- so "same seed" silently
    stopped meaning "same physical trajectory" the moment two policies
    disagreed on how much randomness they use. That breaks any paired
    same-seed comparison between policies, not just this one. Separate
    streams prevent policy RNG consumption from perturbing world randomness.
    Actions can still change the world's outcomes. Policies may implement
    start_episode() and end_episode(final_state) lifecycle hooks; ordinary
    functions need neither. A run ends as an episodic terminal boundary."""
    seed_seq = np.random.SeedSequence(seed)
    world_seed, policy_seed = seed_seq.spawn(2)
    world_rng = np.random.default_rng(world_seed)
    policy_rng = np.random.default_rng(policy_seed)
    state = WorldState.initial(cfg)
    start_episode = getattr(policy_fn, "start_episode", None)
    if start_episode is not None:
        start_episode()
    for _ in range(cfg.n_ticks):
        mitigate = policy_fn(state, policy_rng)
        step(state, mitigate, world_rng)
    end_episode = getattr(policy_fn, "end_episode", None)
    if end_episode is not None:
        end_episode(state)
    return state

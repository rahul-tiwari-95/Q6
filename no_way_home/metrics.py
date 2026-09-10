"""Primary and diagnostic metrics, derived from raw per-tick events only
(I-11: append-only truth, metrics reproducible from the event stream)."""

from __future__ import annotations

from no_way_home.world import WorldState


def need_shortfall_per_10k(state: WorldState, tick_range: tuple | None = None) -> float:
    """Primary metric: need-shortfall agent-ticks per 10,000 world-ticks,
    optionally restricted to a [start, end) tick range (e.g. post-shift
    only). Lower is better."""
    events = state.events
    if tick_range is not None:
        start, end = tick_range
        events = [e for e in events if start <= e["tick"] < end]
    if not events:
        return float("nan")
    total_shortfall = sum(e["shortfall_this_tick"] for e in events)
    return total_shortfall / len(events) * 10_000


def mitigation_rate(state: WorldState, tick_range: tuple | None = None) -> float:
    events = state.events
    if tick_range is not None:
        start, end = tick_range
        events = [e for e in events if start <= e["tick"] < end]
    if not events:
        return float("nan")
    return sum(1 for e in events if e["mitigate"]) / len(events)

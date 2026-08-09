"""Message/lineage mechanism, per PREREGISTRATION.md §2 (I-7, I-8) and §6.2.

I-7, verbatim from the source design: "Ten forwards of one test are one
evidence lineage, not ten independent tests." This module is what makes that
checkable instead of just written down: every message carries an origin_id,
a REPORT creates a fresh one, a FORWARD reuses its source's. Two ways to
count "how much evidence do we have" -- raw message count, and unique-origin
count -- diverge exactly when forwarding is unbalanced across sources, which
is the realistic case (some localities/agents have more reach than others).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Literal

import numpy as np

_origin_id_counter = count()


def _next_origin_id() -> int:
    return next(_origin_id_counter)


LineageRole = Literal["Initiates", "Happens-only"]


@dataclass
class Message:
    tick: int
    origin_id: int          # shared by a REPORT and every FORWARD of it
    locality: int            # where the ORIGINAL report came from
    is_forward: bool

    @property
    def lineage_role(self) -> LineageRole:
        """Event Calculus framing (Kowalski & Sergot 1986), per
        ENVIRONMENT_REDESIGN.md §2: a REPORT Initiates a fresh evidential
        fluent (a brand-new origin_id); a FORWARD only Happens relative to
        one that already exists and creates no new evidence. Derived
        directly from is_forward -- the field this generalizes -- so this is
        zero new state and zero behavior change, only a named invariant."""
        return "Happens-only" if self.is_forward else "Initiates"


@dataclass
class MessageLog:
    """Append-only message history plus a small recent-reports buffer used
    for sampling forwards. Not a full I-11 event-sourced log -- this is a
    smoke test, not the confirmatory kernel -- but same spirit: nothing here
    is mutated after the fact, only appended."""
    messages: list = field(default_factory=list)
    _recent_report_origins: list = field(default_factory=list)  # (origin_id, locality)

    def report(self, tick: int, locality: int) -> None:
        origin_id = _next_origin_id()
        self.messages.append(Message(tick=tick, origin_id=origin_id, locality=locality, is_forward=False))
        self._recent_report_origins.append((origin_id, locality))
        # keep the forward-sampling pool bounded
        if len(self._recent_report_origins) > 200:
            self._recent_report_origins = self._recent_report_origins[-200:]

    def maybe_forward(self, tick: int, rng: np.random.Generator, hub_locality: int | None,
                       hub_forward_boost: float) -> None:
        """With some probability, forward a random recent report. Reports
        originating in hub_locality are forwarded hub_forward_boost times
        more often than others -- the 'high-reach source' scenario."""
        if not self._recent_report_origins:
            return
        weights = np.array([
            hub_forward_boost if locality == hub_locality else 1.0
            for _, locality in self._recent_report_origins
        ])
        weights = weights / weights.sum()
        idx = rng.choice(len(self._recent_report_origins), p=weights)
        origin_id, locality = self._recent_report_origins[idx]
        self.messages.append(Message(tick=tick, origin_id=origin_id, locality=locality, is_forward=True))

    def raw_message_rate(self, current_tick: int, window: int) -> float:
        """Naive count: every message (original or copy) counts as one unit
        of evidence."""
        recent = [m for m in self.messages if current_tick - window <= m.tick < current_tick]
        return len(recent) / window

    def unique_origin_rate(self, current_tick: int, window: int) -> float:
        """Lineage-aware count: dedupe by origin_id first."""
        recent = [m for m in self.messages if current_tick - window <= m.tick < current_tick]
        return len({m.origin_id for m in recent}) / window

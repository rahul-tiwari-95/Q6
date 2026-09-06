"""Manipulability classification registry for No Way Home's live emitters,
per ENVIRONMENT_REDESIGN.md §2 (Maynard Smith & Harper's cue/index/signal
continuum, sharpened by Biernaskie/Perry/Grafen 2018 -- see CITATIONS.md).

This is a static audit, not a continuous cost-function apparatus: the only
enforcement is the invariant checked in tests/test_no_way_home_smoke.py --
no allogenic (agent-driven) emitter may write to an index-tagged channel.
Deliberately NOT the full 8-tuple Channel type from ENVIRONMENT_REDESIGN.md
§2 -- that abstraction is explicitly parked (ENGINEERING_NWH_PHASE_PLAN.md
§5) until there's a concrete reason to need it. Four live emitters only,
matching Increment 1's scope; the Mandate/Ballot Instrument (channel 6) and
CORROBORATION (channel 5, doesn't exist yet) are later increments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

EmitterClass = Literal["autogenic", "allogenic"]
Manipulability = Literal["cue", "index", "index-of-a-claim", "signal"]


@dataclass(frozen=True)
class ChannelTag:
    emitter_class: EmitterClass
    manipulability: Manipulability


CHANNELS: dict[str, ChannelTag] = {
    # WorldState.blight_high -- written only by step()'s own regime-shift
    # logic (world.py), never by a policy or message-generation code path.
    # Hard-tied to the world's actual hidden regime, so nothing can fake it.
    "blight_exposure": ChannelTag(emitter_class="autogenic", manipulability="index"),

    # WorldState.wealth / food_stock / medicine_stock -- byproducts of the
    # economy running, not transmitted to anyone on purpose.
    "resource_ledger": ChannelTag(emitter_class="autogenic", manipulability="cue"),

    # messages.py::MessageLog.report() -- a locality's deliberate, fresh
    # claim. Initiates a lineage (see Message.lineage_role).
    "report": ChannelTag(emitter_class="allogenic", manipulability="signal"),

    # messages.py::MessageLog.maybe_forward() -- a content-frozen relay of
    # an existing claim. Only as trustworthy as the report it points at, so
    # it's an index of that claim rather than a fresh signal in its own
    # right (see ENVIRONMENT_REDESIGN.md §2 for the full reasoning).
    "forward": ChannelTag(emitter_class="allogenic", manipulability="index-of-a-claim"),
}

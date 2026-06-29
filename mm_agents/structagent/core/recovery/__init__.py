"""Recovery subsystem.

- ``dispatcher.Recovery`` — the mixin that decides per-turn planner
  mode (the 5 stuck categories + initial / progress_check). Folded
  into ``StructAgent`` via inheritance.
- ``state.RecoveryState`` — frozen dataclass mirroring the
  dispatcher's counters + last-transition receipt. Refreshed once
  per turn by ``agent._finalize_recovery_state``.
- ``transitions.Transition`` — Literal enumerating the 5 stuck
  categories + 2 normal-flow labels.

This module re-exports the public names so existing
``from mm_agents.structagent.core.recovery import Recovery`` keeps
working after the file-to-folder restructure.
"""

from mm_agents.structagent.core.recovery.dispatcher import Recovery
from mm_agents.structagent.core.recovery.state import RecoveryState
from mm_agents.structagent.core.recovery.transitions import (
    STUCK_CATEGORIES,
    Transition,
)

__all__ = [
    "Recovery",
    "RecoveryState",
    "STUCK_CATEGORIES",
    "Transition",
]

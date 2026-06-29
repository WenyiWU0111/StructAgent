"""Recovery subsystem. Re-exports the public names below.

- ``dispatcher.Recovery``: mixin deciding per-turn planner mode (5 stuck
  categories + initial / progress_check); mixed into ``StructAgent``.
- ``state.RecoveryState``: frozen snapshot of the dispatcher's counters +
  last-transition receipt, refreshed per turn by ``agent._finalize_recovery_state``.
- ``transitions.Transition``: Literal over the 5 stuck categories + 2 normal labels.
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

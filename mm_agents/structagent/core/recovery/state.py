"""Recovery dispatcher state — per-turn snapshot of counters + last
transition receipt.

Frozen dataclass aggregating the recovery counters used by
``recovery.dispatcher._decide_planner_mode_and_reason`` plus a receipt
of the transition that fired. Bundles the scattered ``self._<counter>``
fields into one value that's easy to construct in tests, advance via
``dataclasses.replace``, and read downstream (T3 done-auditor).

Currently a read-only mirror: ``_finalize_recovery_state`` refreshes it
once per ``predict()`` after the dispatcher picks the turn's mode; the
agent's ``self._*`` counters stay source of truth.
"""

from dataclasses import dataclass, replace
from typing import Optional

from mm_agents.structagent.core.recovery.transitions import Transition


@dataclass(frozen=True)
class RecoveryState:
    """Per-turn snapshot of recovery counters + transition receipt.

    ``last_transition`` is written *after* the mode dispatcher runs, so it
    records why the previous turn fired what it did (not what the next
    turn should do). Tests can assert on it without inspecting messages.
    """

    # Stuck-related counters mirrored from agent.* (which still owns
    # increment/reset).
    replans_since_last_progress: int = 0
    last_search_fired_at_replan: int = -99  # matches agent.__init__ default
    total_replan_count: int = 0
    done_rejected_count: int = 0
    replans_within_current_strategy: int = 0

    # Receipt of the most recent transition decided by the dispatcher.
    # None before the first turn.
    last_transition: Optional[Transition] = None

    # Step index of the most recent turn whose transition was recorded.
    # Lets downstream code compare counters relative to "when" without
    # needing a separate wall-clock counter.
    turn_step_idx: int = -1

    @classmethod
    def bootstrap(cls) -> "RecoveryState":
        """Initial state for a fresh task. Matches the values that
        ``agent.__init__`` writes to its ``self.*`` counters."""
        return cls()

    def with_transition(
        self,
        transition: Transition,
        step_idx: int,
        *,
        replans_since_last_progress: Optional[int] = None,
        last_search_fired_at_replan: Optional[int] = None,
        total_replan_count: Optional[int] = None,
        done_rejected_count: Optional[int] = None,
        replans_within_current_strategy: Optional[int] = None,
    ) -> "RecoveryState":
        """Return a new ``RecoveryState`` with ``transition`` recorded
        and any counters updated. Unspecified counters are preserved
        from the prior state.
        """
        return replace(
            self,
            last_transition=transition,
            turn_step_idx=step_idx,
            replans_since_last_progress=(
                self.replans_since_last_progress
                if replans_since_last_progress is None
                else replans_since_last_progress
            ),
            last_search_fired_at_replan=(
                self.last_search_fired_at_replan
                if last_search_fired_at_replan is None
                else last_search_fired_at_replan
            ),
            total_replan_count=(
                self.total_replan_count
                if total_replan_count is None
                else total_replan_count
            ),
            done_rejected_count=(
                self.done_rejected_count
                if done_rejected_count is None
                else done_rejected_count
            ),
            replans_within_current_strategy=(
                self.replans_within_current_strategy
                if replans_within_current_strategy is None
                else replans_within_current_strategy
            ),
        )

"""Recovery dispatcher state — per-turn snapshot of counters + last
transition receipt.

A frozen dataclass that aggregates the recovery counters used by
``recovery.dispatcher._decide_planner_mode_and_reason`` plus the
receipt of which transition fired most recently. Replaces today's
scattered ``self._<counter>`` surface on the agent with a single
value that can be:

  - constructed in unit tests (no need to mock ``self.*`` fields)
  - copied via ``dataclasses.replace`` to advance to the next turn
  - read by downstream escalation (T3 done-auditor) via a single
    field rather than 5 separate ``self.*`` lookups

T1 scope: this is a *read-only mirror*, refreshed once per
``predict()`` call by ``_finalize_recovery_state`` immediately after
the dispatcher decides this turn's mode. The agent's ``self._*``
counters remain source of truth — ``RecoveryState`` mirrors them at a
fixed checkpoint. Future work can migrate ownership onto
``self._recovery_state`` and convert recovery handlers into pure
functions.

See plan file Part I.1.
"""

from dataclasses import dataclass, replace
from typing import Optional

from mm_agents.structagent.core.recovery.transitions import Transition


@dataclass(frozen=True)
class RecoveryState:
    """Per-turn snapshot of recovery counters + transition receipt.

    Receipt model (Claude Code T1 pattern):
        ``last_transition`` is written *after* the turn's mode
        dispatcher ran, so it answers "why did the previous turn fire
        what it fired" — not "what should the next turn do". Tests can
        assert ``state.last_transition == 'done_rejected'`` to check a
        recovery path fired without inspecting message contents.
    """

    # 5 stuck-related counters mirrored from agent.* (T1 scope:
    # read-only mirror; agent.* still owns the increment/reset).
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

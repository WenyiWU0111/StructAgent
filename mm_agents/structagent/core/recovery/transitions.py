"""Recovery transition labels.

A transition is a receipt written after a turn fires a stuck category
(or normal-flow label), so tests and downstream escalation (T3
done-auditor) can see which path fired without parsing messages.

Stuck categories match ``self._force_replan_category`` set by
``recovery.dispatcher._decide_planner_mode_and_reason``; normal-flow
labels match that dispatcher's ``mode`` strings.
"""

from typing import Literal


Transition = Literal[
    # === The 6 stuck categories ===
    "actor_failure",        # Rule #2 — actor reported IMPOSSIBLE / no-bash-body
    "done_rejected",        # Rule #3 — DONE rejected by ledger gate
    "done_audit_failed",    # Rule #3 — DONE refuted by adversarial auditor (T3)
    "budget_exhausted",     # Rule #4 — subgoal step budget exhausted
    "queue_empty_no_done",  # Rule #5 — queue drained but task not VERIFIED
    "replan_cadence",       # Rule #6 — N REPLANs within same strategy
    # === Normal-flow labels (also returned by the dispatcher) ===
    "initial",              # Rule #1 — first turn, no prior state
    "progress_check",       # Default — normal CONTINUE/REPLAN/DONE dispatch
]


# Stuck categories as a frozenset, for "did the last turn fire recovery?" filters.
STUCK_CATEGORIES: "frozenset[str]" = frozenset({
    "actor_failure",
    "done_rejected",
    "done_audit_failed",
    "budget_exhausted",
    "queue_empty_no_done",
    "replan_cadence",
})

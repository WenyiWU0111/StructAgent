"""Failure attribution — first-class subsystem (extracted from
ledger/verification/ in Phase 1, where it was nested 3 levels deep).

  - failure_attribution        : 5-stage diagnosis
                                 (RECALL→OBSERVE→CRITIQUE→ATTRIBUTE→ACT),
                                 role×category attribution.
  - failure_attribution_router : maps attribution verdict → routed intervention.
  - failure_attribution_executor: applies the chosen intervention.
  - stuck_diagnosis            : stuck-state detection + diagnosis injection.

Imports one-way from ledger.core (timeline types) — no ledger→attribution
cycle (ledger.core only stores StuckDiagnosis *results* as dict fields)."""

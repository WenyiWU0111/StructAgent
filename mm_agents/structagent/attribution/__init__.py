"""Failure attribution subsystem.

  - failure_attribution         : 5-stage diagnosis (RECALL→OBSERVE→CRITIQUE
                                  →ATTRIBUTE→ACT), role×category attribution.
  - failure_attribution_router  : attribution verdict → routed intervention.
  - failure_attribution_executor: applies the chosen intervention.
  - stuck_diagnosis             : stuck-state detection + diagnosis injection.

Imports one-way from ledger.core (timeline types) to avoid a cycle; ledger.core
only stores StuckDiagnosis results as dict fields."""

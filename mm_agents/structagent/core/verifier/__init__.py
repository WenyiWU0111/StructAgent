"""Verifier role — checks milestones and drives the progress ledger.

Two layers:
  - orchestrator.py (LedgerOrchestrator mixin) : ledger init / accumulate
    observations / fire keynode detection + summarizer / gate planner DONE.
  - the verification ENGINE (pure functions) that the orchestration calls:
      * verifiers.py          : deterministic verify-op registry
                                (file_grep / url_match / a11y_match / shell /
                                calc / impress / writer), 3-state True/False/None.
      * key_node_detector.py  : per-step milestone verification -> KeyNode events.
      * key_node_debug.py     : per-step KeyNode dumps for offline triage.

Relationship to ``ledger/``: the verifier PRODUCES KeyNode events; the
ledger CONSUMES them to derive milestone state. Import is one-way
(verifier -> ledger.core types); the ledger never imports the verifier.
"""
from .orchestrator import LedgerOrchestrator

__all__ = ["LedgerOrchestrator"]

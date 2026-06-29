"""Verifier role — checks milestones and drives the progress ledger.

  - orchestrator.py (LedgerOrchestrator mixin): ledger init, accumulate
    observations, fire keynode detection + summarizer, gate planner DONE.
  - verifiers.py: deterministic verify-op registry (file_grep / url_match /
    a11y_match / shell / calc / impress / writer), 3-state True/False/None.
  - key_node_detector.py: per-step milestone verification -> KeyNode events.
  - key_node_debug.py: per-step KeyNode dumps for offline triage.

Import is one-way: verifier PRODUCES KeyNode events (verifier -> ledger.core
types); the ledger CONSUMES them and never imports the verifier.
"""
from .orchestrator import LedgerOrchestrator

__all__ = ["LedgerOrchestrator"]

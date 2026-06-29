"""Progress Ledger package — structured proprioceptive state for GUI agents.

Public API:

    from mm_agents.structagent.ledger import (
        ProgressLedger, Outcome, DoneEntry, FailedPath, InitialContext,
        init_ledger, update_ledger,
    )
"""
from mm_agents.structagent.ledger.core.ledger import (
    ProgressLedger,
    Outcome,
    VerifySpec,
    DoneEntry,
    FailedPath,
    InitialContext,
)
from mm_agents.structagent.ledger.core.initializer import init_ledger
from mm_agents.structagent.ledger.core.summarizer import update_ledger

__all__ = [
    "ProgressLedger",
    "Outcome",
    "VerifySpec",
    "DoneEntry",
    "FailedPath",
    "InitialContext",
    "init_ledger",
    "update_ledger",
]

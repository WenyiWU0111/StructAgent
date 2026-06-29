"""Done-Auditor subsystem (T3).

Adversarial second-opinion check on DONE claims (see plan file Part
I.2). Default on in ``StructAgent``; it can also be enabled explicitly
via ``ENABLE_DONE_AUDITOR=1`` in env-driven runs.

Public surface:

  - ``DoneAuditSnapshot`` — frozen dataclass of state at the DONE
    moment (task, screenshots, a11y, outcome proofs, perceiver focus).
    Same shape whether built offline (from a run dir) or live (from
    in-flight agent attrs).
  - ``build_snapshot_from_run_dir(run_dir)`` — offline replay builder.
  - ``build_snapshot_from_live_state(agent)`` — runtime builder.
  - ``audit_done(agent)`` — top-level entry from ``agent._pa_predict``.
    Returns ``(verdict, reason, AuditResult)``; verdict is always one of
    PASS / FAIL / PARTIAL (errors and parser-fails map to PARTIAL).
  - ``run_audit(snapshot)`` — lower-level entry that takes any snapshot
    (used by ``tests/replay_done_audit.py``).
  - ``parse_verdict(text)`` — last-line VERDICT extractor.
"""

from mm_agents.structagent.core.final_audit.done_auditor import (
    AuditResult,
    Verdict,
    audit_done,
    parse_verdict,
    run_audit,
)
from mm_agents.structagent.core.final_audit.prompts import (
    SYSTEM_PROMPT,
    build_chat_messages,
    build_user_message_text,
)
from mm_agents.structagent.core.final_audit.snapshot import (
    DoneAuditSnapshot,
    SnapshotSourceMeta,
    build_snapshot_from_live_state,
    build_snapshot_from_run_dir,
)

__all__ = [
    "AuditResult",
    "DoneAuditSnapshot",
    "SnapshotSourceMeta",
    "SYSTEM_PROMPT",
    "Verdict",
    "audit_done",
    "build_chat_messages",
    "build_snapshot_from_live_state",
    "build_snapshot_from_run_dir",
    "build_user_message_text",
    "parse_verdict",
    "run_audit",
]

"""Done-Auditor subsystem (T3).

Adversarial second-opinion check on DONE claims (see plan file Part I.2).
On by default in ``StructAgent``; also gated by ``ENABLE_DONE_AUDITOR=1`` in
env-driven runs.

Public surface:
  - ``DoneAuditSnapshot`` — frozen state at the DONE moment (task, screenshots,
    a11y, outcome proofs, perceiver focus). Same shape offline or live.
  - ``build_snapshot_from_run_dir`` / ``build_snapshot_from_live_state`` —
    offline-replay vs runtime builders.
  - ``audit_done(agent)`` — entry from ``agent._pa_predict``. Returns
    ``(verdict, reason, AuditResult)``; verdict is PASS/FAIL/PARTIAL (errors and
    parse-fails map to PARTIAL).
  - ``run_audit(snapshot)`` — lower-level entry for any snapshot (used by
    ``tests/replay_done_audit.py``).
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

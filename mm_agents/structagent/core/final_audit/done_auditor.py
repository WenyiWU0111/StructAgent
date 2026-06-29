"""Done-Auditor LLM call + VERDICT parser + top-level audit_done.

Used by ``tests/replay_done_audit.py`` (offline replay) and
``agent._pa_predict``'s DONE-acceptance hook (runtime). Both build a
``DoneAuditSnapshot`` and call ``audit_done``. Runtime uses the agent's
``done_auditor_model`` over the shared ``LLMClient`` routing, so auditor
endpoints can't drift from the main dispatch table.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from mm_agents.structagent.core.final_audit.prompts import build_chat_messages
from mm_agents.structagent.core.final_audit.snapshot import DoneAuditSnapshot
from mm_agents.structagent.utils.llm_client import LLMClient


Verdict = Literal["PASS", "FAIL", "PARTIAL"]


# Match 'VERDICT: <X>' at line start (ignores inline mentions). Last match
# wins, handling "almost FAIL, reconsidered to PARTIAL" reasoning chains.
_VERDICT_RE = re.compile(
    r"^\s*VERDICT:\s*(PASS|FAIL|PARTIAL)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_verdict(text: str) -> Optional[Verdict]:
    """Return the LAST VERDICT in ``text``, or None if absent.

    None (parser found nothing) should be treated as PARTIAL by callers; it's
    kept distinct from explicit PARTIAL so batch metrics can flag parser-fails.
    """
    if not text:
        return None
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()  # type: ignore[return-value]


class _StandaloneAuditorClient(LLMClient):
    """Adapter for offline replay (no agent instance).

    Runtime calls go through ``agent.call_llm`` directly; this supplies the
    attributes ``LLMClient`` expects while using the same routing table.
    """

    def __init__(self, *, max_tokens: int, temperature: float, top_p: float):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p


def call_auditor_llm(
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout_s: float = 120.0,
    model_alias: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Call the auditor LLM via the shared LLM client.

    Returns ``(response_text, meta)`` (meta: elapsed_s, model, route_source).
    ``temperature=0.0`` so a snapshot always yields the same verdict —
    deterministic for offline replay and repeatable for A/B in production.
    """
    model = model_alias or "vllm_qwen35-vl"
    client = _StandaloneAuditorClient(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    t0 = time.time()
    text = client.call_llm(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        model,
    )
    elapsed = time.time() - t0
    meta: Dict[str, Any] = {
        "elapsed_s": round(elapsed, 2),
        "model": model,
        "route_source": "mm_agents.structagent.utils.llm_client.LLMClient",
        "timeout_s": timeout_s,
    }
    return text or "", meta


@dataclass
class AuditResult:
    """Result of one audit call.

    ``verdict`` is None when the LLM responded but no VERDICT line parsed.
    ``error`` is set on transport errors (network, timeout, backend exception).
    Treat both as "auditor unavailable", not as task refutations.
    """

    verdict: Optional[Verdict]
    response_text: str
    llm_meta: Dict[str, Any]
    error: Optional[str] = None


def run_audit(snap: DoneAuditSnapshot, model_alias: Optional[str] = None) -> AuditResult:
    """Build messages, call LLM, parse verdict.

    Any LLM error is returned in ``error`` rather than raised — the DONE site
    must not crash on a transient auditor failure.
    """
    msgs = build_chat_messages(snap)
    try:
        text, llm_meta = call_auditor_llm(msgs, model_alias=model_alias)
    except Exception as e:  # noqa: BLE001 — report any LLM error as-is
        return AuditResult(
            verdict=None,
            response_text="",
            llm_meta={},
            error=f"{type(e).__name__}: {e}",
        )
    verdict = parse_verdict(text)
    return AuditResult(
        verdict=verdict,
        response_text=text,
        llm_meta=llm_meta,
    )


# === Top-level entry used by agent.py ============================ #


def audit_done(agent: Any) -> Tuple[Verdict, str, AuditResult]:
    """Run the auditor at the agent's current DONE attempt.

    Returns ``(verdict, reason, full_result)``. verdict is always
    PASS/FAIL/PARTIAL (errors and parser-fails → PARTIAL). reason is a short
    summary for ``self._force_replan_reason``. Persist full_result to
    ``<results_dir>/audit_debug/`` for offline triage.
    """
    from mm_agents.structagent.core.final_audit.snapshot import (
        build_snapshot_from_live_state,
    )
    snap = build_snapshot_from_live_state(agent)
    model_alias = (
        getattr(agent, "done_auditor_model", None)
        or getattr(agent, "verifier_model", None)
        or "vllm_qwen35-vl"
    )
    msgs = build_chat_messages(snap)
    try:
        t0 = time.time()
        text = agent.call_llm(
            {
                "model": model_alias,
                "messages": msgs,
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_p": 1.0,
            },
            model_alias,
        )
        result = AuditResult(
            verdict=parse_verdict(text or ""),
            response_text=text or "",
            llm_meta={
                "elapsed_s": round(time.time() - t0, 2),
                "model": model_alias,
                "route_source": "agent.call_llm",
            },
        )
    except Exception as e:  # noqa: BLE001
        result = AuditResult(
            verdict=None,
            response_text="",
            llm_meta={
                "model": model_alias,
                "route_source": "agent.call_llm",
            },
            error=f"{type(e).__name__}: {e}",
        )

    if result.error:
        return (
            "PARTIAL",
            f"done auditor errored: {result.error}",
            result,
        )

    verdict: Verdict = result.verdict or "PARTIAL"
    if verdict == "PASS":
        return verdict, "done audit accepted DONE", result

    # FAIL/PARTIAL — one-line reason from GAP ANALYSIS or the last non-VERDICT
    # line; full reasoning stays in ``result.response_text``.
    reason = _summarize_reason(result.response_text, verdict)
    return verdict, reason, result


# Match the GAP ANALYSIS header WITH its trailing colon (``:?``), so iteration
# starts on the next line. Without it the regex stops at ``S`` and the lone
# ``:`` becomes the reason — logs like ``done audit FAIL: :`` hiding the real
# message.
_GAP_HEADER_RE = re.compile(r"(?im)^\s*GAP ANALYSIS\s*:?")


def _summarize_reason(text: str, verdict: Verdict) -> str:
    """Extract a 1-line gap summary: first non-empty bullet under GAP ANALYSIS,
    else the last non-VERDICT line."""
    if not text:
        return f"done audit returned {verdict} with empty response"

    m = _GAP_HEADER_RE.search(text)
    if m:
        tail = text[m.end():]
        for line in tail.splitlines():
            stripped = line.strip().lstrip("-* ").strip()
            # Skip empties + punctuation-only residue (lone ``:`` from a split
            # header, or a separator bullet).
            if not stripped or all(c in ":-*. " for c in stripped):
                continue
            if stripped.startswith("VERDICT:"):
                break
            return (f"done audit {verdict}: " + stripped[:240])

    # Fallback: last non-VERDICT, non-empty, non-punct-only line.
    for line in reversed(text.splitlines()):
        stripped = line.strip().lstrip("-* ").strip()
        if not stripped or all(c in ":-*. " for c in stripped):
            continue
        if stripped.startswith("VERDICT:"):
            continue
        return (f"done audit {verdict}: " + stripped[:240])

    return f"done audit {verdict} (no extractable reason)"

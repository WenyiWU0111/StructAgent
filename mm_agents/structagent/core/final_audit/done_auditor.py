"""Done-Auditor LLM call + VERDICT parser + top-level audit_done.

Used by both:

  - ``tests/replay_done_audit.py`` (offline replay, Phase A validation)
  - ``agent._pa_predict`` DONE-acceptance hook (runtime, Phase B)

Both paths build a ``DoneAuditSnapshot`` (see ``snapshot.py``) and then
call ``audit_done(snapshot)``. Runtime calls use the agent's configured
``done_auditor_model`` and the shared causal-agent ``LLMClient`` routing,
so auditor endpoints cannot drift from the main model dispatch table.
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


# Verdict parser — match the LAST 'VERDICT: <X>' line in the response.
# Matches at line start, optional whitespace; ignores inline mentions.
# If multiple matches, last one wins (handles "I almost decided FAIL,
# reconsidered to PARTIAL" reasoning chains).
_VERDICT_RE = re.compile(
    r"^\s*VERDICT:\s*(PASS|FAIL|PARTIAL)\b",
    re.IGNORECASE | re.MULTILINE,
)


def parse_verdict(text: str) -> Optional[Verdict]:
    """Return the LAST VERDICT line in ``text``, or None if absent.

    None means the parser found nothing — callers should treat as
    PARTIAL (conservative refusal). The distinction "no verdict" vs
    "explicit PARTIAL" is preserved so batch metrics can flag
    parser-fail runs separately.
    """
    if not text:
        return None
    matches = _VERDICT_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()  # type: ignore[return-value]


class _StandaloneAuditorClient(LLMClient):
    """Small adapter for offline replay.

    Live runtime calls pass through ``agent.call_llm`` directly. Offline
    replay has no agent instance, so this adapter supplies the attributes
    expected by ``LLMClient`` while still using the same central routing table.
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
    """Call the auditor LLM via the shared causal-agent LLM client.

    Returns ``(response_text, meta)``. ``meta`` includes elapsed_s,
    model, and route_source. ``temperature=0.0`` so the same snapshot
    always produces the same verdict — we want deterministic behavior
    for offline replay AND repeatable A/B vs the structured verifier
    in production.
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

    ``verdict`` is None when the LLM responded but the parser couldn't
    find a VERDICT line. ``error`` is non-None on transport errors
    (network, timeout, OpenAI-compatible backend exception). Runtime
    callers should treat errors and parser failures as unavailable, not
    as task refutations.
    """

    verdict: Optional[Verdict]
    response_text: str
    llm_meta: Dict[str, Any]
    error: Optional[str] = None


def run_audit(snap: DoneAuditSnapshot, model_alias: Optional[str] = None) -> AuditResult:
    """End-to-end: build messages, call LLM, parse verdict.

    Best-effort: any LLM-level error is returned in ``error`` rather
    than raising — the agent loop's DONE site must not crash on a
    transient auditor failure.
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

    Builds the snapshot from live agent state, calls the LLM, returns
    ``(verdict, reason, full_result)`` where ``verdict`` is always one
    of PASS/FAIL/PARTIAL (errors and parser-fails map to PARTIAL —
    conservative uncertainty). ``reason`` is a short summary suitable for
    staging into ``self._force_replan_reason``.

    The ``full_result`` includes the auditor's reasoning + LLM meta;
    callers should persist it to ``<results_dir>/audit_debug/`` for
    offline triage.
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

    # FAIL or PARTIAL — extract a one-line "reason" by grabbing the
    # GAP ANALYSIS or last non-VERDICT line of the response. The full
    # reasoning is in ``result.response_text``.
    reason = _summarize_reason(result.response_text, verdict)
    return verdict, reason, result


# Match the GAP ANALYSIS header WITH the trailing colon so the iterator
# below starts at the next line. Without ``:?`` the regex stops at the
# letter ``S`` and the next "line" is just ``:``, which then becomes the
# extracted reason — yielding logs like ``done audit FAIL: :`` and
# hiding the real auditor message.
_GAP_HEADER_RE = re.compile(r"(?im)^\s*GAP ANALYSIS\s*:?")


def _summarize_reason(text: str, verdict: Verdict) -> str:
    """Extract a 1-line gap summary from the auditor response.

    Strategy: prefer the first non-empty bullet under GAP ANALYSIS.
    Fallback: the last non-VERDICT line of the response.
    """
    if not text:
        return f"done audit returned {verdict} with empty response"

    m = _GAP_HEADER_RE.search(text)
    if m:
        tail = text[m.end():]
        for line in tail.splitlines():
            stripped = line.strip().lstrip("-* ").strip()
            # Skip empties + punctuation-only residue (e.g. a lone ``:``
            # left over when the GAP-ANALYSIS header was split across
            # lines, or a bullet that's literally a separator).
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

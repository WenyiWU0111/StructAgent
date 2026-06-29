"""VLM verifier for one actor turn inside a subgoal.

Narrower than the boundary verifier: checks whether the last action helped
the current subgoal, from before/after screenshots plus a short turn history.
Does not touch the ledger or certify task-level completion.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


VALID_STATUSES = {"complete", "continue", "no_effect", "off_track", "stuck"}

SYSTEM = """You are a strict low-level verifier for a computer-use agent.

Judge only the CURRENT SUBGOAL, not the whole task. Use the recent actor-turn
history for context, then judge the CURRENT actor turn by comparing the CURRENT
BEFORE and CURRENT AFTER screenshots with the actor's last action.

Return:
- complete: the current subgoal is visibly satisfied now.
- continue: the action made useful progress, but the subgoal is not complete.
- no_effect: the action appears to have no useful visible effect.
- off_track: the agent moved to the wrong app/page/dialog or changed the wrong thing.
- stuck: the screen is blocked, loading, errored, or needs higher-level replanning.

Be conservative. Do not mark complete unless the AFTER screenshot clearly
supports the expected post-state. Output JSON only.
"""


@dataclass
class LowLevelVerifyContext:
    instruction: str
    subgoal: str
    expected_post_state: Optional[str]
    actor_response: str
    actions: List[str]
    before_screenshot_b64: str
    after_screenshot_b64: str
    prior_screenshots_b64: Optional[List[str]] = None
    recent_actor_actions: Optional[List[str]] = None
    actor_terminal: Optional[str] = None
    prior_feedback: Optional[str] = None


@dataclass
class LowLevelVerdict:
    status: str
    reason: str = ""
    evidence: str = ""
    feedback_to_actor: Optional[str] = None
    raw: str = ""
    error: Optional[str] = None


def _b64_to_url(b64: str) -> str:
    if b64.startswith("data:image"):
        return b64
    return f"data:image/png;base64,{b64}"


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    dec = json.JSONDecoder()
    i = s.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(s[i:])
            return obj if isinstance(obj, dict) else None
        except Exception:
            i = s.find("{", i + 1)
    return None


def parse_low_level_verdict(raw: str) -> LowLevelVerdict:
    item = _extract_json_object(raw)
    if not isinstance(item, dict):
        return LowLevelVerdict(
            status="continue",
            reason="verifier response was not parseable; keep actor control",
            raw=raw or "",
            error="parse_error",
        )
    status = str(item.get("status") or "").strip().lower()
    if status not in VALID_STATUSES:
        status = "continue"
        error = "invalid_status"
    else:
        error = None
    feedback = item.get("feedback_to_actor")
    if feedback is not None:
        feedback = str(feedback).strip() or None
    return LowLevelVerdict(
        status=status,
        reason=str(item.get("reason") or "").strip(),
        evidence=str(item.get("evidence") or "").strip(),
        feedback_to_actor=feedback,
        raw=raw or "",
        error=error,
    )


def build_low_level_messages(ctx: LowLevelVerifyContext) -> List[Dict[str, Any]]:
    lines = [
        f"[TASK]\n{ctx.instruction}",
        f"[CURRENT SUBGOAL]\n{ctx.subgoal}",
    ]
    if ctx.expected_post_state:
        lines.append(f"[EXPECTED POST-STATE]\n{ctx.expected_post_state}")
    if ctx.prior_feedback:
        lines.append(f"[PREVIOUS VERIFIER FEEDBACK]\n{ctx.prior_feedback}")
    if ctx.actor_terminal:
        lines.append(f"[ACTOR TERMINAL CLAIM]\n{ctx.actor_terminal}")
    recent_actions = list(ctx.recent_actor_actions or [])[-3:]
    if recent_actions:
        lines.append(
            "[RECENT ACTOR TURN HISTORY before the current turn]\n"
            + "\n".join(f"  - {a}" for a in recent_actions)
        )
    lines.append(f"[ACTOR RESPONSE]\n{ctx.actor_response or ''}")
    if ctx.actions:
        lines.append("[EXECUTED ACTION CODE]\n" + "\n".join(ctx.actions))
    else:
        lines.append("[EXECUTED ACTION CODE]\n(no executable action)")
    lines.append(
        "Output exactly one JSON object:\n"
        "{\n"
        '  "status": "complete|continue|no_effect|off_track|stuck",\n'
        '  "reason": "one short sentence",\n'
        '  "evidence": "brief visible evidence from the AFTER screenshot",\n'
        '  "feedback_to_actor": "specific next low-level hint, or null"\n'
        "}"
    )
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": "\n\n".join(lines)}
    ]
    prior_screenshots = list(ctx.prior_screenshots_b64 or [])[-3:]
    for i, b64 in enumerate(prior_screenshots, start=1):
        if not b64:
            continue
        user_content.extend([
            {"type": "text", "text": (
                f"[PRIOR actor turn after screenshot {i} — context only; "
                "do not judge completion from this frame alone]"
            )},
            {"type": "image_url",
             "image_url": {"url": _b64_to_url(b64)}},
        ])
    user_content.extend([
            {"type": "text", "text": "[BEFORE screenshot]"},
            {"type": "image_url",
             "image_url": {"url": _b64_to_url(ctx.before_screenshot_b64)}},
            {"type": "text", "text": "[AFTER screenshot]"},
            {"type": "image_url",
             "image_url": {"url": _b64_to_url(ctx.after_screenshot_b64)}},
    ])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_content},
    ]


def verify_actor_turn(
    ctx: LowLevelVerifyContext,
    *,
    call_llm: Any,
    model: str,
    logger: Any = None,
) -> LowLevelVerdict:
    messages = build_low_level_messages(ctx)
    try:
        raw = call_llm(
            {"model": model, "messages": messages,
             "max_tokens": 700, "temperature": 0.0, "top_p": 0.9},
            model,
        )
    except Exception as e:  # verifier infra should not crash the agent loop
        if logger:
            logger.warning("[ActorBurst] low-level verifier call failed: %s", e)
        return LowLevelVerdict(
            status="continue",
            reason="low-level verifier call failed; keep actor control",
            error=str(e),
        )
    verdict = parse_low_level_verdict(raw or "")
    if logger:
        logger.info(
            "[ActorBurst] low-level verdict=%s reason=%s",
            verdict.status,
            verdict.reason[:180],
        )
    return verdict

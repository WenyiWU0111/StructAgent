"""Auditor system prompt + user-message templating.

Validated in Phase A (tests/replay_done_audit.py + replay_done_audit_labels.csv).
The known-artifact table was dropped (task-spec leak); the 3-check BEFORE-FAIL
gate (RELEVANCE / GROUNDING / AVAILABILITY) is its abstract replacement.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from mm_agents.structagent.core.final_audit.snapshot import DoneAuditSnapshot


# Adversarial system prompt (verificationAgent pattern, adapted to OSWorld).
# Phase-A: 4/6 detection on false-DONEs, 1/6 FP on true-DONEs, 0/3 control fires.
SYSTEM_PROMPT = """\
You are the Done-Auditor for an OSWorld GUI agent. The agent has just
claimed the task is DONE. A separate structured verifier has already
approved this claim against its own checklist. Your job is to
adversarially refute that approval by examining only the raw evidence
you receive — the user's literal task text, the screenshots you see,
the accessibility tree, and the proof traces the structured verifier
already accepted.

You do NOT see the planner's chain of thought, the plan steps, the
current subgoal text, or any strategy notes. This is by design — you
are a fresh pair of eyes.

You have two documented failure patterns. Recognize them and do the
opposite:
- VERIFICATION AVOIDANCE: you read the outcome proofs the structured
  verifier already accepted and say "ok the verifier verified it, looks
  fine". The structured verifier is an LLM/heuristic that already made
  one mistake getting here. Verify independently against the raw a11y
  and screenshot.
- SEDUCED BY THE FIRST 80%: the final URL matches, the page heading
  matches, the structured verifier passed — feels like done. The first
  80% is the easy part. Your value is in the last 20%: did the agent
  actually reach the SPECIFIC state the task asks for, or just one that
  superficially resembles it?

TASK TEXT IS TRUTH; verifier_specs are ONE LLM's guess at success
written at task start. When the spec marks an outcome PASS but the
task text names a target (a specific identifier, value, location,
property) that the snapshot does NOT show, trust the task text and
FAIL. Always parse the task into concrete deliverables FIRST, then
check each against the snapshot — do not chain off "spec passed".

=== RECOGNIZE YOUR OWN RATIONALIZATIONS ===
You will feel the urge to skip the hard check. These are the exact
excuses you reach for — recognize them and do the opposite:
- "The structured verifier already passed" — the structured verifier is
  an LLM, not ground truth. Re-verify against a11y + screenshot.
- "The URL looks right" — URL on its own is rarely the deliverable.
  Check the page CONTENT.
- "The heading matches" — a heading is a UI affordance, not the user's
  target state. Verify the specific content the task asks for.
- "Looks complete in the screenshot" — pixel-glance is not verification.
  Quote the exact a11y row or screenshot region.
- "The formula / setting / property text matches" — text matching
  is shallow. A formula whose text matches the expected pattern
  can still evaluate to an error code, point at the wrong source,
  or live in the wrong target cell; a setting whose label matches
  can still be the "off" state. Check the rendered value / state,
  not just the symbol. For toggles/checkboxes/switches/radios:
  the ON state requires the a11y row to contain `checked` /
  `pressed` / `selected` — `enabled` alone means the control is
  clickable but currently OFF, and a closed confirmation dialog
  only proves the user acknowledged intent, not that the underlying
  toggle flipped.
- "This is probably fine" — probably is not verified. State the
  evidence.

=== OUTPUT FORMAT (REQUIRED) ===

Structure your response as:

  TASK PARSE:
    <one bullet per required deliverable in the task wording>

  EVIDENCE CROSS-CHECK:
    <one bullet per task requirement, citing the specific a11y row /
     outcome_proof / screenshot observation that supports or refutes it.
     If something is unsupported, say so explicitly.>

  GAP ANALYSIS (if any):
    <one bullet per identified gap; quote the exact snapshot content
     that reveals the gap>

  Final line MUST be on its own line, exactly one of:

    VERDICT: PASS
    VERDICT: FAIL
    VERDICT: PARTIAL

  Use PASS when the visible evidence directly supports every
  requirement of the task wording.

  Use FAIL when at least one concrete requirement is not satisfied AND
  the gap is supported by a verbatim snapshot quote.

  Use PARTIAL when the evidence is genuinely insufficient to decide
  — e.g. the relevant property lives off-screen, requires filesystem
  access, or the snapshot is incomplete. PARTIAL is NOT a hedge for
  "I'm unsure if it's a bug." If you can decide PASS or FAIL from the
  snapshot, you must.

=== BEFORE returning VERDICT: FAIL, run these three checks ===

(1) RELEVANCE: Is the gap you identified a property the TASK TEXT
    actually asked for? Only the task text is authoritative here —
    verifier_specs are one LLM's t=0 guess and may name properties
    the task never required, or omit ones it did. If the task did
    not name this property (a specific style, UI state, rendering
    detail), do not FAIL on it; many environment-side properties
    are incidental rendering choices, not task requirements.

(2) GROUNDING: Re-quote the exact a11y row, screenshot observation, or
    outcome_proofs string that supports your gap. If you cannot quote
    snapshot content directly — only "the screenshot probably shows" or
    "I assume the page is" — downgrade FAIL to PARTIAL.

(3) AVAILABILITY: Could the gap be confirmed ONLY by a signal that is
    NOT in the snapshot (filesystem state, clipboard content,
    off-screen widgets)? If yes, return PARTIAL, not FAIL — the
    structured verifier (not you) owns predicates the snapshot cannot
    observe.

Only after all three checks pass, return VERDICT: FAIL.
"""


def _render_outcome_proof_line(p: Dict[str, Any]) -> str:
    state = p.get("state", "?")
    step = p.get("verified_at_step") or p.get("reverted_at_step")
    ev = (p.get("evidence", "") or "").strip()
    return f"  - {p.get('outcome_id')} [{state}] @ step {step}: {ev}"


def _render_verifier_spec_line(s: Dict[str, Any]) -> str:
    kind = (s.get("verify") or {}).get("kind", "?")
    desc = (s.get("description", "") or "").strip()
    spec_obj = s.get("verify") or {}
    spec_short = json.dumps(spec_obj, separators=(",", ":"))[:240]
    return (
        f"  - id={s.get('id')} kind={kind}\n"
        f"    description: {desc}\n"
        f"    verify: {spec_short}"
    )


def build_user_message_text(snap: DoneAuditSnapshot) -> str:
    """Render the snapshot as the auditor's user-message text.

    Screenshots are appended separately as image_url parts (build_chat_messages);
    this covers everything else.
    """
    parts: List[str] = []

    parts.append("TASK (verbatim user request):")
    parts.append("```")
    parts.append(snap.task_instruction.strip())
    parts.append("```")
    parts.append("")

    parts.append("INITIAL ENVIRONMENT (at task start):")
    parts.append("```json")
    parts.append(json.dumps(snap.initial_context, indent=2, default=str))
    parts.append("```")
    parts.append("")

    parts.append("STRUCTURED VERIFIER SPECS (what the in-loop verifier was checking):")
    if snap.verifier_specs:
        for s in snap.verifier_specs:
            parts.append(_render_verifier_spec_line(s))
    else:
        parts.append("  (none)")
    parts.append("")

    parts.append(
        f"CURRENT ACTIVE URL: {snap.current_url if snap.current_url else '(none)'}"
    )
    parts.append("")

    parts.append(
        "CURRENT ACCESSIBILITY TREE (capped, last 6000 chars shown if longer):"
    )
    parts.append("```")
    a11y = snap.current_a11y
    parts.append(a11y if len(a11y) < 6000 else a11y[-6000:])
    parts.append("```")
    parts.append("")

    parts.append(
        "STRUCTURED VERIFIER's PROOF CLAIMS (to cross-check against "
        "the snapshot below — these are the verifier's claims, NOT "
        "ground truth):"
    )
    if snap.outcome_proofs:
        for p in snap.outcome_proofs:
            parts.append(_render_outcome_proof_line(p))
    else:
        parts.append(
            "  (none — agent declared DONE with zero verified outcomes)"
        )
    parts.append("")

    parts.append("PERCEIVER FOCUS BLOCK AT DONE STEP:")
    pfb = snap.perceiver_focus_block
    pfb_trim = {
        k: v for k, v in pfb.items()
        if k not in ("relevant_controls", "visual_focus", "a11y_full_rows")
    }
    parts.append("```json")
    parts.append(json.dumps(pfb_trim, indent=2, default=str)[:2000])
    parts.append("```")
    parts.append("")

    parts.append(
        "Now run TASK PARSE / EVIDENCE CROSS-CHECK / GAP ANALYSIS as "
        "instructed in the system prompt and emit your verdict."
    )

    return "\n".join(parts)


def build_chat_messages(
    snap: DoneAuditSnapshot,
    *,
    include_screenshots: bool = True,
    max_screenshots: int = 6,   # initial + last 5 frames (the workflow into DONE)
) -> List[Dict[str, Any]]:
    """Compose [system, user] OpenAI-style chat messages.

    User message is multimodal: rendered-snapshot text + up to ``max_screenshots``
    image_url parts (initial + last K frames), b64 data-URLs for any
    OpenAI-compatible vLLM endpoint.
    """
    user_text = build_user_message_text(snap)
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": user_text}
    ]

    if include_screenshots:
        ss: List[str] = []
        if snap.initial_screenshot_b64:
            ss.append(snap.initial_screenshot_b64)
        ss.extend(snap.recent_screenshots_b64)
        # Dedup adjacent dupes (some runs have initial == first recent);
        # strict equality is enough — same b64 payload byte-for-byte.
        deduped: List[str] = []
        for s in ss:
            if not deduped or deduped[-1] != s:
                deduped.append(s)
        for img in deduped[:max_screenshots]:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img}"},
            })

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

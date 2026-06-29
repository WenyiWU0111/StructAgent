"""Render router-decision payloads as recovery prompt blocks.

When the failure-attribution router picks ``actor_redo`` /
``planner_replan`` / ``fallback_planner``, the target agent needs to see
the diagnosis. Raw JSON gets ignored, so we wrap it in a reasoning
scaffold (acknowledge cause / cite memory / name a different next step)
under a "[Recovery context]" header.

render_actor_recovery_block: actor decomposer prompt. No coord hints —
the actor's own grounding handles HOW.
render_planner_recovery_block: planner force_replan prompt. Adds
attributed role/category so the planner knows whether the bad loop was
its own plan vs actor/env/verifier.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def _bullets(items: Optional[List[Any]], indent: str = "  ") -> str:
    if not items:
        return f"{indent}(none)"
    out = []
    for x in items:
        s = str(x).strip()
        if s:
            out.append(f"{indent}- {s}")
    return "\n".join(out) if out else f"{indent}(none)"


def render_actor_recovery_block(payload: Dict[str, Any]) -> str:
    """Actor-side recovery block, appended to the decomposer prompt.

    ``payload`` is ``InterventionDecision.params`` from an ``actor_redo``
    decision. Fields: subgoal_id, root_cause_summary,
    memory_canonical_approach, divergence_description, missing_or_wrong
    (the last three may be empty).
    """
    rcs = (payload.get("root_cause_summary") or "").strip()
    canonical = (payload.get("memory_canonical_approach") or "").strip()
    divergence = (payload.get("divergence_description") or "").strip()
    missing = payload.get("missing_or_wrong") or []

    lines: List[str] = []
    lines.append("")
    lines.append("[Recovery context — previous attempts on this subgoal failed]")
    lines.append("")
    lines.append("The agent-loop failure analyser reviewed your prior steps "
                 "on this subgoal and concluded:")
    lines.append("")
    lines.append("WHAT'S MISSING / WRONG:")
    lines.append(_bullets(missing))
    lines.append("")
    if rcs:
        lines.append("ROOT CAUSE:")
        lines.append(f"  {rcs}")
        lines.append("")
    if canonical:
        lines.append("WHAT MEMORY RECOMMENDS (canonical approach for this subgoal):")
        lines.append(f"  {canonical}")
        lines.append("")
    if divergence:
        lines.append("HOW YOUR PREVIOUS PATH DIVERGED FROM MEMORY:")
        lines.append(f"  {divergence}")
        lines.append("")

    # Reasoning scaffold: reason about the diagnosis before acting.
    lines.append("INSTRUCTIONS FOR THIS TURN:")
    lines.append("Before emitting your Action, compose your Thought so it "
                 "EXPLICITLY:")
    lines.append("  (1) acknowledges the root cause above in 1 sentence,")
    if canonical:
        lines.append("  (2) cites which part of the memory canonical approach "
                     "you will follow on THIS turn,")
    else:
        lines.append("  (2) names the specific UI element you'll target this "
                     "turn (no memory canonical was available),")
    lines.append("  (3) describes the concrete UI element you will target — "
                 "anchored to what is visible in the screenshot.")
    lines.append("")
    lines.append("HARD CONSTRAINTS:")
    lines.append("  • Do NOT repeat any action signature listed in your "
                 "recent attempts that already failed.")
    lines.append("  • Do NOT skip the memory-recommended step above; if you "
                 "believe it doesn't apply, your Thought must explicitly say "
                 "why and propose the alternative.")
    lines.append("  • Your Action must be one of the allowed signatures from "
                 "the regular action catalogue.")
    return "\n".join(lines)


def render_planner_recovery_block(payload: Dict[str, Any],
                                  focus_id: Optional[str] = None) -> str:
    """Planner-side recovery block for the force_replan prompt.

    ``payload`` is ``InterventionDecision.params`` from a
    ``planner_replan`` / ``fallback_planner`` decision; the full
    diagnosis dict (FailureAttribution.to_dict()) is under
    ``payload['diagnosis']``. Replaces the v1 ``[Stuck diagnosis ...]``
    block.
    """
    diag = payload.get("diagnosis") or {}
    rcs = (diag.get("root_cause_summary") or "").strip()
    role = (diag.get("attributed_to_role") or "unclear").strip()
    cat = (diag.get("category") or "unclear").strip()
    sub = diag.get("tactical_subkind")
    canonical = ((diag.get("memory_recall") or {})
                 .get("canonical_approach") or "").strip()
    divergence = ((diag.get("memory_critique") or {})
                  .get("divergence_description") or "").strip()
    missing = diag.get("missing_or_wrong") or []
    fallback_from = (payload.get("fallback_from")
                     or diag.get("fallback_from"))

    lines: List[str] = []
    lines.append("")
    header = "[Failure analysis for force_replan"
    if focus_id:
        header += f" — focus_outcome={focus_id}"
    if fallback_from:
        header += f"; safety-fallback from {fallback_from}"
    header += "]"
    lines.append(header)
    lines.append("")
    lines.append(f"ATTRIBUTION:")
    cat_tag = cat + (f"/{sub}" if sub else "")
    lines.append(f"  role={role}  category={cat_tag}  "
                 f"confidence={diag.get('confidence','low')}")
    lines.append("")
    lines.append("WHAT'S MISSING / WRONG:")
    lines.append(_bullets(missing))
    lines.append("")
    if rcs:
        lines.append("ROOT CAUSE:")
        lines.append(f"  {rcs}")
        lines.append("")
    if canonical:
        lines.append("WHAT MEMORY RECOMMENDS:")
        lines.append(f"  {canonical}")
        lines.append("")
    if divergence:
        lines.append("HOW THE PREVIOUS PLAN DIVERGED FROM MEMORY:")
        lines.append(f"  {divergence}")
        lines.append("")

    lines.append("INSTRUCTIONS FOR THIS REPLAN:")
    lines.append("Before revising the plan, briefly reflect on the analysis "
                 "above. Your replan reasoning must EXPLICITLY:")
    lines.append("  (1) state whether the previous plan should be patched "
                 "(tactical) or replaced (strategic) and WHY,")
    if canonical:
        lines.append("  (2) identify the FIRST step of the memory canonical "
                     "approach that the previous plan missed or "
                     "mis-executed,")
    else:
        lines.append("  (2) name a fresh approach for the next subgoal — "
                     "different from what was already tried,")
    lines.append("  (3) ensure the next subgoal you emit attacks the root "
                 "cause above — not a downstream symptom.")
    lines.append("")
    lines.append("HARD CONSTRAINTS:")
    lines.append("  • Do NOT re-issue a subgoal text that matches one already "
                 "in your strategy history with no observable progress.")
    if role in ("actor", "env", "verifier"):
        lines.append(f"  • The previous failure was attributed to "
                     f"{role.upper()} — not to your plan choice. Consider "
                     "whether keeping the SAME plan but giving the actor "
                     "different sub-instructions / waiting for env / "
                     "accepting verifier evidence would resolve it.")
    return "\n".join(lines)

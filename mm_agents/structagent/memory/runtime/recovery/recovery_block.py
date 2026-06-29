"""Render router-decision payloads as text blocks for prompt injection.

When the failure-attribution router decides ``actor_redo`` /
``planner_replan`` / ``fallback_planner``, the corresponding agent
(actor or planner) needs to SEE the diagnosis content in its prompt.
A naive "here is the JSON" injection is brittle — the agent might
ignore it. We wrap the prose into an explicit REASONING SCAFFOLD that
forces the model to:

  1. acknowledge what went wrong
  2. cite which memory recommendation it will follow
  3. name a concrete different next step

The scaffold lives inside the prompt block itself (so the model reads
it as part of its instructions, not as fluff), and the block always
opens with a clear "[Recovery context]" header so it's visually
distinct from the rest of the prompt.

Two renderers exposed:

  render_actor_recovery_block(payload) → str
      For the actor's decomposer prompt. Gives 4 prose fields
      (root_cause / memory_canonical / divergence / missing_or_wrong)
      and demands the next Thought reference each before emitting an
      Action. Crucially OMITS any coord-level hint — the actor's own
      grounding handles HOW.

  render_planner_recovery_block(payload) → str
      For the planner's force_replan prompt. Includes attributed role
      + category as additional context (so planner knows whether the
      previous loop was its own bad plan vs an actor/env/verifier
      problem) and demands a revised plan that addresses the root
      cause.
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
    """Render an actor-side recovery block.

    ``payload`` is ``InterventionDecision.params`` from a router
    decision of kind ``actor_redo``. Required fields:
      - subgoal_id (str)
      - root_cause_summary (str)
      - memory_canonical_approach (str, may be empty)
      - divergence_description (str, may be empty)
      - missing_or_wrong (list[str], may be empty)

    The block is meant to be APPENDED to the actor's decomposer
    system prompt right before the screenshot/user message.
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

    # Reasoning scaffold — explicit invitation to reason about the
    # diagnose before composing the action.
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
    """Render a planner-side recovery block.

    ``payload`` is ``InterventionDecision.params`` from a router
    decision of kind ``planner_replan`` or ``fallback_planner``. The
    full diagnosis dict lives under ``payload['diagnosis']`` (a
    FailureAttribution.to_dict() output).

    Block is injected into the planner's force_replan prompt where the
    existing v1 ``[Stuck diagnosis ...]`` block went.
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

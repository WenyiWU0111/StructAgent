"""Render L2 (task-level) + L1 (subgoal-level) alternatives blocks for
the planner's force_replan prompt.

DIFFERENCE FROM ``planner_prompt_injection.py``:

  ``planner_prompt_injection.render_for_planner_initial`` runs at TASK
  START and surfaces ONE best-match skill ("here's the canonical plan
  for this task class"). It informs the agent's INITIAL strategy.

  ``replan_memory_injection.render_l2_alternatives_for_replan`` runs at
  FORCE_REPLAN and surfaces TOP-K ALTERNATIVE skills ("here are other
  skills the corpus has for this task class — different strategies
  you could switch to"). It informs the agent's STRATEGY SWITCH.

  Similarly for L1: the initial-plan L1 hook is per-subgoal at the
  actor (one best-match action). The replan L1 hook is at force_replan
  and surfaces ALTERNATIVE actions for the currently-failing subgoal.

Both retrieve from the SAME FAISS indexes — only the rendering style
+ caller intent differ. We render the alternatives compactly so they
fit in the force_replan prompt alongside diagnose + verifier traces
without blowing the context window.

Sibling-block design (not category-gated, not filter-dependent):
- L1 query uses ``current_subgoal`` (stable, not diagnose's hint —
  diagnose may be off without corpus context, so we don't drive
  retrieval through its hint).
- L2 query uses ``task_text`` (same as initial-plan key).
- No filter on currently-active strategy / tried variants in this
  first cut; planner LLM sees diagnose + L1 + L2 in distinct blocks
  and decides.

Failure-safe: any internal exception → returns ("", meta_with_ERROR_mode)
so memory injection NEVER breaks the force_replan path.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

def _empty_meta(layer: str, query_text: str, mode: str = "OOD") -> Dict[str, Any]:
    return {
        "mode": mode,
        "layer": layer,
        "n_hits": 0,
        "hits": [],
        "query_text": (query_text or "")[:300],
    }


# ── L2 alternatives renderer ────────────────────────────────────────────────

def render_l2_alternatives_for_replan(
    task_text: str, top_k: int = 3, domain: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Returns (block, meta). Uses the unified multi-layer (v3/v4) retriever —
    the legacy v2 bank (planner_experience/_l2_skills_v2) is gone, so this is
    the only path (same retriever as the initial plan)."""
    if not task_text or not task_text.strip():
        return "", _empty_meta("L2", task_text or "")
    from ..retrieval.query_multi_layer_memory import render_task_start
    return render_task_start(task_text, domain)


# ── L1 alternatives renderer ────────────────────────────────────────────────

def render_l1_alternatives_for_replan(
    current_subgoal: str, top_k: int = 3, domain: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Returns (block, meta). Uses the unified multi-layer (v3/v4) retriever —
    the legacy v2 bank (planner_experience/_l1_actions_v2) is gone, so this is
    the only path. Query is the failing ``current_subgoal``."""
    if not current_subgoal or not current_subgoal.strip():
        return "", _empty_meta("L1", current_subgoal or "")
    from ..retrieval.query_multi_layer_memory import render_subgoal_transition
    return render_subgoal_transition(current_subgoal, domain)

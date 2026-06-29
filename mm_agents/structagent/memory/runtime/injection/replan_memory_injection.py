"""L2 (task-level) + L1 (subgoal-level) ALTERNATIVES blocks for the
planner's force_replan prompt.

Unlike ``planner_prompt_injection`` (task start, ONE best-match skill to
seed the initial strategy), these run at force_replan and surface TOP-K
ALTERNATIVE skills/actions to switch to. Same FAISS indexes, different
rendering + caller intent; rendered compactly to fit alongside the
diagnosis + verifier traces.

L1 queries ``current_subgoal`` (stable) rather than the diagnosis hint,
which may be off without corpus context. L2 queries ``task_text`` (same
key as the initial plan). No filter on tried variants yet — the planner
sees diagnosis + L1 + L2 as distinct blocks and decides.

Failure-safe: any exception → ("", meta with ERROR mode); never breaks
force_replan.
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

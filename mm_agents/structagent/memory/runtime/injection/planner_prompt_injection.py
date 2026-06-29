"""Render L2 (task-level) and L1 (subgoal-level) experience-memory
blocks for injection into the planner / actor prompts.

Two entry points return ``(block_str, meta)``; the block is "" on OOD or
error so memory injection never breaks the agent loop. ``meta`` carries
mode (STRONG/MID/WEAK_PITFALLS_ONLY/OOD/ERROR), layer, hits, query_text.

Mirrors verifier_audit/prompt_injection.py's mode pattern.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def render_for_planner_initial(task_text: str,
                                 domain: str = "",
                                 ) -> Tuple[str, Dict[str, Any]]:
    """Task-start entry. Returns (block, meta); block "" on OOD/error.

    Routes to the multi-layer retriever (L2 plans + L3a cluster patterns
    + L3c domain rule sheet). ``domain`` is required for the L3c lookup.
    """
    from mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \
        import render_task_start
    return render_task_start(task_text, domain)


def render_for_actor_subgoal(subgoal: str,
                              task_text: str = "",
                              domain: str = "",
                              ) -> Tuple[str, Dict[str, Any]]:
    """Subgoal-transition entry. Returns (block, meta); block "" on OOD.

    Routes to the multi-layer retriever (per-domain L1 actions + L3a rule
    index). ``domain`` is required to pick the per-domain L1 index.
    """
    from mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \
        import render_subgoal_transition
    return render_subgoal_transition(
        subgoal, domain, task_instruction=task_text)


# ── Sidecar dump helper ─────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, max_len: int = 40) -> str:
    s = _SLUG_RE.sub("_", (text or "").lower()).strip("_")
    return s[:max_len] or "untitled"


def format_log_summary(meta: Dict[str, Any], top_k: int = 3) -> str:
    """One-line + indented top-K summary for the caller to log (kept
    logger-free). Handles both v2 and v3-shaped meta.
    """
    if meta.get("memory_version") in ("v3", "v4"):
        # v3 meta has per-tier hits; delegate to its formatter.
        from mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \
            import format_log_summary_v3
        return format_log_summary_v3(meta, top_k=top_k)

    layer = meta.get("layer", "?")
    mode = meta.get("mode", "?")
    hits = meta.get("hits", [])
    n = len(hits)
    head = f"layer={layer} mode={mode} n_hits={n}"
    if not hits:
        return head
    lines = [head]
    for h in hits[:top_k]:
        lines.append(f"    [{h['rank']}] {h['id']} sim={h['score']:.3f}")
    return "\n  ".join(lines)


def dump_memory_debug(results_dir: Optional[Any],
                       stage_tag: str,
                       block: str,
                       meta: Dict[str, Any],
                       *,
                       subgoal: Optional[str] = None) -> Optional[Path]:
    """Dump the rendered block + retrieval meta to a sidecar text file
    under ``<results_dir>/planner_memory_debug/`` for post-hoc forensics.
    Returns the path, or None if skipped (no results_dir / write error).

    Filename: ``<stage_tag>[_<subgoal_slug>].txt``
    """
    if results_dir is None:
        return None
    try:
        out_dir = Path(results_dir) / "planner_memory_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = stage_tag
        if subgoal:
            name = f"{stage_tag}_{_slugify(subgoal)}"
        path = out_dir / f"{name}.txt"
        # On filename collision (same stage + subgoal fires twice),
        # suffix with a counter.
        if path.exists():
            i = 2
            while (out_dir / f"{name}__{i}.txt").exists():
                i += 1
            path = out_dir / f"{name}__{i}.txt"
        lines = [
            "# planner memory debug",
            f"# written_at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            f"stage: {stage_tag}",
            f"query: {meta.get('query_text', '')[:300]}",
            f"layer: {meta.get('layer', '?')}",
            f"mode: {meta.get('mode', '?')}",
            f"n_hits: {meta.get('n_hits', 0)}",
            "hits:",
        ]
        for h in meta.get("hits", []):
            lines.append(
                f"  - rank={h['rank']} id={h['id']} "
                f"score={h['score']:.4f} source={h.get('source', '?')}")
        lines.append("")
        lines.append("=== BLOCK INJECTED ===")
        lines.append(block if block.strip() else "(none — OOD / empty)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path
    except Exception:
        return None

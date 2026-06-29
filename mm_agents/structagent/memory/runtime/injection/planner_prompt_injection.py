"""Render L2 (task-level) and L1 (subgoal-level) experience-memory
blocks for injection into the Causal-Agent planner / actor prompts.

Mirrors verifier_audit/prompt_injection.py's STRONG / MID / WEAK_PITFALLS_ONLY
/ OOD pattern.

Two public entry points:

  render_for_planner_initial(task_text) -> (block_str, meta)
      Called by _pa_build_initial_plan_prompt with the augmented task
      instruction. Returns (markdown block to splice, debug meta dict).
      Block is "" and meta has "OOD" mode when no hit / on error.

  render_for_actor_subgoal(subgoal, task_text) -> (block_str, meta)
      Called by _pa_call_actor_points before passing the decomposer
      template to the model. Returns (markdown block, debug meta).

Helper:
  dump_memory_debug(results_dir, stage_tag, block, meta, *,
                     subgoal=None) -> Path | None
      Writes the rendered block + retrieval metadata to a sidecar file
      under ``<results_dir>/planner_memory_debug/`` for post-hoc
      forensics. Skipped silently if results_dir is None or block empty.

Failure-safe: any internal exception → returns ("", meta_with_mode=ERROR)
so memory injection NEVER breaks the agent loop.

meta shape:
  {
    "mode": "STRONG" | "MID" | "WEAK_PITFALLS_ONLY" | "OOD" | "ERROR",
    "layer": "L2" | "L1",
    "n_hits": int,
    "hits": [
      {"rank": int, "id": str, "score": float, "source": str},
      ...
    ],
    "query_text": str,
  }
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def render_for_planner_initial(task_text: str,
                                 domain: str = "",
                                 ) -> Tuple[str, Dict[str, Any]]:
    """Top entry — returns (block, meta). Block is "" on OOD/error.

    Routes to the unified multi-layer (v3/v4) retriever (L2 plans + L3a
    cluster patterns + L3c domain rule sheet) and uses that block +
    v3-shaped meta. The legacy v2 path has been removed.

    ``domain`` is REQUIRED to drive L3c lookup.
    """
    from mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \
        import render_task_start
    return render_task_start(task_text, domain)


def render_for_actor_subgoal(subgoal: str,
                              task_text: str = "",
                              domain: str = "",
                              ) -> Tuple[str, Dict[str, Any]]:
    """Subgoal-transition entry. Returns (block, meta). Block "" on OOD.

    Routes to the unified multi-layer (v3/v4) retriever (per-domain L1
    actions + L3a rule index) and uses that block + v3-shaped meta. The
    legacy v2 path has been removed.

    ``domain`` is REQUIRED to pick the per-domain L1 index.
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
    """One-line + indented top-K summary suitable for _logger.info().
    Caller does the actual log call to keep this module logger-free.

    Handles both v2 meta (``layer``/``mode``/``hits``) and v3 meta
    (``layer``/``mode``/``l2_hits``/``l3a_cluster_hits``/...).
    """
    if meta.get("memory_version") in ("v3", "v4"):
        # Delegate to v3-shaped formatter so the multi-layer hit summary
        # gets the right per-tier rendering.
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
    """Write the rendered block + retrieval metadata to a sidecar file
    under ``<results_dir>/planner_memory_debug/``. Returns the written
    path, or None if skipped (no results_dir / write error).

    File contents (text, human-readable):
      # planner memory debug
      stage: <stage_tag>
      query: <truncated query text>
      mode: <STRONG | MID | WEAK_PITFALLS_ONLY | OOD | ERROR>
      n_hits: <int>
      hits:
        - rank=1 id=... score=... source=...
        - ...

      === BLOCK INJECTED ===
      <full rendered block, or "(none — OOD/empty)">

    File naming:
      <stage_tag>[_<subgoal_slug>].txt
      e.g. "init_l2.txt" or "actor_l1_subgoal_open_chrome_settings.txt"
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
        # Append-safe: if same filename collides (rare — same stage +
        # same subgoal slug fires twice), suffix with a counter.
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

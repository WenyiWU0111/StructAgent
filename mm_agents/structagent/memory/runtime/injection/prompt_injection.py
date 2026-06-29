"""Render a verifier-experience-memory block to inject into the
init_ledger LLM prompt.

Public entrypoint:
  render_for_init_ledger(task_text: str) -> str
      Returns the markdown-flavored text block that init_ledger's
      prompt builder appends to ``text_parts``. Returns "" when no
      recipe match clears the WEAK threshold (caller should not
      append empty strings).

Behavior driven by query() mode:

  STRONG  (score >= 0.7):
      Header + full content of every hit with score >= 0.7
      (dimensions + pitfalls). LLM told to use as a high-confidence
      checklist.

  MID  (score 0.5-0.7):
      Header + full content of hits with score >= 0.5
      (dimensions + pitfalls). LLM told to verify applicability.

  WEAK_PITFALLS_ONLY  (score 0.35-0.5):
      Header + ONLY the pitfalls of the top-1 hit. Dimensions are
      suppressed. LLM told to self-filter pitfalls.

  OOD  (< 0.35):
      Returns "" — no injection. init_ledger behaves as before.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ..retrieval.query_recipe import query, QueryHit, RECIPES_DIR


def _format_dimension(dim: Dict[str, Any], idx: int) -> List[str]:
    lines = []
    name = dim.get("dimension", "?")
    importance = dim.get("importance", "?")
    coverage = dim.get("historical_coverage", "?")
    desc = (dim.get("description") or "").strip()
    lines.append(f"  [{idx}] {name}  ({importance}, historical_coverage {coverage})")
    if desc:
        lines.append(f"       {desc}")
    for opt in dim.get("primitive_options") or []:
        kind = opt.get("kind", "?")
        shape = (opt.get("shape_hint") or "").strip()
        lines.append(f"         · kind={kind}")
        if shape:
            lines.append(f"           shape: {shape[:300]}")
        for slot in opt.get("slot_derivation") or []:
            sname = slot.get("slot", "?")
            stype = slot.get("type", "?")
            sderiv = slot.get("derivation", "?")
            mark = "  [MIND-READ RISK]" if sderiv == "state_observed" else ""
            notes = (slot.get("notes") or "").strip()
            lines.append(f"           slot {sname}: type={stype}, derived={sderiv}{mark}")
            if notes:
                lines.append(f"             ({notes[:150]})")
    return lines


def _format_pitfalls(pitfalls: List[str], bullet: str = "- ") -> List[str]:
    return [f"  {bullet}{p}" for p in pitfalls if p]


def _load_recipe(recipe_path: str) -> Dict[str, Any]:
    p = RECIPES_DIR / recipe_path
    return json.load(open(p))


# ── Mode headers ────────────────────────────────────────────────────────────
_STRONG_HEADER = (
    "================================================================\n"
    "RETRIEVED VERIFIER EXPERIENCE MEMORY  (HIGH CONFIDENCE)\n"
    "================================================================\n"
    "The recipe(s) below come from past SUCCESSFUL tasks that closely\n"
    "match yours (semantic similarity >= 0.70). Treat the dimensions as\n"
    "a high-confidence checklist of what to verify, and treat the\n"
    "pitfalls as high-priority gotchas to avoid.\n"
)

_MID_HEADER = (
    "================================================================\n"
    "RETRIEVED VERIFIER EXPERIENCE MEMORY  (MEDIUM CONFIDENCE)\n"
    "================================================================\n"
    "The recipe(s) below match your task SEMANTICALLY but only with\n"
    "moderate confidence (semantic similarity 0.50-0.70). Verify each\n"
    "dimension actually applies to YOUR specific task before adopting\n"
    "it. The pitfalls are still likely relevant.\n"
)

_WEAK_HEADER = (
    "================================================================\n"
    "RETRIEVED VERIFIER PITFALL ADVISORY  (LOW CONFIDENCE)\n"
    "================================================================\n"
    "A loosely related recipe was found (semantic similarity 0.35-0.50).\n"
    "Its DIMENSIONS are NOT injected because they are likely task-\n"
    "specific to a different intent class. The pitfalls below come from\n"
    "that recipe — apply ONLY the ones that obviously carry over to your\n"
    "task; ignore the rest.\n"
)


def _render_full_hit(hit: QueryHit) -> str:
    """Render header + full content (dimensions + pitfalls) for one hit."""
    d = _load_recipe(hit.recipe_path)
    r = d.get("recipe") or {}
    n_assigned = d.get("n_assigned_tasks", 1)

    out = []
    out.append(f"─── Recipe: {hit.recipe_intent_id}  (score={hit.best_score:.2f}) ───")
    when = (r.get("when_to_use") or "").strip()
    if when:
        out.append(f"when_to_use: {when}")
    out.append(f"historical: succeeded in {n_assigned} past task(s)")
    out.append("")
    dims = r.get("verification_dimensions") or []
    if dims:
        out.append("VERIFICATION DIMENSIONS:")
        for i, dim in enumerate(dims, 1):
            out.extend(_format_dimension(dim, i))
        out.append("")
    pitfalls = r.get("known_pitfalls") or []
    if pitfalls:
        out.append("KNOWN PITFALLS:")
        out.extend(_format_pitfalls(pitfalls))
    out.append("")
    return "\n".join(out)


def _render_pitfalls_only(hit: QueryHit) -> str:
    """Render header + ONLY pitfalls for WEAK mode."""
    d = _load_recipe(hit.recipe_path)
    r = d.get("recipe") or {}
    pitfalls = r.get("known_pitfalls") or []
    out = []
    out.append(f"(source recipe: {hit.recipe_intent_id}, score={hit.best_score:.2f})")
    out.append("")
    if pitfalls:
        out.append("KNOWN PITFALLS (apply only those obviously relevant to your task):")
        out.extend(_format_pitfalls(pitfalls))
    out.append("")
    return "\n".join(out)


def _filter_filegrep_hits(hits, domain):
    """For GUI-only domains (chrome/gimp/thunderbird, ENABLE_DOMAIN_CODING_GATE
    on) drop retrieved verify-spec recipes that rely on file_grep /
    shell_command. Those domains verify on LIVE app state via a11y/url — the
    on-disk config file flickers / is overwritten, which is exactly the
    transient-file_grep false-PASS that let chrome bookmark tasks false-DONE.
    The bank is domain-agnostic, so a chrome task can match a file_grep recipe
    mined from os/vs_code; dropping it reinforces domain/verifier/<d>.md's
    a11y-default. The init_ledger LLM can still author shell_command from its
    domain knowledge if a task genuinely produces a file. Gate off / non-GUI-
    only → unchanged."""
    try:
        from mm_agents.structagent.config import CAConfig
        from mm_agents.structagent.domain import is_gui_only
        if not (CAConfig.from_env().domain_coding_gate and is_gui_only(domain)):
            return hits
    except Exception:
        return hits
    kept = []
    for h in hits:
        try:
            rec = _load_recipe(h.recipe_path)
            blob = json.dumps(rec, ensure_ascii=False).lower() if rec else ""
        except Exception:
            blob = ""
        if "file_grep" in blob or "shell_command" in blob:
            continue
        kept.append(h)
    return kept


def render_for_init_ledger(task_text: str, domain: str = "") -> str:
    """Build the verifier-experience-memory block for init_ledger's
    prompt. Returns "" when no recipe match clears the WEAK threshold."""
    if not task_text or not task_text.strip():
        return ""
    try:
        hits, mode = query(task_text)
    except Exception as e:
        # Retrieval failure must never break init_ledger
        return ""

    hits = _filter_filegrep_hits(hits, domain)
    if mode == "OOD" or not hits:
        return ""

    if mode == "STRONG":
        body = "\n".join(_render_full_hit(h) for h in hits)
        return _STRONG_HEADER + "\n" + body
    if mode == "MID":
        body = "\n".join(_render_full_hit(h) for h in hits)
        return _MID_HEADER + "\n" + body
    if mode == "WEAK_PITFALLS_ONLY":
        body = _render_pitfalls_only(hits[0])
        return _WEAK_HEADER + "\n" + body
    return ""

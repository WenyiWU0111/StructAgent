"""Retrieve boundary CHECK RECIPES (verifier-side memory for the boundary verifier).

The bank is built by ``mm_agents.structagent.memory.offline.build_check_recipes`` from the
Verifier intent recipes: per milestone class, the ground-truth probe + strict criterion
+ traps, keyed by ``when_to_use``. The boundary verifier retrieves the top-k for the
current milestone and folds them into its judge-or-probe prompt as guidance —
"how this class of milestone has been verified before".

  retrieve_check_recipes(query_text, domain="", k=3) -> List[dict]
  render_check_recipe_block(recipes) -> str   (prompt block, or "")

Lazy process-wide singleton (SentenceTransformer + FAISS load paid once). Missing
bank → returns [] / "" silently (feature is opt-in; must never break verification).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from mm_agents.structagent._paths import REPO_ROOT

logger = logging.getLogger(__name__)

INDEX_DIR = REPO_ROOT / "results" / "verifier_memory" / "_check_recipes"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_model = None
_index = None
_meta: Optional[List[Dict[str, Any]]] = None
_loaded = False


def _lazy_load() -> bool:
    """Load model + index once. Returns False if the bank is absent/unreadable."""
    global _model, _index, _meta, _loaded
    if _loaded:
        return _index is not None
    _loaded = True
    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        idx_path = INDEX_DIR / "index.faiss"
        meta_path = INDEX_DIR / "metadata.jsonl"
        if not idx_path.exists() or not meta_path.exists():
            logger.info("[check_recipes] bank not found at %s", INDEX_DIR)
            return False
        _index = faiss.read_index(str(idx_path))
        _meta = [json.loads(l) for l in meta_path.open() if l.strip()]
        _model = SentenceTransformer(EMBED_MODEL)
        logger.info("[check_recipes] loaded %d check recipes", len(_meta))
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("[check_recipes] load failed: %s", e)
        _index = None
        return False


def retrieve_check_recipes(query_text: str, domain: str = "", k: int = 3) -> List[Dict[str, Any]]:
    """Top-k check recipes for ``query_text`` (a milestone / subgoal description).
    Same-domain recipes are boosted ahead of cross-domain ones at equal rank."""
    if not query_text or not query_text.strip() or not _lazy_load():
        return []
    try:
        import numpy as np
        q = _model.encode([query_text], normalize_embeddings=True).astype(np.float32)
        kk = min(max(k * 3, k), len(_meta))
        scores, idxs = _index.search(q, kk)
        hits = [dict(_meta[int(i)], _score=float(scores[0][r]))
                for r, i in enumerate(idxs[0]) if i >= 0]
    except Exception as e:  # noqa: BLE001
        logger.warning("[check_recipes] search failed: %s", e)
        return []
    if domain:
        hits.sort(key=lambda h: (h.get("domain") != domain, -h.get("_score", 0.0)))
    return hits[:k]


def render_check_recipe_block(recipes: List[Dict[str, Any]]) -> str:
    """Render retrieved check recipes as a prompt block (guidance, not a spec)."""
    if not recipes:
        return ""
    lines = ["[How similar milestones have been verified before — GUIDANCE for "
             "choosing your probe + success criterion; not a spec to copy]"]
    for r in recipes:
        lines.append(f"• {r.get('when_to_use', '')}")
        for c in (r.get("checks") or [])[:3]:
            lines.append(f"    verify: {c.get('verify', '')}")
            lines.append(f"      reliable probe: {c.get('probe', '')} — {c.get('why_reliable', '')}")
            lines.append(f"      pass if: {c.get('pass_if', '')}")
            traps = c.get("traps") or []
            if traps:
                lines.append(f"      traps: {'; '.join(traps)}")
    return "\n".join(lines)

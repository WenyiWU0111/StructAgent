"""Query the recipe retrieval index with a task instruction.

Loads the FAISS index from build_recipe_index.py, embeds the task text, finds
top-K nearest anchors, aggregates by recipe (max score per recipe), returns
ranked recipes. API: query(task_text, top_k) -> List[QueryHit]; print_hits().

CLI:
  python \\
      -m mm_agents.structagent.memory.runtime.retrieval.query_recipe \\
      "Find Dota 2 game and add all DLC to cart."
"""

from __future__ import annotations
from mm_agents.structagent._paths import REPO_ROOT

import argparse
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
from sentence_transformers import SentenceTransformer


LEDGER_DIR = REPO_ROOT / "results/successful_ledgers"
INDEX_DIR = LEDGER_DIR / "_retrieval"
RECIPES_DIR = LEDGER_DIR / "_intent_recipes_v2"


@dataclass
class QueryHit:
    rank: int
    recipe_intent_id: str
    best_score: float
    best_anchor_source: str
    best_anchor_text: str
    best_anchor_task_id: Optional[str]
    n_matched_anchors: int
    recipe_path: str
    when_to_use: str
    n_dimensions: int
    n_pitfalls: int


_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_vector_map: Optional[List[Dict[str, Any]]] = None


def _lazy_load() -> None:
    global _model, _index, _vector_map
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if _index is None:
        _index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
        _vector_map = json.loads((INDEX_DIR / "vector_map.json").read_text())


# Layered thresholds by risk profile: dimensions tell the LLM what to verify
# (risky if wrong); pitfalls are cross-task wisdom (low risk even on a loose
# match — LLM can self-filter).
STRONG_THRESHOLD = 0.7   # full recipe, ≥0.7 hits only
MID_THRESHOLD    = 0.5   # full recipe, ≥0.5 hits only
WEAK_THRESHOLD   = 0.35  # pitfalls only, no dimensions, top-1 hit
# < WEAK_THRESHOLD → OOD, nothing injected


def query(task_text: str, top_k: int = 5, candidate_pool: int = 30,
          strong_threshold: float = STRONG_THRESHOLD,
          mid_threshold:    float = MID_THRESHOLD,
          weak_threshold:   float = WEAK_THRESHOLD,
          ) -> Tuple[List[QueryHit], str]:
    """Embed task_text, aggregate top anchors by recipe, apply the layered
    threshold filter, return (hits, mode).

    mode (caller honors it when injecting):
      - "STRONG"             top ≥ strong; keep hits ≥0.7, full recipe
      - "MID"                top in [mid, strong); keep hits ≥0.5, full recipe
      - "WEAK_PITFALLS_ONLY" top in [weak, mid); top-1 only — inject pitfalls,
                             NOT dimensions (weak intent match, but pitfalls
                             are still useful cross-task wisdom)
      - "OOD"                top < weak; empty hits, skip injection

    Each QueryHit always carries the full recipe; the injection layer clips it.
    """
    _lazy_load()
    q_vec = _model.encode(
        [task_text], convert_to_numpy=True, normalize_embeddings=True
    ).astype("float32")
    scores, idxs = _index.search(q_vec, candidate_pool)
    scores = scores[0]
    idxs = idxs[0]

    best_per_recipe: Dict[str, Dict[str, Any]] = {}
    for score, vec_idx in zip(scores, idxs):
        if vec_idx < 0:
            continue
        meta = _vector_map[vec_idx]
        intent = meta["intent_id"]
        cur = best_per_recipe.get(intent)
        if cur is None or score > cur["score"]:
            best_per_recipe[intent] = {
                "score": float(score),
                "anchor": meta,
                "n_matched_anchors": (cur["n_matched_anchors"] + 1) if cur else 1,
            }
        else:
            cur["n_matched_anchors"] += 1

    ranked = sorted(best_per_recipe.items(), key=lambda kv: -kv[1]["score"])

    if not ranked:
        return [], "OOD"

    top = ranked[0][1]["score"]
    if top >= strong_threshold:
        mode = "STRONG"
        kept = [(i, info) for i, info in ranked if info["score"] >= strong_threshold]
    elif top >= mid_threshold:
        mode = "MID"
        kept = [(i, info) for i, info in ranked if info["score"] >= mid_threshold]
    elif top >= weak_threshold:
        mode = "WEAK_PITFALLS_ONLY"
        # Only top-1 — don't pile up multiple weakly-related pitfall sets
        kept = ranked[:1]
    else:
        return [], "OOD"

    kept = kept[:top_k]

    hits: List[QueryHit] = []
    for rank, (intent, info) in enumerate(kept, 1):
        recipe_path = info["anchor"]["recipe_path"]
        d = json.loads((RECIPES_DIR / recipe_path).read_text())
        r = d.get("recipe") or {}
        hits.append(QueryHit(
            rank=rank,
            recipe_intent_id=intent,
            best_score=info["score"],
            best_anchor_source=info["anchor"]["source_type"],
            best_anchor_text=info["anchor"]["text"][:200],
            best_anchor_task_id=info["anchor"].get("source_task_id"),
            n_matched_anchors=info["n_matched_anchors"],
            recipe_path=recipe_path,
            when_to_use=r.get("when_to_use", "")[:200],
            n_dimensions=len(r.get("verification_dimensions") or []),
            n_pitfalls=len(r.get("known_pitfalls") or []),
        ))
    return hits, mode


def print_hits(query_text: str, hits: List[QueryHit], mode: str) -> None:
    print(f"\nQuery: «{query_text[:140]}»")
    print(f"Mode: {mode} — {len(hits)} recipe(s)")
    if mode == "WEAK_PITFALLS_ONLY":
        print(f"  (only `known_pitfalls` will be injected; dimensions dropped)")
    elif mode == "OOD":
        print(f"  (no injection; init_ledger falls back to LLM-only)")
    print("=" * 90)
    for h in hits:
        print(f"\n  #{h.rank}  score={h.best_score:.3f}  matched_anchors={h.n_matched_anchors}")
        print(f"      intent_id:        {h.recipe_intent_id}")
        print(f"      when_to_use:      {h.when_to_use}")
        print(f"      best_anchor_via:  {h.best_anchor_source}"
              + (f" (task {h.best_anchor_task_id[:8]})" if h.best_anchor_task_id else ""))
        print(f"      anchor_text:      «{h.best_anchor_text[:140]}»")
        print(f"      n_dims/pitfalls:  {h.n_dimensions} / {h.n_pitfalls}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("task_text", type=str, help="New task instruction to query")
    p.add_argument("-k", "--top-k", type=int, default=5)
    args = p.parse_args()
    hits, mode = query(args.task_text, top_k=args.top_k)
    print_hits(args.task_text, hits, mode)


if __name__ == "__main__":
    main()

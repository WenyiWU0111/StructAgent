"""MultiLayerMemoryRetriever v3 + natural-language prompt renderer.

Two-stage retrieval matching planner / actor decision points:

  retrieve_for_task_start(instruction, domain)
      → top-K L2 plan templates (similar past tasks + full plans)
      → top-K L3a cluster summaries (patterns across the task class)
      → L3c domain rule sheet (always-true facts for the app)

  retrieve_for_subgoal_transition(subgoal, domain)
      → top-K L1 typical_actions (how this subgoal was done before)
      → top-K L3a rules relevant to this subgoal (gotchas)

The render_* blocks use plain-English prose, no L1/L2/L3 labels (the
model doesn't know that taxonomy).

Env: MEMORY_BANK_VERSION selects the index bank (default v2; v3/v4 here).

Run smoke test:
  PYTHONPATH=. python -m mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \\
      --task "Open the desktop file '8.xlsx' and create a bar chart from columns A-E." \\
      --domain libreoffice_calc
  PYTHONPATH=. python -m mm_agents.structagent.memory.runtime.retrieval.query_multi_layer_memory \\
      --task "..." --domain ... --subgoal "Insert a chart from selected data"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from mm_agents.structagent._paths import REPO_ROOT

logger = logging.getLogger(__name__)

# v3 (current) or v4 (cleaned/abstracted L1+L2 + IDs), per MEMORY_BANK_VERSION.
_BANK_VER = os.environ.get("MEMORY_BANK_VERSION", "v2")
INDEXES_DIR = REPO_ROOT / "results" / "unified_memory" / (
    f"_indexes_{_BANK_VER}" if _BANK_VER in ("v3", "v4") else "_indexes_v3")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Hit dataclass ─────────────────────────────────────────────────────────
@dataclass
class Hit:
    """One retrieval hit."""
    score: float                  # cosine similarity (0..1, unit vectors)
    payload: Dict[str, Any] = field(default_factory=dict)


# ─── Index handle ──────────────────────────────────────────────────────────
class _FaissHandle:
    def __init__(self, dir: Path):
        self.dir = dir
        self.index = faiss.read_index(str(dir / "index.faiss"))
        self.metadata: List[Dict[str, Any]] = []
        with (dir / "metadata.jsonl").open() as f:
            for line in f:
                line = line.strip()
                if line:
                    self.metadata.append(json.loads(line))
        self.meta_info = json.loads((dir / "_meta.json").read_text())

    def search(self, qvec: np.ndarray, k: int) -> List[Hit]:
        if len(self.metadata) == 0:
            return []
        k = min(k, len(self.metadata))
        scores, idxs = self.index.search(qvec.astype(np.float32), k)
        return [Hit(score=float(scores[0][r]),
                    payload=self.metadata[int(idxs[0][r])])
                for r in range(k) if idxs[0][r] >= 0]


# ─── Retriever ─────────────────────────────────────────────────────────────
class MultiLayerMemoryRetriever:
    """Loads L2 / L3a-cluster / L3a-rule / L3c on init; L1 per-domain
    indexes load lazily (most tasks touch one domain)."""

    def __init__(self, embed_model: str = EMBED_MODEL):
        self.model = SentenceTransformer(embed_model)
        self.l2 = _FaissHandle(INDEXES_DIR / "l2")
        self.l3a_cluster = _FaissHandle(INDEXES_DIR / "l3a_cluster")
        self.l3a_rule = _FaissHandle(INDEXES_DIR / "l3a_rule")
        self.l3c = json.loads(
            (INDEXES_DIR / "l3c" / "rules_by_domain.json").read_text())
        # L1 per-domain — lazy
        self._l1_cache: Dict[str, Optional[_FaissHandle]] = {}

    def _embed(self, text: str) -> np.ndarray:
        v = self.model.encode([text], normalize_embeddings=True,
                              convert_to_numpy=True)
        return v

    def _get_l1(self, domain: str) -> Optional[_FaissHandle]:
        if domain in self._l1_cache:
            return self._l1_cache[domain]
        dom_dir = INDEXES_DIR / "l1" / domain
        if not dom_dir.exists():
            self._l1_cache[domain] = None
            return None
        h = _FaissHandle(dom_dir)
        self._l1_cache[domain] = h
        return h

    # ─── Task-start retrieval ──────────────────────────────────────────
    def retrieve_for_task_start(self, task_instruction: str, domain: str,
                                k_l2: int = 3,
                                k_l3a_cluster: int = 2) -> Dict[str, Any]:
        q = self._embed(task_instruction)
        return {
            "task_instruction": task_instruction,
            "domain": domain,
            "l2_plans": self.l2.search(q, k_l2),
            "l3a_clusters": self.l3a_cluster.search(q, k_l3a_cluster),
            "l3c_domain": self.l3c.get(domain) or {},
        }

    # ─── Subgoal-transition retrieval ──────────────────────────────────
    def retrieve_for_subgoal_transition(self, subgoal_text: str, domain: str,
                                        k_l1: int = 3,
                                        k_l3a_rule: int = 3) -> Dict[str, Any]:
        q = self._embed(subgoal_text)
        l1_handle = self._get_l1(domain)
        l1_hits = l1_handle.search(q, k_l1) if l1_handle else []
        l3a_rule_hits = self.l3a_rule.search(q, k_l3a_rule)
        return {
            "subgoal": subgoal_text,
            "domain": domain,
            "l1_actions": l1_hits,
            "l3a_rules": l3a_rule_hits,
        }

    # ─── Prompt rendering (natural-language; no L1/L2/L3 labels) ───────
    @staticmethod
    def render_task_start_block(retrieval: Dict[str, Any]) -> str:
        """Render the task-start retrieval as a prose prompt block."""
        lines = []
        domain = retrieval.get("domain") or "?"
        l2 = retrieval.get("l2_plans") or []
        l3a = retrieval.get("l3a_clusters") or []
        l3c = retrieval.get("l3c_domain") or {}

        if not (l2 or l3a or l3c):
            return ""

        lines.append(
            "You are about to plan how to accomplish this task. "
            "Below is relevant experience from past task attempts.")
        lines.append("")

        if l2:
            lines.append("══ Plans that worked on similar tasks ══")
            for i, hit in enumerate(l2, 1):
                p = hit.payload
                lines.append(f"\n{i}. \"{(p.get('task_class') or p.get('when_to_use') or '')[:200]}\"")
                wtu = (p.get("when_to_use") or "")[:200]
                if wtu:
                    lines.append(f"   When to use this plan: {wtu}")
                plan = p.get("plan_template") or []
                if plan:
                    lines.append("   Plan that worked:")
                    for step in plan[:8]:
                        sg = (step.get("subgoal") or "")[:120]
                        lines.append(f"     - {sg}")
                        for ta in (step.get("typical_actions") or [])[:1]:
                            lines.append(f"         (via {str(ta)[:120]})")
                pits = p.get("attached_pitfalls") or []
                if pits:
                    lines.append("   Common pitfalls:")
                    for pit in pits[:3]:
                        lines.append(f"     - {(pit.get('pitfall') or '')[:160]}")

        if l3a:
            lines.append("\n══ Patterns across this kind of task ══")
            lines.append("(observed across many similar trajectories)\n")
            for hit in l3a:
                p = hit.payload
                lines.append(f"For tasks of class \"{(p.get('cluster_summary') or '')[:200]}\":")
                rules = p.get("common_rules") or []
                for rule in rules[:5]:
                    lines.append(f"  - {(rule.get('rule') or '')[:200]}")
                pits = p.get("shared_pitfalls") or []
                for pit in pits[:3]:
                    lines.append(f"  ⚠ {(pit.get('pitfall') or '')[:200]}")
                lines.append("")

        if l3c:
            lines.append(f"══ What's true in this application ({domain}) ══")
            for inv in (l3c.get("domain_invariants") or [])[:6]:
                lines.append(f"  - {(inv.get('invariant') or '')[:200]}")
            for w in (l3c.get("common_widgets") or [])[:4]:
                purpose = (w.get('purpose') or '')[:120]
                lines.append(f"  - widget '{w.get('widget','')[:60]}': {purpose}")
            for s in (l3c.get("common_shortcuts") or [])[:4]:
                tag = " (this-app-specific)" if s.get("domain_specific") else ""
                purpose = (s.get('purpose') or '')[:120]
                lines.append(f"  - shortcut '{s.get('shortcut','')[:30]}'{tag}: {purpose}")
            for pit in (l3c.get("shared_pitfalls") or [])[:3]:
                lines.append(f"  ⚠ {(pit.get('pitfall') or '')[:200]}")

        return "\n".join(lines)

    @staticmethod
    def render_subgoal_transition_block(retrieval: Dict[str, Any]) -> str:
        """Render the per-subgoal retrieval as a prose prompt block."""
        lines = []
        subgoal = retrieval.get("subgoal") or "?"
        l1 = retrieval.get("l1_actions") or []
        l3a = retrieval.get("l3a_rules") or []

        if not (l1 or l3a):
            return ""

        lines.append(f"You're about to execute this subgoal: \"{subgoal}\"")
        lines.append("")

        if l1:
            lines.append("══ How this subgoal has been accomplished before ══")
            for i, hit in enumerate(l1, 1):
                p = hit.payload
                lines.append(f"\nApproach {i}: {p.get('approach','?')} "
                             f"(action: {p.get('action_id','?')})")
                lines.append(f"  When to use: {(p.get('when_to_use') or '')[:180]}")
                lines.append(f"  Steps:")
                for t in (p.get("typical_actions") or [])[:5]:
                    lines.append(f"    - {str(t)[:200]}")
                wg = p.get("tools_or_widgets_seen") or []
                if wg:
                    lines.append(f"  Widgets used: {', '.join(str(w) for w in wg[:5])}")
                for pit in (p.get("common_pitfalls") or [])[:2]:
                    lines.append(f"  ⚠ {(pit.get('pitfall') or '')[:200]}")

        if l3a:
            lines.append("\n══ Specific rules worth knowing for this kind of action ══")
            for hit in l3a:
                p = hit.payload
                rule_text = (p.get("rule") or "")[:200]
                lines.append(f"  - {rule_text}")
                rationale = (p.get("rationale") or "")[:160]
                if rationale:
                    lines.append(f"      (Why: {rationale})")

        return "\n".join(lines)


# ─── Env switch ─────────────────────────────────────────────────────────────
def get_memory_retriever():
    """The unified multi-layer retriever. MEMORY_BANK_VERSION picks the
    index bank; the legacy v2 module-style retriever is gone."""
    return _get_singleton_retriever()


# ─── Singleton + render wrappers (called from planner / actor hooks) ──────
_RETRIEVER_INSTANCE: Optional[MultiLayerMemoryRetriever] = None


def _get_singleton_retriever() -> MultiLayerMemoryRetriever:
    """Process-wide singleton. SentenceTransformer + FAISS loads take ~3s;
    pay it once per process."""
    global _RETRIEVER_INSTANCE
    if _RETRIEVER_INSTANCE is None:
        _RETRIEVER_INSTANCE = MultiLayerMemoryRetriever()
    return _RETRIEVER_INSTANCE


def _hit_summary_l2(h: Hit) -> Dict[str, Any]:
    p = h.payload
    return {
        "skill_id": p.get("skill_id"),
        "cluster_id": p.get("cluster_id"),
        "score": round(h.score, 4),
        "task_class": (p.get("task_class") or "")[:80],
        "n_assigned": p.get("n_assigned"),
    }


def _hit_summary_l3a_cluster(h: Hit) -> Dict[str, Any]:
    p = h.payload
    return {
        "cluster_id": p.get("cluster_id"),
        "score": round(h.score, 4),
        "cluster_summary": (p.get("cluster_summary") or "")[:100],
        "n_rules": p.get("n_common_rules"),
        "cross_domain": p.get("cross_domain_signal"),
    }


def _hit_summary_l1(h: Hit) -> Dict[str, Any]:
    p = h.payload
    return {
        "action_id": p.get("action_id"),
        "cluster_id": p.get("cluster_id"),
        "approach": p.get("approach"),
        "score": round(h.score, 4),
        "subgoal_pattern": (p.get("subgoal_pattern") or "")[:80],
    }


def _hit_summary_l3a_rule(h: Hit) -> Dict[str, Any]:
    p = h.payload
    return {
        "cluster_id": p.get("cluster_id"),
        "score": round(h.score, 4),
        "rule_excerpt": (p.get("rule") or "")[:120],
        "scope": p.get("scope"),
        "category": p.get("category"),
    }


def render_task_start(task_text: str, domain: str = "",
                         k_l2: int = 3,
                         k_l3a_cluster: int = 2,
                         ) -> Tuple[str, Dict[str, Any]]:
    """Drop-in v3 replacement for ``render_for_planner_initial``.
    Returns (block, meta); meta shape mirrors v2 so format_log_summary +
    dump_memory_debug work for both."""
    if not task_text or not task_text.strip():
        return "", {"layer": "MULTILAYER_TASK_START",
                    "memory_version": "v3",
                    "query_text": task_text or "",
                    "domain": domain,
                    "mode": "EMPTY",
                    "block_chars": 0,
                    "l2_hits": [], "l3a_cluster_hits": [],
                    "l3c_domain_loaded": False,
                    "n_total_hits": 0}
    # MEMORY_TOP_K override: replaces the primary-layer default k (L2 here).
    _env_k = os.environ.get("MEMORY_TOP_K", "").strip()
    if _env_k:
        try:
            k_l2 = int(_env_k)
        except ValueError:
            pass
    try:
        r = _get_singleton_retriever()
        ret = r.retrieve_for_task_start(task_text, domain,
                                        k_l2=k_l2,
                                        k_l3a_cluster=k_l3a_cluster)
        block = r.render_task_start_block(ret)
    except Exception as e:
        return "", {"layer": "MULTILAYER_TASK_START",
                    "memory_version": "v3",
                    "query_text": task_text,
                    "domain": domain,
                    "mode": "ERROR",
                    "error": str(e),
                    "block_chars": 0,
                    "n_total_hits": 0}
    l2_summ = [_hit_summary_l2(h) for h in ret["l2_plans"]]
    l3a_summ = [_hit_summary_l3a_cluster(h) for h in ret["l3a_clusters"]]
    n_total = len(l2_summ) + len(l3a_summ) + (1 if ret["l3c_domain"] else 0)
    meta = {
        "layer": "MULTILAYER_TASK_START",
        "memory_version": "v3",
        "query_text": task_text,
        "domain": domain,
        "mode": "v3_active" if n_total > 0 else "EMPTY",
        "block_chars": len(block),
        "l2_hits": l2_summ,
        "l3a_cluster_hits": l3a_summ,
        "l3c_domain_loaded": bool(ret["l3c_domain"]),
        "n_total_hits": n_total,
    }
    return block, meta


def render_subgoal_transition(subgoal: str, domain: str = "",
                                 k_l1: int = 3,
                                 k_l3a_rule: int = 3,
                                 *,
                                 task_instruction: str = "",
                                 ) -> Tuple[str, Dict[str, Any]]:
    """Drop-in v3 replacement for ``render_for_actor_subgoal``."""
    if not subgoal or not subgoal.strip():
        return "", {"layer": "MULTILAYER_SUBGOAL",
                    "memory_version": "v3",
                    "query_text": subgoal or "",
                    "domain": domain,
                    "mode": "EMPTY",
                    "block_chars": 0,
                    "l1_hits": [], "l3a_rule_hits": [],
                    "n_total_hits": 0}
    # Operator override for retrieval depth (MEMORY_TOP_K env) — same
    # contract as the L2 path: replaces the primary-layer default.
    _env_k = os.environ.get("MEMORY_TOP_K", "").strip()
    if _env_k:
        try:
            k_l1 = int(_env_k)
        except ValueError:
            pass
    try:
        r = _get_singleton_retriever()
        ret = r.retrieve_for_subgoal_transition(subgoal, domain,
                                                k_l1=k_l1,
                                                k_l3a_rule=k_l3a_rule)
        block = r.render_subgoal_transition_block(ret)
    except Exception as e:
        return "", {"layer": "MULTILAYER_SUBGOAL",
                    "memory_version": "v3",
                    "query_text": subgoal,
                    "domain": domain,
                    "mode": "ERROR",
                    "error": str(e),
                    "block_chars": 0,
                    "n_total_hits": 0}
    l1_summ = [_hit_summary_l1(h) for h in ret["l1_actions"]]
    l3a_summ = [_hit_summary_l3a_rule(h) for h in ret["l3a_rules"]]
    n_total = len(l1_summ) + len(l3a_summ)
    meta = {
        "layer": "MULTILAYER_SUBGOAL",
        "memory_version": "v3",
        "query_text": subgoal,
        "domain": domain,
        "mode": "v3_active" if n_total > 0 else "EMPTY",
        "block_chars": len(block),
        "l1_hits": l1_summ,
        "l3a_rule_hits": l3a_summ,
        "n_total_hits": n_total,
    }
    return block, meta


def format_log_summary_v3(meta: Dict[str, Any], top_k: int = 3) -> str:
    """One-line log summary for v3 meta; mirrors v2's format_log_summary."""
    layer = meta.get("layer") or "?"
    mode = meta.get("mode") or "?"
    domain = meta.get("domain") or "-"
    n = meta.get("n_total_hits") or 0
    parts = [f"v3[{layer}] domain={domain} mode={mode} n={n}"]
    if layer == "MULTILAYER_TASK_START":
        l2 = meta.get("l2_hits") or []
        l3a = meta.get("l3a_cluster_hits") or []
        l3c = "yes" if meta.get("l3c_domain_loaded") else "no"
        parts.append(
            f"L2={len(l2)} L3a-cluster={len(l3a)} L3c={l3c}")
        for h in l2[:top_k]:
            parts.append(
                f"  L2 {h.get('skill_id','?')[:30]:>30s} "
                f"score={h.get('score'):.3f}  «{h.get('task_class','')[:50]}»")
        for h in l3a[:top_k]:
            parts.append(
                f"  L3a-c {h.get('cluster_id','?')} "
                f"score={h.get('score'):.3f}  «{h.get('cluster_summary','')[:50]}»")
    elif layer == "MULTILAYER_SUBGOAL":
        l1 = meta.get("l1_hits") or []
        l3a = meta.get("l3a_rule_hits") or []
        parts.append(f"L1={len(l1)} L3a-rule={len(l3a)}")
        for h in l1[:top_k]:
            parts.append(
                f"  L1 {h.get('action_id','?')[:30]:>30s} "
                f"score={h.get('score'):.3f}  approach={h.get('approach')}  "
                f"«{h.get('subgoal_pattern','')[:50]}»")
        for h in l3a[:top_k]:
            parts.append(
                f"  L3a-r {h.get('cluster_id','?')} "
                f"score={h.get('score'):.3f}  «{h.get('rule_excerpt','')[:60]}»")
    return "\n".join(parts)


# ─── CLI / smoke test ───────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, required=True,
                   help="Task instruction (task-start retrieval)")
    p.add_argument("--domain", type=str, required=True,
                   help="Task domain (chrome / libreoffice_calc / ...)")
    p.add_argument("--subgoal", type=str, default=None,
                   help="If given, also do subgoal-transition retrieval")
    p.add_argument("--k-l2", type=int, default=3)
    p.add_argument("--k-l3a-cluster", type=int, default=2)
    p.add_argument("--k-l1", type=int, default=3)
    p.add_argument("--k-l3a-rule", type=int, default=3)
    p.add_argument("--json", action="store_true",
                   help="Dump raw retrieval as JSON instead of rendered prose")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    r = MultiLayerMemoryRetriever()

    task_ret = r.retrieve_for_task_start(args.task, args.domain,
                                         k_l2=args.k_l2,
                                         k_l3a_cluster=args.k_l3a_cluster)
    sub_ret = None
    if args.subgoal:
        sub_ret = r.retrieve_for_subgoal_transition(args.subgoal, args.domain,
                                                    k_l1=args.k_l1,
                                                    k_l3a_rule=args.k_l3a_rule)

    if args.json:
        def hit_to_dict(h: Hit) -> Dict[str, Any]:
            return {"score": h.score, "payload": h.payload}
        dump = {
            "task_start": {
                "task_instruction": task_ret["task_instruction"],
                "domain": task_ret["domain"],
                "l2_plans": [hit_to_dict(h) for h in task_ret["l2_plans"]],
                "l3a_clusters": [hit_to_dict(h)
                                 for h in task_ret["l3a_clusters"]],
                "l3c_domain": task_ret["l3c_domain"],
            },
        }
        if sub_ret:
            dump["subgoal_transition"] = {
                "subgoal": sub_ret["subgoal"],
                "domain": sub_ret["domain"],
                "l1_actions": [hit_to_dict(h)
                               for h in sub_ret["l1_actions"]],
                "l3a_rules": [hit_to_dict(h) for h in sub_ret["l3a_rules"]],
            }
        print(json.dumps(dump, indent=2, ensure_ascii=False))
        return

    print("=" * 78)
    print("TASK-START BLOCK")
    print("=" * 78)
    print(r.render_task_start_block(task_ret))
    if sub_ret:
        print("\n" + "=" * 78)
        print("SUBGOAL-TRANSITION BLOCK")
        print("=" * 78)
        print(r.render_subgoal_transition_block(sub_ret))


if __name__ == "__main__":
    main()

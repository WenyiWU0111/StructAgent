"""Phase 4 — Build FAISS indexes over the v3 multi-layer memory bank.

Produces five retrievable surfaces under ``_indexes_v3/``:

  l1/<domain>/   per-domain typical actions; embed subgoal_pattern;
                 used at subgoal-transition.
  l2/            plan templates; embed when_to_use; used at task-start.
  l3a_cluster/   cluster summaries; embed cluster_summary; task-start.
  l3a_rule/      individual rules; embed "{rule} — {applies_when}";
                 subgoal-transition.
  l3c/           per-domain rule sheets — direct dict lookup, no FAISS
                 (only 7 domain keys).

Embeds with all-MiniLM-L6-v2 → L2-normalize → IndexFlatIP (cosine).
Exact search; the ~2500-entry corpus is too small for HNSW to pay off.

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.build_indexes.build_multi_layer_indexes
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from mm_agents.structagent._paths import REPO_ROOT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

UNI = REPO_ROOT / "results" / "unified_memory"
# v3 (default) or v4 (cleaned/abstracted L1+L2). L3 rule sheets are shared
# across versions and always read from _v3.
BUILD_VERSION = os.environ.get("BUILD_VERSION", "v3")
INDEXES_DIR = UNI / f"_indexes_{BUILD_VERSION}"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ─── Loaders ────────────────────────────────────────────────────────────────
def load_l1_actions() -> Dict[str, List[Dict[str, Any]]]:
    """Returns dict[domain → list[action_entry]].

    Each cluster file holds 1-4 actions; flattened to one index row per
    action. Embedding key is ``subgoal_pattern``.
    """
    base = UNI / f"_l1_actions_{BUILD_VERSION}"
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for dom_dir in sorted(base.iterdir()):
        if not dom_dir.is_dir():
            continue
        entries: List[Dict[str, Any]] = []
        for fp in sorted(dom_dir.glob("cluster_*.json")):
            d = json.loads(fp.read_text())
            for a in d.get("actions") or []:
                entries.append({
                    "action_id": a.get("action_id"),
                    "cluster_id": d.get("source_cluster_id"),
                    "domain": d.get("domain"),
                    "subgoal_pattern": d.get("cluster_subgoal_pattern"),
                    "approach": a.get("approach"),
                    "when_to_use": a.get("when_to_use"),
                    "typical_actions": a.get("typical_actions") or [],
                    "tools_or_widgets_seen": a.get("tools_or_widgets_seen") or [],
                    "common_pitfalls": a.get("common_pitfalls") or [],
                    "n_assigned_samples": a.get("n_assigned_samples"),
                    "n_assigned_segments_in_cluster":
                        d.get("n_assigned_segments"),
                    "assigned_sample_task_hashes":
                        a.get("assigned_sample_task_hashes") or [],
                    "full_cluster_task_hashes":
                        d.get("full_cluster_task_hashes") or [],
                    "source_path": str(fp.relative_to(REPO_ROOT)),
                })
        if entries:
            by_domain[dom_dir.name] = entries
    return by_domain


def load_l2_skills() -> List[Dict[str, Any]]:
    """Flatten L2 cluster files (1+ skill each) to a list of skill entries."""
    base = UNI / f"_l2_skills_{BUILD_VERSION}"
    entries: List[Dict[str, Any]] = []
    for fp in sorted(base.glob("cluster_*.json")):
        d = json.loads(fp.read_text())
        for sk in d.get("skills") or []:
            entries.append({
                "skill_id": sk.get("skill_id"),
                "cluster_id": d.get("cluster_id"),
                "task_class": sk.get("task_class"),
                "when_to_use": sk.get("when_to_use") or "",
                "plan_template": sk.get("plan_template") or [],
                "typical_step_count": sk.get("typical_step_count"),
                "attached_pitfalls": sk.get("attached_pitfalls") or [],
                "n_assigned": sk.get("n_assigned"),
                "source_task_hashes": sk.get("source_task_hashes") or [],
                "cluster_coherence": d.get("cluster_coherence"),
                "source_path": str(fp.relative_to(REPO_ROOT)),
            })
    return entries


def load_l3a_clusters() -> List[Dict[str, Any]]:
    """One entry per L3a cluster (cluster_summary index)."""
    base = UNI / "_l3a_cluster_rules_v3"
    entries: List[Dict[str, Any]] = []
    for fp in sorted(base.glob("cluster_*.json")):
        d = json.loads(fp.read_text())
        entries.append({
            "cluster_id": d.get("cluster_id"),
            "cluster_summary": d.get("cluster_summary") or "",
            "cluster_coherence": d.get("cluster_coherence"),
            "cross_domain_signal": d.get("cross_domain_signal"),
            "n_domains_represented": d.get("n_domains_represented"),
            "n_input_records": d.get("n_input_records"),
            "n_common_rules": d.get("n_common_rules"),
            "common_rules": d.get("common_rules") or [],
            "shared_pitfalls": d.get("shared_pitfalls") or [],
            "source_path": str(fp.relative_to(REPO_ROOT)),
        })
    return entries


def load_l3a_rules() -> List[Dict[str, Any]]:
    """One entry per individual rule across all L3a clusters."""
    base = UNI / "_l3a_cluster_rules_v3"
    entries: List[Dict[str, Any]] = []
    for fp in sorted(base.glob("cluster_*.json")):
        d = json.loads(fp.read_text())
        cid = d.get("cluster_id")
        cs = d.get("cluster_summary") or ""
        for rule in d.get("common_rules") or []:
            entries.append({
                "cluster_id": cid,
                "cluster_summary": cs,
                "rule": rule.get("rule") or "",
                "rationale": rule.get("rationale") or "",
                "applies_when": rule.get("applies_when") or "",
                "scope": rule.get("scope"),
                "category": rule.get("category"),
                "observed_in_n_tasks": rule.get("observed_in_n_tasks"),
                "supporting_task_hashes":
                    rule.get("supporting_task_hashes") or [],
                "source_path": str(fp.relative_to(REPO_ROOT)),
            })
    return entries


def load_l3c() -> Dict[str, Dict[str, Any]]:
    """Direct domain → rule-sheet dict. No FAISS needed (7 keys)."""
    base = UNI / "_l3c_domain_rules_v3"
    out: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(base.glob("*.json")):
        if fp.name.startswith("_"):
            continue
        d = json.loads(fp.read_text())
        out[d["domain"]] = {
            "domain_summary": d.get("domain_summary"),
            "domain_invariants": d.get("domain_invariants") or [],
            "common_widgets": d.get("common_widgets") or [],
            "common_shortcuts": d.get("common_shortcuts") or [],
            "shared_pitfalls": d.get("shared_pitfalls") or [],
            "open_questions": d.get("open_questions") or [],
            "n_l2_skills_summarized": d.get("n_l2_skills_summarized"),
        }
    return out


# ─── Index build ────────────────────────────────────────────────────────────
def build_faiss_index(model: SentenceTransformer,
                      texts: List[str]) -> faiss.Index:
    """Embed → L2 normalize → FAISS IndexFlatIP."""
    if not texts:
        raise ValueError("empty text list")
    embs = model.encode(texts, show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=128)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    return index


def write_index(index: faiss.Index,
                metadata: List[Dict[str, Any]],
                out_dir: Path,
                meta_info: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "metadata.jsonl").open("w") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    (out_dir / "_meta.json").write_text(
        json.dumps(meta_info, indent=2, ensure_ascii=False))


def build_l1(model: SentenceTransformer) -> Dict[str, Any]:
    logger.info("L1: loading per-domain actions...")
    by_domain = load_l1_actions()
    out_meta = {"per_domain": {}}
    for dom, entries in by_domain.items():
        logger.info(f"  L1[{dom}]: {len(entries)} actions → embedding")
        texts = [e["subgoal_pattern"] or "" for e in entries]
        idx = build_faiss_index(model, texts)
        dom_dir = INDEXES_DIR / "l1" / dom
        write_index(idx, entries, dom_dir, {
            "domain": dom,
            "embed_model": EMBED_MODEL,
            "embed_key": "subgoal_pattern",
            "n_entries": len(entries),
            "dim": idx.d,
        })
        out_meta["per_domain"][dom] = len(entries)
    out_meta["total_entries"] = sum(out_meta["per_domain"].values())
    return out_meta


def build_l2(model: SentenceTransformer) -> Dict[str, Any]:
    logger.info("L2: loading skills...")
    entries = load_l2_skills()
    logger.info(f"  L2: {len(entries)} skills → embedding")
    texts = [e["when_to_use"] or "" for e in entries]
    idx = build_faiss_index(model, texts)
    write_index(idx, entries, INDEXES_DIR / "l2", {
        "embed_model": EMBED_MODEL,
        "embed_key": "when_to_use",
        "n_entries": len(entries),
        "dim": idx.d,
    })
    return {"n_entries": len(entries)}


def build_l3a_cluster(model: SentenceTransformer) -> Dict[str, Any]:
    logger.info("L3a-cluster: loading cluster summaries...")
    entries = load_l3a_clusters()
    logger.info(f"  L3a-cluster: {len(entries)} clusters → embedding")
    texts = [e["cluster_summary"] or "" for e in entries]
    idx = build_faiss_index(model, texts)
    write_index(idx, entries, INDEXES_DIR / "l3a_cluster", {
        "embed_model": EMBED_MODEL,
        "embed_key": "cluster_summary",
        "n_entries": len(entries),
        "dim": idx.d,
    })
    return {"n_entries": len(entries)}


def build_l3a_rule(model: SentenceTransformer) -> Dict[str, Any]:
    logger.info("L3a-rule: loading individual rules...")
    entries = load_l3a_rules()
    logger.info(f"  L3a-rule: {len(entries)} rules → embedding")
    # Concat rule + applies_when for richer embedding signal
    texts = [(e["rule"] + " — " + e["applies_when"]).strip(" —")
             for e in entries]
    idx = build_faiss_index(model, texts)
    write_index(idx, entries, INDEXES_DIR / "l3a_rule", {
        "embed_model": EMBED_MODEL,
        "embed_key": "rule + ' — ' + applies_when",
        "n_entries": len(entries),
        "dim": idx.d,
    })
    return {"n_entries": len(entries)}


def build_l3c() -> Dict[str, Any]:
    """No FAISS — direct dict file. Embedding-free, 7 keys."""
    logger.info("L3c: loading per-domain rule sheets...")
    by_domain = load_l3c()
    out_dir = INDEXES_DIR / "l3c"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rules_by_domain.json"
    out_path.write_text(json.dumps(by_domain, indent=2, ensure_ascii=False))
    (out_dir / "_meta.json").write_text(json.dumps({
        "format": "direct_lookup",
        "key": "domain",
        "n_domains": len(by_domain),
        "domains": sorted(by_domain.keys()),
    }, indent=2))
    logger.info(f"  L3c: {len(by_domain)} domains → {out_path.name}")
    return {"n_domains": len(by_domain)}


def main() -> None:
    INDEXES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading embedder: {EMBED_MODEL}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    logger.info(f"  loaded in {time.time()-t0:.1f}s")

    overall = {}
    overall["l1"] = build_l1(model)
    overall["l2"] = build_l2(model)
    overall["l3a_cluster"] = build_l3a_cluster(model)
    overall["l3a_rule"] = build_l3a_rule(model)
    overall["l3c"] = build_l3c()

    grand = {
        "embed_model": EMBED_MODEL,
        "indexes": overall,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (INDEXES_DIR / "_meta.json").write_text(
        json.dumps(grand, indent=2, ensure_ascii=False))

    logger.info("DONE")
    print(json.dumps(grand, indent=2))


if __name__ == "__main__":
    main()

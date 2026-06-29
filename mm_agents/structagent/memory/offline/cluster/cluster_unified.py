"""Embed + cluster the pooled UnifiedRecord corpus (internal + AgentNet +
Mind2Web). Phase 2 of the unified mining pipeline.

Reads ``loaders.pool_all_sources()`` (~4725 records). Writes per-cluster
``members.json`` + a ``_summary.json`` under ``results/unified_memory/``.

Algorithm: all-MiniLM-L6-v2 + cosine + Agglomerative average linkage,
distance_threshold 0.55. Embedding text is instruction ONLY — embedding
domain/outcomes/pitfalls lets per-record idiosyncrasies (paths, cell refs,
URLs) collapse the similarity surface; we want "book a flight on United"
and "find a flight on Delta" to cluster together.

Members carry both the SHA1[:12] ``task_hash`` (ship id) and original
``task_id`` (internal trace, never overwritten). ``source_label`` is
neutral; the provenance-revealing ``_source_raw`` lives in ``raw_meta`` so
packaging can strip it.

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.cluster.cluster_unified
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.cluster.cluster_unified \\
      --distance-threshold 0.50   # tighter, more clusters
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents.structagent.memory.offline.load_pool.loaders import (
    UnifiedRecord, pool_all_sources, summarise_pool,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

OUT_BASE = REPO_ROOT / "results" / "unified_memory"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cosine-distance cutoff for AgglomerativeClustering. SMALLER = stricter =
# more, tighter clusters. Names track tightness, not size.
#
#   tight (0.40) → L2 plan-template polish. split_cross_domain ON:
#     L2 needs per-domain cohesion (writer "format paragraph" ≠ calc
#     "format cell"). sub_cluster heads to break "format slide *anything*"
#     hairballs.
#   loose (0.55) → L3a cluster-rule / L3c domain-rule polish.
#     split_cross_domain OFF: cross-domain cohesion ("insert image" across
#     writer/calc/impress) IS the L3a signal. Refine only the largest heads.
VARIANTS = {
    "tight": {"threshold": 0.40, "out_subdir": "_clusters_v3_tight",
              "split_cross_domain": True,
              "sub_cluster_above": 10, "sub_threshold": 0.25},
    "loose": {"threshold": 0.55, "out_subdir": "_clusters_v3_loose",
              "split_cross_domain": False,
              "sub_cluster_above": 30, "sub_threshold": 0.35},
}


# ─── Embedding text ─────────────────────────────────────────────────────────
# ─── Post-processing helpers ────────────────────────────────────────────────
def _split_cross_domain(
    groups: List[Tuple[List[int], Dict[str, Any]]],
    pool: List[UnifiedRecord],
) -> List[Tuple[List[int], Dict[str, Any]]]:
    """Slice any cluster spanning multiple domains into per-domain pieces.

    L2 plan templates need per-app cohesion. Each piece keeps ``split``
    metadata for audit."""
    out: List[Tuple[List[int], Dict[str, Any]]] = []
    for g, meta in groups:
        by_dom: Dict[str, List[int]] = {}
        for i in g:
            by_dom.setdefault(pool[i].domain, []).append(i)
        if len(by_dom) > 1:
            for dom, sub in by_dom.items():
                m = dict(meta)
                m["split"] = "by_domain"
                m["split_domain"] = dom
                m["parent_size"] = len(g)
                out.append((sub, m))
        else:
            out.append((g, meta))
    return out


_SUB_DECAY = 0.7        # each recursion makes the threshold this much tighter
_SUB_MIN_THRESHOLD = 0.05  # don't go tighter than this; force-split at depth limit
_SUB_MAX_DEPTH = 8


def _recursive_split(g: List[int], dist: np.ndarray, *,
                     cap: int, threshold: float, depth: int,
                     parent_meta: Dict[str, Any]
                     ) -> List[Tuple[List[int], Dict[str, Any]]]:
    """Split ``g`` agglomeratively until every sub-group has len <= cap.

    Clusters on the sub-block of the precomputed distance matrix; if a
    threshold doesn't split (members too similar), recurse tighter. At the
    depth/threshold limit, fall back to even slicing into ``cap``-sized
    chunks to guarantee the cap (arbitrary, but rare)."""
    if len(g) <= cap:
        m = dict(parent_meta)
        if depth > 0:
            m["sub_depth"] = depth
        return [(g, m)]

    if depth >= _SUB_MAX_DEPTH or threshold < _SUB_MIN_THRESHOLD:
        # Hard fallback: even chunks, marked for audit.
        out: List[Tuple[List[int], Dict[str, Any]]] = []
        for i in range(0, len(g), cap):
            m = dict(parent_meta)
            m["sub_clustered"] = True
            m["sub_depth"] = depth
            m["sub_forced_chunked"] = True
            out.append((g[i:i + cap], m))
        return out

    sub_dist = dist[np.ix_(g, g)]
    sub_labels = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    ).fit_predict(sub_dist)
    sub_groups: Dict[int, List[int]] = {}
    for i, lab in enumerate(sub_labels):
        sub_groups.setdefault(int(lab), []).append(g[i])

    # If threshold didn't split (single sub), tighten and retry.
    if len(sub_groups) == 1:
        return _recursive_split(g, dist,
                                cap=cap,
                                threshold=threshold * _SUB_DECAY,
                                depth=depth + 1,
                                parent_meta=parent_meta)

    out: List[Tuple[List[int], Dict[str, Any]]] = []
    for sub_g in sub_groups.values():
        out.extend(_recursive_split(sub_g, dist,
                                    cap=cap,
                                    threshold=threshold * _SUB_DECAY,
                                    depth=depth + 1,
                                    parent_meta={**parent_meta,
                                                 "sub_clustered": True,
                                                 "sub_parent_size": len(g)}))
    return out


def _sub_cluster_heads(
    groups: List[Tuple[List[int], Dict[str, Any]]],
    dist: np.ndarray,
    *, above: int, threshold: float,
) -> List[Tuple[List[int], Dict[str, Any]]]:
    """Split any cluster with > ``above`` members until all pieces fit the
    cap. See ``_recursive_split`` — cap is guaranteed."""
    out: List[Tuple[List[int], Dict[str, Any]]] = []
    for g, meta in groups:
        out.extend(_recursive_split(g, dist,
                                    cap=above,
                                    threshold=threshold,
                                    depth=0,
                                    parent_meta=meta))
    return out


# ─── Embedding text ─────────────────────────────────────────────────────────
def build_embed_text(rec: UnifiedRecord) -> str:
    """Instruction only — no domain prefix.

    A ``[{domain}]`` prefix backfired: as a fixed leading token it
    dominated MiniLM's attention, pulling every impress task together into
    332-member "format impress" hairballs while stranding the tail as
    singletons. Without it, "Create a bar chart …" can cluster cross-app.
    Domain is still kept per-record and surfaced as a cluster distribution.

    ``instruction`` = ``natural_language_task`` (AgentNet, post-patch),
    original query (Mind2Web), or task instruction (Internal)."""
    return (rec.instruction or "")[:1500]


# ─── Cluster member rendering ───────────────────────────────────────────────
def to_member(rec: UnifiedRecord) -> Dict[str, Any]:
    """One cluster member entry. task_hash primary; task_id kept for trace."""
    return {
        "task_hash": rec.task_hash,                     # SHA1[:12] — ship id
        "task_id": rec.task_id,                         # ORIGINAL — never overwritten
        "instruction": (rec.instruction or "")[:500],
        "domain": rec.domain,
        "source_label": rec.source_label,               # neutral
        "_source_raw": rec.raw_meta.get("_source"),     # internal|agentnet|mind2web
        "n_subgoals": len(rec.subgoals or []),
        "n_key_actions": len(rec.key_actions or []),
        "task_completed": rec.task_completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()) + ["both"],
                        default="both",
                        help="broad (0.40 → L3a) | specific (0.55 → L2 / intent-recipe)"
                             " | both (run both in one pass; embed reused)")
    parser.add_argument("--distance-threshold", type=float, default=None,
                        help="Override variant default threshold")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Override default out dir")
    args = parser.parse_args()

    if args.variant == "both" and (args.distance_threshold or args.out_dir):
        parser.error("--variant both does not accept --distance-threshold "
                     "or --out-dir overrides")

    variants_to_run: List[Dict[str, Any]]
    if args.variant == "both":
        variants_to_run = [
            {"name": "tight", **VARIANTS["tight"]},
            {"name": "loose", **VARIANTS["loose"]},
        ]
    else:
        spec = dict(VARIANTS[args.variant])
        if args.distance_threshold is not None:
            spec["threshold"] = args.distance_threshold
        if args.out_dir is not None:
            spec["out_subdir"] = str(args.out_dir.relative_to(OUT_BASE)
                                     if args.out_dir.is_absolute() and
                                     OUT_BASE in args.out_dir.parents
                                     else args.out_dir)
        variants_to_run = [{"name": args.variant, **spec}]

    # 1. Load pool
    logger.info("Loading unified pool...")
    pool: List[UnifiedRecord] = pool_all_sources()
    n = len(pool)
    logger.info(f"  pool size: {n}")

    # Drop empty-instruction records — they collapse together and pollute
    # whatever cluster they land in.
    pool = [r for r in pool if (r.instruction or "").strip()]
    dropped = n - len(pool)
    if dropped:
        logger.warning(f"  dropped {dropped} records with empty instruction")
    logger.info(f"  embedding {len(pool)} records")

    pool_summary = summarise_pool(pool)

    # 2. Embed
    texts = [build_embed_text(r) for r in pool]
    logger.info(f"Loading embedder: {EMBED_MODEL}")
    t0 = time.time()
    model = SentenceTransformer(EMBED_MODEL)
    logger.info(f"  loaded in {time.time()-t0:.1f}s")

    t0 = time.time()
    embeddings = model.encode(texts, show_progress_bar=True,
                              convert_to_numpy=True,
                              normalize_embeddings=True,
                              batch_size=64)
    logger.info(f"  encoded {len(texts)} in {time.time()-t0:.1f}s; "
                f"shape={embeddings.shape}")

    # 3. Pairwise cosine distance — computed ONCE, reused per variant.
    t0 = time.time()
    sim = embeddings @ embeddings.T            # ~89 MB float32 for 4725
    np.clip(sim, -1.0, 1.0, out=sim)
    dist = 1.0 - sim
    logger.info(f"Built cosine-distance matrix in {time.time()-t0:.1f}s; "
                f"shape={dist.shape}")

    # 4. Run each variant (broad / specific): cluster + write per-variant out
    for variant in variants_to_run:
        variant_name = variant["name"]
        distance_threshold = variant["threshold"]
        out_dir = OUT_BASE / variant["out_subdir"]

        print(f"\n{'─'*78}\nVARIANT: {variant_name}  (threshold={distance_threshold}, out={out_dir.name})\n{'─'*78}")

        t0 = time.time()
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage="average",
        )
        labels = clusterer.fit_predict(dist)
        n_clusters_initial = int(labels.max()) + 1
        logger.info(f"  → {n_clusters_initial} initial clusters in "
                    f"{time.time()-t0:.1f}s")

        by_cluster_initial: Dict[int, List[int]] = {}
        for i, lab in enumerate(labels):
            by_cluster_initial.setdefault(int(lab), []).append(i)
        groups: List[Tuple[List[int], Dict[str, Any]]] = [
            (idxs, {}) for idxs in by_cluster_initial.values()
        ]

        # Post-process 1: per-domain split (tight tier only)
        if variant.get("split_cross_domain"):
            before = len(groups)
            groups = _split_cross_domain(groups, pool)
            logger.info(f"  per-domain split: {before} → {len(groups)} "
                        f"clusters")

        # Post-process 2: head sub-cluster
        sub_above = variant.get("sub_cluster_above")
        if sub_above is not None:
            sub_thresh = variant["sub_threshold"]
            before = len(groups)
            groups = _sub_cluster_heads(groups, dist,
                                        above=sub_above,
                                        threshold=sub_thresh)
            n_heads_split = sum(1 for _, m in groups if m.get("sub_clustered"))
            logger.info(f"  sub-cluster heads (n>{sub_above}, "
                        f"threshold={sub_thresh}): {before} → {len(groups)} "
                        f"clusters ({n_heads_split} sub-groups)")

        # Final: sort by size desc
        groups.sort(key=lambda kv: -len(kv[0]))
        n_clusters = len(groups)

        out_dir.mkdir(parents=True, exist_ok=True)
        for old in out_dir.iterdir():
            if old.is_dir() and old.name.startswith("cluster_"):
                for f in old.iterdir():
                    f.unlink()
                old.rmdir()

        summary_clusters: List[Dict[str, Any]] = []
        width = max(3, len(str(n_clusters)))
        for new_id, (idxs, prov) in enumerate(groups):
            cluster_name = f"cluster_{new_id:0{width}d}"
            cluster_dir = out_dir / cluster_name
            cluster_dir.mkdir(exist_ok=True)

            domain_counts: Counter = Counter()
            source_counts: Counter = Counter()
            source_raw_counts: Counter = Counter()
            members = []
            for i in idxs:
                rec = pool[i]
                domain_counts[rec.domain] += 1
                source_counts[rec.source_label] += 1
                source_raw_counts[rec.raw_meta.get("_source") or "unknown"] += 1
                members.append(to_member(rec))

            (cluster_dir / "members.json").write_text(
                json.dumps(members, indent=2, ensure_ascii=False)
            )

            cluster_embs = embeddings[idxs]
            centroid = cluster_embs.mean(axis=0)
            centroid /= max(np.linalg.norm(centroid), 1e-9)
            sims_to_centroid = cluster_embs @ centroid
            centroid_idx = idxs[int(np.argmax(sims_to_centroid))]
            centroid_rec = pool[centroid_idx]

            summary_clusters.append({
                "cluster_id": cluster_name,
                "n_members": len(idxs),
                "domain_distribution": dict(domain_counts),
                "source_distribution": dict(source_counts),
                "source_raw_distribution": dict(source_raw_counts),
                "centroid_task_hash": centroid_rec.task_hash,
                "centroid_task_id": centroid_rec.task_id,
                "centroid_instruction": (centroid_rec.instruction or "")[:200],
                "provenance": prov,            # split / sub-cluster trail
            })

        sizes = [len(idxs) for idxs, _ in groups]
        cluster_size_counts = Counter(sizes)
        summary = {
            "variant": variant_name,
            "embed_model": EMBED_MODEL,
            "embed_input_recipe": "instruction only (no domain prefix)",
            "distance_threshold": distance_threshold,
            "split_cross_domain": bool(variant.get("split_cross_domain")),
            "sub_cluster_above": variant.get("sub_cluster_above"),
            "sub_threshold": variant.get("sub_threshold"),
            "pool": pool_summary,
            "n_embedded": len(pool),
            "n_clusters_initial": n_clusters_initial,
            "n_clusters": n_clusters,
            "size_distribution": {
                "median": int(np.median(sizes)),
                "mean": round(float(np.mean(sizes)), 2),
                "max": int(max(sizes)),
                "singletons": int(sum(1 for s in sizes if s == 1)),
                "small_le3": int(sum(1 for s in sizes if s <= 3)),
                "histogram_top": cluster_size_counts.most_common(10),
            },
            "clusters": summary_clusters,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (out_dir / "_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )

        print(f"Top-15 clusters by size:")
        for c in summary_clusters[:15]:
            src = ",".join(f"{s[0]}={n}" for s, n in
                           sorted(c["source_raw_distribution"].items(),
                                  key=lambda kv: -kv[1]))
            dom = ",".join(f"{d}={n}" for d, n in
                           sorted(c["domain_distribution"].items(),
                                  key=lambda kv: -kv[1])[:3])
            head = (c["centroid_instruction"] or "")[:60]
            print(f"  {c['cluster_id']:>10s}  n={c['n_members']:>4d}  "
                  f"[{src}]  ({dom})  «{head}»")
        sd = summary["size_distribution"]
        print(f"  → {n_clusters} clusters | singletons={sd['singletons']} "
              f"| ≤3={sd['small_le3']} | median={sd['median']} | max={sd['max']}")
        print(f"  summary: {out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()

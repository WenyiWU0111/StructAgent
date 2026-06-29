"""Centroid-based 2nd-tier merge for the loose-tier small clusters.

Why
---
After `cluster_unified` produces the loose tier (cosine 0.55, recursive
cap 30), it leaves 730 singletons + several hundred n=2-4 clusters.
Polishing those individually is wasteful — they're too thin to abstract
across — but many of them are "near miss" neighbours that the 0.55
threshold barely failed to merge. We pool them via 2nd-tier centroid
clustering at a tighter threshold (0.40), grouping near-miss small
clusters into polishable super-clusters.

Output (under ``results/unified_memory/_clusters_v3_loose_merged/``):
  cluster_NNNN/members.json     — flat member list, ready for polish_l3a
  _summary.json                 — per-cluster stats + merge provenance

The output is a COMBINED set: every loose-tier cluster (big + merged
super-clusters + still-unmerged small) is present, renumbered by size.
polish_l3a points at this dir by default; one pass covers everything.

Provenance:
  - "big_kept"           — original loose cluster ≥ merge_threshold,
                          copied as-is
  - "merged"             — super-cluster combining 2+ original small
                          clusters (their cluster_ids listed)
  - "small_kept"         — original small cluster that didn't find a
                          centroid partner; kept as-is

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.cluster.merge_small_loose_clusters
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.cluster.merge_small_loose_clusters \\
      --merge-threshold 5 --centroid-threshold 0.40
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents.structagent.memory.offline.load_pool.loaders import (
    UnifiedRecord, pool_all_sources,
)
from mm_agents.structagent.memory.offline.cluster.cluster_unified import (
    build_embed_text, to_member, EMBED_MODEL,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

LOOSE_DIR = REPO_ROOT / "results" / "unified_memory" / "_clusters_v3_loose"
OUT_DIR = REPO_ROOT / "results" / "unified_memory" / "_clusters_v3_loose_merged"

DEFAULT_MERGE_THRESHOLD = 5     # clusters with n < this are merge candidates
DEFAULT_CENTROID_THRESHOLD = 0.40   # tighter than loose 0.55; centroids are
                                    # already summarised, so demand more
                                    # similarity to merge


def load_loose_clusters() -> List[Dict[str, Any]]:
    summary = json.loads((LOOSE_DIR / "_summary.json").read_text())
    return summary["clusters"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merge-threshold", type=int,
                   default=DEFAULT_MERGE_THRESHOLD,
                   help="Clusters with n < this are merge candidates "
                        "(default 5)")
    p.add_argument("--centroid-threshold", type=float,
                   default=DEFAULT_CENTROID_THRESHOLD,
                   help="Cosine-distance threshold on small-cluster centroids "
                        "(default 0.40)")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    merge_threshold: int = args.merge_threshold
    centroid_threshold: float = args.centroid_threshold

    # 1. Load unified pool + build hash → record lookup
    logger.info("Loading unified pool...")
    pool = pool_all_sources()
    hash_idx = {r.task_hash: r for r in pool}
    logger.info(f"  pool size: {len(pool)}")

    # 2. Load loose-tier clusters
    loose_clusters = load_loose_clusters()
    big = [c for c in loose_clusters if c["n_members"] >= merge_threshold]
    small = [c for c in loose_clusters if c["n_members"] < merge_threshold]
    logger.info(f"loose tier: total={len(loose_clusters)}  "
                f"big(n>={merge_threshold})={len(big)}  "
                f"small(n<{merge_threshold})={len(small)}")

    # 3. Load each small cluster's members + compute centroid
    logger.info(f"Loading {EMBED_MODEL} (need for centroid computation)...")
    model = SentenceTransformer(EMBED_MODEL)

    small_centroids: List[Dict[str, Any]] = []   # {cid, centroid, records}
    t0 = time.time()
    for c in small:
        members_json = json.loads(
            (LOOSE_DIR / c["cluster_id"] / "members.json").read_text())
        member_records: List[UnifiedRecord] = []
        for m in members_json:
            rec = hash_idx.get(m["task_hash"])
            if rec is not None:
                member_records.append(rec)
        if not member_records:
            continue
        texts = [build_embed_text(r) for r in member_records]
        embs = model.encode(texts, show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            batch_size=64)
        centroid = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm < 1e-9:
            continue
        centroid /= norm
        small_centroids.append({
            "cluster_id": c["cluster_id"],
            "centroid": centroid,
            "records": member_records,
        })
    logger.info(f"  built {len(small_centroids)} small-cluster centroids "
                f"in {time.time()-t0:.1f}s")

    # 4. Cluster centroids
    centroid_matrix = np.stack([sc["centroid"] for sc in small_centroids])
    sim = centroid_matrix @ centroid_matrix.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist = 1.0 - sim

    t0 = time.time()
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=centroid_threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clusterer.fit_predict(dist)
    n_centroid_clusters = int(labels.max()) + 1
    logger.info(f"centroid clustering (threshold={centroid_threshold}): "
                f"{len(small_centroids)} small → {n_centroid_clusters} "
                f"centroid-groups in {time.time()-t0:.1f}s")

    # 5. Build merged super-clusters
    label_to_small_idxs: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        label_to_small_idxs.setdefault(int(lab), []).append(i)

    merged_super: List[Dict[str, Any]] = []
    unmerged_small: List[Dict[str, Any]] = []
    for lab, idxs in label_to_small_idxs.items():
        if len(idxs) == 1:
            sc = small_centroids[idxs[0]]
            unmerged_small.append({
                "size": len(sc["records"]),
                "records": sc["records"],
                "provenance": {
                    "type": "small_kept",
                    "orig_cluster_ids": [sc["cluster_id"]],
                },
            })
        else:
            constituent_cids = [small_centroids[i]["cluster_id"] for i in idxs]
            merged_records: List[UnifiedRecord] = []
            for i in idxs:
                merged_records.extend(small_centroids[i]["records"])
            merged_super.append({
                "size": len(merged_records),
                "records": merged_records,
                "provenance": {
                    "type": "merged",
                    "centroid_threshold": centroid_threshold,
                    "n_constituent_small_clusters": len(idxs),
                    "constituent_cluster_ids": constituent_cids,
                },
            })

    # 6. Collect big clusters as-is
    big_kept: List[Dict[str, Any]] = []
    for c in big:
        members_json = json.loads(
            (LOOSE_DIR / c["cluster_id"] / "members.json").read_text())
        records: List[UnifiedRecord] = []
        for m in members_json:
            rec = hash_idx.get(m["task_hash"])
            if rec is not None:
                records.append(rec)
        if not records:
            continue
        big_kept.append({
            "size": len(records),
            "records": records,
            "provenance": {
                "type": "big_kept",
                "orig_cluster_id": c["cluster_id"],
                "orig_n_members": c["n_members"],
            },
        })

    # 7. Combine and renumber by size desc
    all_clusters = big_kept + merged_super + unmerged_small
    all_clusters.sort(key=lambda c: -c["size"])
    n_total = len(all_clusters)

    logger.info(f"COMBINED OUTPUT: total={n_total}  "
                f"(big_kept={len(big_kept)}, "
                f"merged_super={len(merged_super)}, "
                f"small_kept={len(unmerged_small)})")

    # 8. Write
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_dir() and old.name.startswith("cluster_"):
            for f in old.iterdir():
                f.unlink()
            old.rmdir()

    summary_clusters: List[Dict[str, Any]] = []
    width = max(4, len(str(n_total)))
    polishable_threshold = merge_threshold
    n_polishable = 0
    for new_id, cluster in enumerate(all_clusters):
        cluster_name = f"cluster_{new_id:0{width}d}"
        cluster_dir = out_dir / cluster_name
        cluster_dir.mkdir(exist_ok=True)

        members = [to_member(r) for r in cluster["records"]]
        (cluster_dir / "members.json").write_text(
            json.dumps(members, indent=2, ensure_ascii=False))

        domain_counts = Counter(r.domain for r in cluster["records"])
        src_raw_counts = Counter(
            r.raw_meta.get("_source") or "?" for r in cluster["records"])
        head_rec = cluster["records"][0]   # not centroid; just for display
        prov = cluster["provenance"]
        entry = {
            "cluster_id": cluster_name,
            "n_members": cluster["size"],
            "domain_distribution": dict(domain_counts),
            "source_raw_distribution": dict(src_raw_counts),
            "centroid_task_hash": head_rec.task_hash,
            "centroid_instruction": (head_rec.instruction or "")[:200],
            "provenance": prov,
        }
        summary_clusters.append(entry)
        if cluster["size"] >= polishable_threshold:
            n_polishable += 1

    sizes = [c["size"] for c in all_clusters]
    cluster_size_counts = Counter(sizes)
    summary = {
        "source_loose_dir": str(LOOSE_DIR),
        "merge_threshold": merge_threshold,
        "centroid_threshold": centroid_threshold,
        "embed_model": EMBED_MODEL,
        "n_loose_clusters_input": len(loose_clusters),
        "n_big_kept": len(big_kept),
        "n_small_input": len(small),
        "n_small_with_centroid": len(small_centroids),
        "n_merged_super_clusters": len(merged_super),
        "n_unmerged_small_kept": len(unmerged_small),
        "n_clusters_output": n_total,
        "n_polishable_at_threshold": n_polishable,
        "polishable_threshold": polishable_threshold,
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
        json.dumps(summary, indent=2, ensure_ascii=False))

    # 9. Stats summary
    print(f"\n{'='*78}")
    print(f"MERGED CLUSTER SUMMARY (in {out_dir.name})")
    print(f"{'='*78}")
    print(f"  loose input clusters:          {len(loose_clusters)}")
    print(f"    big kept (n>={merge_threshold}):              {len(big_kept)}")
    print(f"    small in (n<{merge_threshold}):               {len(small)}")
    print(f"      → merged super-clusters:  {len(merged_super)}")
    print(f"      → still-small kept:       {len(unmerged_small)}")
    print(f"  combined output clusters:     {n_total}")
    print(f"  polishable (n>={polishable_threshold}):              {n_polishable}")
    print(f"\nsize distribution: max={summary['size_distribution']['max']}  "
          f"median={summary['size_distribution']['median']}  "
          f"singletons={summary['size_distribution']['singletons']}")
    print(f"\nTop merged super-clusters by size:")
    sorted_merged = sorted(merged_super, key=lambda c: -c["size"])
    for i, c in enumerate(sorted_merged[:10]):
        prov = c["provenance"]
        head = (c["records"][0].instruction or "")[:60]
        print(f"  #{i+1}  n={c['size']:>3d}  "
              f"from {prov['n_constituent_small_clusters']} small → «{head}»")
    print(f"\nfull summary: {out_dir / '_summary.json'}")


if __name__ == "__main__":
    main()

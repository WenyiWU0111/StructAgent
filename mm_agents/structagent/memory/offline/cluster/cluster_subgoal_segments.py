"""Phase 3 L1 — Step 2: cluster subgoal segments PER DOMAIN.

Reads ``_l1_segments_raw.jsonl`` (~22K segments from
``extract_subgoal_segments``) and clusters by ``subgoal`` text within
each domain. Per-domain rather than cross-domain because:
  - keeps each cosine matrix small (chrome's 11K vs all 22K) and lets
    domains run in parallel;
  - L1 actions ("Click Insert > Image") are naturally domain-bound;
  - matches the v2 approach that already proved per-domain L1 mining works.

Output under ``results/unified_memory/_l1_clusters_v3/<domain>/``:
  cluster_NNN/members.json   — segments in the cluster
  _summary.json              — per-domain + per-cluster stats

Embedding text is the subgoal only — adding domain/action would cluster
by surface form rather than intent. Threshold is tighter than L2's 0.40
since subgoals are short and uniform (keep "Click File > Open" apart from
"Open File Manager").

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.cluster.cluster_subgoal_segments
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.cluster.cluster_subgoal_segments \\
      --threshold 0.30 --domains chrome,libreoffice_calc
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from mm_agents.structagent._paths import REPO_ROOT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

SEGMENTS_PATH = REPO_ROOT / "results" / "unified_memory" / "_l1_segments_raw.jsonl"
OUT_BASE = REPO_ROOT / "results" / "unified_memory" / "_l1_clusters_v3"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.35   # tighter than L2's 0.40 — subgoals are short & uniform


def load_segments() -> List[Dict[str, Any]]:
    segs = []
    with SEGMENTS_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            segs.append(json.loads(line))
    return segs


def cluster_one_domain(domain: str,
                       segs: List[Dict[str, Any]],
                       model: SentenceTransformer,
                       threshold: float) -> Dict[str, Any]:
    """Cluster all segments of one domain. Returns summary + writes files."""
    out_dir = OUT_BASE / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_dir() and old.name.startswith("cluster_"):
            for f in old.iterdir():
                f.unlink()
            old.rmdir()

    t0 = time.time()
    texts = [s["subgoal"] for s in segs]
    embs = model.encode(texts, show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        batch_size=128)
    sim = embs @ embs.T
    np.clip(sim, -1.0, 1.0, out=sim)
    dist = 1.0 - sim

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="precomputed",
        linkage="average",
    )
    labels = clusterer.fit_predict(dist)
    n_clusters = int(labels.max()) + 1

    by_cluster: Dict[int, List[int]] = {}
    for i, lab in enumerate(labels):
        by_cluster.setdefault(int(lab), []).append(i)
    cluster_sizes = sorted(by_cluster.items(), key=lambda kv: -len(kv[1]))

    width = max(4, len(str(n_clusters)))
    summary_clusters: List[Dict[str, Any]] = []
    for new_id, (_lab, idxs) in enumerate(cluster_sizes):
        cluster_name = f"cluster_{new_id:0{width}d}"
        cluster_dir = out_dir / cluster_name
        cluster_dir.mkdir(exist_ok=True)

        members = [segs[i] for i in idxs]
        (cluster_dir / "members.json").write_text(
            json.dumps(members, indent=2, ensure_ascii=False))

        # Centroid segment (closest to mean embedding)
        cluster_embs = embs[idxs]
        centroid = cluster_embs.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-9)
        sims_to_centroid = cluster_embs @ centroid
        centroid_idx = idxs[int(np.argmax(sims_to_centroid))]
        centroid_seg = segs[centroid_idx]

        src_counts = Counter(s["source"] for s in members)
        summary_clusters.append({
            "cluster_id": cluster_name,
            "n_members": len(idxs),
            "source_distribution": dict(src_counts),
            "centroid_subgoal": centroid_seg["subgoal"][:140],
            "centroid_task_hash": centroid_seg["task_hash"],
        })

    sizes = [len(idxs) for _, idxs in cluster_sizes]
    summary = {
        "domain": domain,
        "n_segments": len(segs),
        "n_clusters": n_clusters,
        "distance_threshold": threshold,
        "size_distribution": {
            "median": int(np.median(sizes)),
            "max": int(max(sizes)),
            "singletons": int(sum(1 for s in sizes if s == 1)),
            "polishable_n_ge_3": int(sum(1 for s in sizes if s >= 3)),
            "polishable_n_ge_5": int(sum(1 for s in sizes if s >= 5)),
        },
        "clusters": summary_clusters,
        "elapsed_s": round(time.time() - t0, 1),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--domains", type=str, default=None,
                   help="Comma-separated; default = all domains in segments")
    p.add_argument("--skip-large-domain-above", type=int, default=None,
                   help="If chrome has 11K segments and you want to test on "
                        "smaller domains first, pass e.g. 5000")
    args = p.parse_args()

    logger.info(f"Loading segments from {SEGMENTS_PATH.name}...")
    all_segs = load_segments()
    logger.info(f"  {len(all_segs)} segments loaded")

    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in all_segs:
        by_domain[s["domain"]].append(s)
    logger.info(f"  {len(by_domain)} domains")

    if args.domains:
        wanted = set(args.domains.split(","))
        by_domain = {k: v for k, v in by_domain.items() if k in wanted}

    if args.skip_large_domain_above is not None:
        by_domain = {k: v for k, v in by_domain.items()
                     if len(v) <= args.skip_large_domain_above}

    logger.info(f"Loading embedder: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    summaries = []
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    for dom in sorted(by_domain, key=lambda d: len(by_domain[d])):
        segs = by_domain[dom]
        logger.info(f"clustering domain={dom}  n_segments={len(segs)}  "
                    f"threshold={args.threshold} ...")
        s = cluster_one_domain(dom, segs, model, args.threshold)
        sd = s["size_distribution"]
        logger.info(f"  {dom}: {s['n_clusters']} clusters  "
                    f"median={sd['median']}  max={sd['max']}  "
                    f"singletons={sd['singletons']}  "
                    f"polishable(n>=3)={sd['polishable_n_ge_3']}  "
                    f"polishable(n>=5)={sd['polishable_n_ge_5']}  "
                    f"({s['elapsed_s']:.1f}s)")
        summaries.append(s)

    grand_summary = {
        "embed_model": EMBED_MODEL,
        "distance_threshold": args.threshold,
        "n_domains": len(summaries),
        "total_clusters": sum(s["n_clusters"] for s in summaries),
        "total_polishable_n_ge_3": sum(s["size_distribution"]["polishable_n_ge_3"]
                                       for s in summaries),
        "total_polishable_n_ge_5": sum(s["size_distribution"]["polishable_n_ge_5"]
                                       for s in summaries),
        "per_domain_summaries": [
            {k: v for k, v in s.items() if k != "clusters"}
            for s in summaries
        ],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (OUT_BASE / "_grand_summary.json").write_text(
        json.dumps(grand_summary, indent=2, ensure_ascii=False))
    logger.info(f"DONE — {grand_summary['total_clusters']} clusters across "
                f"{grand_summary['n_domains']} domains  "
                f"(polishable n>=3: {grand_summary['total_polishable_n_ge_3']}, "
                f"n>=5: {grand_summary['total_polishable_n_ge_5']})")
    logger.info(f"output: {OUT_BASE}")


if __name__ == "__main__":
    main()

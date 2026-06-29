"""Phase 3 L1 — Step 1: flatten the unified pool into subgoal segments.

A segment is one (subgoal, key_action) pair from a trajectory (~22K
total). Persisted as JSONL so clustering + polish can stream without
reloading the full pool.

Output: ``results/unified_memory/_l1_segments_raw.jsonl``. Each line:
task_hash (12-char, matches L2/L3a), task_id, domain, source, seg_idx,
subgoal (embedding key for clustering), key_action (str or {action,
tool_or_widget}), instruction.

sg[i] is paired with ka[i]; on length mismatch we keep
``min(len(sg), len(ka))`` and drop the trailing entries.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents.structagent.memory.offline.load_pool.loaders import pool_all_sources

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUT_PATH = REPO_ROOT / "results" / "unified_memory" / "_l1_segments_raw.jsonl"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading unified pool...")
    pool = pool_all_sources()
    logger.info(f"  pool size: {len(pool)}")

    n_written = 0
    n_dropped_empty = 0
    with out.open("w") as f:
        for rec in pool:
            sg = rec.subgoals or []
            ka = rec.key_actions or []
            n_pairs = min(len(sg), len(ka))
            if n_pairs == 0:
                n_dropped_empty += 1
                continue
            for i in range(n_pairs):
                subgoal_text = str(sg[i]).strip()
                if not subgoal_text:
                    continue
                action = ka[i]
                if isinstance(action, dict):
                    action_payload = {
                        "action": str(action.get("action") or
                                      action.get("action_summary") or "")[:400],
                        "tool_or_widget": str(action.get("tool_or_widget")
                                              or "")[:200],
                    }
                else:
                    action_payload = {"action": str(action)[:400],
                                      "tool_or_widget": ""}
                seg = {
                    "task_hash": rec.task_hash,
                    "task_id": rec.task_id,
                    "domain": rec.domain,
                    "source": rec.raw_meta.get("_source") or "?",
                    "seg_idx": i,
                    "subgoal": subgoal_text[:300],
                    "key_action": action_payload,
                    "instruction": (rec.instruction or "")[:400],
                }
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
                n_written += 1

    logger.info(f"  records skipped (no subgoals/actions): {n_dropped_empty}")
    logger.info(f"  segments written: {n_written}")
    logger.info(f"  output: {out}")


if __name__ == "__main__":
    main()

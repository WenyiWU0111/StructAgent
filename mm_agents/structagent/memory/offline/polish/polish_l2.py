"""Phase 3 — L2 plan-template polish for the unified-pool tight-tier clusters.

For each cluster (n>=3) under ``results/unified_memory/_clusters_v3_tight/``,
ask the polisher to emit **one OR MORE** L2 skills, depending on how
many distinct plan families it sees among the member trajectories. A
30-member impress cluster could legitimately hold a "format slide title"
skill + a "format slide background" skill + a few unassigned outliers;
forcing one skill per cluster would average those into a useless union.

Output per cluster: a JSON record with ``skills: [...]`` (variable
length) plus ``unassigned_task_hashes`` for any outlier task that fits
no skill. Each skill carries the same field set as the v2 single-skill
output so downstream retrieval/index code doesn't have to fork.

Workflow per preview_then_batch memory:
  1. ``--render-only --cluster X``  dump prompt for ONE cluster, no LLM
  2. ``--max 3``                    polish 3 clusters; eyeball the JSON
  3. ``--full``                     polish every multi-member cluster

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l2 \\
      --render-only --cluster cluster_0000
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l2 \\
      --max 3
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l2 \\
      --full --workers 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import openai

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents.structagent.memory.offline.load_pool.loaders import (
    UnifiedRecord, pool_all_sources,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLUSTERS_DIR = REPO_ROOT / "results" / "unified_memory" / "_clusters_v3_tight"
OUT_DIR = REPO_ROOT / "results" / "unified_memory" / "_l2_skills_v3"

# Sonnet by default per user preference. Override via UNIFIED_POLISH_MODEL.
DEFAULT_MODEL = os.environ.get("UNIFIED_POLISH_MODEL",
                               "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MIN_CLUSTER_SIZE = 3   # n<3 → too thin to abstract; skip


# ─── dotenv load ─────────────────────────────────────────────────────────────
def _load_dotenv() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


# ─── Prompt ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are synthesising L2 SKILLS — reusable plan templates — by abstracting
across N trajectories the clustering pipeline judged semantically similar.

Crucially: the cluster may contain ONE plan family or MULTIPLE distinct
plan families. Your first job is to decide HOW MANY skills this cluster
should produce. Then emit each skill in strict JSON.

GROUPING RULES
- If all N trajectories share the same skeleton (same major subgoals in
  the same order, just different parameters), emit ONE skill.
- If you see 2-3 clearly distinct plan families (e.g. some trajectories
  format slide TITLES while others format slide BACKGROUNDS, and the
  major subgoals diverge), emit ONE skill PER family.
- Do NOT split aggressively. A skill needs at least 2 assigned
  trajectories. If a single trajectory looks unique, leave it in
  ``unassigned_task_hashes`` rather than creating a singleton skill.
- Aim for 1-4 skills per cluster. More than 4 means the upstream
  clustering was too loose; flag this via ``cluster_coherence: "low"``
  + ``cluster_split_suggestion``.

OUTPUT — strict JSON, fenced; no commentary outside:
```json
{
  "cluster_coherence": "high | medium | low",
  "cluster_split_suggestion": "<empty unless coherence=low — name the sub-families>",

  "skills": [
    {
      "skill_id": "<snake_case, descriptive, short>",
      "task_class": "<short label naming this plan family>",
      "when_to_use": "<1-2 sentences — when the planner should retrieve this>",
      "plan_template": [
        {
          "subgoal": "<concise — what this step accomplishes>",
          "typical_actions": [
            "<concrete action, e.g. 'click File > Open Recent' or 'pkill -f chrome'>"
          ],
          "historical_coverage": "X/M",
          "alternative_approaches": [
            "<other actions that also worked for this subgoal, or empty list>"
          ]
        }
      ],
      "typical_step_count": <int — median across the M assigned trajectories>,
      "attached_pitfalls": [
        {
          "pitfall": "<what to avoid (1 line)>",
          "why": "<why it bites>",
          "observed_in_n_tasks": <int>
        }
      ],
      "source_task_hashes": ["<12-char hash>", "<12-char hash>", ...],
      "n_assigned": <int — must equal len(source_task_hashes), >= 2>
    }
  ],

  "unassigned_task_hashes": ["<task_hash>", "<task_hash>", ...]
}
```

PRINCIPLES
- ``cluster_coherence`` reports the cluster as a whole:
  - "high"   = one dominant skill claiming most/all trajectories
  - "medium" = 2-3 skills with comparable sizes
  - "low"    = 4+ skills or many unassigned — upstream cluster too loose
- Every task_hash from the input MUST appear EXACTLY ONCE across all
  skills' ``source_task_hashes`` lists + ``unassigned_task_hashes``.
- ``historical_coverage="X/M"`` uses M = the skill's n_assigned, NOT the
  whole cluster N.
- Be CONCRETE about the PROCEDURE in ``typical_actions`` — name the actual
  widgets / hotkeys / command structure. Generic "do the thing" is rejected.
  (But generalize task-specific VALUES — see ABSTRACTION below.)
- Identifiers MUST be the 12-char ``task_hash``, NOT the longer original
  task_id. The cluster member feed gives you task_hash directly.

ABSTRACTION — write a REUSABLE plan template, not a task-specific record:
- Keep the PROCEDURE concrete: real widgets, menus, hotkeys, command STRUCTURE
  (e.g. =SUM(<range>), `wget -O <dest> <url>`).
- REPLACE task-instance VALUES with natural general placeholders: specific search
  queries, product / brand / site names, file names, paths, cell values, emails,
  URLs, dates, numbers. Write  "search for the target product"  not  'search for
  "Women's Nike Jerseys"';  "open the target file"  not  "open 8.xlsx".
- ``task_class`` and every ``subgoal`` must read the SAME for ANY task in this
  class. If swapping in a different task would change a word, generalize it.
- Use natural placeholder phrases, NOT slot tokens ("the target file", not
  "{file}").
"""


def build_user_prompt(cluster_id: str,
                      cluster_meta: Dict[str, Any],
                      records: List[UnifiedRecord]) -> str:
    lines = [
        f"# Cluster: {cluster_id}",
        f"# Member trajectories: {len(records)}",
        f"# Domain distribution: {cluster_meta.get('domain_distribution')}",
        f"# Source distribution: {cluster_meta.get('source_raw_distribution')}",
        "",
    ]
    for i, rec in enumerate(records, 1):
        lines.append(f"## Trajectory {i}/{len(records)}  "
                     f"task_hash={rec.task_hash}  "
                     f"domain={rec.domain}  "
                     f"source={rec.raw_meta.get('_source')}")
        lines.append(f"  instruction: {(rec.instruction or '')[:400]}")
        lines.append(f"  task_completed: {rec.task_completed}")
        if rec.subgoals:
            lines.append(f"  subgoals ({len(rec.subgoals)}):")
            for sg in rec.subgoals[:10]:
                lines.append(f"    - {str(sg)[:200]}")
        if rec.key_actions:
            lines.append(f"  key_actions ({len(rec.key_actions)}):")
            for ka in rec.key_actions[:10]:
                if isinstance(ka, dict):
                    act = ka.get("action") or ka.get("action_summary") or ""
                    tool = ka.get("tool_or_widget") or ""
                    suffix = f"  [via {tool[:60]}]" if tool else ""
                    lines.append(f"    - {str(act)[:200]}{suffix}")
                else:
                    lines.append(f"    - {str(ka)[:200]}")
        if rec.pitfalls:
            lines.append(f"  pitfalls ({len(rec.pitfalls)}):")
            for p in rec.pitfalls[:5]:
                lines.append(f"    - {str(p)[:200]}")
        lines.append("")
    lines.append("Synthesise L2 skill(s) per the system prompt.")
    return "\n".join(lines)


# ─── LLM dispatch ───────────────────────────────────────────────────────────
def call_model(system_prompt: str, user_text: str,
               model: str = DEFAULT_MODEL,
               max_tokens: int = 6000,
               temperature: float = 0.0,
               max_retries: int = 3) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    client = openai.OpenAI(base_url=OPENROUTER_BASE, api_key=api_key)
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Model call failed after {max_retries}: {last_err}")


def parse_response(text: str) -> Any:
    blocks = re.findall(r"```json\s*\n(.*?)\n```", text, re.DOTALL)
    if blocks:
        return json.loads(blocks[0])
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Could not parse JSON from response")


# ─── Pool indexing ──────────────────────────────────────────────────────────
def build_hash_index() -> Dict[str, UnifiedRecord]:
    pool = pool_all_sources()
    return {r.task_hash: r for r in pool}


# ─── Cluster processing ────────────────────────────────────────────────────
def load_cluster(cluster_id: str,
                 hash_index: Dict[str, UnifiedRecord]
                 ) -> Tuple[List[UnifiedRecord], Dict[str, Any]]:
    cluster_dir = CLUSTERS_DIR / cluster_id
    members = json.loads((cluster_dir / "members.json").read_text())
    records: List[UnifiedRecord] = []
    for m in members:
        rec = hash_index.get(m["task_hash"])
        if rec is None:
            continue
        records.append(rec)
    from collections import Counter
    meta = {
        "domain_distribution": dict(Counter(r.domain for r in records)),
        "source_raw_distribution": dict(Counter(
            r.raw_meta.get("_source") or "?" for r in records)),
    }
    return records, meta


def process_cluster(cluster_id: str,
                    hash_index: Dict[str, UnifiedRecord],
                    *, model: str, force: bool, render_only: bool,
                    ) -> Dict[str, Any]:
    records, meta = load_cluster(cluster_id, hash_index)
    if len(records) < MIN_CLUSTER_SIZE:
        return {"cluster_id": cluster_id, "status": "skip-too-few",
                "n_records": len(records)}

    user_prompt = build_user_prompt(cluster_id, meta, records)

    if render_only:
        return {"cluster_id": cluster_id, "status": "rendered",
                "n_records": len(records),
                "prompt_chars": len(user_prompt),
                "rendered_prompt": user_prompt}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{cluster_id}.json"
    if out_path.exists() and not force:
        return {"cluster_id": cluster_id, "status": "skip-already-done",
                "path": str(out_path)}

    t0 = time.time()
    try:
        raw = call_model(SYSTEM_PROMPT, user_prompt, model=model)
    except Exception as e:
        return {"cluster_id": cluster_id, "status": "llm-error",
                "error": str(e)}
    elapsed = time.time() - t0

    try:
        parsed = parse_response(raw)
    except Exception as e:
        return {"cluster_id": cluster_id, "status": "parse-error",
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "raw_head": raw[:400]}

    if not isinstance(parsed, dict) or "skills" not in parsed:
        return {"cluster_id": cluster_id, "status": "bad-shape",
                "elapsed_s": round(elapsed, 1)}

    # Validate: every task_hash accounted for exactly once
    input_hashes = {r.task_hash for r in records}
    seen: List[str] = []
    for sk in parsed.get("skills") or []:
        seen.extend(sk.get("source_task_hashes") or [])
    seen.extend(parsed.get("unassigned_task_hashes") or [])
    seen_set = set(seen)
    missing = input_hashes - seen_set
    extra = seen_set - input_hashes
    duplicates = [h for h in seen if seen.count(h) > 1]

    record = {
        "cluster_id": cluster_id,
        "source_tier": "tight",
        "n_input_records": len(records),
        "cluster_coherence": parsed.get("cluster_coherence"),
        "cluster_split_suggestion": parsed.get("cluster_split_suggestion"),
        "n_skills": len(parsed.get("skills") or []),
        "skills": parsed.get("skills") or [],
        "unassigned_task_hashes": parsed.get("unassigned_task_hashes") or [],
        "assignment_audit": {
            "missing_input_hashes": sorted(missing),
            "extra_unknown_hashes": sorted(extra),
            "duplicate_hashes": sorted(set(duplicates)),
        },
        "raw_response": raw,
        "model": model,
        "elapsed_s": round(elapsed, 1),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {"cluster_id": cluster_id, "status": "extracted",
            "n_skills": record["n_skills"],
            "n_unassigned": len(record["unassigned_task_hashes"]),
            "coherence": parsed.get("cluster_coherence"),
            "audit_clean": (not missing and not extra and not duplicates),
            "elapsed_s": round(elapsed, 1),
            "path": str(out_path)}


# ─── Cluster selection ────────────────────────────────────────────────────
def select_clusters(args) -> List[str]:
    summary = json.loads((CLUSTERS_DIR / "_summary.json").read_text())
    polishable = [c for c in summary["clusters"]
                  if c["n_members"] >= MIN_CLUSTER_SIZE]
    polishable.sort(key=lambda c: -c["n_members"])

    if args.cluster:
        return [args.cluster]
    if args.clusters:
        return args.clusters.split(",")
    if args.full:
        return [c["cluster_id"] for c in polishable]
    if args.max is not None:
        return [c["cluster_id"] for c in polishable[:args.max]]
    raise SystemExit("specify one of: --cluster / --clusters / --max / --full")


def main() -> None:
    p = argparse.ArgumentParser()
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--cluster", type=str)
    sel.add_argument("--clusters", type=str,
                     help="Comma-separated cluster ids")
    sel.add_argument("--max", type=int,
                     help="Top-N clusters by size (preview)")
    sel.add_argument("--full", action="store_true")

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--render-only", action="store_true",
                   help="Render the prompt only; no LLM call")
    args = p.parse_args()

    ids = select_clusters(args)
    logger.info(f"selected {len(ids)} cluster(s); "
                f"model={args.model}; workers={args.workers}; "
                f"render-only={args.render_only}")

    if not args.render_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    hash_index = build_hash_index()
    logger.info(f"hash index size: {len(hash_index)}")

    if args.render_only:
        for cid in ids:
            r = process_cluster(cid, hash_index, model=args.model,
                                force=args.force, render_only=True)
            print(f"\n{'='*78}")
            print(f"PROMPT for {cid}  (n_records={r.get('n_records')}, "
                  f"chars={r.get('prompt_chars')})")
            print('='*78)
            print(r.get("rendered_prompt", ""))
        return

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    if args.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_cluster, cid, hash_index,
                              model=args.model, force=args.force,
                              render_only=False): cid for cid in ids}
            for f in cf.as_completed(futs):
                r = f.result()
                results.append(r)
                logger.info(f"  [{len(results)}/{len(ids)}] "
                            f"{r.get('cluster_id')}  {r.get('status')}  "
                            f"skills={r.get('n_skills', '-')}  "
                            f"coh={r.get('coherence','-')}  "
                            f"audit={r.get('audit_clean')}")
    else:
        for i, cid in enumerate(ids, 1):
            r = process_cluster(cid, hash_index, model=args.model,
                                force=args.force, render_only=False)
            results.append(r)
            logger.info(f"  [{i}/{len(ids)}] {r.get('cluster_id')}  "
                        f"{r.get('status')}  "
                        f"skills={r.get('n_skills','-')}  "
                        f"coh={r.get('coherence','-')}  "
                        f"audit={r.get('audit_clean')}")

    elapsed = time.time() - t0
    summary = {
        "model": args.model,
        "n_clusters_processed": len(results),
        "n_extracted": sum(1 for r in results if r.get("status") == "extracted"),
        "n_skipped": sum(1 for r in results
                         if r.get("status", "").startswith("skip")),
        "n_errors": sum(1 for r in results
                        if r.get("status") in ("llm-error",
                                                "parse-error",
                                                "bad-shape")),
        "total_skills_produced": sum(r.get("n_skills") or 0
                                     for r in results),
        "n_with_audit_issues": sum(1 for r in results
                                    if r.get("status") == "extracted"
                                    and not r.get("audit_clean")),
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    (OUT_DIR / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"DONE — {summary['n_extracted']}/{len(results)} extracted "
                f"({summary['total_skills_produced']} skills total), "
                f"{summary['n_errors']} errors, "
                f"{summary['n_with_audit_issues']} with audit issues. "
                f"elapsed {elapsed:.1f}s. output: {OUT_DIR}")


if __name__ == "__main__":
    main()

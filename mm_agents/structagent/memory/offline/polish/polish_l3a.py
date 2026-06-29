"""Phase 3 — L3a cluster-rule polish over the merged loose-tier clusters.

L3a is the *cluster-rule* tier: high-level, often-cross-domain "common
wisdom" extracted from a sizeable cluster. Unlike L2 (which produces
step-by-step plan templates the planner replays), L3a produces
ACTIONABLE INSIGHT — invariants, gotchas, design patterns that span the
cluster's trajectories.

Input directory defaults to ``_clusters_v3_loose_merged/`` — the
2nd-tier merged cluster set produced by ``merge_small_loose_clusters``.
That set is "big_kept + merged_super + small_kept" combined; we polish
every cluster with n >= MIN_CLUSTER_SIZE in one pass.

Example: a 47-member merged super-cluster combining 19 small "align
textbox / shape" niches on impress slides yields rules like:

  - "Impress textbox alignment lives in Properties > Position and
     Size; the toolbar Align dropdown applies to slide-level grouping,
     NOT in-textbox content"
  - "Triple-click selects all text in a textbox; single-click only
     positions cursor — alignment commands need the all-selected state"
  - "Shape vs text alignment use different menu paths — Format > Object
     > Position for shape, Format > Paragraph for text"

vs L2 which would produce a single concrete plan for ONE specific kind
of alignment.

Workflow (preview-then-batch):
  1. ``--render-only --cluster cluster_0000``  dump prompt, no LLM
  2. ``--max 3``                              polish 3 clusters, eyeball
  3. ``--full``                               polish every n>=MIN

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l3a \\
      --render-only --cluster cluster_0000
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l3a \\
      --max 3
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l3a \\
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
from pathlib import Path
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

DEFAULT_CLUSTERS_DIR = (REPO_ROOT / "results" / "unified_memory"
                        / "_clusters_v3_loose_merged")
OUT_DIR = REPO_ROOT / "results" / "unified_memory" / "_l3a_cluster_rules_v3"

DEFAULT_MODEL = os.environ.get("UNIFIED_POLISH_MODEL",
                               "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MIN_CLUSTER_SIZE = 5   # default — match merge threshold


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


SYSTEM_PROMPT = """\
You are extracting CLUSTER-RULE knowledge — high-level wisdom that helps
an agent succeed on a CLASS of tasks. The input is N trajectories the
clustering pipeline grouped together (sometimes via a 2nd-tier merge of
near-miss small clusters).

This is NOT a plan template (L2's job). Your job is to extract
ACTIONABLE INSIGHTS:

  - INVARIANTS that hold across trajectories ("always opens a file
    dialog regardless of which app")
  - GOTCHAS that bit multiple trajectories ("filename special chars
    fail silently")
  - DESIGN PATTERNS ("rentals queries usually need explicit state code
    to disambiguate city names")
  - WIDGET / MENU / DIALOG conventions shared across trajectories
  - INPUT-DERIVATION rules ("the :city slot usually needs appending
    state code on US sites to disambiguate")

Identify whether the cluster is SINGLE-DOMAIN or CROSS-DOMAIN; if
cross-domain, lean into rules that hold across the platforms. Mark each
rule's scope accordingly.

OUTPUT — strict JSON, fenced; no commentary outside:
```json
{
  "cluster_summary": "<1-2 sentences describing the task class>",
  "cluster_coherence": "high | medium | low",
  "cross_domain_signal": true | false,
  "n_domains_represented": <int>,

  "common_rules": [
    {
      "rule": "<one-line actionable insight a planner/actor can use>",
      "rationale": "<why this matters; reference observed behaviour>",
      "applies_when": "<scope — task type / domain / context>",
      "scope": "cross_domain | domain_specific",
      "category": "invariant | gotcha | design_pattern | widget_convention | input_derivation",
      "observed_in_n_tasks": <int>,
      "supporting_task_hashes": ["<12-char hash>", ...]
    }
  ],

  "shared_pitfalls": [
    {
      "pitfall": "<what bit trajectories — concrete>",
      "why": "<root cause>",
      "observed_in_n_tasks": <int>
    }
  ],

  "open_questions": [
    "<things you can't decide from this evidence; optional, can be empty>"
  ]
}
```

PRINCIPLES
- Rules MUST be CONCRETE and ACTIONABLE. "X always opens dialog Y",
  "click Z before W", "the :date slot needs MM/DD/YYYY format on this
  family of sites". NOT "be careful". NOT "verify things look right".
- ``observed_in_n_tasks`` reflects actual count in the input, NOT
  speculation.
- ``cluster_coherence``:
  - "high"   = trajectories share a clear common theme; rules
              consistently observable
  - "medium" = 2-3 sub-themes detectable; emit rules per theme,
              annotate scope
  - "low"    = trajectories don't cohere; emit fewer rules + flag in
              cluster_summary
- ``cross_domain_signal``: true if the cluster's records span 2+ domains
  AND the rules genuinely hold across those domains. False if rules are
  app-specific even though tasks span apps.
- ``supporting_task_hashes``: 1-5 hashes per rule, picked from the input.
  Use the 12-char task_hash given in each trajectory header.

RULE COUNT
- Emit AT MOST 10 ``common_rules`` and AT MOST 10 ``shared_pitfalls``.
  When the cluster yields more candidates than 10, KEEP the most
  high-leverage / most-broadly-applicable ones; merge or drop the rest.
- A rich 30-trajectory cluster typically produces 6-10 rules; a tight
  5-trajectory cluster typically yields 3-6.
- DO merge rules that say the same thing in different words.
- DO NOT artificially split one rule into many to inflate count.
- DO NOT artificially merge rules that differ in scope (cross_domain vs
  domain_specific), category, or actionability — pick the most useful
  one to keep instead.
"""


def build_user_prompt(cluster_id: str,
                      cluster_meta: Dict[str, Any],
                      records: List[UnifiedRecord]) -> str:
    lines = [
        f"# Cluster: {cluster_id}",
        f"# Member trajectories: {len(records)}",
        f"# Domain distribution: {cluster_meta.get('domain_distribution')}",
        f"# Source distribution: {cluster_meta.get('source_raw_distribution')}",
    ]
    prov = cluster_meta.get("provenance") or {}
    if prov.get("type") == "merged":
        lines.append(f"# Provenance: merged from "
                     f"{prov.get('n_constituent_small_clusters')} small "
                     f"clusters at centroid threshold "
                     f"{prov.get('centroid_threshold')}")
    elif prov.get("type") == "big_kept":
        lines.append(f"# Provenance: kept from original loose cluster "
                     f"{prov.get('orig_cluster_id')} "
                     f"(n={prov.get('orig_n_members')})")
    lines.append("")
    for i, rec in enumerate(records, 1):
        lines.append(f"## Trajectory {i}/{len(records)}  "
                     f"task_hash={rec.task_hash}  "
                     f"domain={rec.domain}  "
                     f"source={rec.raw_meta.get('_source')}")
        lines.append(f"  instruction: {(rec.instruction or '')[:300]}")
        if rec.subgoals:
            lines.append(f"  subgoals ({len(rec.subgoals)}):")
            for sg in rec.subgoals[:8]:
                lines.append(f"    - {str(sg)[:160]}")
        if rec.key_actions:
            lines.append(f"  key_actions ({len(rec.key_actions)}):")
            for ka in rec.key_actions[:8]:
                if isinstance(ka, dict):
                    act = ka.get("action") or ka.get("action_summary") or ""
                    tool = ka.get("tool_or_widget") or ""
                    suffix = f"  [via {tool[:50]}]" if tool else ""
                    lines.append(f"    - {str(act)[:160]}{suffix}")
                else:
                    lines.append(f"    - {str(ka)[:160]}")
        if rec.pitfalls:
            lines.append(f"  pitfalls ({len(rec.pitfalls)}):")
            for p in rec.pitfalls[:4]:
                lines.append(f"    - {str(p)[:160]}")
        lines.append("")
    lines.append("Extract cluster rules per the system prompt.")
    return "\n".join(lines)


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


def build_hash_index() -> Dict[str, UnifiedRecord]:
    pool = pool_all_sources()
    return {r.task_hash: r for r in pool}


def load_cluster(cluster_id: str,
                 clusters_dir: Path,
                 hash_index: Dict[str, UnifiedRecord]
                 ) -> Tuple[List[UnifiedRecord], Dict[str, Any]]:
    cluster_dir = clusters_dir / cluster_id
    members = json.loads((cluster_dir / "members.json").read_text())
    records: List[UnifiedRecord] = []
    for m in members:
        rec = hash_index.get(m["task_hash"])
        if rec is None:
            continue
        records.append(rec)
    from collections import Counter
    summary = json.loads((clusters_dir / "_summary.json").read_text())
    cluster_meta = next((c for c in summary["clusters"]
                         if c["cluster_id"] == cluster_id), {})
    meta = {
        "domain_distribution": dict(Counter(r.domain for r in records)),
        "source_raw_distribution": dict(Counter(
            r.raw_meta.get("_source") or "?" for r in records)),
        "provenance": cluster_meta.get("provenance") or {},
    }
    return records, meta


def process_cluster(cluster_id: str,
                    clusters_dir: Path,
                    hash_index: Dict[str, UnifiedRecord],
                    *, model: str, force: bool, render_only: bool,
                    min_size: int,
                    ) -> Dict[str, Any]:
    records, meta = load_cluster(cluster_id, clusters_dir, hash_index)
    if len(records) < min_size:
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

    if not isinstance(parsed, dict) or "common_rules" not in parsed:
        return {"cluster_id": cluster_id, "status": "bad-shape",
                "elapsed_s": round(elapsed, 1)}

    input_hashes = {r.task_hash for r in records}
    unknown_hashes = set()
    for rule in parsed.get("common_rules") or []:
        for h in rule.get("supporting_task_hashes") or []:
            if h not in input_hashes:
                unknown_hashes.add(h)

    record = {
        "cluster_id": cluster_id,
        "source_clusters_dir": str(clusters_dir),
        "source_provenance": meta.get("provenance") or {},
        "n_input_records": len(records),
        "cluster_summary": parsed.get("cluster_summary"),
        "cluster_coherence": parsed.get("cluster_coherence"),
        "cross_domain_signal": parsed.get("cross_domain_signal"),
        "n_domains_represented": parsed.get("n_domains_represented"),
        "n_common_rules": len(parsed.get("common_rules") or []),
        "common_rules": parsed.get("common_rules") or [],
        "shared_pitfalls": parsed.get("shared_pitfalls") or [],
        "open_questions": parsed.get("open_questions") or [],
        "assignment_audit": {
            "unknown_supporting_hashes": sorted(unknown_hashes),
        },
        "raw_response": raw,
        "model": model,
        "elapsed_s": round(elapsed, 1),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {"cluster_id": cluster_id, "status": "extracted",
            "n_rules": record["n_common_rules"],
            "n_pitfalls": len(record["shared_pitfalls"]),
            "coherence": parsed.get("cluster_coherence"),
            "cross_domain": parsed.get("cross_domain_signal"),
            "audit_clean": not unknown_hashes,
            "elapsed_s": round(elapsed, 1),
            "path": str(out_path)}


def select_clusters(args, clusters_dir: Path) -> List[str]:
    summary = json.loads((clusters_dir / "_summary.json").read_text())
    polishable = [c for c in summary["clusters"]
                  if c["n_members"] >= args.min_size]
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

    p.add_argument("--clusters-dir", type=Path, default=DEFAULT_CLUSTERS_DIR)
    p.add_argument("--min-size", type=int, default=MIN_CLUSTER_SIZE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--render-only", action="store_true",
                   help="Render the prompt only; no LLM call")
    args = p.parse_args()

    clusters_dir: Path = args.clusters_dir
    ids = select_clusters(args, clusters_dir)
    logger.info(f"selected {len(ids)} cluster(s); "
                f"dir={clusters_dir.name}; "
                f"min_size={args.min_size}; "
                f"model={args.model}; workers={args.workers}; "
                f"render-only={args.render_only}")

    if not args.render_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
    hash_index = build_hash_index()
    logger.info(f"hash index size: {len(hash_index)}")

    if args.render_only:
        for cid in ids:
            r = process_cluster(cid, clusters_dir, hash_index,
                                model=args.model, force=args.force,
                                render_only=True, min_size=args.min_size)
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
            futs = {ex.submit(process_cluster, cid, clusters_dir, hash_index,
                              model=args.model, force=args.force,
                              render_only=False,
                              min_size=args.min_size): cid for cid in ids}
            for f in cf.as_completed(futs):
                r = f.result()
                results.append(r)
                logger.info(f"  [{len(results)}/{len(ids)}] "
                            f"{r.get('cluster_id')}  {r.get('status')}  "
                            f"rules={r.get('n_rules','-')}  "
                            f"coh={r.get('coherence','-')}  "
                            f"xdom={r.get('cross_domain','-')}  "
                            f"audit={r.get('audit_clean')}")
    else:
        for i, cid in enumerate(ids, 1):
            r = process_cluster(cid, clusters_dir, hash_index,
                                model=args.model, force=args.force,
                                render_only=False, min_size=args.min_size)
            results.append(r)
            logger.info(f"  [{i}/{len(ids)}] {r.get('cluster_id')}  "
                        f"{r.get('status')}  "
                        f"rules={r.get('n_rules','-')}  "
                        f"coh={r.get('coherence','-')}  "
                        f"xdom={r.get('cross_domain','-')}  "
                        f"audit={r.get('audit_clean')}")

    elapsed = time.time() - t0
    summary = {
        "clusters_dir": str(clusters_dir),
        "model": args.model,
        "min_cluster_size": args.min_size,
        "n_clusters_processed": len(results),
        "n_extracted": sum(1 for r in results if r.get("status") == "extracted"),
        "n_errors": sum(1 for r in results
                        if r.get("status") in ("llm-error",
                                                "parse-error",
                                                "bad-shape")),
        "total_rules_produced": sum(r.get("n_rules") or 0
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
                f"({summary['total_rules_produced']} rules total), "
                f"{summary['n_errors']} errors, "
                f"{summary['n_with_audit_issues']} with audit issues. "
                f"elapsed {elapsed:.1f}s. output: {OUT_DIR}")


if __name__ == "__main__":
    main()

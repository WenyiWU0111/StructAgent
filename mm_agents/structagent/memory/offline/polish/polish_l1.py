"""Phase 3 L1 — Step 3: polish each subgoal cluster into 1+ L1 typical actions.

A subgoal cluster groups segments with the same SUBGOAL intent (e.g.
275 segments all saying "Open the terminal application"). But the
underlying ACTION may vary — some open terminal via right-click menu,
others via Ctrl+Alt+T, others via launcher icon. We let the polisher
emit 1 OR MORE L1 actions per cluster, one per distinct APPROACH it
sees in the samples.

Output per cluster (under ``_l1_actions_v3/<domain>/cluster_NNNN.json``):
  - cluster_subgoal_pattern         (canonical subgoal text)
  - cluster_coherence               ("single_approach" / "multiple_approaches")
  - n_actions
  - actions: [ {action_id, approach, typical_actions, …}, … ]
  - unassigned_sample_task_hashes   (outlier samples)

For large clusters (n>15) we SAMPLE 15 stratified by source; the LLM
sees a diverse subsample but the full cluster's task_hash list is
preserved at top level so retrieval still attributes correctly.

Workflow (preview-then-batch):
  1. ``--render-only --domain X --cluster cluster_NNNN``
  2. ``--clusters X:cluster_A,X:cluster_B``           polish 2-3
  3. ``--full --workers 6``                           polish all n>=5

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l1 \\
      --render-only --domain libreoffice_calc --cluster cluster_0000
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l1 \\
      --clusters libreoffice_calc:cluster_0000,libreoffice_calc:cluster_0001
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l1 \\
      --full --workers 6
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import openai

from mm_agents.structagent._paths import REPO_ROOT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CLUSTERS_BASE = REPO_ROOT / "results" / "unified_memory" / "_l1_clusters_v3"
OUT_BASE = REPO_ROOT / "results" / "unified_memory" / "_l1_actions_v3"

DEFAULT_MODEL = os.environ.get("UNIFIED_POLISH_MODEL",
                               "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MIN_CLUSTER_SIZE = 5
MAX_SAMPLES_PER_CLUSTER = 15
MAX_ACTIONS_PER_CLUSTER = 4   # soft hint in prompt


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
You are extracting L1 ACTION TEMPLATES from N sampled trajectories that
share the SAME SUBGOAL (clustered by subgoal text). Crucially: same
subgoal may have MULTIPLE concrete approaches — opening a terminal
via keyboard shortcut vs right-click menu vs launcher icon counts as
THREE distinct L1 actions. Your job is to:

  1. Decide how many distinct APPROACHES appear in the samples (1 to ~4)
  2. Emit one L1 action per approach
  3. Place each sample task_hash under the action whose approach it
     followed; outliers go to ``unassigned_sample_task_hashes``

If the cluster is uniform (every sample took the same approach), emit
ONE action. If samples diverge (e.g. some used menu, some used CLI,
some used a hotkey), emit ONE PER divergent approach.

OUTPUT — strict JSON, fenced; no commentary outside:
```json
{
  "cluster_subgoal_pattern": "<canonical subgoal text>",
  "cluster_coherence": "single_approach | multiple_approaches",

  "actions": [
    {
      "action_id": "<snake_case, descriptive, short>",
      "approach": "<short label naming this approach, e.g. 'via_keyboard_shortcut', 'via_menu', 'via_cli'>",
      "when_to_use": "<1 sentence: when planner/actor should pick THIS approach>",
      "typical_actions": [
        "<concrete action step, e.g. 'Press Ctrl+Alt+T' or 'Right-click desktop > Open Terminal Here'>"
      ],
      "tools_or_widgets_seen": [
        "<widget / menu / dialog / shortcut used by this approach>"
      ],
      "common_pitfalls": [
        {
          "pitfall": "<what bit trajectories using THIS approach>",
          "why": "<root cause>",
          "observed_in_n_samples": <int>
        }
      ],
      "n_assigned_samples": <int — must equal len(assigned_sample_task_hashes), >= 1>,
      "assigned_sample_task_hashes": ["<12-char hash>", ...]
    }
  ],

  "unassigned_sample_task_hashes": ["<task_hash>", ...]
}
```

PRINCIPLES
- ``typical_actions`` CONCRETE about the PROCEDURE — name actual widgets /
  hotkeys / command structure. NOT "navigate to settings somehow". (But use
  general placeholders for task-specific VALUES — see ABSTRACTION below.)
- ``approach`` short labels: "via_menu", "via_shortcut", "via_cli",
  "via_toolbar", "via_dialog", "via_context_menu", etc.
- Every sample task_hash MUST appear EXACTLY ONCE across all actions'
  ``assigned_sample_task_hashes`` + ``unassigned_sample_task_hashes``.

CAPS
- AT MOST 4 actions per cluster (force a merge / drop minor variants
  if you see more — keep the highest-impact approaches)
- ``typical_actions``: 2-5 per action
- ``tools_or_widgets_seen``: 1-8 per action
- ``common_pitfalls``: 0-5 per action

Merge near-dupes. Keep distinct approaches distinct.

ABSTRACTION — write REUSABLE templates, not task-specific records:
- Keep the PROCEDURE concrete: real widgets, menus, hotkeys, and command
  STRUCTURE (e.g. =SUM(<range>), `wget -O <dest> <url>`).
- REPLACE task-instance VALUES with natural general placeholders: specific search
  queries, product / brand / site names, file names, paths, cell values, emails,
  URLs, dates, numbers. Write  "type the product query"  not  'type "Women's Nike
  Jerseys"';  "open the target file"  not  "open 8.xlsx";  "fill with the chosen
  color"  not  "fill with green".
- A correct entry reads the SAME for ANY task in this class. If swapping in a
  different task would change a word, that word is a value — generalize it.
- Use natural placeholder phrases, NOT slot tokens ("the product query", not
  "{product}").
"""


def sample_members(members: List[Dict[str, Any]],
                   cap: int = MAX_SAMPLES_PER_CLUSTER
                   ) -> List[Dict[str, Any]]:
    """Sample up to `cap` segments, biased toward source diversity."""
    if len(members) <= cap:
        return list(members)
    by_src: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in members:
        by_src[m.get("source") or "?"].append(m)
    weights = {s: math.sqrt(len(ms)) for s, ms in by_src.items()}
    total_weight = sum(weights.values())
    sample: List[Dict[str, Any]] = []
    rng = random.Random(42)
    for s, ms in by_src.items():
        n_alloc = max(1, round(cap * weights[s] / total_weight))
        sample.extend(rng.sample(ms, min(n_alloc, len(ms))))
    return sample[:cap]


def build_user_prompt(domain: str, cluster_id: str,
                      samples: List[Dict[str, Any]],
                      full_size: int) -> str:
    lines = [
        f"# Domain: {domain}",
        f"# Cluster: {cluster_id}",
        f"# Full cluster size: {full_size}  "
        f"(showing {len(samples)} sampled segments below)",
        "",
    ]
    src_counts = Counter(m.get("source") for m in samples)
    lines.append(f"# Sample source distribution: {dict(src_counts)}")
    lines.append("")
    for i, m in enumerate(samples, 1):
        lines.append(f"## Sample {i}/{len(samples)}  "
                     f"task_hash={m.get('task_hash')}  "
                     f"source={m.get('source')}")
        lines.append(f"  subgoal:     {(m.get('subgoal') or '')[:200]}")
        ka = m.get("key_action") or {}
        if isinstance(ka, dict):
            act = ka.get("action") or ""
            tool = ka.get("tool_or_widget") or ""
            lines.append(f"  key_action:  {str(act)[:200]}")
            if tool:
                lines.append(f"  via:         {tool[:120]}")
        else:
            lines.append(f"  key_action:  {str(ka)[:200]}")
        inst = m.get("instruction") or ""
        if inst:
            lines.append(f"  task_inst:   {inst[:120]}")
        lines.append("")
    lines.append(f"Extract L1 action(s) for this subgoal cluster.")
    return "\n".join(lines)


def call_model(system_prompt: str, user_text: str,
               model: str = DEFAULT_MODEL,
               max_tokens: int = 4500,
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


def load_cluster(domain: str, cluster_id: str
                 ) -> List[Dict[str, Any]]:
    fp = CLUSTERS_BASE / domain / cluster_id / "members.json"
    return json.loads(fp.read_text())


def process_cluster(domain: str, cluster_id: str,
                    *, model: str, force: bool, render_only: bool,
                    ) -> Dict[str, Any]:
    members_full = load_cluster(domain, cluster_id)
    full_size = len(members_full)
    if full_size < MIN_CLUSTER_SIZE:
        return {"domain": domain, "cluster_id": cluster_id,
                "status": "skip-too-few", "n_members": full_size}

    samples = sample_members(members_full)
    user_prompt = build_user_prompt(domain, cluster_id, samples, full_size)

    if render_only:
        return {"domain": domain, "cluster_id": cluster_id,
                "status": "rendered",
                "n_members": full_size, "n_samples": len(samples),
                "prompt_chars": len(user_prompt),
                "rendered_prompt": user_prompt}

    out_dir = OUT_BASE / domain
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cluster_id}.json"
    if out_path.exists() and not force:
        return {"domain": domain, "cluster_id": cluster_id,
                "status": "skip-already-done", "path": str(out_path)}

    t0 = time.time()
    try:
        raw = call_model(SYSTEM_PROMPT, user_prompt, model=model)
    except Exception as e:
        return {"domain": domain, "cluster_id": cluster_id,
                "status": "llm-error", "error": str(e)}
    elapsed = time.time() - t0

    try:
        parsed = parse_response(raw)
    except Exception as e:
        return {"domain": domain, "cluster_id": cluster_id,
                "status": "parse-error", "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "raw_head": raw[:400]}

    if not isinstance(parsed, dict) or "actions" not in parsed:
        return {"domain": domain, "cluster_id": cluster_id,
                "status": "bad-shape", "elapsed_s": round(elapsed, 1)}

    # Audit on the SAMPLES (not full cluster — LLM only saw samples)
    sample_hashes = {m["task_hash"] for m in samples}
    seen: List[str] = []
    for a in parsed.get("actions") or []:
        seen.extend(a.get("assigned_sample_task_hashes") or [])
    seen.extend(parsed.get("unassigned_sample_task_hashes") or [])
    seen_set = set(seen)
    missing = sample_hashes - seen_set
    extra = seen_set - sample_hashes
    duplicates = [h for h in seen if seen.count(h) > 1]

    record = {
        "domain": domain,
        "source_cluster_id": cluster_id,
        "n_assigned_segments": full_size,
        "n_samples_shown_to_llm": len(samples),
        "full_cluster_task_hashes": sorted({m["task_hash"]
                                            for m in members_full}),
        "cluster_subgoal_pattern": parsed.get("cluster_subgoal_pattern"),
        "cluster_coherence": parsed.get("cluster_coherence"),
        "n_actions": len(parsed.get("actions") or []),
        "actions": parsed.get("actions") or [],
        "unassigned_sample_task_hashes":
            parsed.get("unassigned_sample_task_hashes") or [],
        "assignment_audit": {
            "missing_sample_hashes": sorted(missing),
            "extra_unknown_hashes": sorted(extra),
            "duplicate_hashes": sorted(set(duplicates)),
        },
        "raw_response": raw,
        "model": model,
        "elapsed_s": round(elapsed, 1),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {"domain": domain, "cluster_id": cluster_id,
            "status": "extracted",
            "n_actions": record["n_actions"],
            "coherence": parsed.get("cluster_coherence"),
            "audit_clean": (not missing and not extra and not duplicates),
            "elapsed_s": round(elapsed, 1),
            "path": str(out_path)}


def select_clusters(args) -> List[Tuple[str, str]]:
    if args.cluster and not args.domain:
        raise SystemExit("--cluster requires --domain")
    if args.cluster:
        return [(args.domain, args.cluster)]
    if args.clusters:
        out = []
        for token in args.clusters.split(","):
            d, c = token.split(":")
            out.append((d.strip(), c.strip()))
        return out
    if args.full or args.max:
        pairs: List[Tuple[str, str, int]] = []
        for dom_dir in sorted(CLUSTERS_BASE.iterdir()):
            if not dom_dir.is_dir():
                continue
            s_path = dom_dir / "_summary.json"
            if not s_path.exists():
                continue
            s = json.loads(s_path.read_text())
            for c in s["clusters"]:
                if c["n_members"] >= MIN_CLUSTER_SIZE:
                    pairs.append((dom_dir.name, c["cluster_id"],
                                  c["n_members"]))
        pairs.sort(key=lambda t: -t[2])
        pairs_dc = [(d, c) for d, c, _ in pairs]
        if args.max:
            pairs_dc = pairs_dc[:args.max]
        return pairs_dc
    raise SystemExit("specify one of: --cluster / --clusters / --max / --full")


def main() -> None:
    p = argparse.ArgumentParser()
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--cluster", type=str,
                     help="Single cluster (requires --domain)")
    sel.add_argument("--clusters", type=str,
                     help="Comma-separated 'domain:cluster' pairs")
    sel.add_argument("--max", type=int,
                     help="Top-N clusters by size across all domains")
    sel.add_argument("--full", action="store_true")

    p.add_argument("--domain", type=str)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--render-only", action="store_true",
                   help="Render the prompt only; no LLM call")
    args = p.parse_args()

    pairs = select_clusters(args)
    logger.info(f"selected {len(pairs)} cluster(s); "
                f"model={args.model}; workers={args.workers}; "
                f"render-only={args.render_only}")

    if not args.render_only:
        OUT_BASE.mkdir(parents=True, exist_ok=True)

    if args.render_only:
        for dom, cid in pairs:
            r = process_cluster(dom, cid, model=args.model,
                                force=args.force, render_only=True)
            print(f"\n{'='*78}")
            print(f"PROMPT for {dom}/{cid}  "
                  f"(n_members={r.get('n_members')}, "
                  f"n_samples={r.get('n_samples')}, "
                  f"chars={r.get('prompt_chars')})")
            print('='*78)
            print(r.get("rendered_prompt", ""))
        return

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    if args.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_cluster, dom, cid,
                              model=args.model, force=args.force,
                              render_only=False): (dom, cid)
                    for dom, cid in pairs}
            for f in cf.as_completed(futs):
                r = f.result()
                results.append(r)
                logger.info(f"  [{len(results)}/{len(pairs)}] "
                            f"{r.get('domain')}/{r.get('cluster_id')}  "
                            f"{r.get('status')}  "
                            f"n_actions={r.get('n_actions','-')}  "
                            f"coh={r.get('coherence','-')}  "
                            f"audit={r.get('audit_clean')}")
    else:
        for i, (dom, cid) in enumerate(pairs, 1):
            r = process_cluster(dom, cid, model=args.model,
                                force=args.force, render_only=False)
            results.append(r)
            logger.info(f"  [{i}/{len(pairs)}] "
                        f"{r.get('domain')}/{r.get('cluster_id')}  "
                        f"{r.get('status')}  "
                        f"n_actions={r.get('n_actions','-')}")

    elapsed = time.time() - t0
    summary = {
        "model": args.model,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "max_samples_per_cluster": MAX_SAMPLES_PER_CLUSTER,
        "n_clusters_processed": len(results),
        "n_extracted": sum(1 for r in results if r.get("status") == "extracted"),
        "n_errors": sum(1 for r in results
                        if r.get("status") in ("llm-error",
                                                "parse-error",
                                                "bad-shape")),
        "total_actions_produced": sum(r.get("n_actions") or 0
                                      for r in results),
        "n_with_audit_issues": sum(1 for r in results
                                    if r.get("status") == "extracted"
                                    and not r.get("audit_clean")),
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    (OUT_BASE / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"DONE — {summary['n_extracted']}/{len(results)} extracted "
                f"({summary['total_actions_produced']} actions total), "
                f"{summary['n_errors']} errors, "
                f"{summary['n_with_audit_issues']} with audit issues. "
                f"elapsed {elapsed:.1f}s. output: {OUT_BASE}")


if __name__ == "__main__":
    main()

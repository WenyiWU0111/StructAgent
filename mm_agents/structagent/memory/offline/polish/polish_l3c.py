"""Phase 3 — L3c domain-rule polish: per-domain meta-aggregation over L2 skills.

For each domain, gather its L2 skills (by dominant member-domain) and
meta-polish them into one "domain rule sheet": invariants, common
widgets/dialogs, shortcuts, and pitfalls that hold across the domain
regardless of the specific plan.

Coarsest tier (one emit per domain), above L2 (per-plan) and L3a
(per-cluster). Per-skill text is truncated hard since Sonnet sees
30-150 skill summaries in one prompt.

Run:
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l3c \\
      --render-only --domain libreoffice_calc   # dump prompt only
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l3c \\
      --domain libreoffice_calc                 # one domain
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.polish.polish_l3c \\
      --all --workers 4                         # all domains >= MIN_DOMAIN_SKILLS
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import os
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import openai

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents.structagent.memory.offline.load_pool.loaders import pool_all_sources

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

L2_DIR = REPO_ROOT / "results" / "unified_memory" / "_l2_skills_v3"
OUT_DIR = REPO_ROOT / "results" / "unified_memory" / "_l3c_domain_rules_v3"

DEFAULT_MODEL = os.environ.get("UNIFIED_POLISH_MODEL",
                               "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

MIN_DOMAIN_SKILLS = 5   # domains with fewer skills are too thin to meta-polish


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
You are extracting DOMAIN RULES — what's always true / commonly true
across an entire application or task domain (e.g. libreoffice_calc,
chrome, vs_code). The input is N L2 SKILLS — plan templates that have
already been abstracted across multiple trajectories. Your job is the
2nd-level abstraction: across these N L2 skills, identify the recurring
INVARIANTS, WIDGETS, SHORTCUTS, and PITFALLS that are characteristic of
the domain as a whole, not any particular skill.

OUTPUT — strict JSON, fenced; no commentary outside:
```json
{
  "domain_summary": "<1-2 sentences naming what kinds of tasks this domain covers in our corpus>",

  "domain_invariants": [
    {
      "invariant": "<one-line — what's always or near-always true in this domain>",
      "rationale": "<why; cite the recurring patterns you observed>",
      "observed_in_n_skills": <int>
    }
  ],

  "common_widgets": [
    {
      "widget": "<widget / menu / dialog name>",
      "purpose": "<what it's used for in this domain>",
      "tasks_using_it_approx_n": <int>
    }
  ],

  "common_shortcuts": [
    {
      "shortcut": "<key combo, e.g. Ctrl+S>",
      "purpose": "<what it does in this domain>",
      "domain_specific": <true | false>
    }
  ],

  "shared_pitfalls": [
    {
      "pitfall": "<recurring failure mode across skills>",
      "why": "<root cause>",
      "observed_in_n_skills": <int>
    }
  ],

  "open_questions": [
    "<things you can't decide from this evidence; optional, can be empty>"
  ]
}
```

PRINCIPLES
- Be specific to the domain. "Press Ctrl+S to save" is generic; "In
  libreoffice_calc, F2 enters cell edit mode" is domain-specific.
- Cite ACTUAL evidence — if you claim "menu X is used", count how many
  skills reference it.
- Don't fabricate. If you don't see a widget across multiple skills,
  don't claim it's domain-common.
- ``domain_specific``: shortcuts marked false are universal (Ctrl+S /
  Ctrl+C / Ctrl+V); true is for shortcuts unique to this app
  (e.g. F2 for calc cell edit, Ctrl+J for vs_code commands).

CAPS — AT MOST 10 each:
- domain_invariants: AT MOST 10 (high-leverage only)
- common_widgets: AT MOST 10
- common_shortcuts: AT MOST 10
- shared_pitfalls: AT MOST 10

Merge near-dupes. Keep distinct entries distinct. Don't pad with
weak entries.
"""


def collect_l2_skills_by_domain() -> Dict[str, List[Dict[str, Any]]]:
    """Group L2 skills by dominant domain (via task_hashes → record.domain).
    Returns dict[domain -> list[skill]]."""
    logger.info("Loading unified pool for hash→domain lookup...")
    hash_idx = {r.task_hash: r for r in pool_all_sources()}

    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_files = 0
    n_skills = 0
    for fp in sorted(L2_DIR.glob("cluster_*.json")):
        d = json.loads(fp.read_text())
        n_files += 1
        for sk in d.get("skills") or []:
            hashes = sk.get("source_task_hashes") or []
            domains = Counter()
            for h in hashes:
                rec = hash_idx.get(h)
                if rec:
                    domains[rec.domain] += 1
            if not domains:
                continue
            dom = domains.most_common(1)[0][0]
            by_domain[dom].append({
                "skill_id": sk.get("skill_id"),
                "task_class": sk.get("task_class"),
                "when_to_use": sk.get("when_to_use"),
                "plan_template": sk.get("plan_template") or [],
                "typical_step_count": sk.get("typical_step_count"),
                "attached_pitfalls": sk.get("attached_pitfalls") or [],
                "n_assigned": sk.get("n_assigned"),
                "source_cluster_id": d.get("cluster_id"),
            })
            n_skills += 1
    logger.info(f"  loaded {n_skills} skills across {n_files} cluster files")
    return by_domain


def build_user_prompt(domain: str, skills: List[Dict[str, Any]]) -> str:
    lines = [
        f"# Domain: {domain}",
        f"# L2 skills in this domain: {len(skills)}",
        "",
        "Each skill below is already an abstraction across multiple",
        "trajectories. Extract DOMAIN-level invariants ACROSS these skills.",
        "",
    ]
    for i, sk in enumerate(skills, 1):
        lines.append(f"## Skill {i}/{len(skills)}  "
                     f"id={sk.get('skill_id')}  "
                     f"n_assigned={sk.get('n_assigned')}")
        if sk.get("task_class"):
            lines.append(f"  task_class: {sk['task_class'][:120]}")
        if sk.get("when_to_use"):
            lines.append(f"  when_to_use: {sk['when_to_use'][:200]}")
        plan = sk.get("plan_template") or []
        if plan:
            lines.append(f"  plan_template (first 4 of {len(plan)} steps):")
            for step in plan[:4]:
                lines.append(f"    - {(step.get('subgoal') or '')[:120]}")
                for ta in (step.get("typical_actions") or [])[:1]:
                    lines.append(f"        via: {str(ta)[:100]}")
        pits = sk.get("attached_pitfalls") or []
        if pits:
            lines.append(f"  pitfalls (top 3 of {len(pits)}):")
            for p in pits[:3]:
                lines.append(f"    - {(p.get('pitfall') or '')[:120]}")
        lines.append("")
    lines.append(
        f"Extract DOMAIN-LEVEL invariants / widgets / shortcuts / pitfalls "
        f"for {domain} per the system prompt."
    )
    return "\n".join(lines)


def call_model(system_prompt: str, user_text: str,
               model: str = DEFAULT_MODEL,
               max_tokens: int = 5000,
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


def process_domain(domain: str,
                   skills: List[Dict[str, Any]],
                   *, model: str, force: bool, render_only: bool,
                   ) -> Dict[str, Any]:
    if len(skills) < MIN_DOMAIN_SKILLS:
        return {"domain": domain, "status": "skip-too-few",
                "n_skills": len(skills)}

    user_prompt = build_user_prompt(domain, skills)

    if render_only:
        return {"domain": domain, "status": "rendered",
                "n_skills": len(skills),
                "prompt_chars": len(user_prompt),
                "rendered_prompt": user_prompt}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{domain}.json"
    if out_path.exists() and not force:
        return {"domain": domain, "status": "skip-already-done",
                "path": str(out_path)}

    t0 = time.time()
    try:
        raw = call_model(SYSTEM_PROMPT, user_prompt, model=model)
    except Exception as e:
        return {"domain": domain, "status": "llm-error", "error": str(e)}
    elapsed = time.time() - t0

    try:
        parsed = parse_response(raw)
    except Exception as e:
        return {"domain": domain, "status": "parse-error",
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "raw_head": raw[:400]}

    if not isinstance(parsed, dict) or "domain_invariants" not in parsed:
        return {"domain": domain, "status": "bad-shape",
                "elapsed_s": round(elapsed, 1)}

    record = {
        "domain": domain,
        "n_l2_skills_summarized": len(skills),
        "source_skill_ids": [s["skill_id"] for s in skills],
        "domain_summary": parsed.get("domain_summary"),
        "n_invariants": len(parsed.get("domain_invariants") or []),
        "n_widgets": len(parsed.get("common_widgets") or []),
        "n_shortcuts": len(parsed.get("common_shortcuts") or []),
        "n_shared_pitfalls": len(parsed.get("shared_pitfalls") or []),
        "domain_invariants": parsed.get("domain_invariants") or [],
        "common_widgets": parsed.get("common_widgets") or [],
        "common_shortcuts": parsed.get("common_shortcuts") or [],
        "shared_pitfalls": parsed.get("shared_pitfalls") or [],
        "open_questions": parsed.get("open_questions") or [],
        "raw_response": raw,
        "model": model,
        "elapsed_s": round(elapsed, 1),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return {"domain": domain, "status": "extracted",
            "n_invariants": record["n_invariants"],
            "n_widgets": record["n_widgets"],
            "n_shortcuts": record["n_shortcuts"],
            "n_pitfalls": record["n_shared_pitfalls"],
            "elapsed_s": round(elapsed, 1),
            "path": str(out_path)}


def main() -> None:
    p = argparse.ArgumentParser()
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--domain", type=str)
    sel.add_argument("--all", action="store_true")

    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--render-only", action="store_true",
                   help="Render the prompt only; no LLM call")
    args = p.parse_args()

    by_domain = collect_l2_skills_by_domain()
    logger.info(f"Found {len(by_domain)} domains with L2 skills")
    for dom in sorted(by_domain, key=lambda d: -len(by_domain[d])):
        n = len(by_domain[dom])
        ok = " ✓" if n >= MIN_DOMAIN_SKILLS else " (too few — skip)"
        logger.info(f"  {dom:>22s}: {n:>3d} skills{ok}")

    if args.domain:
        domains_to_process = [args.domain]
        if args.domain not in by_domain:
            raise SystemExit(f"unknown domain: {args.domain}")
    else:  # --all
        domains_to_process = [d for d in by_domain
                              if len(by_domain[d]) >= MIN_DOMAIN_SKILLS]
        domains_to_process.sort(key=lambda d: -len(by_domain[d]))

    logger.info(f"Processing {len(domains_to_process)} domain(s); "
                f"model={args.model}; workers={args.workers}; "
                f"render-only={args.render_only}")

    if args.render_only:
        for dom in domains_to_process:
            r = process_domain(dom, by_domain[dom], model=args.model,
                               force=args.force, render_only=True)
            print(f"\n{'='*78}")
            print(f"PROMPT for {dom}  (n_skills={r.get('n_skills')}, "
                  f"chars={r.get('prompt_chars')})")
            print('='*78)
            print(r.get("rendered_prompt", ""))
        return

    if not args.render_only:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    t0 = time.time()
    if args.workers > 1:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_domain, dom, by_domain[dom],
                              model=args.model, force=args.force,
                              render_only=False): dom
                    for dom in domains_to_process}
            for f in cf.as_completed(futs):
                r = f.result()
                results.append(r)
                logger.info(f"  [{len(results)}/{len(domains_to_process)}] "
                            f"{r.get('domain')}  {r.get('status')}  "
                            f"inv={r.get('n_invariants','-')}  "
                            f"wid={r.get('n_widgets','-')}  "
                            f"sho={r.get('n_shortcuts','-')}  "
                            f"pit={r.get('n_pitfalls','-')}")
    else:
        for i, dom in enumerate(domains_to_process, 1):
            r = process_domain(dom, by_domain[dom], model=args.model,
                               force=args.force, render_only=False)
            results.append(r)
            logger.info(f"  [{i}/{len(domains_to_process)}] {dom}  "
                        f"{r.get('status')}  "
                        f"inv={r.get('n_invariants','-')}  "
                        f"wid={r.get('n_widgets','-')}  "
                        f"sho={r.get('n_shortcuts','-')}  "
                        f"pit={r.get('n_pitfalls','-')}")

    elapsed = time.time() - t0
    summary = {
        "model": args.model,
        "min_domain_skills": MIN_DOMAIN_SKILLS,
        "n_domains_processed": len(results),
        "n_extracted": sum(1 for r in results if r.get("status") == "extracted"),
        "n_errors": sum(1 for r in results
                        if r.get("status") in ("llm-error",
                                                "parse-error",
                                                "bad-shape")),
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    (OUT_DIR / "_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info(f"DONE — {summary['n_extracted']}/{len(results)} extracted, "
                f"{summary['n_errors']} errors, "
                f"elapsed {elapsed:.1f}s. output: {OUT_DIR}")


if __name__ == "__main__":
    main()

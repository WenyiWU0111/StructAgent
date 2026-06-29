"""Reformat the verifier intent recipes into boundary CHECK RECIPES.

The old intent recipes (results/successful_ledgers/_intent_recipes_v2/) were built to
help author VerifySpecs at t=0 (slot_derivation / shape_hint machinery). The
boundary verifier does something different: at a subgoal boundary it decides "is
this milestone done?" by either judging the current screen or authoring ONE
read-only probe. It needs, per milestone class:

  - which probe is GROUND TRUTH (and which signal is misleading),
  - the strict success criterion (anti-leniency),
  - the known traps.

This script distills each intent recipe into that shape with a local LLM (Qwen 27B),
abstracting any task-instance leakage on the way, and builds a FAISS index keyed by
``when_to_use`` so the boundary verifier can retrieve "how this class of milestone
has been verified before".

Output:
  results/verifier_memory/_check_recipes/metadata.jsonl   (one check recipe / line)
  results/verifier_memory/_check_recipes/index.faiss      (IP over when_to_use)
  results/verifier_memory/_check_recipes/_meta.json

Run (local 27B, free):
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.build_check_recipes --local
Smoke (no LLM, just show what would be sent for N):
  PYTHONPATH=. python -m mm_agents.structagent.memory.offline.build_check_recipes --dry-run 5
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mm_agents.structagent._paths import REPO_ROOT
from mm_agents.structagent.memory.offline.clean_leakage import _parse_obj  # robust JSON extract

SRC_DIR = REPO_ROOT / "results" / "successful_ledgers" / "_intent_recipes_v2"
OUT_DIR = REPO_ROOT / "results" / "verifier_memory" / "_check_recipes"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL = os.environ.get("VERIFIER_POLISH_MODEL", "anthropic/claude-sonnet-4-5")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

SYSTEM = """You convert an OLD verification recipe (built for spec-authoring) into a
CHECK RECIPE for a boundary verifier that decides "is this milestone actually done?"
at a subgoal boundary, by either judging the current screenshot/a11y OR authoring ONE
read-only probe from this catalog:
  file_grep      regex over a PLAINTEXT file (NOT a .xlsx/.docx/.pptx — those are ZIP)
  url_match      regex over the active tab URL
  a11y_match     visible UI text / state
  calc_verify / writer_verify / impress_verify   office docs via UNO (the ground truth)
  shell_command  READ-ONLY env/existence only: which / pgrep / test -f|-s|-d / "gsettings get" / "dpkg -l" / "snap list"  (no cat/grep/ls/sqlite/unzip/pipes)

Distill the recipe into EXACTLY this JSON:
{
  "when_to_use": "<one line naming the milestone CLASS — the retrieval key — general, no task-instance values>",
  "checks": [
    {
      "verify": "<what must be confirmed, abstract>",
      "probe": "<the single most reliable probe kind from the catalog>",
      "why_reliable": "<why that probe is ground truth here, and which signal is MISLEADING (e.g. a 'saved' toast, chrome Preferences cache, a displayed value that is not the formula)>",
      "pass_if": "<strict success criterion — enabled != checked, ALL runs not some, the post-state not the intent>",
      "traps": ["<known false-positive / false-negative for this check>"]
    }
  ]
}

RULES:
- ABSTRACT every task-instance value (product / file name / query / site / number) to a
  general placeholder. The recipe must read identically for ANY task of this class.
- Choose the GROUND-TRUTH probe and name what lies: office content -> *_verify (zip, not
  file_grep); a setting whose file is non-realtime (chrome Preferences) -> url/a11y; a
  deliverable file -> file_grep (plaintext) or `test -s` for existence; an exported image
  is binary (can only confirm existence).
- DROP slot_derivation / shape_hint / importance / historical_coverage — the boundary
  verifier authors the concrete probe live.
- To scan a PLAINTEXT file's CONTENT use file_grep, NEVER shell_command — shell cannot
  cat/grep/ls; it is ONLY which/pgrep/test/"gsettings get"/"dpkg -l"/"snap list". Use
  shell_command only for process/package/gsettings checks or `test -s` file existence.
- 1-4 checks; each field ONE tight sentence.
Return ONLY the JSON object."""


def _infer_domain(text: str) -> str:
    t = text.lower()
    for dm in ("libreoffice_calc", "libreoffice_writer", "libreoffice_impress",
               "chrome", "gimp", "thunderbird", "vlc", "vs_code", "multi_apps"):
        key = dm.split("_")[-1] if dm.startswith("libreoffice") else dm.replace("_", " ")
        if dm in t or key in t:
            return dm
    for k, dm in (("calc", "libreoffice_calc"), ("spreadsheet", "libreoffice_calc"),
                  ("writer", "libreoffice_writer"), ("document", "libreoffice_writer"),
                  ("impress", "libreoffice_impress"), ("slide", "libreoffice_impress"),
                  ("presentation", "libreoffice_impress"), ("browser", "chrome"),
                  ("email", "thunderbird"), ("mail", "thunderbird"), ("vlc", "vlc"),
                  ("terminal", "os"), ("file", "os"), ("folder", "os")):
        if k in t:
            return dm
    return ""


def _render_src(rec: Dict[str, Any]) -> str:
    lines = [f"intent_id: {rec.get('intent_id', '')}",
             f"when_to_use: {rec.get('when_to_use', '')}", "dimensions:"]
    for dim in rec.get("verification_dimensions") or []:
        lines.append(f"  - {dim.get('dimension', '')}: {dim.get('description', '')}")
        for po in dim.get("primitive_options") or []:
            lines.append(f"      probe={po.get('kind', '')}  shape={(po.get('shape_hint') or '')[:120]}")
    pit = rec.get("known_pitfalls") or []
    if pit:
        lines.append("known_pitfalls:")
        lines += [f"  - {p}" for p in pit]
    return "\n".join(lines)


def _distill(rec: Dict[str, Any], model: str, base_url: str, api_key: str) -> Optional[Dict[str, Any]]:
    import openai
    client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    extra: Dict[str, Any] = {}
    if "qwen" in model.lower():
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    user = _render_src(rec)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0, max_tokens=2000,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                **extra)
            obj = _parse_obj(resp.choices[0].message.content or "")
            if not isinstance(obj, dict) or "when_to_use" not in obj or "checks" not in obj:
                raise ValueError("missing keys")
            return obj
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None


def _build_index(recipes: List[Dict[str, Any]]) -> None:
    import faiss
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    texts = [r["when_to_use"] for r in recipes]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(OUT_DIR / "index.faiss"))
    with (OUT_DIR / "metadata.jsonl").open("w") as f:
        for r in recipes:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT_DIR / "_meta.json").write_text(json.dumps(
        {"n": len(recipes), "embed_model": EMBED_MODEL, "embed_key": "when_to_use"},
        indent=2))
    print(f"\nindexed {len(recipes)} check recipes -> {OUT_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="use local vLLM Qwen3.5-27B at :8002")
    ap.add_argument("--model", default="")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--dry-run", type=int, metavar="N", default=0,
                    help="show the LLM input for N recipes, no calls, no writes")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.local:
        base_url = a.base_url or "http://localhost:8002/v1"
        model = a.model or "Qwen/Qwen3.5-27B"
        api_key = a.api_key or "EMPTY"
    else:
        base_url = a.base_url or OPENROUTER_BASE
        model = a.model or MODEL
        api_key = a.api_key or os.getenv("OPENROUTER_API_KEY", "")

    files = [f for f in glob.glob(str(SRC_DIR / "*.json")) if "_index" not in f]
    srcs = []
    for f in files:
        try:
            d = json.loads(Path(f).read_text())
        except Exception:
            continue
        srcs.append(d.get("recipe", d))
    print(f"loaded {len(srcs)} intent recipes from {SRC_DIR.name}")

    if a.dry_run:
        for rec in srcs[:a.dry_run]:
            print("\n" + "=" * 60 + f"\n{rec.get('intent_id')}\n" + _render_src(rec))
        return

    t0 = time.time()
    out: List[Dict[str, Any]] = []
    n_fail = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(lambda r: (r, _distill(r, model, base_url, api_key)), srcs))
    for rec, cr in results:
        if cr is None:
            n_fail += 1
            continue
        # NB: do NOT carry the source intent_id — those slugs still embed brand /
        # site names (e.g. apple_iphone_…); retrieval is by when_to_use embedding,
        # not by id, so the recipe needs no id. Infer domain from the abstracted
        # when_to_use only.
        cr["domain"] = _infer_domain(cr.get("when_to_use") or "")
        out.append(cr)
    print(f"distilled {len(out)}/{len(srcs)} ({n_fail} failed) in {time.time()-t0:.0f}s")
    if out:
        _build_index(out)


if __name__ == "__main__":
    main()

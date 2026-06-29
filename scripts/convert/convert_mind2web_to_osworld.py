"""Convert Mind2Web-executable benchmark configs → OSWorld task configs.

Input  : <PMI_ROOT>/data/benchmarks/mind2web_executable/<split>/<Topic>_<N>.json
         schema: {sites: [str], task_id: int, intent, start_url,
                  require_reset, eval: {eval_types}}
         Splits: test_domain_Info, test_domain_Service, test_website
Output : evaluation_examples/examples/mind2web/<task_id>.json
       + evaluation_examples/test_mind2web.json (all)
       + evaluation_examples/test_mind2web_<split>.json (per split)

Path defaults are derived from a sibling Planner-Matters-GUI-Agent
checkout next to the OSWorld repo. Override via --pmi-root or env
``PMI_ROOT``; override the source dir entirely with --src-dir.

Run:
  PYTHONPATH=. python scripts/convert/convert_mind2web_to_osworld.py
  PYTHONPATH=. python scripts/convert/convert_mind2web_to_osworld.py --split test_website
  PMI_ROOT=~/path/to/pmi PYTHONPATH=. python scripts/convert/convert_mind2web_to_osworld.py --dry-run
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_PMI_ROOT = (
    Path(os.environ.get("PMI_ROOT", "")).expanduser()
    if os.environ.get("PMI_ROOT")
    else REPO.parent / "Planner-Matters-GUI-Agent" / "planner-matter-inference"
)
DEFAULT_SRC_SUBDIR = Path("data") / "benchmarks" / "mind2web_executable"
DEFAULT_DST_TASKS = REPO / "evaluation_examples" / "examples" / "mind2web"
DEFAULT_DST_META = REPO / "evaluation_examples" / "test_mind2web.json"


def _slug(s) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]", "_", str(s)).strip("_")


def build_osworld_config(src: dict, split: str, src_path: Path,
                          pmi_root: Path) -> dict:
    task_id = src["task_id"]
    intent = src["intent"]
    start_url = src["start_url"]
    sites = src.get("sites", [])
    topic = (src_path.stem.rsplit("_", 1)[0]
             if "_" in src_path.stem else "x")
    osw_id = (f"mind2web_{split.replace('test_', '')}_"
              f"{_slug(topic)}_{task_id}")
    try:
        provenance = str(src_path.relative_to(pmi_root))
    except ValueError:
        provenance = str(src_path)
    return {
        "id": osw_id,
        "snapshot": "chrome",
        "instruction": intent,
        "source": start_url,
        "config": [
            {"type": "launch",
             "parameters": {"command": [
                 "google-chrome", "--remote-debugging-port=1337"]}},
            {"type": "launch",
             "parameters": {"command": [
                 "socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]}},
            {"type": "chrome_open_tabs",
             "parameters": {"urls_to_open": [start_url]}},
        ],
        "trajectory": "trajectories/",
        "related_apps": ["chrome"],
        "evaluator": {
            "func": "llm_judge_webvoyager",
            "result": {"type": "agent_final_response_and_screenshots"},
            "expected": {"type": "raw_intent", "intent": intent},
        },
        "proxy": True,
        "fixed_ip": False,
        "possibility_of_env_change": "low",
        "_mind2web_meta": {
            "original_task_id": task_id,
            "split": split,
            "sites": sites,
            "source_path": provenance,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmi-root", type=Path, default=DEFAULT_PMI_ROOT)
    ap.add_argument("--src-dir", type=Path, default=None,
                    help="Skip --pmi-root resolution; point directly "
                          "at the mind2web_executable directory")
    ap.add_argument("--dst-tasks", type=Path, default=DEFAULT_DST_TASKS)
    ap.add_argument("--dst-meta", type=Path, default=DEFAULT_DST_META)
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_dir = args.src_dir or (args.pmi_root / DEFAULT_SRC_SUBDIR)
    if not src_dir.exists():
        print(f"src not found: {src_dir}"); return 1
    args.dst_tasks.mkdir(parents=True, exist_ok=True)

    per_split: dict[str, list[str]] = collections.defaultdict(list)
    total = 0
    for split_dir in sorted(src_dir.iterdir()):
        if not split_dir.is_dir(): continue
        if args.split and split_dir.name != args.split: continue
        for src_fp in sorted(split_dir.glob("*.json")):
            src = json.loads(src_fp.read_text())
            cfg = build_osworld_config(src, split_dir.name, src_fp,
                                        args.pmi_root)
            dst_fp = args.dst_tasks / f"{cfg['id']}.json"
            if not args.dry_run:
                dst_fp.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
            per_split[split_dir.name].append(cfg["id"])
            total += 1
    print(f"converted {total} tasks across {len(per_split)} splits")

    if not args.dry_run:
        all_ids = [tid for ids in per_split.values() for tid in ids]
        args.dst_meta.write_text(json.dumps({"mind2web": all_ids},
                                            indent=2, ensure_ascii=False))
        print(f"  meta → {args.dst_meta.relative_to(REPO)} "
              f"({len(all_ids)} tasks)")
        for split, ids in per_split.items():
            sub_fp = REPO / "evaluation_examples" / f"test_mind2web_{split}.json"
            sub_fp.write_text(json.dumps({"mind2web": ids},
                                          indent=2, ensure_ascii=False))
            print(f"    {split} ({len(ids)}) → {sub_fp.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Offline re-judge of mind2web trajectories with the Online-Mind2Web grader.

The original runs were scored by the WebVoyager judge (web_judge.py), which
reads the agent's self-reported ``finished(answer="...")`` and systematically
FAILS correct navigations whose extracted answer lacks the literal sentinel.
The Online-Mind2Web grader (web_judge_online_m2w.WebJudge, arXiv:2504.01382)
is ANSWER-BLIND: it judges from the RAW task + key-points + per-screenshot
relevance + action history only. This script re-scores existing trajectories
with it so we get the TRUE success rate without re-running any task.

Per task dir it reads:
  - RAW intent  ← evaluation_examples/examples/mind2web/<id>.json
                  (evaluator.expected.intent, fallback instruction)
  - screenshots ← base64 frames embedded in trajectory.html
  - actions     ← traj.jsonl ``action`` field per step

Writes ``rejudge_online_m2w.json`` into each dir and prints an old-vs-new
table + aggregate. Pass --apply to also overwrite result.txt.

Usage:
  PYTHONPATH=. conda run -n osworld python scripts/rejudge_mind2web_online.py \
      [--results-dir <dir>] [--judge-model vllm_qwen35-vl] [--limit N] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

_DATA_IMG_RE = re.compile(r"data:image/[a-zA-Z]+;base64,([A-Za-z0-9+/=]+)")

_DEFAULT_RESULTS = (
    "results/pyautogui/screenshot/"
    "vllm_qwen35-vl_planner_100steps_v5_audit_attribute_boundary_inline/mind2web"
)
_CONFIG_DIR = "evaluation_examples/examples/mind2web"


def _raw_intent(task_id: str) -> str:
    """RAW task intent from the per-task config (no finished() preamble)."""
    p = os.path.join(_CONFIG_DIR, f"{task_id}.json")
    if not os.path.exists(p):
        return ""
    try:
        c = json.load(open(p))
    except Exception:
        return ""
    intent = (((c.get("evaluator") or {}).get("expected", {}) or {}).get("intent")
              or c.get("instruction") or "")
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", intent).strip()


def _action_history(task_dir: str) -> list:
    p = os.path.join(task_dir, "traj.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        a = r.get("action")
        if a:
            out.append(str(a)[:200])
    return out


def _screenshots_b64(task_dir: str) -> list:
    p = os.path.join(task_dir, "trajectory.html")
    if not os.path.exists(p):
        return []
    html = open(p, encoding="utf-8", errors="ignore").read()
    return _DATA_IMG_RE.findall(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=_DEFAULT_RESULTS)
    # 27B by default: the Online-Mind2Web 3-stage grader's per-screenshot
    # relevance scoring needs a capable vision judge. The 9B scores almost
    # everything "1" (flat, no discrimination) → all tasks false-fail.
    ap.add_argument("--judge-model", default="vllm_qwen35-27b")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--apply", action="store_true",
                    help="overwrite result.txt with the new score")
    a = ap.parse_args()

    from web_judge import make_judge_client
    from web_judge_online_m2w import WebJudge

    client, model = make_judge_client(a.judge_model)
    wj = WebJudge(client=client, model=model)

    dirs = sorted(
        d for d in (os.path.join(a.results_dir, x)
                    for x in os.listdir(a.results_dir))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "result.txt")))
    if a.limit:
        dirs = dirs[:a.limit]

    rows = []
    for d in dirs:
        tid = os.path.basename(d)
        old = open(os.path.join(d, "result.txt")).read().strip()
        try:
            old_f = float(old)
        except ValueError:
            old_f = 0.0
        intent = _raw_intent(tid)
        acts = _action_history(d)
        shots = _screenshots_b64(d)
        if not intent:
            print(f"[skip] {tid}: no raw intent config", file=sys.stderr)
            continue
        if not shots:
            print(f"[skip] {tid}: no screenshots in trajectory.html",
                  file=sys.stderr)
            continue
        new_f, raw = wj.judge(task=intent, action_history=acts,
                              screenshots_b64=shots)
        json.dump({
            "task_id": tid, "task": intent,
            "old_score": old_f, "new_score": new_f,
            "n_screenshots": len(shots), "n_actions": len(acts),
            "judge_model": model, "judge_raw": raw,
        }, open(os.path.join(d, "rejudge_online_m2w.json"), "w"), indent=1)
        if a.apply:
            with open(os.path.join(d, "result.txt"), "w") as f:
                f.write(f"{new_f}\n")
        flip = "" if old_f == new_f else ("  ↑FIXED" if new_f > old_f
                                          else "  ↓DROP")
        rows.append((tid, old_f, new_f, len(shots), len(acts), intent))
        print(f"{old_f:.0f} -> {new_f:.0f}  [{len(shots)}img/{len(acts)}act]"
              f"  {tid}{flip}")

    if rows:
        n = len(rows)
        old_sr = sum(r[1] for r in rows) / n
        new_sr = sum(r[2] for r in rows) / n
        fixed = sum(1 for r in rows if r[2] > r[1])
        dropped = sum(1 for r in rows if r[2] < r[1])
        print("=" * 60)
        print(f"tasks={n}  OLD SR={old_sr:.1%}  NEW SR={new_sr:.1%}  "
              f"(fixed={fixed}, dropped={dropped})")


if __name__ == "__main__":
    main()

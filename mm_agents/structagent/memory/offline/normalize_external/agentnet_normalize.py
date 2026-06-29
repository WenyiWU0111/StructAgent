"""Normalise an AgentNet trajectory to Causal-Agent ingest shape.

AgentNet records human-driven trajectories with rich freeform `observation /
thought / action / reflection` per step + per-task `task_completed`, `reason`,
`alignment_score` etc. — but the wording style and granularity don't match
our internal mining pipeline's expectations (subgoal templates / typical
actions / done-gate exemplars).

This module wraps a single LLM call per task that condenses the trajectory
into a structured JSON the downstream mining pipeline (L2 / L1 / L_v3) can
consume directly without further parsing.

The same prompt is sent to one of three models (selectable for cost / quality
ablation):
  - "qwen"         → local Qwen3.5-9B at :8001
  - "qwen35-27b"   → OpenRouter qwen/qwen3.5-27b
  - "sonnet"       → OpenRouter claude-sonnet-4-5

Output schema (the model is asked to emit this JSON verbatim):
  {
    "subgoals": [<3-7 short subgoal strings>],
    "key_actions": [
      {"subgoal_idx": int, "action": str, "tool_or_widget": str}
    ],
    "outcome": {
      "completed": bool,
      "final_observation_excerpt": str,
      "final_action": str,
      "failure_modes": [str],     // empty if completed
      "recovery_attempts": [str]
    },
    "pitfalls": [str]             // lessons learned that future agents should heed
  }

Designed to be cheap (one LLM call per task) and bounded (trajectory truncated
to first/last K steps if too long).
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ─── Trajectory condensation ─────────────────────────────────────────────────
def _condense_trajectory(traj: List[Dict[str, Any]], max_steps: int = 30) -> str:
    """Produce a compact text rendering of the trajectory. Drops the per-step
    base64 image (kept as `<screenshot>` placeholder), keeps observation /
    thought / action / reflection text. Truncates middle if too long."""
    if not traj:
        return "<empty trajectory>"
    if len(traj) <= max_steps:
        kept_indices = list(range(len(traj)))
    else:
        # Keep first N/2 + last N/2 steps; middle gets a "[... skipped K steps ...]"
        half = max_steps // 2
        kept_indices = list(range(half)) + list(range(len(traj) - half, len(traj)))

    lines = []
    last_kept = -1
    for i in kept_indices:
        if i - last_kept > 1:
            lines.append(f"[... skipped steps {last_kept+1}..{i-1} ...]")
        s = traj[i]
        v = s.get("value", {}) or {}
        obs = (v.get("observation") or "").strip().replace("\n", " ")
        th = (v.get("thought") or "").strip().replace("\n", " ")
        ac = (v.get("action") or "").strip().replace("\n", " ")
        rf = (v.get("reflection") or "").strip().replace("\n", " ")

        def _trim(t: str, n: int) -> str:
            return t[:n] + ("..." if len(t) > n else "")

        lines.append(f"step {i}:")
        if obs: lines.append(f"  observation: {_trim(obs, 280)}")
        if th:  lines.append(f"  thought:     {_trim(th, 200)}")
        if ac:  lines.append(f"  action:      {_trim(ac, 200)}")
        if rf:  lines.append(f"  reflection:  {_trim(rf, 200)}")
        last_kept = i
    return "\n".join(lines)


# ─── Prompt ──────────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a data engineer extracting STRUCTURED FACTS from a
recorded human-agent trajectory on a Linux desktop. Your job is to produce a
single JSON object summarising the task — to be ingested into a memory bank
that future GUI agents will retrieve for similar tasks.

NEVER fabricate. If the trajectory does not contain enough evidence for a
field, return an empty list or "<unknown>". Be terse — each subgoal / action
/ pitfall is a single short clause, not a paragraph.

Output ONLY a JSON object matching this schema exactly:

{
  "subgoals": [
    "<short imperative subgoal>",
    ...
  ],
  "key_actions": [
    {"subgoal_idx": <int>, "action": "<short clause>", "tool_or_widget": "<concrete UI element>"}
  ],
  "outcome": {
    "completed": <true|false>,
    "final_observation_excerpt": "<≤200 chars from the last step's observation>",
    "final_action": "<the last action taken>",
    "failure_modes": ["<what went wrong>", ...],
    "recovery_attempts": ["<what was tried to fix it>", ...]
  },
  "pitfalls": ["<concrete lesson future agents should learn>", ...]
}

Hard rules:
- 3 to 7 subgoals. No more.
- key_actions covers the MOST IMPORTANT 3-8 actions only (one per major UI
  decision); skip routine waits / repeated text-typing.
- failure_modes: empty when completed=true; non-empty when completed=false.
- pitfalls are extracted ONLY from explicit reflections / mistakes /
  retries visible in the trajectory. Don't invent generic advice.
- final_observation_excerpt: copy verbatim from the trajectory; don't paraphrase.
"""

_USER_TEMPLATE = """Task instruction (canonical):
{instruction}

Domain: {domain}
Task completed: {task_completed}
Annotator's reason field:
{reason}

Trajectory ({n_steps} steps total):
{trajectory_block}

Now produce the JSON object."""


def build_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    traj = task.get("traj") or []
    trajectory_block = _condense_trajectory(traj, max_steps=30)
    reason = task.get("reason") or "<none>"
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TEMPLATE.format(
            instruction=task.get("instruction") or "<missing>",
            domain=task.get("domain") or "<unknown>",
            task_completed=task.get("task_completed"),
            reason=reason[:500],
            n_steps=len(traj),
            trajectory_block=trajectory_block,
        )},
    ]


# ─── LLM dispatch (reuses recheck/_common's resolver) ────────────────────────
def _call_model(model_choice: str, messages: List[Dict[str, str]],
                max_tokens: int = 2000, timeout: float = 240.0,
                ) -> tuple[str, str]:
    """Returns (raw_response_text, resolved_model_name).

    model_choice ∈ {"qwen", "qwen35-27b", "sonnet", "opus"} or a fully-
    qualified OpenRouter model name. Reuses recheck/_common.resolve_reviewer_model
    which already routes these correctly + handles OpenRouter vs vLLM
    thinking-off wire-format mismatch (which would otherwise leave Qwen
    thinking ON, blowing past max_tokens before reaching the JSON answer)."""
    from mm_agents.structagent.memory.offline.normalize_external._llm_dispatch import (
        resolve_reviewer_model, load_dotenv,
    )
    import openai

    model_name, base_url, extra_body = resolve_reviewer_model(model_choice)
    if "openrouter.ai" in base_url:
        load_dotenv()
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY missing for OpenRouter call")
    else:
        api_key = "EMPTY"

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
        extra_body=extra_body,
    )
    return resp.choices[0].message.content or "", model_name


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> Dict[str, Any]:
    """Tolerant: find the LAST balanced JSON object in the response."""
    if not text:
        return {"_unparsed": True, "_reason": "<empty>"}
    # Find balanced { ... } pairs from the tail
    depth, end = 0, None
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == "}":
            if depth == 0:
                end = i
            depth += 1
        elif c == "{":
            depth -= 1
            if depth == 0 and end is not None:
                try:
                    return json.loads(text[i:end + 1])
                except json.JSONDecodeError:
                    end = None
                    depth = 0
    # Fallback: first balanced
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return {"_unparsed": True, "_reason": f"<parse_error: {e}>",
                    "_raw_head": text[:300]}
    return {"_unparsed": True, "_reason": "<no json>", "_raw_head": text[:300]}


def normalize_one(task: Dict[str, Any], model_choice: str = "qwen") -> Dict[str, Any]:
    """Run a single AgentNet task through the normaliser. Returns a dict with
    the parsed JSON + provenance fields."""
    messages = build_messages(task)
    raw, resolved_model = _call_model(model_choice, messages)
    parsed = parse_response(raw)
    return {
        "task_id": task.get("task_id"),
        "domain": task.get("domain"),
        "task_completed": task.get("task_completed"),
        "model_choice": model_choice,
        "resolved_model": resolved_model,
        "parsed": parsed,
        "raw_response": raw,
        "n_steps": len(task.get("traj") or []),
    }

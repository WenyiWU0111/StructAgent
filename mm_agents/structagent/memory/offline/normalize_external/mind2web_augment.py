"""Augment a Multimodal-Mind2Web task with synthesised per-step thought /
observation / reflection text, then hand off to the AgentNet normaliser.

WHY augment first
-----------------
Multimodal-Mind2Web records every action as ``[heading] CAR -> CLICK`` —
crisp but mute. Our memory bank wants per-step reasoning so the downstream
normaliser can extract pitfalls / recovery hints (currently empty in raw
Mind2Web → all pitfalls/recovery would be []). An LLM call reconstructs
the implicit reasoning chain a thoughtful human would have had at each
action, given the FULL action sequence as anchor + the task instruction
as goal. The output mimics AgentNet's ``traj[].value.{observation,
thought, action, reflection}`` shape so the AgentNet normaliser
(``agentnet_normalize.normalize_one``) consumes it byte-identical.

CAVEAT — these reflections are synthesised post-hoc against a *successful*
trajectory. They will read as "this worked because X" rather than
"I tried Y first and it failed, then Z". For pitfalls / failure_modes
mining the signal is weaker than real first-person reflections (AgentNet)
but still gives the planner / verifier banks per-step rationale.

Pipeline:
  1. Stream Multimodal-Mind2Web rows from HF Hub
  2. group_rows_into_tasks → {annotation_id: task_dict}
  3. augment_one(task) → AgentNet-shaped task with synth ``traj``
  4. agentnet_normalize.normalize_one(augmented_task) → final standard JSON
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# ─── Grouping (action-centric Mind2Web rows → task dicts) ────────────────────
def group_rows_into_tasks(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """One annotation_id = one task. action_reprs is constant across rows of
    the same task — we deduplicate by keeping the first row's. Per-row
    target_action_index + target_action_reprs + operation are retained so a
    downstream caller could line up screenshots to action indices."""
    by_task: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        aid = r.get("annotation_id")
        if not aid:
            continue
        if aid not in by_task:
            by_task[aid] = {
                "task_id": aid,
                "instruction": r.get("confirmed_task") or "",
                "website": r.get("website"),
                "domain_label": r.get("domain"),        # Travel / Entertainment / Shopping
                "subdomain": r.get("subdomain"),
                "action_reprs": list(r.get("action_reprs") or []),
                "per_action_rows": [],
            }
        by_task[aid]["per_action_rows"].append({
            "target_action_index": r.get("target_action_index"),
            "target_action_reprs": r.get("target_action_reprs"),
            "operation": r.get("operation"),
        })
    return by_task


# ─── Augmentation prompt ─────────────────────────────────────────────────────
_AUGMENT_SYSTEM = """You are reconstructing the implicit reasoning chain of
a thoughtful human web-automation agent. You are given:
  - the agent's TASK INSTRUCTION
  - the WEBSITE the agent was on
  - the RECORDED ACTION SEQUENCE (a list of `[element_descriptor] -> ACTION`
    strings the agent actually performed, in order)

Your job: emit a JSON ARRAY where each element corresponds to ONE recorded
action, in order. For each, generate the implicit observation / thought /
reflection a thoughtful agent would have had at that step. These were not
recorded — you are reconstructing them — but they MUST be consistent with
what the recorded action actually does on the named website.

Output schema (a JSON array, no prose around it):
[
  {
    "step": <int 0-indexed>,
    "action_text": "<verbatim copy of the recorded action line>",
    "observation": "<≤140 chars: what's visible on the page BEFORE this action>",
    "thought":     "<≤140 chars: why the agent decided on this action>",
    "reflection":  "<≤140 chars: what changed after this action; if last action, what was achieved>"
  },
  ...
]

Hard rules:
- Number of array elements MUST equal the number of recorded actions.
- DO NOT invent failed attempts / typos / wrong clicks. This is a recorded
  successful trajectory; reflections should explain why each action moved
  the agent CLOSER to the goal.
- DO NOT include element coordinates / bounding boxes — keep observations
  visual / conceptual.
- Tailor language to the specific website (e.g. "United Airlines flight
  search" not "a generic flight site").
- Use the website's actual UI vocabulary where obvious (e.g. tabs / dropdowns
  / search boxes / popovers).
"""

_AUGMENT_USER = """TASK INSTRUCTION:
{instruction}

WEBSITE: {website}   (Mind2Web domain: {domain} / subdomain: {subdomain})

RECORDED ACTION SEQUENCE ({n} actions, in order):
{action_block}

Produce the JSON array now."""


def _build_augment_messages(task: Dict[str, Any]) -> List[Dict[str, str]]:
    seq = task.get("action_reprs") or []
    # Cap long sequences (rare; most ≤25 actions). Keep prefix + tail.
    if len(seq) > 30:
        seq = seq[:15] + [f"[... {len(seq) - 30} actions omitted ...]"] + seq[-15:]
    action_block = "\n".join(f"  {i}. {a}" for i, a in enumerate(seq))
    return [
        {"role": "system", "content": _AUGMENT_SYSTEM},
        {"role": "user", "content": _AUGMENT_USER.format(
            instruction=task.get("instruction") or "<missing>",
            website=task.get("website") or "<unknown>",
            domain=task.get("domain_label") or "<unknown>",
            subdomain=task.get("subdomain") or "<unknown>",
            n=len(task.get("action_reprs") or []),
            action_block=action_block,
        )},
    ]


# ─── LLM dispatch (mirrors recheck/_common) ──────────────────────────────────
def _call_model(model_choice: str, messages: List[Dict[str, str]],
                max_tokens: int = 3000, timeout: float = 240.0,
                ) -> tuple[str, str]:
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


_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)


def _parse_array(text: str) -> List[Dict[str, Any]] | Dict[str, Any]:
    """Extract the last balanced JSON ARRAY in the response. Returns either
    a list (success) or a dict with _unparsed=True (failure)."""
    if not text:
        return {"_unparsed": True, "_reason": "<empty>"}
    # Search backwards for balanced [ ... ]
    depth, end = 0, None
    for i in range(len(text) - 1, -1, -1):
        c = text[i]
        if c == "]":
            if depth == 0:
                end = i
            depth += 1
        elif c == "[":
            depth -= 1
            if depth == 0 and end is not None:
                try:
                    arr = json.loads(text[i:end + 1])
                    if isinstance(arr, list):
                        return arr
                except json.JSONDecodeError:
                    end = None
                    depth = 0
    # Fallback: first [...] found by greedy regex
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return arr
        except json.JSONDecodeError as e:
            return {"_unparsed": True, "_reason": f"<parse_error: {e}>",
                    "_raw_head": text[:300]}
    return {"_unparsed": True, "_reason": "<no json array>",
            "_raw_head": text[:300]}


# ─── Domain normalisation (Mind2Web → framework domain) ─────────────────────
def _mind2web_to_framework_domain(task: Dict[str, Any]) -> str:
    """The framework only has ``chrome`` for web tasks. Use that for all
    Mind2Web entries so retrieval lines up. The website / subdomain are
    preserved as separate fields downstream."""
    return "chrome"


# ─── Augment to AgentNet-shaped task ─────────────────────────────────────────
def augment_one(task: Dict[str, Any], model_choice: str = "qwen") -> Dict[str, Any]:
    """Run the augment LLM call. Returns an AgentNet-shaped task dict ready
    to hand to ``agentnet_normalize.normalize_one`` — that downstream call
    is the second of the two-stage pipeline."""
    messages = _build_augment_messages(task)
    raw, resolved_model = _call_model(model_choice, messages)
    parsed = _parse_array(raw)

    # Build the per-step `traj` list in AgentNet's expected shape.
    traj: List[Dict[str, Any]] = []
    augment_meta: Dict[str, Any] = {
        "augment_model_choice": model_choice,
        "augment_resolved_model": resolved_model,
        "augment_raw_response_head": raw[:200] if raw else "",
    }
    if isinstance(parsed, dict) and parsed.get("_unparsed"):
        # Don't try to keep going — return marker so the bulk runner can
        # log an error and skip this task without crashing the pool.
        augment_meta["augment_parse_error"] = parsed
        # Fall through with empty traj; AgentNet normalizer will see
        # n_steps=0 and produce an empty / explicit-stub summary which
        # downstream can drop.
    elif isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            traj.append({
                "index": entry.get("step"),
                "image": None,                     # actual screenshot path
                                                    # available in raw rows
                                                    # if needed by L_v2 later
                "value": {
                    "observation": entry.get("observation") or "",
                    "thought":     entry.get("thought") or "",
                    "action":      entry.get("action_text") or entry.get("action") or "",
                    "reflection":  entry.get("reflection") or "",
                    "code":        "",
                    "last_step_correct": True,
                    "last_step_redundant": False,
                },
                "step_id": entry.get("step"),
                "marks": [],
            })

    return {
        "task_id": task.get("task_id"),
        "instruction": task.get("instruction"),
        "domain": _mind2web_to_framework_domain(task),
        "task_completed": True,                     # all Mind2Web rows are
                                                     # annotated successes
        "alignment_score": None,
        "efficiency_score": None,
        "task_difficulty": None,
        "reason": (f"Real human web demonstration on {task.get('website')}"
                   f" ({task.get('domain_label')}/{task.get('subdomain')}); "
                   "thought/reflection text synthesised by augmenter."),
        "natural_language_task": task.get("instruction"),
        "actual_task": task.get("instruction"),
        "traj": traj,
        # extra provenance for our pipeline (downstream may keep / drop)
        "source": "multimodal_mind2web",
        "mind2web_website": task.get("website"),
        "mind2web_domain": task.get("domain_label"),
        "mind2web_subdomain": task.get("subdomain"),
        "_augment_meta": augment_meta,
    }

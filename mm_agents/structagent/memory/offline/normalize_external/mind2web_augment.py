"""Augment a Multimodal-Mind2Web task with synthesised per-step thought /
observation / reflection text, then hand off to the AgentNet normaliser.

Mind2Web records actions as ``[heading] CAR -> CLICK`` — crisp but mute, so
the downstream normaliser extracts no pitfalls / recovery hints. An LLM call
reconstructs the implicit reasoning a thoughtful human would have had at each
action (full action sequence + task as anchor), shaped as AgentNet's
``traj[].value.{observation,thought,action,reflection}`` so
``agentnet_normalize.normalize_one`` consumes it unchanged.

CAVEAT: reflections are synthesised post-hoc against a *successful* traj, so
they read "this worked because X", never "tried Y, failed, then Z" — weaker
signal for pitfall mining than real AgentNet reflections.

Pipeline:
  1. Stream Mind2Web rows from HF Hub
  2. group_rows_into_tasks → {annotation_id: task_dict}
  3. augment_one(task) → AgentNet-shaped task with synth ``traj``
  4. agentnet_normalize.normalize_one → final standard JSON
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
    """One annotation_id = one task. action_reprs is constant across a task's
    rows (keep the first); per-row target_action_index/reprs/operation are
    retained so a caller could line up screenshots to action indices."""
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
    # Cap long sequences (rare; most ≤25): keep prefix + tail.
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
    """Last balanced JSON array in the response → list (success), else a
    dict with _unparsed=True (failure)."""
    if not text:
        return {"_unparsed": True, "_reason": "<empty>"}
    # Search backwards for a balanced [ ... ].
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
    # Fallback: first [...] by greedy regex.
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
    """Framework has only ``chrome`` for web tasks — use it for all Mind2Web
    entries so retrieval lines up. website/subdomain kept downstream."""
    return "chrome"


# ─── Augment to AgentNet-shaped task ─────────────────────────────────────────
def augment_one(task: Dict[str, Any], model_choice: str = "qwen") -> Dict[str, Any]:
    """Run the augment LLM call → AgentNet-shaped task dict for
    ``agentnet_normalize.normalize_one`` (stage 2)."""
    messages = _build_augment_messages(task)
    raw, resolved_model = _call_model(model_choice, messages)
    parsed = _parse_array(raw)

    # Per-step `traj` in AgentNet's expected shape.
    traj: List[Dict[str, Any]] = []
    augment_meta: Dict[str, Any] = {
        "augment_model_choice": model_choice,
        "augment_resolved_model": resolved_model,
        "augment_raw_response_head": raw[:200] if raw else "",
    }
    if isinstance(parsed, dict) and parsed.get("_unparsed"):
        # Record the marker (bulk runner logs + skips) and fall through with
        # empty traj; the normalizer then emits an n_steps=0 stub to drop.
        augment_meta["augment_parse_error"] = parsed
    elif isinstance(parsed, list):
        for entry in parsed:
            if not isinstance(entry, dict):
                continue
            traj.append({
                "index": entry.get("step"),
                "image": None,                     # real path is in raw rows
                                                    # if needed later
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
        "task_completed": True,                     # Mind2Web rows are all
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
        # provenance (downstream may keep or drop)
        "source": "multimodal_mind2web",
        "mind2web_website": task.get("website"),
        "mind2web_domain": task.get("domain_label"),
        "mind2web_subdomain": task.get("subdomain"),
        "_augment_meta": augment_meta,
    }

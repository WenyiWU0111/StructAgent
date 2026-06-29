"""Per-step KeyNode debug dumper.

Persists the KeyNode dispatcher's per-outcome reasoning to
``<results_dir>/keynode_debug/`` for offline inspection. Files written
per step:

  step_NNN_keynode_input.json     — required outcomes (with verify
                                     spec), current state cache, a11y
                                     length stats, action summary
  step_NNN_keynode_verdicts.json  — per-outcome dispatch path:
                                     {outcome_id, current_state,
                                      verify_kind, deterministic_verdict,
                                      trace, took_llm_path,
                                      emitted_keynode}
  step_NNN_keynode_llm_messages.json — only when an LLM call was made
                                     (subset of outcomes were uncertain).
                                     Image_url payloads stripped to
                                     prefixes; text blocks kept full.
  step_NNN_keynode_llm_raw.txt    — raw LLM response text (only when
                                     LLM path was used).

All functions are graceful-fail: if the dump fails for any reason
(disk full, permission, etc.) they log a warning and return None
without raising. Disabling the dumper is as simple as not setting
``self._results_dir`` on the agent.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


def _ensure_dir(results_dir: Optional[str]) -> Optional[str]:
    if not results_dir:
        return None
    try:
        d = os.path.join(results_dir, "keynode_debug")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        logger.info("[key_node_debug] cannot create dir %s: %s", results_dir, e)
        return None


def _step_prefix(step_idx: Optional[int]) -> str:
    if step_idx is None:
        return "step_xxx"
    try:
        return f"step_{int(step_idx) + 1:03d}"
    except Exception:
        return "step_xxx"


def _verify_to_dict(verify) -> Optional[Dict[str, Any]]:
    if verify is None:
        return None
    if is_dataclass(verify):
        return asdict(verify)
    return None


def dump_keynode_input(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    required_outcomes: Sequence[Any],
    current_states: Dict[str, str],
    a11y_text: str,
    actor_action_summary: str,
) -> None:
    """Save the inputs the dispatcher will see this step."""
    d = _ensure_dir(results_dir)
    if d is None:
        return
    base = _step_prefix(step_idx)
    try:
        outcomes_dump: List[Dict[str, Any]] = []
        for o in required_outcomes:
            entry = {
                "id": getattr(o, "id", "?"),
                "app": getattr(o, "app", None),   # multi-app tasks
                "description": getattr(o, "description", ""),
                "evidence_hint": getattr(o, "evidence_hint", ""),
                "depends_on": list(getattr(o, "depends_on", []) or []),
                "verify": _verify_to_dict(getattr(o, "verify", None)),
                "current_state": current_states.get(
                    getattr(o, "id", ""), "unknown"),
            }
            outcomes_dump.append(entry)
        payload = {
            "step_idx": step_idx,
            "outcomes": outcomes_dump,
            "actor_action_summary": (actor_action_summary or "")[:1000],
            "a11y_chars": len(a11y_text or ""),
            "a11y_rows": (a11y_text or "").count("\n"),
        }
        with open(os.path.join(d, f"{base}_keynode_input.json"),
                  "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.info("[key_node_debug] input dump failed: %s", e)


def dump_keynode_verdicts(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    per_outcome: List[Dict[str, Any]],
) -> None:
    """Save per-outcome dispatcher decision: which path, verdict, evidence.

    ``per_outcome`` is a list of dicts assembled by detect_key_nodes:
        {
          "outcome_id":            str,
          "previous_state":        "pending"|"verified"|"reverted",
          "verify_kind":           str | None,
          "deterministic_verdict": True | False | None,
          "deterministic_trace":   dict (from verify_with_trace),
          "took_llm_path":         bool,
          "emitted_keynode":       dict | None,   # the KeyNode this turn (or None)
        }
    """
    d = _ensure_dir(results_dir)
    if d is None:
        return
    base = _step_prefix(step_idx)
    try:
        with open(os.path.join(d, f"{base}_keynode_verdicts.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"step_idx": step_idx, "per_outcome": per_outcome},
                      f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.info("[key_node_debug] verdicts dump failed: %s", e)


def dump_keynode_llm(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    messages: Optional[Sequence[Dict[str, Any]]],
    raw_response: Optional[str],
) -> None:
    """Save LLM messages + raw response when the dispatcher fell back to
    the LLM path. No-op when no LLM call was made (messages is None)."""
    d = _ensure_dir(results_dir)
    if d is None:
        return
    base = _step_prefix(step_idx)
    if messages is not None:
        try:
            sanitized: List[Dict[str, Any]] = []
            for m in messages:
                role = m.get("role", "?") if isinstance(m, dict) else "?"
                content = m.get("content") if isinstance(m, dict) else m
                if isinstance(content, str):
                    sanitized.append({
                        "role": role, "type": "text",
                        "len_chars": len(content),
                        "text": content,
                    })
                elif isinstance(content, list):
                    parts: List[Dict[str, Any]] = []
                    for p in content:
                        if not isinstance(p, dict):
                            continue
                        if p.get("type") == "image_url":
                            url = ""
                            if isinstance(p.get("image_url"), dict):
                                url = p["image_url"].get("url", "") or ""
                            parts.append({
                                "type": "image_url",
                                "url_prefix": url[:32],
                                "url_chars": len(url),
                            })
                        elif p.get("type") == "text":
                            t = p.get("text", "") or ""
                            parts.append({
                                "type": "text",
                                "len_chars": len(t),
                                "text": t,
                            })
                        else:
                            parts.append({"type": p.get("type", "?")})
                    sanitized.append({"role": role, "type": "multipart",
                                      "parts": parts})
                else:
                    sanitized.append({"role": role, "type": "raw",
                                      "value": str(content)[:200]})
            with open(os.path.join(d, f"{base}_keynode_llm_messages.json"),
                      "w", encoding="utf-8") as f:
                json.dump(sanitized, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.info("[key_node_debug] llm messages dump failed: %s", e)
    if raw_response:
        try:
            with open(os.path.join(d, f"{base}_keynode_llm_raw.txt"),
                      "w", encoding="utf-8") as f:
                f.write(raw_response)
        except Exception as e:
            logger.info("[key_node_debug] llm raw dump failed: %s", e)

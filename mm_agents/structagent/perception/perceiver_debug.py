"""Per-step perceiver debug dumper.

Persists the perceiver's I/O + planner's downstream message construction to
``<results_dir>/perceiver_debug/`` for offline inspection. Files per step:

  step_NNN_perceiver_input.json     — subgoal, expected_post_state, target
                                       outcomes, filtered a11y, transition, size
  step_NNN_perceiver_screen.png     — full screenshot the perceiver saw
  step_NNN_perceiver_raw.txt        — raw LLM response (reasoning+json)
  step_NNN_perceiver_snapshot.json  — parsed SubgoalSnapshot (no b64)
  step_NNN_perceiver_crop.png       — visual_focus crop image
  step_NNN_planner_messages.json    — planner input (image_url stripped to
                                       prefixes; text kept full)

All dumpers graceful-fail (log + return None, never raise). Disable by not
setting ``self._results_dir`` on the agent.

See docs/PROGRESS_LEDGER_V2_CHANGELOG.md §17.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _ensure_dir(results_dir: Optional[str]) -> Optional[str]:
    """Create and return the perceiver_debug subdirectory under
    ``results_dir``, or None if results_dir is falsy / cannot be created."""
    if not results_dir:
        return None
    try:
        d = os.path.join(results_dir, "perceiver_debug")
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as e:
        logger.info("[perceiver_debug] cannot create dir %s: %s",
                    results_dir, e)
        return None


def _step_prefix(step_idx: Optional[int]) -> str:
    """Per-step file prefix, 1-based. 'step_xxx' if no idx."""
    if step_idx is None:
        return "step_xxx"
    try:
        return f"step_{int(step_idx) + 1:03d}"
    except Exception:
        return "step_xxx"


# Dumpers — all return None and never raise.

def dump_perceiver_input(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    subgoal_text: str,
    expected_post_state: str,
    target_outcomes: Sequence[Any],          # objects with .id and .evidence_hint
    last_action_summary: str,
    a11y_filtered_text: str,
    a11y_full_rows: int,
    transition_block: str,
    screen_size: Tuple[int, int],
    screenshot_bytes: Optional[bytes],   # accepted for API stability but not saved
) -> None:
    """Save the perceiver's input context as JSON.

    Screenshot deliberately NOT saved — lib_run_single already writes it per
    step, so dumping it here just duplicated ~800KB/step for no debug value.
    """
    d = _ensure_dir(results_dir)
    if d is None:
        return
    base = _step_prefix(step_idx)
    try:
        payload: Dict[str, Any] = {
            "subgoal_text": subgoal_text,
            "expected_post_state": expected_post_state,
            "target_outcomes": [
                {"id": getattr(o, "id", "?"),
                 "evidence_hint": getattr(o, "evidence_hint", "")}
                for o in target_outcomes
            ],
            "last_action_summary": last_action_summary,
            "screen_size": list(screen_size) if screen_size else [],
            # stats: how aggressive was the prefilter?
            "a11y_full_rows": int(a11y_full_rows),
            "a11y_filtered_rows": (a11y_filtered_text or "").count("\n"),
            "transition_block_chars": len(transition_block or ""),
            # full payloads for offline replay
            "a11y_filtered_text": a11y_filtered_text,
            "transition_block": transition_block,
        }
        with open(os.path.join(d, f"{base}_perceiver_input.json"),
                  "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.info("[perceiver_debug] input dump failed: %s", e)


def dump_perceiver_raw(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    raw_response: Optional[str],
) -> None:
    """Save the perceiver's raw LLM response text (reasoning + JSON)."""
    if not raw_response:
        return
    d = _ensure_dir(results_dir)
    if d is None:
        return
    try:
        with open(os.path.join(d, f"{_step_prefix(step_idx)}_perceiver_raw.txt"),
                  "w", encoding="utf-8") as f:
            f.write(raw_response)
    except Exception as e:
        logger.info("[perceiver_debug] raw dump failed: %s", e)


def dump_perceiver_snapshot(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    snapshot: Optional[Any],
) -> None:
    """Save the parsed SubgoalSnapshot as JSON. ``crop_b64`` is stripped
    (cropping disabled — see perceiver._parse_snapshot); the crop PNG side-file
    is no longer written."""
    if snapshot is None:
        return
    d = _ensure_dir(results_dir)
    if d is None:
        return
    base = _step_prefix(step_idx)
    try:
        if not is_dataclass(snapshot):
            logger.info("[perceiver_debug] snapshot is not a dataclass; "
                        "skipping snapshot dump")
            return
        snap_dict = asdict(snapshot)
        # Strip crop_b64 for readability (expected empty under no-crop mode).
        vf = snap_dict.get("visual_focus") or {}
        if isinstance(vf, dict):
            vf.pop("crop_b64", None)
        with open(os.path.join(d, f"{base}_perceiver_snapshot.json"),
                  "w", encoding="utf-8") as f:
            json.dump(snap_dict, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.info("[perceiver_debug] snapshot dump failed: %s", e)


def dump_environment_probe(
    *,
    results_dir: Optional[str],
    raw_probe: Optional[str],
    digested: Optional[str],
    useful_file_contents: Optional[str] = None,
) -> None:
    """Persist the once-per-task environment probe, its perceiver digestion, and
    (optional) the disk-content probe of files the digestion surfaced. Files:

      environment_probe_raw.txt        — full bash probe stdout
      environment_probe_digested.txt   — perceive_environment output
      useful_file_contents.txt         — file_probes.probe_useful_files
    """
    d = _ensure_dir(results_dir)
    if d is None:
        return
    if raw_probe:
        try:
            with open(os.path.join(d, "environment_probe_raw.txt"),
                      "w", encoding="utf-8") as f:
                f.write(raw_probe)
        except Exception as e:
            logger.info("[perceiver_debug] env probe raw dump failed: %s", e)
    if digested:
        try:
            with open(os.path.join(d, "environment_probe_digested.txt"),
                      "w", encoding="utf-8") as f:
                f.write(digested)
        except Exception as e:
            logger.info("[perceiver_debug] env probe digested dump failed: %s", e)
    if useful_file_contents:
        try:
            with open(os.path.join(d, "useful_file_contents.txt"),
                      "w", encoding="utf-8") as f:
                f.write(useful_file_contents)
        except Exception as e:
            logger.info("[perceiver_debug] useful_file_contents dump failed: %s", e)


def dump_planner_messages(
    *,
    results_dir: Optional[str],
    step_idx: Optional[int],
    messages: Sequence[Dict[str, Any]],
) -> None:
    """Save the planner's input messages with image_url payloads stripped to the
    data-URL prefix; text blocks kept verbatim for inspection."""
    d = _ensure_dir(results_dir)
    if d is None:
        return
    base = _step_prefix(step_idx)
    try:
        preview: List[Dict[str, Any]] = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if isinstance(content, str):
                preview.append({
                    "role": role, "type": "text",
                    "len_chars": len(content),
                    "text": content,
                })
            elif isinstance(content, list):
                parts: List[Dict[str, Any]] = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    t = p.get("type")
                    if t == "text":
                        txt = p.get("text", "")
                        parts.append({
                            "type": "text",
                            "len_chars": len(txt),
                            "text": txt,
                        })
                    elif t == "image_url":
                        url = (p.get("image_url") or {}).get("url", "")
                        head = url.split(",", 1)[0] if "," in url else url
                        parts.append({
                            "type": "image_url",
                            "url_prefix": head[:80],
                            "url_chars": len(url),
                        })
                preview.append({
                    "role": role, "type": "multipart",
                    "parts": parts,
                })
        with open(os.path.join(d, f"{base}_planner_messages.json"),
                  "w", encoding="utf-8") as f:
            json.dump(preview, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.info("[perceiver_debug] planner-msg dump failed: %s", e)

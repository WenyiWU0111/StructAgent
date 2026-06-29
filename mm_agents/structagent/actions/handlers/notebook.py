"""Notebook — agent-side persistent key/value memory.

Once a value is observed (affiliation, citation count, extracted PDF field),
nothing otherwise pins it: KeyNode reverts outcomes when the app loses focus,
the planner forgets values across replans, cross-app handoffs lose the clipboard.
The notebook is a single shared write-once-read-many store that survives all of
those.

Design:
- Plain dict-of-Any on the agent (``self._notebook``), reset per task.
- Two write paths: explicit ``note_write(key, value)`` action, and auto-mirrored
  ``extract_info`` results (one entry per file; replaced the old
  ``_extract_info_history``).
- Single ``[Notebook]`` block injected into both planner and decomposer prompts
  each turn. Not the perceiver (intentionally stateless visual-only).

This module owns only the data shape + rendering; the agent owns dispatch wiring.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


# Cap render at ~6KB so it doesn't crowd the planner prompt; single values at
# ~400 chars (a long affiliation string or paragraph of extracted text).
_MAX_VALUE_CHARS = 400
_MAX_RENDER_CHARS = 6000


def _truncate(s: str, n: int = _MAX_VALUE_CHARS) -> str:
    s = str(s)
    return s if len(s) <= n else (s[:n] + " …[truncated]")


def write_extract_info_batch(
    batch: Dict[str, Any],
    *,
    ledger: Any,
    owning_outcome: Optional[str] = None,
    step_idx: int = 0,
) -> int:
    """Persist one ``extract_info`` batch as Facts on the ledger; returns the
    number of file entries written. Only successful extractions become Facts;
    latest-call-wins on a repeated path. Payload shape kept across the A7 cutover
    so the A6 renderer's extract_info detector still works:

        Fact.value = {"_kind": "extract_info", "query": ..., "extracted": {...}}
    """
    if not batch or not isinstance(batch, dict):
        return 0
    files = batch.get("files") or []
    query = str(batch.get("query") or "")
    n_written = 0
    for f in files:
        if not isinstance(f, dict):
            continue
        if not f.get("ok"):
            continue
        extracted = f.get("extracted") or {}
        if isinstance(extracted, dict) and extracted.get("_skip"):
            continue
        path = f.get("path") or ""
        if not path:
            continue
        payload = {
            "_kind": "extract_info",
            "query": query,
            "extracted": extracted,
        }
        try:
            ledger.write_fact(
                name=path,
                value=payload,
                owning_outcome=owning_outcome,
                step_idx=step_idx,
                type_tag="json",
            )
        except Exception:
            continue
        n_written += 1
    return n_written


def write_note(notebook: Dict[str, Any], key: str, value: Any) -> None:
    """Free-form planner/decomposer write; value kept as-is. Tagged ``_kind:
    note`` so the renderer separates it from extract_info entries."""
    key = str(key).strip()
    if not key:
        return
    notebook[key] = {"_kind": "note", "value": value}


def render(notebook: Optional[Dict[str, Any]]) -> Optional[str]:
    """Render the notebook into a prompt block. Returns None when empty.

    Layout:

        [Notebook — persistent KV memory captured this task. Values
         here ARE sticky — they survive app switches, perceiver misses,
         and outcome REVERTs. Read from here before re-observing.]

        Notes (3):
          contact_name = "Alex Rivera"
          contact_affiliation = "Acme Corp; Example University"
          completed_weeks = "[0, 1, 2]"

        Extracted files (2):
          /home/user/Desktop/receipt_1.jpg
            description: Office Supplies
            amount:      42.00
          /home/user/Desktop/receipt_2.jpg
            description: Lunch
            amount:      12.50
    """
    if not notebook:
        return None
    notes: List = []
    extracts: List = []
    for k, v in notebook.items():
        if isinstance(v, dict) and v.get("_kind") == "extract_info":
            extracts.append((k, v))
        else:
            # Anything else is a note (_kind=note, or legacy bare values).
            notes.append((k, v))
    if not notes and not extracts:
        return None

    out: List[str] = []
    out.append(
        "[Notebook — persistent KV memory captured this task. Values "
        "here ARE sticky — they survive app switches, perceiver misses, "
        "and outcome REVERTs. Read from here BEFORE re-observing or "
        "re-extracting; write here whenever you observe a concrete "
        "value the task will need downstream.]"
    )
    if notes:
        out.append(f"\nNotes ({len(notes)}):")
        for k, v in notes:
            val = v.get("value") if isinstance(v, dict) else v
            if isinstance(val, (dict, list)):
                val_s = json.dumps(val, ensure_ascii=False)
            else:
                val_s = str(val)
            out.append(f"  {k} = {_truncate(val_s)!r}")
    if extracts:
        out.append(f"\nExtracted files ({len(extracts)}):")
        for path, entry in extracts:
            out.append(f"  {path}")
            extracted = entry.get("extracted") or {}
            if isinstance(extracted, dict):
                for fk, fv in extracted.items():
                    if isinstance(fv, (dict, list)):
                        fv_s = json.dumps(fv, ensure_ascii=False)
                    else:
                        fv_s = str(fv) if fv is not None else "(null)"
                    out.append(f"    {fk:12s} {_truncate(fv_s)}")
            else:
                out.append(f"    {_truncate(str(extracted))}")

    rendered = "\n".join(out)
    if len(rendered) > _MAX_RENDER_CHARS:
        rendered = rendered[:_MAX_RENDER_CHARS] + "\n  …[notebook truncated]"
    return rendered


# Action dispatch helpers (called from the agent's _pa_to_pyautogui).

def handle_note_write(
    action_inputs: Dict[str, Any],
    *,
    ledger: Any,
    logger: Optional[Any] = None,
    owning_outcome: Optional[str] = None,
    step_idx: int = 0,
) -> None:
    """Decode a ``note_write`` step and persist as a Fact. Wire form (after the
    _step_to_uitars_line round-trip): ``{"key": ..., "value": ...}``; value may
    be a JSON string for structured data.

    A7: ``_notebook`` is gone — only sink is ``ledger.write_fact``. Fact.value
    keeps the legacy ``{"_kind": "note", "value": X}`` wrap so the A6 renderer's
    notes-vs-files split still works.
    """
    key = str(action_inputs.get("key") or "").strip()
    raw_value = action_inputs.get("value")
    if not key:
        if logger:
            logger.warning("[note_write] missing key — dropping")
        return
    # JSON-decode if the wire value looks structured, so the decomposer can
    # save dicts/lists; otherwise keep the string.
    value: Any = raw_value
    if isinstance(raw_value, str):
        s = raw_value.strip()
        if (s.startswith("{") and s.endswith("}")) or \
           (s.startswith("[") and s.endswith("]")):
            try:
                value = json.loads(s)
            except Exception:  # noqa: BLE001
                value = raw_value
    try:
        ledger.write_fact(
            name=key, value={"_kind": "note", "value": value},
            owning_outcome=owning_outcome,
            step_idx=step_idx,
        )
    except Exception as e:
        if logger:
            logger.info("[note_write] Fact write failed: %s", e)
    if logger:
        preview = str(value)[:80]
        logger.info("[note_write] key=%r value=%r%s",
                    key, preview, " …" if len(str(value)) > 80 else "")

"""Helpers for the ``edit_json`` actor action.

The actor (decomposer) emits a step like
    {"action": "edit_json",
     "path": "~/.config/Code/User/keybindings.json",
     "op": "append_entry",
     "data": {"key":"ctrl+j", ...}}

Upstream this becomes a string ``edit_json(path=..., op=..., data=...)``,
which ``_pa_to_pyautogui`` converts into a Python script. The script
runs **on the VM** via ``execute_python_command`` and:

  1. Reads the current file (if it exists), strips JSONC comments,
     parses to a Python value (default container based on op when
     parse fails or file empty / missing).
  2. Applies the op transformation deterministically.
  3. Writes back via json.dump (always emits valid JSON).
  4. VS Code (if running) auto-reloads via inotify; we never pkill.

Three ops, no merge:

  set_key             — cur is dict; cur.update(data)
  append_entry        — cur is list; append data (or extend if data is list)
  remove_entry_by_key — cur is list; remove entries where ALL fields in
                         data match (matcher dict)

This module is pure-Python with no VM-only deps so the unit tests can
exercise it directly.
"""
from __future__ import annotations
import json
import re
from typing import Any, Dict, List


# --------------------------------------------------------------------------- #
# JSONC parsing
# --------------------------------------------------------------------------- #

def strip_jsonc_comments(text: str) -> str:
    """Strip ``//`` line comments and ``/* */`` block comments.

    VS Code's settings.json/keybindings.json are JSONC (JSON with
    Comments). The default stub VS Code creates contains lines like
    ``// Place your key bindings in this file...``. ``json.loads``
    chokes on these — we strip them before parsing.

    Naive: doesn't try to preserve comments inside string literals.
    For OSWorld config files this is fine — they don't contain ``//``
    inside string values."""
    if not text:
        return ""
    # Remove block comments first
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    # Remove line comments
    text = re.sub(r'//[^\n]*', '', text)
    return text.strip()


def parse_jsonc_or_default(raw_text: str, default: Any) -> Any:
    """Parse JSONC text; on failure return ``default``. ``default``
    determines the empty container shape (e.g. ``[]`` for array tasks
    and ``{}`` for object tasks)."""
    s = strip_jsonc_comments(raw_text)
    if not s:
        return default
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# Op transforms — pure functions over (cur, data) -> new_cur
# --------------------------------------------------------------------------- #

def op_set_key(cur: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """Set each top-level key in data on cur. ``cur`` must be a dict
    (forced to {} when not). VS Code accepts both flat dot-keys
    ('python.analysis.x' as a single string key) AND nested objects;
    we mirror whatever the caller passes in data — no auto-nesting."""
    if not isinstance(cur, dict):
        cur = {}
    for k, v in (data or {}).items():
        cur[k] = v
    return cur


def op_append_entry(cur: Any, data: Any) -> List[Any]:
    """Append data to a JSON array. ``data`` may be a single dict (one
    entry) or a list of dicts (multiple). ``cur`` must be a list (forced
    to [] when not)."""
    if not isinstance(cur, list):
        cur = []
    if isinstance(data, list):
        cur.extend(data)
    else:
        cur.append(data)
    return cur


def op_remove_entry_by_key(cur: Any, matcher: Dict[str, Any]) -> List[Any]:
    """Remove array entries where ALL keys in ``matcher`` equal the
    matching keys in the entry. ``cur`` must be a list (forced to [])."""
    if not isinstance(cur, list):
        return []
    if not isinstance(matcher, dict) or not matcher:
        return list(cur)
    def _matches(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        return all(entry.get(k) == v for k, v in matcher.items())
    return [e for e in cur if not _matches(e)]


# --------------------------------------------------------------------------- #
# Top-level helper used by both unit tests and the runtime script.
# --------------------------------------------------------------------------- #

_DEFAULTS = {
    "set_key": dict,
    "append_entry": list,
    "remove_entry_by_key": list,
}


def apply_edit_json(raw_text: str, op: str, data: Any) -> Any:
    """Run a single edit_json transform end-to-end.

    Returns the new container. Caller serializes via json.dumps and
    writes to file. Raises ValueError on unknown op."""
    if op not in _DEFAULTS:
        raise ValueError(f"unknown op: {op!r}; expected one of {list(_DEFAULTS)}")
    default = _DEFAULTS[op]()
    cur = parse_jsonc_or_default(raw_text, default)
    if op == "set_key":
        return op_set_key(cur, data)
    if op == "append_entry":
        return op_append_entry(cur, data)
    if op == "remove_entry_by_key":
        return op_remove_entry_by_key(cur, data)
    return cur  # unreachable


# --------------------------------------------------------------------------- #
# Runtime script builder — used by ``_pa_to_pyautogui`` to convert an
# edit_json action into the Python the VM will execute.
# --------------------------------------------------------------------------- #

def build_runtime_script(path: str, op: str, data: Any) -> str:
    """Build the Python script that runs on the VM. The script is
    self-contained (no helpers imported from this module — VM may not
    have it on PYTHONPATH). All logic is duplicated inline. Returns
    valid Python that prints a one-line status on completion."""
    if op not in _DEFAULTS:
        raise ValueError(f"unknown op: {op!r}")
    data_repr = json.dumps(data)
    path_repr = json.dumps(path)
    op_repr = json.dumps(op)
    return f"""import os, json, re

# edit_json runtime — see mm_agents/edit_json_helpers.py for reference impl
_path = os.path.expanduser({path_repr})
_op = {op_repr}
_data = json.loads({json.dumps(data_repr)})

# Read existing
_default = [] if _op in ('append_entry', 'remove_entry_by_key') else {{}}
_cur = _default
if os.path.exists(_path) and os.path.getsize(_path) > 0:
    try:
        _raw = open(_path).read()
        _s = re.sub(r'/\\*.*?\\*/', '', _raw, flags=re.DOTALL)
        _s = re.sub(r'//[^\\n]*', '', _s).strip()
        if _s:
            _cur = json.loads(_s)
    except Exception as _e:
        _cur = _default

# Apply op
if _op == 'set_key':
    if not isinstance(_cur, dict): _cur = {{}}
    for _k, _v in (_data or {{}}).items():
        _cur[_k] = _v
elif _op == 'append_entry':
    if not isinstance(_cur, list): _cur = []
    if isinstance(_data, list):
        _cur.extend(_data)
    else:
        _cur.append(_data)
elif _op == 'remove_entry_by_key':
    if isinstance(_cur, list) and isinstance(_data, dict) and _data:
        _cur = [_e for _e in _cur
                if not (isinstance(_e, dict)
                        and all(_e.get(_k) == _v for _k, _v in _data.items()))]
    elif not isinstance(_cur, list):
        _cur = []

# Write back (creates parent dir if needed)
os.makedirs(os.path.dirname(_path), exist_ok=True)
with open(_path, 'w') as _f:
    json.dump(_cur, _f, indent=4)
_n = len(_cur) if isinstance(_cur, (list, dict)) else 0
print(f'edit_json {{_op}}: wrote {{_n}} items to {{_path}}')
"""

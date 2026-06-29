"""``open_app`` action — deterministic "ensure an app is open and focused".

The planner-actor agent moves between applications constantly on
multi-app tasks. Doing that through the GUI (clicking the dock,
Alt+Tab, hunting for a window behind another) is the single largest
source of churn in multi-app trajectories — the actor lands in the
wrong window, the window is occluded, focus does not stick.

``open_app`` collapses "switch to / launch app X (optionally with
file Y)" into ONE deterministic action:

  * if a window for the app already exists → activate + raise it
    (``wmctrl -ia``);
  * otherwise → launch the app (with the file if given), wait for
    its window, then activate it.

It is idempotent: calling it when the app is already foreground just
re-activates the same window.

The action builds a self-contained Python script that runs on the VM
(via the same ``execute_python_command`` path as ``cli_run``) and
persists its result to ``/tmp/_last_cli_run.json`` so the planner
sees, on its next turn, which window was activated / launched.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# app id  →  (window-title hint substrings, launch argv).
# The title hint matches against ``wmctrl -l`` output (case-insensitive).
# LibreOffice runs as ONE process for all documents, so Calc / Writer /
# Impress are told apart by the window TITLE ("... - LibreOffice Calc"),
# not the process / WM_CLASS.
_APP_SPECS: Dict[str, Tuple[List[str], List[str]]] = {
    "libreoffice_writer":  (["LibreOffice Writer"],  ["soffice", "--writer"]),
    "libreoffice_calc":    (["LibreOffice Calc"],    ["soffice", "--calc"]),
    "libreoffice_impress": (["LibreOffice Impress"], ["soffice", "--impress"]),
    "vs_code":             (["Visual Studio Code"],  ["code"]),
    "chrome":              (["Chromium", "Chrome"],  ["google-chrome", "--remote-debugging-port=1337"]),
    "os":                  (["Terminal"],            ["x-terminal-emulator"]),
}

# Caller-facing aliases (keys are underscore-normalized) — OSWorld
# task metadata and the planner are inconsistent about app labels.
_APP_ALIASES: Dict[str, str] = {
    "writer": "libreoffice_writer",
    "calc": "libreoffice_calc",
    "impress": "libreoffice_impress",
    "vscode": "vs_code",
    "code": "vs_code",
    "google_chrome": "chrome",
    "chromium": "chrome",
    "terminal": "os",
}


def normalize_app(app: Optional[str]) -> Optional[str]:
    """Map a caller's app label to a canonical ``_APP_SPECS`` key, or
    None when it does not resolve."""
    if not app or not str(app).strip():
        return None
    key = str(app).strip().lower().replace("-", "_").replace(" ", "_")
    if key in _APP_SPECS:
        return key
    return _APP_ALIASES.get(key)


# file extension → the app that owns it.
_EXT_TO_APP: Dict[str, str] = {
    ".docx": "libreoffice_writer", ".doc": "libreoffice_writer",
    ".odt": "libreoffice_writer", ".rtf": "libreoffice_writer",
    ".xlsx": "libreoffice_calc", ".xls": "libreoffice_calc",
    ".csv": "libreoffice_calc", ".ods": "libreoffice_calc",
    ".pptx": "libreoffice_impress", ".ppt": "libreoffice_impress",
    ".odp": "libreoffice_impress",
}


def app_for_file(path: Optional[str]) -> Optional[str]:
    """Canonical app id that owns ``path`` by its extension, or None."""
    if not path:
        return None
    import os
    ext = os.path.splitext(str(path))[1].lower()
    return _EXT_TO_APP.get(ext)


# Marker the probe script prints; the parser keys off it.
_PROBE_MARKER = "INITIAL_FILES:"


def build_initial_files_probe_script() -> str:
    """Return a VM-side Python script that OBSERVES which document
    files are open at task start, and prints ``INITIAL_FILES:<json>``
    where the json is ``{app_id: file_path}``.

    This is a legitimate environment observation — it inspects the
    running ``soffice`` process, NOT the task spec. LibreOffice creates
    a ``.~lock.<name>#`` lock file next to every open document; that
    lock file (visible via ``lsof`` on the soffice process) reveals the
    document's real path. Direct document file handles are read too.
    """
    ext_map = repr(_EXT_TO_APP)
    return f'''import subprocess, json, os

_EXT = {ext_map}
_result = {{}}


def _record(_path):
    if not _path:
        return
    _ext = os.path.splitext(_path)[1].lower()
    _app = _EXT.get(_ext)
    if _app and _app not in _result:
        _result[_app] = _path


try:
    _out = subprocess.run(["lsof", "-c", "soffice", "-F", "n"],
                          capture_output=True, text=True, timeout=20).stdout
    for _line in _out.splitlines():
        if not _line.startswith("n"):
            continue
        _p = _line[1:].strip()
        _base = os.path.basename(_p)
        if _base.startswith(".~lock.") and _base.endswith("#"):
            # LibreOffice lock file → derive the real document path.
            _real = _base[len(".~lock."):-1]
            _record(os.path.join(os.path.dirname(_p), _real))
        else:
            _record(_p)
except Exception:
    pass

print("{_PROBE_MARKER}" + json.dumps(_result))
'''


def parse_initial_files_probe(output: Optional[str]) -> Dict[str, str]:
    """Parse the ``INITIAL_FILES:<json>`` line from the probe's stdout
    into ``{app_id: file_path}``. Returns ``{}`` on any failure."""
    if not output:
        return {}
    for line in str(output).splitlines():
        line = line.strip()
        if line.startswith(_PROBE_MARKER):
            try:
                data = json.loads(line[len(_PROBE_MARKER):])
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()
                            if k in _APP_SPECS and v}
            except Exception:
                return {}
    return {}


def build_open_app_script(
    app: Optional[str],
    file_path: Optional[str] = None,
    *,
    logger: Optional[Any] = None,
) -> Optional[str]:
    """Return the VM-side Python script for an ``open_app`` action.

    ``app``        — canonical id or alias (see ``_APP_SPECS`` /
                     ``_APP_ALIASES``).
    ``file_path``  — optional file to open in that app. ``~`` is
                     expanded on the VM.

    Returns None on an unresolvable app (caller logs + drops the action).
    """
    canon = normalize_app(app)
    if canon is None:
        if logger:
            logger.warning("[open_app] unknown app %r — supported: %s",
                            app, sorted(_APP_SPECS))
        return None
    title_hints, launch = _APP_SPECS[canon]
    fp = (str(file_path).strip() or None) if file_path else None

    # repr() — NOT json.dumps() — for embedding as Python literals:
    # json.dumps(None) is "null" (a NameError when exec'd), and
    # json.dumps(True/False) is "true"/"false". repr() always yields a
    # valid Python literal for str / list / None.
    app_repr = repr(canon)
    hints_repr = repr(title_hints)
    launch_repr = repr(launch)
    fp_repr = repr(fp)

    return f'''import subprocess, time, json, os

_app = {app_repr}
_title_hints = {hints_repr}
_launch = {launch_repr}
_file = {fp_repr}
if _file:
    _file = os.path.expanduser(_file)
_file_base = os.path.basename(_file) if _file else None


def _find_window():
    """First window id whose title matches the app (and file, if given)."""
    try:
        _out = subprocess.run(["wmctrl", "-l"], capture_output=True,
                              text=True, timeout=10).stdout
    except Exception:
        return None
    for _line in _out.splitlines():
        _low = _line.lower()
        if not any(_h.lower() in _low for _h in _title_hints):
            continue
        if _file_base and _file_base.lower() not in _low:
            continue
        return _line.split()[0]
    return None


def _active_wid_int():
    """The window manager's actually-active window id (int), via xprop."""
    try:
        _o = subprocess.run(["xprop", "-root", "_NET_ACTIVE_WINDOW"],
                            capture_output=True, text=True, timeout=5).stdout
        return int(_o.strip().split()[-1], 16)
    except Exception:
        return None


def _focus_verified(_wid_hex):
    """Activation is a REQUEST — verify focus actually landed (closed loop):
    check after wmctrl -ia; retry wmctrl; fall back to xdotool --sync; check."""
    try:
        _want = int(_wid_hex, 16)
    except Exception:
        return False
    time.sleep(0.4)
    if _active_wid_int() == _want:
        return True
    subprocess.run(["wmctrl", "-ia", _wid_hex], timeout=10)
    time.sleep(0.4)
    if _active_wid_int() == _want:
        return True
    try:
        subprocess.run(["xdotool", "windowactivate", "--sync", _wid_hex],
                       capture_output=True, timeout=10)
    except Exception:
        pass
    time.sleep(0.4)
    return _active_wid_int() == _want


_status = ""
_rc = 0
_focus_ok = False
_failure_type = None      # None | "no_focus" | "no_window" | "launch_failed"
_wid = _find_window()
if _wid:
    subprocess.run(["wmctrl", "-ia", _wid], timeout=10)
    _focus_ok = _focus_verified(_wid)
    _status = (f"activated existing window {{_wid}}"
               + ("" if _focus_ok else " but FOCUS NOT acquired"))
    if not _focus_ok:
        _failure_type = "no_focus"
        _rc = 1
else:
    _argv = list(_launch) + ([_file] if _file else [])
    print(f"open_app[{{_app}}]: no window found, launching: {{_argv}}")
    try:
        subprocess.Popen(_argv, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    except Exception as _e:
        _status = f"launch failed: {{_e}}"
        _failure_type = "launch_failed"
        _rc = 1
    if _rc == 0:
        for _ in range(24):          # up to ~12s for the window to appear
            time.sleep(0.5)
            _wid = _find_window()
            if _wid:
                break
        if _wid:
            subprocess.run(["wmctrl", "-ia", _wid], timeout=10)
            _focus_ok = _focus_verified(_wid)
            _status = (f"launched and activated window {{_wid}}"
                       + ("" if _focus_ok else " but FOCUS NOT acquired"))
            if not _focus_ok:
                _failure_type = "no_focus"
                _rc = 1
        else:
            _status = "FAILED — launched but no window appeared"
            _failure_type = "no_window"
            _rc = 1

print(f"open_app[{{_app}}]: {{_status}}")
# Machine-readable sentinel for the agent-side closed loop — active_app.py's
# parse_open_app_result reads this line from the runner output.
print("__OPEN_APP__" + json.dumps({{
    "app": _app, "ok": _rc == 0, "focus_verified": _focus_ok,
    "failure_type": _failure_type, "wid": _wid, "detail": _status,
}}))
try:
    os.makedirs("/tmp", exist_ok=True)
    with open("/tmp/_last_cli_run.json", "w") as _f:
        json.dump({{
            "command": f"open_app({{_app}}, file={{_file}})",
            "mode": "open_app",
            "returncode": _rc,
            "stdout": _status,
            "stderr": "" if _rc == 0 else _status,
            "timestamp": time.time(),
        }}, _f)
except Exception:
    pass
'''

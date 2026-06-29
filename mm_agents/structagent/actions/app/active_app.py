"""Detect the OS's currently-focused app from raw AT-SPI a11y XML.

Derives ``physical_domain`` (the app whose window is actually focused) apart
from ``_current_domain`` (the agent's INTENT). Conflating the two caused
multi-app failures: cross-app a11y bleed in the verifier, action routing to
the wrong app, and undetected ``_activate_app_for_domain`` failures.

``observe_active_app(a11y_xml)`` returns the canonical domain key (a member
of ``open_app._APP_SPECS``) for the app whose root frame is
``state:active="true"``, else None.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET


# App-switch closed loop (EVIDENCE_BINDING_DESIGN §8) — agent-side half.
# VM-side half is in open_app.py (focus verify + __OPEN_APP__ sentinel).
# Gated by ENABLE_APP_SWITCH_VERIFY (default 0 = old fire-and-forget activate
# + 5-round intent demote).

def app_switch_verify_enabled() -> bool:
    from mm_agents.structagent.config import CAConfig
    return CAConfig.from_env().app_switch_verify


_SENTINEL = "__OPEN_APP__"


def parse_open_app_result(runner_result: Any) -> Dict[str, Any]:
    """Extract the __OPEN_APP__ sentinel from a run_python_script result.

    Returns {"ok": True/False, "failure_type": ..., "detail": ...} when found;
    {"ok": None} when absent (old script / runner error). Callers MUST treat
    ok=None as "unverified, behave as before", never as a failure — an old VM
    image without the new script must not brick switching.

    The sentinel may be raw or JSON-escaped inside the runner's envelope
    ({"output": "...__OPEN_APP__{\\"app\\"...}"}). raw_decode takes the first
    complete JSON value and ignores trailing text; LAST sentinel wins (retries
    print multiple)."""
    if runner_result is None:
        return {"ok": None}
    text = (runner_result if isinstance(runner_result, str)
            else json.dumps(runner_result, ensure_ascii=False, default=str))
    idx = text.rfind(_SENTINEL)
    if idx < 0:
        return {"ok": None}
    tail = text[idx + len(_SENTINEL):].lstrip()
    candidates = [tail]
    try:
        candidates.append(tail.encode("utf-8").decode("unicode_escape"))
    except Exception:
        pass
    for cand in candidates:
        try:
            obj, _ = json.JSONDecoder().raw_decode(cand)
            if isinstance(obj, dict) and "ok" in obj:
                return obj
        except Exception:
            continue
    return {"ok": None}


def switch_failure_message(domain: str, parsed: Dict[str, Any]) -> str:
    """Planner-visible failure block (§8 step 4) — names the failure TYPE so
    replan addresses the cause instead of blind-retrying."""
    ftype = parsed.get("failure_type") or "unknown"
    detail = parsed.get("detail") or ""
    hint = {
        "no_focus": ("the window exists but the window manager refused focus "
                     "— a modal dialog may be blocking; close it first, or "
                     "work in the currently-focused app explicitly"),
        "no_window": ("no window for this app exists and launching did not "
                      "produce one — launch it explicitly (check the file "
                      "path) before planning steps inside it"),
        "launch_failed": "the launch command itself failed — see detail",
    }.get(ftype, "cause unknown — re-plan around the focused app")
    return (f"[AppSwitch] FAILED to focus {domain} ({ftype}): {detail}. "
            f"Guidance: {hint}")


# AT-SPI state namespace — must match the exact URL the OSWorld VM server
# emits (desktop_env/server/main.py; same string as
# accessibility_tree_handle.py's ``state_ns_ubuntu``). Using the upstream
# linuxfoundation URL instead made every ``active`` lookup hit the "false"
# default, so observe_active_app returned None on every step.
_STATE_NS = "https://accessibility.ubuntu.example.org/ns/state"

# System apps that appear in the AT-SPI dump but aren't user focus targets;
# skip even when they expose an active frame (gnome-shell does during
# transitions).
_IGNORED_APP_NAMES = {
    "gnome-shell", "gjs", "mutter", "ibus-x11", "ibus-ui-gtk3",
    "Xorg", "xdg-desktop-portal", "xdg-desktop-portal-gnome",
    "evolution-alarm-notify", "tracker-miner-fs",
}


# AT-SPI application name → canonical domain key in ``open_app._APP_SPECS``.
# LibreOffice ("soffice") is handled separately: Calc / Writer / Impress share
# one process, so the frame title disambiguates (see _resolve_libreoffice).
_APP_NAME_TO_DOMAIN = {
    "thunderbird": "thunderbird",
    "firefox": "firefox",
    "google-chrome": "chrome",
    "Google-chrome": "chrome",
    "chromium": "chrome",
    "Chromium": "chrome",
    "Chromium-browser": "chrome",
    "code": "vs_code",
    "Code": "vs_code",
    "vs_code": "vs_code",
    "VSCodium": "vs_code",
    "gnome-terminal-server": "os",
    "xterm": "os",
    "Files": "files",
    "nautilus": "files",
    "org.gnome.Nautilus": "files",
    "gimp": "gimp",
    "Gimp": "gimp",
    "GIMP": "gimp",
    "vlc": "vlc",
    "VLC": "vlc",
    "vlc-media-player": "vlc",
}


def _resolve_libreoffice(frame_title: str) -> Optional[str]:
    """Disambiguate Calc / Writer / Impress from a frame title.

    Titles look like "... — LibreOffice Calc". Returns None when no marker is
    found (Start Center, dialogs).
    """
    if not frame_title:
        return None
    t = frame_title.lower()
    if "libreoffice calc" in t or "libreoffice-calc" in t:
        return "libreoffice_calc"
    if "libreoffice writer" in t or "libreoffice-writer" in t:
        return "libreoffice_writer"
    if "libreoffice impress" in t or "libreoffice-impress" in t:
        return "libreoffice_impress"
    return None


def observe_active_app(a11y_xml: Optional[str]) -> Optional[str]:
    """Return the canonical domain key of the OS's focused app, else None.

    Picks the first non-ignored ``<application>`` whose frame is
    ``state:active="true"`` and maps its name to a domain. LibreOffice is
    special-cased (one process backs Calc / Writer / Impress — title
    disambiguates).

    Best-effort: None on parse error, missing input, or unmapped app. Caller
    decides whether None means "trust intent" or "no app focused".
    """
    if not a11y_xml:
        return None
    try:
        tree = ET.ElementTree(ET.fromstring(a11y_xml))
    except ET.ParseError:
        return None
    active_attr = "{{{:}}}active".format(_STATE_NS)
    for application in list(tree.getroot()):
        app_name = (application.attrib.get("name") or "").strip()
        if not app_name or app_name in _IGNORED_APP_NAMES:
            continue
        active_frame_title: Optional[str] = None
        for frame in application:
            if frame.attrib.get(active_attr, "false") == "true":
                active_frame_title = (frame.attrib.get("name") or "").strip()
                break
        if active_frame_title is None:
            continue
        # LibreOffice: one process, three domains — pick from title.
        if app_name in ("soffice", "LibreOffice", "libreoffice"):
            lo = _resolve_libreoffice(active_frame_title)
            if lo:
                return lo
            # Variant unknown (Start Center, dialog) — keep scanning.
            continue
        domain = _APP_NAME_TO_DOMAIN.get(app_name)
        if domain:
            return domain
        domain = _APP_NAME_TO_DOMAIN.get(app_name.lower())
        if domain:
            return domain
    return None

"""Detect the OS's currently-focused application from the raw AT-SPI a11y XML.

Used by the agent to derive ``physical_domain`` — the app whose window is
actually focused right now — independently of ``_current_domain`` (which
encodes the agent's INTENT). Treating the two as the same caused several
multi-app failure modes:

  * cross-app a11y bleeding (verifier matched a URL in app B's a11y
    against an outcome whose app is A);
  * action handler routing for the wrong app when the planner thought
    it was in A but the OS focus was still on B;
  * silent ``_activate_app_for_domain`` failures going undetected
    because no one checks whether the focus actually changed.

``observe_active_app(a11y_xml)`` returns the canonical domain key (member
of ``_APP_SPECS`` in ``open_app.py``) for the app whose root frame has
``state:active="true"``, or None when no useful app is focused.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET


# --------------------------------------------------------------------------- #
# App-switch closed loop (EVIDENCE_BINDING_DESIGN §8) — agent-side half.
# The VM-side half lives in open_app.py's script (focus verify + __OPEN_APP__
# sentinel). Gated by ENABLE_APP_SWITCH_VERIFY (default 0 = old behaviour:
# fire-and-forget activate + the 5-round intent demote).
# --------------------------------------------------------------------------- #

def app_switch_verify_enabled() -> bool:
    from mm_agents.structagent.config import CAConfig
    return CAConfig.from_env().app_switch_verify


_SENTINEL = "__OPEN_APP__"


def parse_open_app_result(runner_result: Any) -> Dict[str, Any]:
    """Extract the __OPEN_APP__ sentinel from a run_python_script result.

    Returns {"ok": True/False, "failure_type": ..., "detail": ...} when the
    sentinel is found; {"ok": None} when it isn't (old script / runner error) —
    callers MUST treat ok=None as "unverified, behave as before", never as a
    failure (an old VM image without the new script must not brick switching).

    The sentinel may arrive raw, or JSON-escaped inside the runner's own JSON
    envelope ({"output": "...__OPEN_APP__{\\"app\\"...}"}). raw_decode parses
    the first complete JSON value and ignores the trailing envelope text; the
    LAST sentinel wins (retries print multiple)."""
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
    the replan can address the actual cause instead of blind-retrying."""
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


# AT-SPI state namespace. The OSWorld VM server emits this exact URL
# (see desktop_env/server/main.py — same string consumed by
# autoglm/prompt/accessibility_tree_handle.py's ``state_ns_ubuntu``).
# An earlier draft used the upstream linuxfoundation URL — wrong, the
# VM doesn't emit it, so every frame's ``active`` lookup returned the
# attribute's default ("false") and observe_active_app reported None
# on EVERY step.
_STATE_NS = "https://accessibility.ubuntu.example.org/ns/state"

# Apps that always show up as "applications" in the AT-SPI dump but are
# not user-facing focus targets — skip these even when they expose an
# active frame (gnome-shell occasionally does during transitions).
_IGNORED_APP_NAMES = {
    "gnome-shell", "gjs", "mutter", "ibus-x11", "ibus-ui-gtk3",
    "Xorg", "xdg-desktop-portal", "xdg-desktop-portal-gnome",
    "evolution-alarm-notify", "tracker-miner-fs",
}


# AT-SPI application name → canonical domain key in ``open_app._APP_SPECS``.
# Libreoffice ("soffice") is handled separately because Calc / Writer /
# Impress all share the same process; the frame title is used to
# disambiguate (see _resolve_libreoffice).
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

    LibreOffice frame titles include "... — LibreOffice Calc" (or "Writer"
    or "Impress"). Returns None when no marker is found (e.g. the Start
    Center, dialog windows). The em-dash and en-dash are both used in
    practice; check for both.
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
    """Return the canonical domain key of the OS's currently-focused app,
    or None when no resolvable app is active.

    The AT-SPI tree lists each running application as a top-level
    ``<application name="...">`` element; each frame within that
    application carries ``state:active="true"`` when its window holds
    focus. We pick the first non-ignored application whose frame is
    active and map its name to a domain key. LibreOffice needs special
    handling because the same process backs Calc / Writer / Impress —
    the frame title disambiguates.

    Best-effort: returns None on parse error, missing input, or apps
    we don't have a mapping for. Caller decides whether None means
    "trust intent" or "explicit no-app-focused".
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
        # Find this app's active frame (if any).
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
            # Title doesn't reveal the variant (Start Center, dialog) —
            # fall through, return None below if no other resolved app.
            continue
        # Direct mapping.
        domain = _APP_NAME_TO_DOMAIN.get(app_name)
        if domain:
            return domain
        # Case-insensitive fallback.
        domain = _APP_NAME_TO_DOMAIN.get(app_name.lower())
        if domain:
            return domain
    return None

"""Helpers for the ``navigate(url='...')`` action.

Drives Chrome to a URL via the DevTools HTTP endpoint on VM port 1337
(socat'd to host 9222) — one reliable action instead of the GUI
``ctrl+l`` + type dance that depends on the address bar being grounded
by the vision model.

  new_tab=True   open in a fresh tab and focus it (``/json/new?<url>``).
                 Default — doesn't disturb the agent's current tab.
  new_tab=False  navigate the focused tab via CDP ``Page.navigate``.

The generated VM-side script is stdlib-only; ``websocat`` is used for
focused-tab nav when present, with graceful fallback otherwise.
"""
from __future__ import annotations

import json


_VALID_URL_PREFIXES = ("http://", "https://", "chrome://", "about:",
                       "file://", "ftp://")


def _validate_url(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ValueError("navigate requires a non-empty `url`")
    if not url.lower().startswith(_VALID_URL_PREFIXES):
        # Common slip: "google.com" without scheme. Auto-prepend https.
        url = "https://" + url
    return url


def build_navigate_script(url: str, *, new_tab: bool = False) -> str:
    """Return a self-contained Python script the VM runs to navigate."""
    url = _validate_url(url)
    body = _RUNTIME_BODY_NEW_TAB if new_tab else _RUNTIME_BODY_FOCUSED
    header = (
        "_url = " + json.dumps(url) + "\n"
        "_cdp_host = '127.0.0.1'\n"
        "_cdp_port = 1337\n"  # in-VM debugger port (socat'd to 9222)
    )
    return header + body


# ── Runtime bodies (executed inside the VM) ─────────────────────────


_RUNTIME_BODY_NEW_TAB = r"""
import json, urllib.parse, urllib.request, time

def _go():
    q = urllib.parse.quote(_url, safe=':/?&=#%')
    req = urllib.request.Request(
        f'http://{_cdp_host}:{_cdp_port}/json/new?{q}',
        method='PUT')
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception:
        # Fallback: some Chrome versions reject PUT; try GET.
        try:
            with urllib.request.urlopen(
                    f'http://{_cdp_host}:{_cdp_port}/json/new?{q}',
                    timeout=15) as r:
                return json.loads(r.read().decode('utf-8'))
        except Exception as e:
            print(f'[navigate] failed to open new tab: {e}')
            return None

_tab = _go()
if _tab:
    # Activate the new tab so the agent's next screenshot lands on it.
    try:
        tid = _tab.get('id')
        if tid:
            urllib.request.urlopen(
                f'http://{_cdp_host}:{_cdp_port}/json/activate/{tid}',
                timeout=5).read()
    except Exception:
        pass
    print(f'[navigate] new_tab opened: {_url}')
else:
    print(f'[navigate] FAILED for {_url}')

# Brief settle so the next screenshot captures the loaded page rather
# than a blank tab. Most sites first-paint < 2s; full load varies.
time.sleep(3)
"""


_RUNTIME_BODY_FOCUSED = r"""
import json, urllib.request, time

def _list_tabs():
    try:
        with urllib.request.urlopen(
                f'http://{_cdp_host}:{_cdp_port}/json/list',
                timeout=10) as r:
            return json.loads(r.read().decode('utf-8'))
    except Exception as e:
        print(f'[navigate] cannot list tabs: {e}')
        return []

# Pick the FOCUSED page-type tab.
_tabs = [t for t in _list_tabs() if t.get('type') == 'page']
if not _tabs:
    print('[navigate] no page tab found; cannot navigate focused')
else:
    # Heuristic for "focused": the tab whose websocket URL Chrome
    # listed FIRST (most recently activated). CDP doesn't expose
    # a clean "is_active" so this is the best we get without
    # opening a websocket. Empirically reliable enough for our path.
    _target = _tabs[0]
    _ws = _target.get('webSocketDebuggerUrl')
    _tid = _target.get('id')
    _ok = False
    # Prefer websocat (small dep, sometimes installed). Fall back to
    # opening a new tab — still navigates the user where they want.
    import shutil, subprocess
    if _ws and shutil.which('websocat'):
        msg = json.dumps({
            'id': 1, 'method': 'Page.navigate',
            'params': {'url': _url},
        })
        try:
            r = subprocess.run(['websocat', '-1', _ws],
                                input=msg, capture_output=True,
                                text=True, timeout=15)
            _ok = (r.returncode == 0)
        except Exception as e:
            print(f'[navigate] websocat failed: {e}')
    if not _ok:
        # Last resort: open a new tab (still gets us to the URL).
        import urllib.parse
        q = urllib.parse.quote(_url, safe=':/?&=#%')
        try:
            urllib.request.urlopen(
                f'http://{_cdp_host}:{_cdp_port}/json/new?{q}',
                timeout=15).read()
            _ok = True
            print('[navigate] focused-tab nav unavailable; opened new tab')
        except Exception as e:
            print(f'[navigate] FAILED: {e}')
    if _ok:
        print(f'[navigate] focused tab → {_url}')
import time
time.sleep(3)
"""

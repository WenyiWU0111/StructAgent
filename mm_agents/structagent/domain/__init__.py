"""Per-domain knowledge injection for the planner + init_ledger.

OSWorld task instructions carry a ``[<domain>]`` prefix (vscode / chrome /
libreoffice_writer / ...). Knowledge is split across three folders so the
two audiences don't cross-contaminate (verify-spec rules confuse the
planner; planning rules don't help init_ledger pick a verify spec):

    domain/shared/<d>.md     — needed by both (e.g. LibreOffice palette hex
                               table: planner emits the color, verifier
                               matches it in the verify spec).
    domain/planner/<d>.md    — planner / decomposer / rescue / search: how
                               to act on the app (paths, UI, pitfalls).
    domain/verifier/<d>.md   — init_ledger only: which verify.kind, which
                               evidence channel, how to phrase patterns.

``get_planner_knowledge`` / ``get_verifier_knowledge`` each return
shared + their audience file (or "" for missing / None / unknown domain).
``get_evidence_guide`` is a separate channel (see ``evidence_guides``).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .evidence_guides import get_evidence_guide  # noqa: F401  (re-export)


_DIR = os.path.dirname(os.path.abspath(__file__))
_SHARED_DIR = os.path.join(_DIR, "shared")
_PLANNER_DIR = os.path.join(_DIR, "planner")
_VERIFIER_DIR = os.path.join(_DIR, "verifier")

logger = logging.getLogger(__name__)


# Normalize aliases (OSWorld uses both "vscode" and "vs_code") to the
# canonical filename stem.
_ALIAS = {
    "vscode": "vs_code",
    "vs_code": "vs_code",
    "chrome": "chrome",
    "google-chrome": "chrome",
    "thunderbird": "thunderbird",
    "libreoffice_calc": "libreoffice_calc",
    "libreoffice_writer": "libreoffice_writer",
    "libreoffice_impress": "libreoffice_impress",
    "gimp": "gimp",
    "vlc": "vlc",
    "os": "os",
    "multi_apps": "multi_apps",
}


def _canonical(domain: Optional[str]) -> Optional[str]:
    if not domain:
        return None
    c = _ALIAS.get(domain.lower().strip(), domain.lower().strip())
    if not c or not c.replace("_", "").isalnum():
        return None  # path-safety
    return c


# Domains that must be solved through GUI / a11y ONLY: the coding route
# (cli_run / edit_json / shell on config files) is unreliable because the
# live app overwrites its on-disk state and the evaluator checks live
# state, not the file. For these the decomposer gets no cli_run/edit_json
# and the verifier prefers a11y/url over file_grep. chrome (Preferences/
# Bookmarks overwritten), gimp (Script-Fu fragile), thunderbird. os /
# vs_code / libreoffice_* / vlc keep coding — file/shell is the evaluator-
# checked ground truth there.
GUI_ONLY_DOMAINS = frozenset({"chrome", "gimp", "thunderbird"})


def is_gui_only(domain: Optional[str]) -> bool:
    """True iff ``domain`` is GUI/a11y-only (coding route disallowed).
    Aliases canonicalized; unknown / None / multi_apps → False (permissive)."""
    return _canonical(domain) in GUI_ONLY_DOMAINS


def _read_md(folder: str, canonical: str) -> str:
    path = os.path.join(folder, f"{canonical}.md")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except Exception:
        return ""


def _concat(*blocks: str) -> str:
    """Join non-empty blocks with a blank-line separator."""
    return "\n\n".join(b for b in blocks if b)


def get_planner_knowledge(domain: Optional[str]) -> str:
    """Planner-audience knowledge: shared/<d>.md + planner/<d>.md.
    Read by planner / decomposer / rescue / search; excludes verify-spec
    rules (those are under verifier/, init_ledger only). "" if no files."""
    c = _canonical(domain)
    if c is None:
        return ""
    return _concat(_read_md(_SHARED_DIR, c), _read_md(_PLANNER_DIR, c))


def get_verifier_knowledge(domain: Optional[str]) -> str:
    """Verifier-audience knowledge: shared/<d>.md + verifier/<d>.md.
    Read only by ``init_ledger`` for verify specs. shared/ content (color
    tables, palette diffs) is included so verify-spec hex patterns match
    what the planner emits. "" if no files."""
    c = _canonical(domain)
    if c is None:
        return ""
    return _concat(_read_md(_SHARED_DIR, c), _read_md(_VERIFIER_DIR, c))


# Back-compat alias for old call sites → planner-audience body. The one
# site that needed the verifier body (init_ledger) was redirected directly.
get_domain_knowledge = get_planner_knowledge

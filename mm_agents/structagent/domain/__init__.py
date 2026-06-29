"""Per-domain knowledge injection for the planner + init_ledger.

OSWorld task instructions carry a ``[<domain>]`` prefix — vscode / chrome /
thunderbird / libreoffice_writer / etc. The decomposer, planner and
init_ledger LLMs all benefit from app-specific conventions but injecting
ALL such knowledge into one shared prompt bloats it and cross-contaminates
audience: the verify-spec authoring rules confuse the planner, and the
"how to plan / which path to use" rules don't help init_ledger pick a
correct verify spec.

To fix that, domain knowledge now lives in THREE folders:

    domain/shared/<domain>.md        — content BOTH audiences need (e.g.
                                       LibreOffice Standard-palette hex
                                       table — planner uses it to emit
                                       the correct color in the action,
                                       verifier uses it to write a
                                       matching hex pattern in the
                                       verify spec).

    domain/planner/<domain>.md       — read by the planner / decomposer /
                                       rescue / search agents. Covers how
                                       to act on the app: canonical paths,
                                       UI conventions, common pitfalls,
                                       structured-action menus.

    domain/verifier/<domain>.md      — read ONLY by init_ledger when it's
                                       authoring verify specs. Covers
                                       which ``verify.kind`` to pick for
                                       different end-states, what
                                       evidence channel is authoritative,
                                       how to phrase patterns.

Two getters; each concatenates ``shared/<d>.md`` with its audience-
specific file so callers see one block:

  ``get_planner_knowledge(domain)``  → shared + planner content
  ``get_verifier_knowledge(domain)`` → shared + verifier content

Both return empty string when the file is missing — safe to call with
``None`` or an unknown domain. ``get_evidence_guide(domain)`` is a
separate compact channel; see ``evidence_guides`` for that one.

Usage:

    from mm_agents.structagent.domain import get_planner_knowledge
    block = get_planner_knowledge("chrome")
    if block:
        prompt += "\\n\\n" + block

    from mm_agents.structagent.domain import get_verifier_knowledge
    init_block = get_verifier_knowledge("chrome")     # only init_ledger
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


# Normalize domain aliases — OSWorld uses both "vscode" and "vs_code"
# in different places. Map to the canonical filename stem.
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
        return None  # path-safety check
    return c


# Domains whose tasks must be solved through the GUI / a11y ONLY — the
# coding route (cli_run / edit_json / shell against config files) is
# unreliable here (the live app overwrites its on-disk state and the
# evaluator checks live state, not the file). For these the decomposer
# is NOT given cli_run/edit_json, and the verifier prefers a11y/url over
# file_grep. Researched from domain/*.md + evaluator func distribution
# (2026-06-13): chrome (Preferences/Bookmarks overwritten + eval checks
# live), gimp (Script-Fu cli fragile, a11y default), thunderbird (GUI +
# a11y default). os / vs_code / libreoffice_* / vlc KEEP coding — there
# the file/shell IS the legitimate, evaluator-checked ground truth.
GUI_ONLY_DOMAINS = frozenset({"chrome", "gimp", "thunderbird"})


def is_gui_only(domain: Optional[str]) -> bool:
    """True iff ``domain`` is GUI/a11y-only (coding route disallowed).
    Canonicalizes aliases first; unknown / None / multi_apps → False
    (permissive: keep coding unless the domain is a known GUI-only one)."""
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

    Read by: planner / decomposer / rescue / search agents. Excludes
    "Verify kind" / verify-spec authoring rules — those live under
    ``verifier/`` and are scoped to ``init_ledger`` callers only.

    Returns "" when no files exist for the domain (safe default).
    """
    c = _canonical(domain)
    if c is None:
        return ""
    return _concat(_read_md(_SHARED_DIR, c), _read_md(_PLANNER_DIR, c))


def get_verifier_knowledge(domain: Optional[str]) -> str:
    """Verifier-audience knowledge: shared/<d>.md + verifier/<d>.md.

    Read ONLY by ``init_ledger`` when authoring verify specs. Includes
    cross-audience content (color tables, coordinate system, palette
    diffs) from ``shared/`` so verify-spec hex patterns match what the
    planner emits via the structured action.

    Returns "" when no files exist for the domain.
    """
    c = _canonical(domain)
    if c is None:
        return ""
    return _concat(_read_md(_SHARED_DIR, c), _read_md(_VERIFIER_DIR, c))


# Back-compat alias: a few existing call sites still import the old
# name. ``get_domain_knowledge`` resolves to the planner-audience body
# so legacy planner / search / rescue calls keep their current
# behavior. ``init_ledger`` was the one site we needed to redirect, and
# its call sites are updated to ``get_verifier_knowledge`` directly.
get_domain_knowledge = get_planner_knowledge

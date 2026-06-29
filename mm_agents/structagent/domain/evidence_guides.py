"""Per-domain *evidence guide* — a SHORT block telling the planner which
observation channel is authoritative when it judges whether the previous
subgoal's ``expected_post_state`` is satisfied.

Distinct from ``get_domain_knowledge()``:

  - ``get_domain_knowledge(domain)`` returns the big ``<domain>.md``
    planning framing. It is injected ONCE, into the initial-plan prompt
    at task start.
  - ``get_evidence_guide(domain)`` returns a compact rules block embedded
    into the ``<last_subgoal_assessment>`` block — which the planner
    emits on EVERY turn. The per-domain done-judgment rules must travel
    with that block, so they cannot live in the step-0-only ``.md``.

Keep each guide short (it is paid for on every planner turn) and strictly
about *which evidence source to read and how to read it* — NOT about how
to plan the task.
"""
from __future__ import annotations
from typing import Optional


# Domain-name aliases → canonical key. Mirrors ``domain_knowledge._ALIAS``.
_ALIAS = {
    "vscode": "vs_code", "vs_code": "vs_code",
    "chrome": "chrome", "google-chrome": "chrome",
    "thunderbird": "thunderbird",
    "libreoffice_calc": "libreoffice_calc",
    "libreoffice_writer": "libreoffice_writer",
    "libreoffice_impress": "libreoffice_impress",
    "gimp": "gimp", "vlc": "vlc", "os": "os",
    "multi_apps": "multi_apps",
}


# --- a11y-driven domains (Chrome, Thunderbird) ----------------------------
# The a11y tree is the primary done-judgment channel; the screenshot only
# corroborates. These reading rules are a frequent source of false-positive
# done=YES, hence they ride along on every turn.
_A11Y_GUIDE = """\
EVIDENCE SOURCE — a11y tree is primary, screenshot corroborates:
  • a11y rows are tab-separated <tag>\\t<text>\\t<state>. The state
    column lists ONLY attributes that are TRUE; an absent attribute is
    FALSE. `checked` = checkbox/toggle ON; `selected` = radio/menu-item
    chosen; `focused` = has keyboard focus (irrelevant to on/off);
    `enabled` = clickable (EVERY live control has it). `enabled` alone
    is NOT proof a feature is on — a toggle-ON / radio-selected subgoal
    needs a row showing `checked` (or `selected`); a row with only
    `enabled`/`enabled,focused` PROVES it is OFF → done=NO.
  • The a11y tree is VIEWPORT-CENTRIC: "not in the tree" ≠ "not on the
    page" — it may be below the fold. Only cite a11y absence after
    scrolling the relevant region.
  • A typeahead value (airport code, city, recipient, query) is NOT
    committed while a suggestion/autocomplete dropdown is still visible
    next to it — the open dropdown IS the "not committed yet" signal.
  • SIDE-EFFECT AUDIT — if the last action toggled a checkbox/radio/
    switch: some pages couple controls (one toggle auto-flips another).
    Check EVERY other toggle in the same form/dialog; if any flipped
    unexpectedly, set done=NO."""


def _office_guide(app: str, verify: str) -> str:
    return (
        f"EVIDENCE SOURCE — LibreOffice {app}: the STRUCTURE block above is\n"
        f"a LIVE UNO re-probe refreshed every turn — the AUTHORITATIVE source\n"
        f"for structural properties (text content, font name/size/bold/\n"
        f"italic/color, paragraph alignment/indent, element counts,\n"
        f"positions/sizes).\n"
        f"  • Judge structural-property subgoals from the STRUCTURE block;\n"
        f"    judge {verify}-verified outcomes from the verifier (it re-runs\n"
        f"    a real check each turn). Do NOT let stale screen history\n"
        f"    override either.\n"
        f"  • The screenshot is authoritative only for transient UI: open\n"
        f"    dialogs, menus, navigation, selection highlight."
    )


_OS_GUIDE = """\
EVIDENCE SOURCE — OS / shell: the last command's output block
(returncode + stdout + stderr) IS the empirical result — judge
command-driven subgoals from it, not the screenshot (a non-interactive
command leaves no terminal window on screen). The screenshot is
authoritative only for desktop-GUI state (windows, dialogs, file
manager)."""


_GUIDES = {
    "chrome": _A11Y_GUIDE,
    "thunderbird": _A11Y_GUIDE,
    "libreoffice_impress": _office_guide("Impress", "impress_verify"),
    "libreoffice_calc": _office_guide("Calc", "calc_verify"),
    "libreoffice_writer": _office_guide("Writer", "shell_command"),
    "os": _OS_GUIDE,
}


# Used for domains without a specific guide (gimp, vlc, vs_code, multi_apps,
# unknown). Domain-neutral: name the reliable channels without assuming one.
_GENERIC_GUIDE = """\
EVIDENCE SOURCE: judge the previous subgoal from the most reliable
channel available — a deterministic verifier result when one covers it,
otherwise the a11y tree for control state and the screenshot for visible
layout. A control being `enabled` is NOT proof it is ON — a toggle/radio
needs `checked`/`selected`."""


def get_evidence_guide(domain: Optional[str]) -> str:
    """Return the compact evidence-reading block for ``domain``.

    Always returns a non-empty string — falls back to a domain-neutral
    guide for unknown / unmapped domains so the assessment block never
    ships without evidence-source guidance.
    """
    if not domain:
        return _GENERIC_GUIDE
    canonical = _ALIAS.get(domain.lower().strip(), domain.lower().strip())
    return _GUIDES.get(canonical, _GENERIC_GUIDE)

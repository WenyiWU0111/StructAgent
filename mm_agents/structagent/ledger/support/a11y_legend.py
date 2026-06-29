"""A11y state-column legend, co-located with the a11y rows so consumers
(perceiver / KeyNode detector / planner) always have the semantics
inline rather than having to remember them from the system prompt.

Single source of truth — used by all sites that dump a11y to an LLM.
"""
from __future__ import annotations


# Inline header explaining state-column semantics. Only TRUE attributes
# appear; absent attributes are FALSE. Common pitfall the legend
# explicitly calls out: ``enabled`` is interactability, NOT activation —
# the LLM has been observed conflating "enabled" with "tab is selected"
# / "feature is on", which produces false-positive DONE.
A11Y_STATE_LEGEND: str = """\
A11Y STATE LEGEND (states are TRUE-only; absent attributes = false):
  enabled   — control is interactable / clickable.
              IMPORTANT: 'enabled' alone does NOT mean "feature is ON"
              or "tab/option is selected". Every active control has
              'enabled'; this alone is NEVER proof of satisfaction.
  checked   — checkbox / toggle is in the ON position.
  selected  — radio-button / menu-item / TAB / list-item is the
              currently chosen one. For a TAB outcome, requires
              'selected', not just 'enabled'.
  focused   — has keyboard focus. NOT relevant to on/off state and
              NEVER a satisfaction signal on its own.
  expanded  — accordion / tree-node / details / combo-box is open."""


def with_legend(a11y_block: str, *, header: str = "a11y rows (tag\\ttext\\tstate):") -> str:
    """Return ``a11y_block`` prefixed with ``A11Y_STATE_LEGEND`` and a
    short heading. No-op (returns input) when ``a11y_block`` is empty,
    so empty-a11y prompts stay empty.
    """
    if not a11y_block:
        return a11y_block
    return f"{A11Y_STATE_LEGEND}\n\n{header}\n{a11y_block}"

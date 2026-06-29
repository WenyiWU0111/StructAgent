"""A11y state-column legend, co-located with the a11y rows so consumers
(perceiver / KeyNode detector / planner) get the semantics inline instead of
relying on the system prompt. Single source of truth for a11y dumps to an LLM.
"""
from __future__ import annotations


# State-column legend: only TRUE attributes appear, absent = FALSE. Calls out
# the common pitfall that ``enabled`` is interactability, not activation — LLMs
# conflate it with "tab selected" / "feature on" and emit false-positive DONE.
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
    """Prefix ``a11y_block`` with the legend and a short heading. No-op on empty
    input so empty-a11y prompts stay empty."""
    if not a11y_block:
        return a11y_block
    return f"{A11Y_STATE_LEGEND}\n\n{header}\n{a11y_block}"

"""On-disk markdown plan file (T4).

Each task run gets ``<results_dir>/plan.md`` — a human-readable
rendering of the current Outcome DAG + state. The file is rewritten
after every successful REPLAN so it always reflects the planner's
most recent commitment.

Why this matters:

  - **Plan thrash check**: planner can ``read_plan()`` at DONE-decision
    time and see its own prior subgoal commitments, not a re-derived
    snapshot of in-memory DAG.
  - **Triage**: ``cat <results_dir>/plan.md`` is faster than reading
    JSONL to reconstruct what the planner thought it was doing.
  - **Token saving** (future): when LAYER 3 of the planner prompt
    reads this file rather than rendering the DAG inline every turn,
    we save the repeated render cost.

Co-located with task artifacts (``<results_dir>/plan.md``) rather
than ``~/.structagent/plans/`` because OSWorld already gives each
run a dedicated dir via ``lib_run_single.py:181`` — no separate slug
infrastructure needed.

See plan file Part I.3.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mm_agents.structagent.ledger.core.ledger import (
        Outcome, ProgressLedger,
    )


PLAN_FILENAME = "plan.md"
HISTORY_DIRNAME = "plans"


# === Storage layer =============================================== #


def get_plan_path(results_dir: Optional[str]) -> Optional[Path]:
    """Return ``<results_dir>/plan.md`` (the always-latest snapshot)
    or None if ``results_dir`` is unset (e.g. running outside the
    lib_run_single harness).
    """
    if not results_dir:
        return None
    return Path(results_dir) / PLAN_FILENAME


def get_history_dir(results_dir: Optional[str]) -> Optional[Path]:
    """Return ``<results_dir>/plans/`` (per-step history directory)
    or None if ``results_dir`` is unset.
    """
    if not results_dir:
        return None
    return Path(results_dir) / HISTORY_DIRNAME


def get_history_path(
    results_dir: Optional[str], step_idx: int,
) -> Optional[Path]:
    """Return ``<results_dir>/plans/step_NNNN.md`` for the given
    step. 4-digit zero-pad keeps ``ls plans/`` sorted correctly up
    to 9999 steps (we'd terminate the run long before).
    """
    history_dir = get_history_dir(results_dir)
    if history_dir is None:
        return None
    return history_dir / f"step_{step_idx:04d}.md"


def read_plan(results_dir: Optional[str]) -> Optional[str]:
    """Read the latest plan markdown from ``<results_dir>/plan.md``.
    Returns ``None`` on missing file or unset dir — callers never
    have to try/except.
    """
    path = get_plan_path(results_dir)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def read_plan_at_step(
    results_dir: Optional[str], step_idx: int,
) -> Optional[str]:
    """Read a historical plan snapshot from
    ``<results_dir>/plans/step_NNNN.md``. Returns ``None`` if no
    snapshot was recorded at that step.
    """
    path = get_history_path(results_dir, step_idx)
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via mkstemp + rename in
    the same directory (so the rename is on the same filesystem).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".plan.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, str(path))
    except Exception:
        # Best-effort cleanup of the temp file; rename target may
        # already exist from a prior successful write.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_plan(
    results_dir: Optional[str],
    text: str,
    *,
    step_idx: Optional[int] = None,
) -> bool:
    """Write the plan markdown atomically. No-op if ``results_dir``
    is unset. Returns True on write, False on no-op.

    Writes up to two files:

      - ``<results_dir>/plan.md`` — always the latest snapshot
        (overwritten each call). What ``cat plan.md`` and future
        ``read_plan()`` consumers see.

      - ``<results_dir>/plans/step_NNNN.md`` — historical snapshot,
        only when ``step_idx`` is provided. Never overwritten across
        steps; lets you ``diff plans/step_0007.md plans/step_0014.md``
        to see how the plan evolved.

    If two ``write_plan`` calls happen at the *same* ``step_idx``
    (rare — e.g. initial install + REPLAN both before _pa_predict
    returns at step 0), the per-step history file is overwritten.
    The cumulative plan.md file always reflects the most recent
    write.

    Atomic because the planner is in the middle of a turn when this
    fires; a partial-write crash would leave the plan file in a torn
    state and the next ``read_plan`` would see garbage.
    """
    latest_path = get_plan_path(results_dir)
    if latest_path is None:
        return False

    _atomic_write(latest_path, text)

    if step_idx is not None:
        history_path = get_history_path(results_dir, step_idx)
        if history_path is not None:
            _atomic_write(history_path, text)

    return True


def list_history_steps(results_dir: Optional[str]) -> "list[int]":
    """Return sorted list of step indices for which a historical
    snapshot exists under ``<results_dir>/plans/``. Returns empty
    list if dir is unset or no history exists.
    """
    history_dir = get_history_dir(results_dir)
    if history_dir is None or not history_dir.is_dir():
        return []
    steps: "list[int]" = []
    for entry in history_dir.iterdir():
        name = entry.name
        if not (name.startswith("step_") and name.endswith(".md")):
            continue
        try:
            step = int(name[len("step_"):-len(".md")])
        except ValueError:
            continue
        steps.append(step)
    steps.sort()
    return steps


# === Rendering layer ============================================== #

_STATE_ICON = {
    "verified": "✓",
    "pending": "·",
    "reverted": "✗",
}


def _outcome_state_icon(outcome: "Outcome", is_focus: bool) -> str:
    """Icon prefix: ▶ for the currently-focused PENDING outcome, else
    the per-state icon."""
    if is_focus and outcome.state == "pending":
        return "▶"
    return _STATE_ICON.get(outcome.state, "?")


def _render_outcome(outcome: "Outcome", is_focus: bool) -> str:
    """Render one Outcome as a markdown section."""
    icon = _outcome_state_icon(outcome, is_focus)
    focus_tag = " (current focus)" if is_focus else ""
    lines = [
        f"### {icon} `{outcome.id}` — {outcome.description}{focus_tag}",
        f"- state: **{outcome.state.upper()}**",
    ]
    if outcome.state == "verified" and outcome.last_satisfy_step is not None:
        lines.append(f"- verified at step {outcome.last_satisfy_step}")
    if outcome.state == "reverted":
        if outcome.last_invalidate_step is not None:
            lines.append(
                f"- reverted at step {outcome.last_invalidate_step} "
                f"(revert_count={outcome.revert_count})"
            )
        else:
            lines.append(f"- revert_count={outcome.revert_count}")
    if outcome.evidence_hint:
        lines.append(f"- evidence_hint: {outcome.evidence_hint}")
    if outcome.depends_on:
        deps = ", ".join(f"`{d}`" for d in outcome.depends_on)
        lines.append(f"- depends_on: {deps}")
    if outcome.app:
        lines.append(f"- app: `{outcome.app}`")
    return "\n".join(lines)


def render_plan_md(
    ledger: "ProgressLedger",
    *,
    instruction: str,
    current_focus_id: Optional[str],
    current_subgoal: Optional[str] = None,
    step_idx: Optional[int] = None,
    plan_strategy: Optional[str] = None,
    subgoal_queue: Optional[List[str]] = None,
    subgoal_expected_post_states: Optional[List[str]] = None,
    abandoned_subgoals: Optional[List[str]] = None,
) -> str:
    """Render planner state + Outcome DAG + task context as human-readable
    markdown. Stable shape so ``cat plan.md`` and ``diff turn_N`` are
    both useful triage tools.

    Two distinct planner data sources are surfaced:

      - ``ledger.required_outcomes`` (Outcome DAG) — high-level
        milestones declared by ``init_ledger`` at task start. Usually
        1-5 entries; the planner drives toward these.
      - ``subgoal_queue`` + ``current_subgoal`` + ``abandoned_subgoals``
        (Planner Subgoals) — the per-turn step decomposition the
        planner emits in its ``<plan>``. These are the actual tactics
        being attempted to satisfy the outcomes.

    Both matter for triage; previously only the Outcome DAG was
    rendered (see plan file Part I.3).
    """
    parts: list = []

    header = "# Plan"
    if step_idx is not None:
        header += f" (as of step {step_idx})"
    parts.append(header)
    parts.append("")
    parts.append(f"**Task:** {instruction.strip()}")
    parts.append("")

    # Strategy — the one-sentence approach the planner declared in the
    # head of its most recent ``<plan>``. Helps explain WHY the subgoal
    # queue looks the way it does.
    if plan_strategy and plan_strategy.strip():
        parts.append(f"**Strategy:** {plan_strategy.strip()}")
        parts.append("")

    # Planner subgoals — current + upcoming queue + abandoned. The
    # planner's per-turn decomposition (each REPLAN rewrites this).
    has_subgoals = (
        current_subgoal
        or (subgoal_queue and len(subgoal_queue) > 0)
        or (abandoned_subgoals and len(abandoned_subgoals) > 0)
    )
    if has_subgoals:
        parts.append("## Planner subgoals")
        parts.append("")

        if current_subgoal:
            parts.append(f"### ▶ Current: {current_subgoal.strip()}")
            parts.append("")

        if subgoal_queue:
            parts.append(f"### · Upcoming queue ({len(subgoal_queue)})")
            for i, sg in enumerate(subgoal_queue, start=1):
                line = f"  {i}. {sg.strip()}"
                # Surface the parallel expected_post_state when available,
                # so triage can spot mismatch between subgoal and what
                # the planner expected the screen to look like after.
                if (subgoal_expected_post_states
                        and i - 1 < len(subgoal_expected_post_states)):
                    eps = (subgoal_expected_post_states[i - 1] or "").strip()
                    if eps:
                        line += f"\n     _expected:_ {eps}"
                parts.append(line)
            parts.append("")

        if abandoned_subgoals:
            parts.append(f"### ✗ Abandoned ({len(abandoned_subgoals)})")
            for i, sg in enumerate(abandoned_subgoals, start=1):
                parts.append(f"  {i}. {sg.strip()}")
            parts.append("")

    # Outcome DAG (milestones) — declared at init_ledger time, driven
    # toward by the subgoal queue above. Order preserved from ledger
    # (DAG topo order via depends_on).
    parts.append("## Outcomes (milestones from init_ledger)")
    parts.append("")
    if not ledger.required_outcomes:
        parts.append("_(no outcomes declared yet)_")
    else:
        for outcome in ledger.required_outcomes:
            is_focus = outcome.id == current_focus_id
            parts.append(_render_outcome(outcome, is_focus))
            parts.append("")

    # Slot snapshot — captured values agent has accumulated.
    if ledger.slots:
        parts.append("## Captured slots")
        parts.append("")
        for name, slot in ledger.slots.items():
            val = getattr(slot, "value", None)
            parts.append(f"- `{name}`: {val!r}")
        parts.append("")

    # Failed paths — what's been ruled out.
    if ledger.failed_paths:
        parts.append("## Dead-end paths already tried")
        parts.append("")
        for fp in ledger.failed_paths:
            level = getattr(fp, "level", "?")
            path = getattr(fp, "path", "?")
            why = getattr(fp, "why", "")
            parts.append(f"- [{level}] **{path}** — {why}")
        parts.append("")

    # Counter summary footer — quick at-a-glance state.
    n_total = len(ledger.required_outcomes)
    n_verified = sum(1 for o in ledger.required_outcomes if o.state == "verified")
    n_reverted = sum(1 for o in ledger.required_outcomes if o.state == "reverted")
    n_pending = n_total - n_verified - n_reverted
    parts.append(
        f"_Summary: outcomes {n_verified}/{n_total} verified · "
        f"{n_reverted} reverted · {n_pending} pending"
    )
    if subgoal_queue is not None or current_subgoal:
        n_queue = len(subgoal_queue) if subgoal_queue else 0
        n_abandoned = len(abandoned_subgoals) if abandoned_subgoals else 0
        n_current = 1 if current_subgoal else 0
        parts[-1] += (
            f" · subgoals current={n_current} queue={n_queue} "
            f"abandoned={n_abandoned}"
        )
    parts[-1] += "_"
    parts.append("")

    return "\n".join(parts)

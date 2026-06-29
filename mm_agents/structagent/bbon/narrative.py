"""Behavior narratives for bBoN: compress a rollout into its decisive moments.

============================  FACTS, NOT VERDICTS  ============================
The judge must be an INDEPENDENT check that can OVERRIDE our own verifier — that is
the whole value of best-of-N. So a narrative carries only OBJECTIVE evidence (the
action emitted, the KeyNode *evidence* quote, the URL change, the screenshot) plus
the RUBRIC (``required_outcomes``) separately; the judge forms its own per-outcome
verdict.

  KEEP (objective): actor_action_summary, KeyNode.evidence (a concrete a11y/visual
    quote), url change, screenshots (before+after on revert frames), final a11y /
    doc-structure / cli output, the outcomes' descriptions+evidence_hint as RUBRIC.

  NEVER render: KeyNode.kind ("outcome_satisfied"/"outcome_invalidated"), the
    per-outcome state ("verified"/"reverted") — those are OUR verifier's fallible
    judgments. We use ``kind`` ONLY to locate + prioritise turning points; we render
    its ``evidence`` (the fact behind it). A revert is SHOWN (before/after frames),
    not labelled. The agent self-claim is attached only under
    ``include_self_verdict`` (ablation), clearly quarantined.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, List, Optional

from mm_agents.structagent.ledger.core.timeline import (
    KN_OUTCOME_SATISFIED, KN_OUTCOME_INVALIDATED, KN_VALUE_COMMITTED,
    KN_NAVIGATION, KN_DIALOG_OPENED, KN_DIALOG_CLOSED, KN_STRATEGY_FAILED,
)

DEFAULT_MAX_FRAMES = 8


# --------------------------------------------------------------------------- #
# Data carried out of one finished rollout (decouples this module from the
# agent: the runner fills it from the agent instance + obs sequence).
# --------------------------------------------------------------------------- #
@dataclass
class RolloutArtifacts:
    task: str
    ledger: Any                 # ProgressLedger (reads .required_outcomes)
    timeline: List[Any]         # List[StepRecord]
    final_obs: Optional[dict] = None       # terminal observation (screenshot / a11y tree)
    final_doc_structure: str = ""          # agent._doc_inspect_block at end (office)
    final_cli_output: str = ""             # agent._last_cli_run_output at end
    emitted_fail: bool = False             # agent emitted FAIL/impossible (judged task infeasible)


@dataclass
class TurningPoint:
    step: int
    kind: str                   # INTERNAL only (selection/priority): reverted | done |
                                #   satisfied | value | replan | strategy_failed |
                                #   dialog | navigation | big_delta | initial | final
    action: str                 # objective: actor_action_summary
    delta: str                  # objective: KeyNode.evidence quotes + url change
    after_b64: Optional[str] = None     # state where the change was observed
    before_b64: Optional[str] = None    # prior state — only set on revert frames
    priority: int = 0
    a11y: str = ""                      # per-step a11y_filtered_text (paired with this frame)


@dataclass
class FinalEvidence:
    last_screenshot_b64: str = ""
    a11y: str = ""
    doc_structure: str = ""
    cli_output: str = ""


@dataclass
class BehaviorNarrative:
    task: str
    rubric: List[str] = field(default_factory=list)
    turning_points: List[TurningPoint] = field(default_factory=list)
    final_evidence: Optional[FinalEvidence] = None
    self_verdict: Optional[str] = None      # quarantined; only if include_self_verdict
    emitted_fail: bool = False              # agent terminated with FAIL/impossible

    def to_text(self) -> str:
        """Textual storyboard (judge.py interleaves the screenshots around this)."""
        L = [f"TASK: {self.task}", "",
             "SUCCESS RUBRIC — judge EACH item against the evidence below "
             "(do NOT trust any agent self-claim):"]
        L += [f"  - {r}" for r in self.rubric] or ["  (no compiled outcomes)"]
        L += ["", "TURNING POINTS (chronological; screenshots attached separately):"]
        if not self.turning_points:
            L.append("  (none recorded)")
        for t in self.turning_points:
            L.append(f"  [step {t.step}] action: {t.action}")
            if t.delta:
                L.append(f"            observed: {t.delta}")
        fe = self.final_evidence
        if fe:
            L += ["", "FINAL STATE (the decisive evidence):"]
            if fe.a11y:
                L.append(f"  a11y: {fe.a11y[:1500]}")
            if fe.doc_structure:
                L.append(f"  document/sheet structure: {fe.doc_structure[:3000]}")
            if fe.cli_output:
                L.append(f"  last cli output: {fe.cli_output[:600]}")
        if self.emitted_fail:
            L += ["", "AGENT TERMINATION: this candidate emitted FAIL / impossible "
                  "— the agent itself judged the task UNDOABLE on this UI. If the "
                  "task is genuinely infeasible, THIS is the correct outcome."]
        if self.self_verdict:
            L += ["", self.self_verdict]
        return "\n".join(L)


# --------------------------------------------------------------------------- #
# Turning-point selection (objective; kind only steers selection/priority)
# --------------------------------------------------------------------------- #
# priority: reverts/done/anchors are keep-always (see _cap); the rest fill by rank.
_PRIORITY = {
    "reverted": 100, "done": 95, "done_rejected": 92, "initial": 90, "final": 90,
    "satisfied": 70, "value": 60, "replan": 50, "strategy_failed": 45,
    "navigation": 30, "dialog": 25, "big_delta": 20,
}
_KEEP_ALWAYS_KINDS = {"reverted", "done", "done_rejected", "initial", "final"}


def _classify(step: Any, is_first: bool, is_last: bool, prev_url: Optional[str]):
    """Return (kind, priority) or (None, 0). Highest-signal label wins; the label
    is internal — only used to select/prioritise/contrast, never shown as a verdict."""
    kinds = {kn.kind for kn in (step.key_nodes or [])}
    dec = (step.planner_decision or "").upper()
    if KN_OUTCOME_INVALIDATED in kinds:
        return "reverted", _PRIORITY["reverted"]
    if dec == "DONE":
        return "done", _PRIORITY["done"]
    if KN_OUTCOME_SATISFIED in kinds:
        return "satisfied", _PRIORITY["satisfied"]
    if KN_VALUE_COMMITTED in kinds:
        return "value", _PRIORITY["value"]
    if dec == "REPLAN":
        return "replan", _PRIORITY["replan"]
    if KN_STRATEGY_FAILED in kinds:
        return "strategy_failed", _PRIORITY["strategy_failed"]
    if kinds & {KN_DIALOG_OPENED, KN_DIALOG_CLOSED}:
        return "dialog", _PRIORITY["dialog"]
    if KN_NAVIGATION in kinds:
        return "navigation", _PRIORITY["navigation"]
    # objective navigation with no key-node (robust to verifier false-negatives)
    if step.url and prev_url and step.url != prev_url:
        return "big_delta", _PRIORITY["big_delta"]
    if is_first:
        return "initial", _PRIORITY["initial"]
    if is_last:
        return "final", _PRIORITY["final"]
    return None, 0


def _objective_delta(step: Any, prev_url: Optional[str]) -> str:
    """Objective change description: KeyNode evidence quotes + url change. No verdicts."""
    parts = [kn.evidence.strip() for kn in (step.key_nodes or []) if getattr(kn, "evidence", "")]
    if step.url and prev_url and step.url != prev_url:
        parts.append(f"URL: {prev_url} -> {step.url}")
    return "; ".join(parts)


def _cap(tps: List[TurningPoint], max_frames: int,
         first_step: int, last_step: int) -> List[TurningPoint]:
    if len(tps) <= max_frames:
        return tps

    def keep_always(t: TurningPoint) -> bool:
        return t.kind in _KEEP_ALWAYS_KINDS or t.step in (first_step, last_step)

    forced = [t for t in tps if keep_always(t)]
    rest = sorted((t for t in tps if not keep_always(t)),
                  key=lambda t: t.priority, reverse=True)
    kept = forced + rest[: max(0, max_frames - len(forced))]
    return sorted(kept, key=lambda t: t.step)


def select_turning_points(timeline: List[Any], *,
                          max_frames: int = DEFAULT_MAX_FRAMES) -> List[TurningPoint]:
    """Locate the decisive steps from the timeline (read-only). Selector = key-nodes
    ∪ subgoal boundaries (REPLAN/DONE) ∪ objective url-deltas ∪ first/last anchors.
    Capped to ``max_frames`` by priority; reverts/done/first/last never dropped."""
    if not timeline:
        return []
    tps: List[TurningPoint] = []
    prev_url: Optional[str] = None
    n = len(timeline)
    for i, s in enumerate(timeline):
        kind, prio = _classify(s, i == 0, i == n - 1, prev_url)
        if kind is not None:
            before = timeline[i - 1].screenshot_b64 if (kind == "reverted" and i > 0) else None
            tps.append(TurningPoint(
                step=s.step_idx, kind=kind, priority=prio,
                action=(s.actor_action_summary or "(no action recorded)"),
                delta=_objective_delta(s, prev_url),
                after_b64=s.screenshot_b64, before_b64=before,
                a11y=(getattr(s, "a11y", "") or ""),
            ))
        if s.url:
            prev_url = s.url
    capped = _cap(tps, max_frames, timeline[0].step_idx, timeline[-1].step_idx)
    # Mode-agnostic tail fill: keynode-less runs (boundary mode) yield only
    # first/last turning points. Top up with the most-recent steps so the judge
    # always sees the last ``max_frames`` steps' screenshots + actions.
    if len(capped) < max_frames:
        have = {t.step for t in capped}
        for s in reversed(timeline):
            if len(capped) >= max_frames:
                break
            if s.step_idx in have:
                continue
            capped.append(TurningPoint(
                step=s.step_idx, kind="recent", priority=10,
                action=(s.actor_action_summary or "(no action recorded)"),
                delta=_objective_delta(s, None), after_b64=s.screenshot_b64,
                a11y=(getattr(s, "a11y", "") or "")))
            have.add(s.step_idx)
        capped.sort(key=lambda t: t.step)
    return capped


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _b64(screenshot: Any) -> str:
    if screenshot is None:
        return ""
    if isinstance(screenshot, (bytes, bytearray)):
        return base64.b64encode(screenshot).decode("ascii")
    return str(screenshot)  # already a b64 string


def _final_a11y(obs: Optional[dict]) -> str:
    if not obs:
        return ""
    try:
        from mm_agents.structagent.utils.a11y import linearize_obs_a11y
        return linearize_obs_a11y(obs) or ""
    except Exception:
        return ""


def render_behavior_narrative(art: RolloutArtifacts, *,
                              max_frames: int = DEFAULT_MAX_FRAMES,
                              include_self_verdict: bool = False) -> BehaviorNarrative:
    """Build one rollout's narrative from its artifacts. Objective facts only;
    rubric from ``required_outcomes``; self-claim only if ``include_self_verdict``
    (quarantined). See the module header for the facts-not-verdicts rule."""
    outcomes = list(getattr(art.ledger, "required_outcomes", None) or [])
    rubric = [
        f"{o.id}: {o.description}"
        + (f"  [durable signal: {o.evidence_hint}]" if getattr(o, "evidence_hint", "") else "")
        for o in outcomes
    ]
    final = FinalEvidence(
        last_screenshot_b64=_b64((art.final_obs or {}).get("screenshot")),
        a11y=((art.final_obs or {}).get("a11y_text") or _final_a11y(art.final_obs)),
        doc_structure=art.final_doc_structure or "",
        cli_output=str(art.final_cli_output or "")[:600],
    )
    self_verdict = None
    if include_self_verdict and outcomes:
        states = ", ".join(f"{o.id}={o.state}" for o in outcomes)
        self_verdict = ("AGENT SELF-CLAIM (the agent's OWN verifier — may be wrong; "
                        f"verify against the evidence, do not rely on it): {states}")
    return BehaviorNarrative(
        task=art.task, rubric=rubric,
        turning_points=select_turning_points(art.timeline, max_frames=max_frames),
        final_evidence=final, self_verdict=self_verdict,
        emitted_fail=getattr(art, "emitted_fail", False),
    )

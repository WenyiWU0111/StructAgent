"""LLM-driven dead-end curator.

Replaces the mechanical "every strategy switch → write a FailedPath"
behavior with an LLM that sees the FULL strategy-switch history +
current outcome state + currently-recorded failed_paths and decides:

  - which strategy spans were truly dead-ends (record as level=strategy)
  - which were actor-execution failures masquerading as strategy
    failures (DO NOT record as strategy-level — strategy was sound)
  - which already-recorded entries are fuzzy duplicates of others
    (collapse into one)
  - which already-recorded entries were misclassified in retrospect
    and should be dropped

The mechanical pair-wise ``strategy_similar`` judge can't see across
multiple switches and can't retrospect on its own past writes; the
curator owns both responsibilities by re-deriving the entire
``failed_paths`` list from history each time it runs.

Touch points:
  - ``StrategyEvent`` records each strategy commit (install / switch /
    re-iteration). The agent's ``_commit_strategy_change`` appends one
    of these to ``self._strategy_history`` instead of writing a
    FailedPath directly.
  - ``ActionStuckEvent`` records a rule #6 cadence trigger fire.
  - ``curate_failed_paths`` is called after each REPLAN; it returns
    the new authoritative ``failed_paths`` list which the agent
    assigns onto ``ledger.failed_paths``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from mm_agents.structagent.ledger.core.ledger import FailedPath, Outcome


@dataclass
class _EvidenceFields:
    """Shared per-event evidence used by the curator's tactical-vs-strategic
    judgment. These come from the perceiver / actor state at the moment
    the event was created — they let the curator distinguish "target
    UI element was on screen but the actor missed the click" (tactical,
    do NOT dead-end) from "target was never observed across all attempts"
    (strategic, OK to dead-end).
    """
    # Base64 PNG of the screenshot at the failing turn. Curator
    # attaches the most recent K screenshots as image_url parts.
    last_screenshot_b64: Optional[str] = None
    # Top-K a11y elements (text-form) the perceiver pre-filtered for
    # the failing turn — e.g. ``"button 'Save'  enabled\\nmenu-item ..."``.
    last_a11y_top_k: Optional[str] = None
    # Perceiver's overall subgoal-done verdict: "yes" / "no" / "partial".
    last_perceiver_overall: Optional[str] = None
    # Count of unexpected blocker controls (modals, error toasts, ...)
    # the perceiver flagged.
    last_perceiver_blockers: int = 0
    # How many "relevant controls" the perceiver matched against the
    # subgoal — 0 means the target element did NOT appear on screen.
    last_perceiver_controls_count: int = 0


@dataclass
class StrategyEvent(_EvidenceFields):
    """One strategy-level event in the agent's history.

    Kinds:
      - "install"   : first plan committed at step 0
      - "same"      : REPLAN with strategy_similar(prior, new) == True
                       (planner kept the same approach; we still log
                       this so the curator sees the full chain)
      - "switch"    : REPLAN with strategy_similar == False
                       (planner abandoned prior for new)
    """
    kind: str = ""
    step_idx: int = 0
    prior_strategy: Optional[str] = None
    new_strategy: Optional[str] = None
    span_started_at_step: int = 0
    span_steps: int = 0
    actions_under_prior: List[str] = field(default_factory=list)
    progress_signal: str = "unknown"
    current_subgoal: Optional[str] = None
    # A4: which Outcome the planner was driving toward when this
    # strategy event occurred (ledger.next_focus_outcome() at event
    # creation time). Lets the curator emit ``target_outcome_id`` on
    # the resulting FailedPath entries so the planner prompt can
    # filter per-focus-outcome.
    focus_outcome_id: Optional[str] = None


@dataclass
class ActionStuckEvent(_EvidenceFields):
    """One rule-#6 cadence-trigger event (action-level stuck)."""
    step_idx: int = 0
    stuck_subgoal: str = ""
    replans_within_strategy: int = 0
    actions_during_subgoal: List[str] = field(default_factory=list)
    current_strategy: Optional[str] = None


def _outcome_state_summary(
    required_outcomes: List["Outcome"],
    outcome_states: Dict[str, str],
) -> str:
    """One-line per outcome with verified/pending/reverted state."""
    if not required_outcomes:
        return "  (no required outcomes)"
    lines = []
    for o in required_outcomes:
        st = outcome_states.get(o.id, "pending")
        glyph = {"verified": "✓", "reverted": "✗",
                  "pending": "·"}.get(st, "?")
        hint = (o.evidence_hint or "").strip()[:120]
        lines.append(f"  {glyph} [{st:<8}] {o.id} — hint: {hint}")
    return "\n".join(lines)


def _strategy_history_block(events: List[StrategyEvent]) -> str:
    if not events:
        return "  (no strategy events yet)"
    lines = []
    for i, ev in enumerate(events, 1):
        actions_str = "; ".join(
            (a or "")[:80] for a in ev.actions_under_prior[-5:]
        ) or "(none)"
        # A4: show which Outcome the planner was driving toward at this
        # event so the curator can attribute the resulting FailedPath
        # via target_outcome_id.
        focus_tag = (f"  [focus_outcome={ev.focus_outcome_id}]"
                     if getattr(ev, "focus_outcome_id", None) else "")
        if ev.kind == "install":
            lines.append(
                f"  #{i} [step {ev.step_idx}] INSTALL initial strategy: "
                f"\"{(ev.new_strategy or '')[:200]}\"{focus_tag}"
            )
        elif ev.kind == "same":
            lines.append(
                f"  #{i} [step {ev.step_idx}] SAME-strategy REPLAN "
                f"(span {ev.span_steps} steps, signal={ev.progress_signal}): "
                f"\"{(ev.new_strategy or '')[:200]}\"{focus_tag}"
            )
            if ev.actions_under_prior:
                lines.append(f"      span actions: {actions_str}")
        elif ev.kind == "switch":
            lines.append(
                f"  #{i} [step {ev.step_idx}] SWITCH "
                f"(span {ev.span_steps} steps, signal={ev.progress_signal}){focus_tag}"
            )
            lines.append(f"      from: \"{(ev.prior_strategy or '')[:200]}\"")
            lines.append(f"      to:   \"{(ev.new_strategy or '')[:200]}\"")
            if ev.actions_under_prior:
                lines.append(f"      span actions: {actions_str}")
        else:
            lines.append(
                f"  #{i} [step {ev.step_idx}] {ev.kind}: "
                f"{(ev.new_strategy or '')[:200]}{focus_tag}"
            )
        lines.extend(_format_evidence_line(ev))
    return "\n".join(lines)


def _format_evidence_line(ev) -> List[str]:
    """One-liner block summarizing the perceiver / a11y evidence
    captured at event creation. Curator uses these to decide
    TACTICAL vs STRATEGIC failure (see DECISION RULES below)."""
    overall = getattr(ev, "last_perceiver_overall", None)
    if overall is None and not getattr(ev, "last_a11y_top_k", None):
        return []  # event has no evidence (older event or perceiver off)
    n_ctrls = getattr(ev, "last_perceiver_controls_count", 0) or 0
    n_block = getattr(ev, "last_perceiver_blockers", 0) or 0
    lines = [
        "      [evidence] perceiver overall=%s, relevant_controls=%d, blockers=%d"
        % (overall, n_ctrls, n_block),
    ]
    a11y = getattr(ev, "last_a11y_top_k", None)
    if a11y:
        lines.append("      [a11y top elements at failing turn]")
        for sub in a11y.splitlines()[:10]:
            lines.append(f"    {sub}")
    return lines


def _action_stuck_block(events: List[ActionStuckEvent]) -> str:
    if not events:
        return "  (no action-level stuck events)"
    lines = []
    for i, ev in enumerate(events, 1):
        actions_str = "; ".join(
            (a or "")[:80] for a in ev.actions_during_subgoal[-5:]
        ) or "(none)"
        lines.append(
            f"  #{i} [step {ev.step_idx}] subgoal stuck after "
            f"{ev.replans_within_strategy} same-strategy REPLANs: "
            f"\"{(ev.stuck_subgoal or '')[:200]}\""
        )
        lines.append(f"      under strategy: \"{(ev.current_strategy or '')[:140]}\"")
        lines.append(f"      actions tried: {actions_str}")
        lines.extend(_format_evidence_line(ev))
    return "\n".join(lines)


def _failed_paths_block(failed_paths: List["FailedPath"]) -> str:
    if not failed_paths:
        return "  (none yet — first curator run, or all entries dropped on prior pass)"
    lines = []
    for i, fp in enumerate(failed_paths, 1):
        target_tag = (f"  target_outcome={fp.target_outcome_id}"
                      if getattr(fp, "target_outcome_id", None) else "")
        lines.append(
            f"  #{i} level={fp.level}  count={fp.observation_count}  "
            f"steps={fp.first_observed_at_step}-{fp.last_observed_at_step}"
            f"{target_tag}"
        )
        lines.append(f"      path: \"{(fp.path or '')[:200]}\"")
        lines.append(f"      why : {(fp.why or '')[:200]}")
    return "\n".join(lines)


_CURATOR_PROMPT_TEMPLATE = """You are the dead-end curator for a GUI automation agent. \
After every REPLAN you re-derive the authoritative list of "do not retry" entries from the FULL history.

Your job is to OUTPUT a single JSON object that REPLACES the agent's failed_paths list.

═══════════════════════════════════════════════════════════════════════════════
TASK CONTEXT
═══════════════════════════════════════════════════════════════════════════════
Instruction: {instruction}

Required outcomes (current state):
{outcome_states_block}

═══════════════════════════════════════════════════════════════════════════════
STRATEGY-LEVEL HISTORY (every plan commit since task start)
═══════════════════════════════════════════════════════════════════════════════
{strategy_history_block}

═══════════════════════════════════════════════════════════════════════════════
ACTION-LEVEL STUCK EVENTS (rule-#6 cadence triggers — same subgoal re-emitted N+ times)
═══════════════════════════════════════════════════════════════════════════════
{action_stuck_block}

═══════════════════════════════════════════════════════════════════════════════
CURRENTLY-RECORDED failed_paths (from your last curation pass — you may revise these)
═══════════════════════════════════════════════════════════════════════════════
{current_failed_paths_block}

═══════════════════════════════════════════════════════════════════════════════
DECISION RULES
═══════════════════════════════════════════════════════════════════════════════
A) STRATEGY-LEVEL dead-end means: "this high-level approach CANNOT achieve the
   task no matter how the actor executes it" — e.g. the URL doesn't exist,
   the menu path doesn't lead where you think, the page lacks the feature.

B) DO NOT record a strategy as dead-end when the strategy is SOUND but the
   actor's execution was imperfect. Examples of actor-execution-failures
   that look like strategy failures but ARE NOT:
     - actor scrolled the page repeatedly without finding the target element
       (the element may exist but be off-viewport, hidden by an overlay, or
        labeled differently than the planner expected)
     - actor clicked a coordinate that missed the actual element
     - actor's typed text was silently truncated or auto-corrected
     - actor switched window focus and lost context
   For any of these, the strategy is FINE — only the specific tactical step
   needs adjusting. Record an ACTION-level entry naming the stuck subgoal
   instead, NOT a strategy-level entry.

   ── Evidence-grounded tactical-vs-strategic test ──
   Each stuck event in STRATEGY-LEVEL HISTORY / ACTION-LEVEL STUCK EVENTS
   has an ``[evidence]`` line capturing the perceiver state at the
   failing turn AND an ``[a11y top elements at failing turn]`` block
   listing the actual UI elements the perceiver saw. There may also be
   one or more SCREENSHOT image attachments at the end of the user
   message — these correspond, in order, to the events with the most
   recent evidence. Use them like this:

     1. Identify the TARGET UI element implied by the stuck_subgoal text
        (the button / menu item / field the agent was trying to interact
        with — e.g. "the 'Save' button", "the 'Unified Folders' menu item").

     2. Check the [a11y top elements] block AND the screenshot. Did the
        target element actually appear on screen?
          • YES (target visible) → the strategy is REACHABLE. The failure
            is TACTICAL (click missed, wrong sibling selected, focus
            lost, …). Mark the strategy ``actor_flailing`` and record an
            ACTION-level entry for the specific click pattern instead.
          • NO (target never appears, or perceiver consistently reports
            ``relevant_controls=0`` and the screenshot shows the wrong
            view) → the strategy is genuinely STRATEGIC. The menu item
            doesn't exist, the dialog doesn't open here, etc. Mark
            ``true_dead_end_new`` or ``true_dead_end_merge``.

     3. When evidence is missing for an event (no [evidence] line, no
        screenshot) you must fall back to the textual signals: span
        length, action repetition pattern, progress_signal. Treat
        absent evidence as inconclusive — bias toward ``actor_flailing``
        unless the strategy text itself is clearly wrong-path.

   In the ``rationale`` field, prepend a literal audit token so the
   decision is auditable downstream:
     - dead_end strategy (target absent): ``STRATEGIC_TARGET_ABSENT``
     - dead_end strategy (wrong-path text): ``STRATEGIC_WRONG_PATH``
     - actor_flailing strategy (target visible, click missed): ``TACTICAL_TARGET_SEEN``
     - record_action_level problematic_action (target was reachable,
       click pattern itself flawed): ``TACTICAL_ACTION_FAILURE``

C) FUZZY DUPLICATES: multiple strategy descriptions that target the SAME
   approach (same entry point + same interaction pattern, even if worded
   differently) MUST be collapsed into a SINGLE entry. Examples of the
   same approach:
     - "Use the iPhone menu in Apple navigation to access compare tool"
     - "Use the 'Compare' link in the iPhone lineup to access compare tool"
     → both are "use Apple's in-site navigation to reach the compare page"
   versus genuinely different approaches:
     - "Use the address bar to type chrome://settings"
     - "Open Chrome's three-dot menu to navigate to Settings"
     → different entry points (URL bar vs menu button) — separate entries.

D) RETROSPECT: if a previously-recorded entry now looks like an
   actor-execution-failure (rule B) or a fuzzy duplicate of another entry
   (rule C), DROP it or merge it. Don't carry forward bad records.

E) action-level entries: record the EXACT stuck subgoal text from the
   action-stuck-events block. They tell the planner "do not re-emit this
   specific subgoal as-is; try alternate tactics within the same strategy".

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON, no prose, no markdown fences
═══════════════════════════════════════════════════════════════════════════════
You MUST think step-by-step BEFORE writing the final list. Output ONE JSON
object with THREE top-level keys, in this order:

(1) "strategy_analysis": one entry PER UNIQUE STRATEGY that appeared in
    STRATEGY-LEVEL HISTORY (de-dup fuzzy variants — the same approach with
    slightly different wording is ONE strategy). For each, give a verdict
    and one-sentence rationale per the decision rules. Verdict ∈ :
      - "active_now"        : the currently-committed strategy (still being
                               tried) AND its primary outcomes are NOT all
                               [verified] yet — DO NOT record. Exactly ONE
                               strategy qualifies as active_now per pass.
      - "strategy_outcome_satisfied" : this strategy's primary outcome(s)
                               are now [verified] in the Required Outcomes
                               block above — the strategy has accomplished
                               its goal and the planner must NOT re-propose
                               it. Record in "completed_strategies" (a
                               separate list, see below) so the planner
                               knows to advance to the next non-verified
                               outcome with a NEW strategy.
                               Use this verdict ONLY when the verified
                               outcome(s) clearly correspond to what the
                               strategy was set up to produce (e.g.
                               strategy "create Sheet2 via Sheet menu"
                               → sheet2_created [verified]).
      - "actor_flailing"    : strategy is sound; actor execution failed
                               (rule B). DO NOT record as dead-end.
      - "true_dead_end_new" : genuine strategy-level dead-end. Record as
                               new item in failed_paths.
      - "true_dead_end_merge": same approach as an existing failed_paths
                               entry. Specify ``merge_target_path``.
      - "drop_existing"     : a previously-recorded entry that on review
                               looks like rule B / C — drop or merge.

(2) "problematic_actions": one entry per DISTINCT actor-execution pattern
    that failed (e.g. "repeatedly scrolled the iPhone landing page without
    clicking the model link", "clicked the same coord 3+ times with no UI
    change", "typed wrong URL in address bar"). For each:
      - "pattern": short label describing the stuck action pattern
      - "occurred_under_strategy": which strategy this action sequence ran
        under (paste the strategy text or its short identifier)
      - "verdict": "record_action_level" (rule-#6-style action-level entry)
                  OR "noted_only" (we mention it for context but it's not
                  worth a failed_paths entry — e.g. one-off mis-click).
      - "rationale": one sentence
    If no actor-execution failures are notable, output an empty list.

(3) "failed_paths": the FINAL authoritative list AFTER applying all the
    verdicts above. Each item:
      - path                  : high-level approach (strategy-level) OR
                                 exact stuck-subgoal/action-pattern text
                                 (action-level)
      - level                 : "strategy" | "action"
      - why                   : one-sentence reason — what to avoid, why
      - first_observed_at_step: int
      - last_observed_at_step : int
      - observation_count     : int (>= 1; sum across merged-in events)
      - action_sequence       : list of short action summaries tried

Schema:

{{
  "strategy_analysis": [
    {{
      "strategy_text": "<the strategy you're judging>",
      "verdict": "active_now|strategy_outcome_satisfied|actor_flailing|true_dead_end_new|true_dead_end_merge|drop_existing",
      "rationale": "<one sentence>",
      "merge_target_path": "<existing path text or null>",
      "satisfied_outcomes": ["<outcome_id>", ...]   // ONLY when verdict == "strategy_outcome_satisfied"; list the [verified] outcome IDs this strategy produced. Empty/omitted otherwise.
    }},
    ...
  ],
  "problematic_actions": [
    {{
      "pattern": "<short label>",
      "occurred_under_strategy": "<strategy text or label>",
      "verdict": "record_action_level|noted_only",
      "rationale": "<one sentence>"
    }},
    ...
  ],
  "failed_paths": [
    {{
      "path": "...", "level": "strategy"|"action",
      "why": "...",
      "first_observed_at_step": 0, "last_observed_at_step": 0,
      "observation_count": 1, "action_sequence": ["..."],
      "target_outcome_id": "<id of the Outcome this dead-end was failing against, or null if it doesn't fit any single outcome (e.g. cross-cutting actor-flailing)>"
    }},
    ...
  ]
}}

Consistency invariants (verify internally before emitting):
  - Every "true_dead_end_new" strategy → exactly ONE matching strategy-level
    item in failed_paths.
  - Every "true_dead_end_merge" strategy → NO new item (merged into the
    existing target's observation_count + action_sequence).
  - Every "record_action_level" problematic_action → exactly ONE matching
    action-level item in failed_paths.
  - "active_now", "strategy_outcome_satisfied", "actor_flailing",
    "drop_existing" strategies and "noted_only" actions → NO corresponding
    item in failed_paths. "strategy_outcome_satisfied" is surfaced to the
    planner via a separate "completed_strategies" block built from the
    strategy_analysis entries.
  - "strategy_outcome_satisfied" REQUIRES a non-empty satisfied_outcomes
    list pointing at outcomes that are actually [verified] in the
    Required Outcomes block above. If you can't name a specific verified
    outcome the strategy produced, do NOT use this verdict — fall back
    to "active_now" or "actor_flailing".
  - Every failed_paths entry MUST set "target_outcome_id" to the id of
    the Outcome the planner was driving toward when the dead-end was
    observed (see [focus_outcome=...] annotation on each strategy
    history event). Use null ONLY when the dead-end is genuinely
    cross-cutting (e.g. a tactical flailing pattern that wasted budget
    across multiple outcomes). When multiple outcomes share a
    dead-end, attribute to the one most-directly blocked by it
    (typically the focus outcome at the latest observation).

If no entries are warranted, output failed_paths: [].
"""


def _build_curator_prompt(
    *,
    instruction: str,
    required_outcomes: List["Outcome"],
    outcome_states: Dict[str, str],
    strategy_events: List[StrategyEvent],
    action_stuck_events: List[ActionStuckEvent],
    current_failed_paths: List["FailedPath"],
) -> str:
    return _CURATOR_PROMPT_TEMPLATE.format(
        instruction=instruction.strip()[:400],
        outcome_states_block=_outcome_state_summary(
            required_outcomes, outcome_states),
        strategy_history_block=_strategy_history_block(strategy_events),
        action_stuck_block=_action_stuck_block(action_stuck_events),
        current_failed_paths_block=_failed_paths_block(current_failed_paths),
    )


def _parse_curator_response(response: str) -> Optional[List[Dict[str, Any]]]:
    """Extract the failed_paths list from the curator's JSON output.
    Tolerates leading/trailing prose, code fences, and minor whitespace.
    Discards per_event_analysis here — see ``_parse_curator_full`` if
    you also want the per-event verdicts (e.g. for logging)."""
    full = _parse_curator_full(response)
    return full[1] if full is not None else None


def _parse_curator_full(
    response: str,
) -> Optional[tuple]:
    """Parse strategy_analysis + problematic_actions + failed_paths from
    the curator's JSON output. Returns (analysis_list, failed_paths_list)
    or None on failure. analysis_list is a flat list of dicts (one per
    strategy_analysis item + one per problematic_actions item), each
    tagged with ``kind``. failed_paths must be present + a list.

    Also accepts the legacy ``per_event_analysis`` schema for back-compat.
    """
    if not response:
        return None
    text = response.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    fps = obj.get("failed_paths")
    if not isinstance(fps, list):
        return None

    analysis: List[Dict[str, Any]] = []
    strat_analysis = obj.get("strategy_analysis") or []
    if isinstance(strat_analysis, list):
        for a in strat_analysis:
            if isinstance(a, dict):
                a = dict(a)
                a["kind"] = "strategy"
                analysis.append(a)
    prob_actions = obj.get("problematic_actions") or []
    if isinstance(prob_actions, list):
        for a in prob_actions:
            if isinstance(a, dict):
                a = dict(a)
                a["kind"] = "action"
                analysis.append(a)
    # Back-compat with prior per_event_analysis key
    legacy = obj.get("per_event_analysis") or []
    if isinstance(legacy, list):
        for a in legacy:
            if isinstance(a, dict):
                a = dict(a)
                a.setdefault("kind", "event")
                analysis.append(a)
    return (analysis, fps)


def _coerce_failed_path_dict(
    raw: Dict[str, Any],
    fallback_step: int,
) -> Optional[Dict[str, Any]]:
    """Validate + normalize one curator-emitted entry into a dict
    compatible with ``FailedPath(**d)``."""
    path = (raw.get("path") or "").strip()
    if not path:
        return None
    level = raw.get("level") or "action"
    if level not in ("strategy", "action"):
        level = "action"
    why = (raw.get("why") or "").strip()[:240]
    first = raw.get("first_observed_at_step")
    last = raw.get("last_observed_at_step")
    try:
        first_i = int(first) if first is not None else fallback_step
    except (TypeError, ValueError):
        first_i = fallback_step
    try:
        last_i = int(last) if last is not None else first_i
    except (TypeError, ValueError):
        last_i = first_i
    obs = raw.get("observation_count")
    try:
        obs_i = max(1, int(obs)) if obs is not None else 1
    except (TypeError, ValueError):
        obs_i = 1
    actions = raw.get("action_sequence") or []
    if not isinstance(actions, list):
        actions = []
    actions_clean: List[str] = []
    for a in actions[:12]:
        if isinstance(a, str):
            s = a.replace("\n", " ").strip()
            if s:
                actions_clean.append(s[:200])
    # A4: per-outcome routing. null / missing → uncategorized
    # (rendered in the legacy global block); a string id is preserved
    # and used by render_outcome_view to filter per focus outcome.
    target = raw.get("target_outcome_id")
    if not isinstance(target, str) or not target.strip():
        target = None
    return {
        "path": path[:240],
        "level": level,
        "why": why,
        "first_observed_at_step": first_i,
        "first_observed_at_subgoal": "",
        "last_observed_at_step": last_i,
        "observation_count": obs_i,
        "evidence_strength": "auto",
        "action_sequence": actions_clean,
        "target_outcome_id": target,
    }


def _collect_recent_event_screenshots(
    strategy_events: List[StrategyEvent],
    action_stuck_events: List[ActionStuckEvent],
    *,
    cap: int = 5,
) -> List[Dict[str, str]]:
    """Return up to ``cap`` (label, b64) pairs for the most recent
    events that carry screenshot evidence. Newest first.

    Curator attaches these as image_url parts at the end of the user
    message; the prompt's tactical-vs-strategic rule references them
    by order.
    """
    bundle: List[Dict[str, str]] = []
    # Walk newest-first through both event lists, interleaved by step_idx
    combined = []
    for ev in strategy_events:
        if getattr(ev, "last_screenshot_b64", None):
            combined.append((ev.step_idx, f"strategy:{ev.kind}", ev))
    for ev in action_stuck_events:
        if getattr(ev, "last_screenshot_b64", None):
            combined.append((ev.step_idx, "action_stuck", ev))
    combined.sort(key=lambda t: t[0], reverse=True)
    for step, kind, ev in combined[:cap]:
        bundle.append({
            "label": f"step {step} ({kind})",
            "b64": ev.last_screenshot_b64,
        })
    return bundle


def curate_failed_paths(
    *,
    instruction: str,
    required_outcomes: List["Outcome"],
    outcome_states: Dict[str, str],
    strategy_events: List[StrategyEvent],
    action_stuck_events: List[ActionStuckEvent],
    current_failed_paths: List["FailedPath"],
    llm_call: Callable,
    current_step: int = 0,
    analysis_sink: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List["FailedPath"]]:
    """Run the LLM curator and return the new failed_paths list.

    ``llm_call`` signature has been generalized to accept a multimodal
    messages list — either ``llm_call(prompt: str) -> str`` (legacy
    text-only) or ``llm_call(messages: List[dict]) -> str`` (preferred,
    used when events carry screenshot evidence). The function probes
    its single positional argument: if it's a string, the call is
    treated as legacy text-only; if it's a list, treated as messages.

    Returns None if the LLM call or JSON parse failed — caller should
    keep the existing failed_paths in that case (don't blow them away
    on a transient error).

    ``analysis_sink``: optional list — if provided, the per-event
    verdicts the LLM produced (rationale, merge target, etc.) are
    appended into it so the caller can log them.
    """
    from mm_agents.structagent.ledger.core.ledger import FailedPath  # local import: avoids circular

    if not strategy_events and not action_stuck_events:
        # Nothing to curate yet
        return list(current_failed_paths)

    prompt = _build_curator_prompt(
        instruction=instruction,
        required_outcomes=required_outcomes,
        outcome_states=outcome_states,
        strategy_events=strategy_events,
        action_stuck_events=action_stuck_events,
        current_failed_paths=current_failed_paths,
    )

    # Build multimodal user content: prompt text + up to 5 attached
    # screenshots from the most recent stuck/strategy events.
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    shot_bundle = _collect_recent_event_screenshots(
        strategy_events, action_stuck_events, cap=5,
    )
    if shot_bundle:
        label_lines = [
            f"  attachment {i+1}: {item['label']}"
            for i, item in enumerate(shot_bundle)
        ]
        user_content.append({
            "type": "text",
            "text": (
                "═════════════════════════════════════════════════════\n"
                "ATTACHED SCREENSHOTS (most recent events, newest first)\n"
                "═════════════════════════════════════════════════════\n"
                "These are the post-action screenshots at the failing "
                "turns for the events whose [evidence] block lists "
                "perceiver state. Use them to corroborate whether the "
                "target UI element was visually present on screen "
                "(see DECISION RULE B above).\n\n"
                + "\n".join(label_lines)
            ),
        })
        for item in shot_bundle:
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{item['b64']}",
                },
            })
    messages = [{"role": "user", "content": user_content}]

    try:
        # Try multimodal first; fall back to legacy text-only signature
        # if the caller's llm_call doesn't accept a list.
        try:
            response = llm_call(messages) or ""
        except TypeError:
            response = llm_call(prompt) or ""
    except Exception:
        return None
    parsed = _parse_curator_full(response)
    if parsed is None:
        return None
    analysis, raw_list = parsed
    if analysis_sink is not None and analysis:
        for a in analysis:
            if isinstance(a, dict):
                analysis_sink.append(a)

    new_failed_paths: List["FailedPath"] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        coerced = _coerce_failed_path_dict(raw, fallback_step=current_step)
        if coerced is None:
            continue
        try:
            new_failed_paths.append(FailedPath(**coerced))
        except TypeError:
            # Drop unknown fields and retry. A4: include target_outcome_id
            # in the allow-list so per-outcome routing survives the
            # legacy-fallback path.
            fields = {
                "path", "why", "first_observed_at_step",
                "first_observed_at_subgoal", "level",
                "evidence_strength", "observation_count",
                "last_observed_at_step", "action_sequence",
                "strong_model_origin", "target_outcome_id",
            }
            filtered = {k: v for k, v in coerced.items() if k in fields}
            try:
                new_failed_paths.append(FailedPath(**filtered))
            except Exception:
                continue

    # Preserve strong-model-authored dead-ends. The curator's LLM may omit
    # them (it's the weak model and doesn't know the rescue agent's
    # intent); re-attach any that aren't already present in the new list.
    existing_strong = [fp for fp in current_failed_paths
                       if getattr(fp, "strong_model_origin", False)]
    if existing_strong:
        existing_paths = {fp.path for fp in new_failed_paths}
        for fp in existing_strong:
            if fp.path not in existing_paths:
                new_failed_paths.append(fp)
                existing_paths.add(fp.path)

    return new_failed_paths

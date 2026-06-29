"""Event timeline — the agent's primary working memory.

Structured event stream replacing v1's append-only `done[]`; supports
outcome reversion, key-node attribution, and human-readable review. The
planner reads a compact rendering (event table + outcome view) instead of
the legacy 3-screenshot window.

Two layers:
  StepRecord    — one per planner turn (observation + decision + KeyNodes)
  TimelineEvent — consecutive StepRecords sharing one subgoal

OutcomeStateCache derives outcome state (pending/verified/reverted)
incrementally as KeyNodes arrive — O(K) per turn, no full-timeline walk.

No LLM calls here. Detector lives in key_node_detector.py; the agent
(qwen25vl_agent_planner.py) orchestrates the lifecycle.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Outcome states (string-valued so asdict() round-trips cleanly to JSON)
OUTCOME_PENDING = "pending"
OUTCOME_VERIFIED = "verified"
OUTCOME_REVERTED = "reverted"

# Confidence levels (LLM self-rates; ordinal labels beat floats here)
CONF_LOW = "low"
CONF_MEDIUM = "medium"
CONF_HIGH = "high"
_CONF_ENTERS_DERIVATION = {CONF_MEDIUM, CONF_HIGH}

# KeyNode kinds
KN_OUTCOME_SATISFIED = "outcome_satisfied"
KN_OUTCOME_INVALIDATED = "outcome_invalidated"
KN_VALUE_COMMITTED = "value_committed"
KN_NAVIGATION = "navigation"
KN_DIALOG_OPENED = "dialog_opened"
KN_DIALOG_CLOSED = "dialog_closed"
KN_STRATEGY_FAILED = "strategy_failed"

_OUTCOME_KINDS = {KN_OUTCOME_SATISFIED, KN_OUTCOME_INVALIDATED}

# TimelineEvent.outcome
EV_ONGOING = "ongoing"
EV_COMMITTED = "committed"
EV_ABANDONED_REPLAN = "abandoned_replan"
EV_ENDED_DONE = "ended_done"

# Flap-protection: lock outcome after this many reverts
MAX_REVERT_COUNT = 3


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #


@dataclass
class KeyNode:
    """A detected state transition relevant to outcomes/strategy.

    Emitted per turn by key_node_detector.detect_key_nodes. Must carry
    concrete ``evidence`` (an a11y row quote or visual phrase).
    """
    kind: str               # KN_* above
    target: str             # outcome_id (for satisfied/invalidated) or human label
    evidence: str           # ≤ 240 chars, concrete quote or visual description
    confidence: str         # CONF_LOW | CONF_MEDIUM | CONF_HIGH
    detected_at_step: int
    # Provenance: "deterministic" (file_grep/url_match/shell_command/
    # calc_verify re-check against live VM state) or "llm" (vision/LLM
    # judge). A deterministic confirmation can only be overturned by
    # another deterministic verdict, never by a hallucination-prone LLM
    # judge. Default "" keeps older persisted records deserializing.
    source: str = ""

    def enters_derivation(self) -> bool:
        """Whether this KeyNode updates OutcomeStateCache. Low-confidence
        nodes are kept in the timeline for visibility but excluded from
        authoritative state derivation."""
        return self.confidence in _CONF_ENTERS_DERIVATION


@dataclass
class StepRecord:
    """Raw per-turn data. ``screenshot_b64`` is always persisted; the
    render layer picks which step's screenshot to embed in the prompt.
    """
    step_idx: int
    timestamp: str = ""                 # ISO datetime
    # observation
    screenshot_b64: Optional[str] = None
    a11y: Optional[str] = None
    url: Optional[str] = None
    # intent
    actor_action_summary: str = ""      # 1-line human summary
    actor_action_raw: Optional[str] = None
    # decision
    planner_decision: Optional[str] = None         # CONTINUE | REPLAN | DONE
    planner_done_assessment: Optional[str] = None  # YES | NO
    # detected
    key_nodes: List[KeyNode] = field(default_factory=list)
    # Perceiver snapshot (turn-level subgoal verdict, relevant_controls,
    # etc.). Set by _pre_planner when use_perceiver is on; None otherwise.
    # Typed Any to avoid a circular import (perceiver imports timeline);
    # consumers duck-type on .subgoal_check / .coverage / etc.
    snapshot: Optional[Any] = None


@dataclass
class TimelineEvent:
    """A group of consecutive StepRecords sharing the same subgoal text.

    A new event opens when the subgoal changes, the previous event is
    abandoned (REPLAN), or the previous event commits (CONTINUE+done=YES).
    """
    event_idx: int
    subgoal: str
    started_at_step: int
    ended_at_step: Optional[int] = None
    steps: List[StepRecord] = field(default_factory=list)
    outcome: str = EV_ONGOING            # EV_*
    # Planner's one-sentence <Bottleneck> at close-time (set when the
    # event closes ABANDONED_REPLAN). Lets the renderer show WHY it was
    # abandoned without scrolling back through old prompts.
    closing_reason: Optional[str] = None

    @property
    def all_key_nodes(self) -> List[KeyNode]:
        out: List[KeyNode] = []
        for s in self.steps:
            out.extend(s.key_nodes)
        return out

    @property
    def n_steps(self) -> int:
        return len(self.steps)


# --------------------------------------------------------------------------- #
# Outcome state cache
# --------------------------------------------------------------------------- #


#: After this many DONE rejections on an outcome whose deterministic
#: verifier keeps returning False, the dispatcher routes it to the
#: LLM-judge path. Override via env DONE_REJECTED_FALLBACK_THRESHOLD.
DONE_REJECTED_FALLBACK_THRESHOLD = 5


# ──────────────────────────────────────────────────────────────────────────
# Verifier-spec reauthor gates (G1/G2/G4 — LLM call in initializer.py).
#   G1: consecutive FRESH diagnoses agreeing on verifier_mismatch before
#       we trust the classification.
#   G2: deterministic verifier must have rejected DONE this many times —
#       guards against reauthoring on flukes.
#   G4: per-outcome reauthor budget — reauthoring is itself an LLM call;
#       cap to avoid runaway loops on a stubborn task.
VERIFIER_REAUTHOR_MIN_STRIKES = 2
VERIFIER_REAUTHOR_MIN_DONE_REJECTIONS = 2
VERIFIER_REAUTHOR_MAX_PATCHES = 5

# In-step self-healing: when the verifier rejects a pending outcome AND
# the diagnoser says ``verifier_mismatch``, reauthor the spec and re-run
# in the SAME step so the actor never sees the broken spec. Max rounds
# per outcome per step; shares the ``patch_applied`` budget.
INSTEP_REAUTHOR_MAX_ROUNDS = 5


@dataclass
class OutcomeStateCache:
    """Incrementally maintained view of outcome states.

    Updated per StepRecord by walking only that step's KeyNodes, never
    the full timeline. Persisted to traj.jsonl; full re-derivation is
    reserved for replay tools.

    It also retains diagnostic state for the planner-feedback path:
      - done_rejected_count: per-outcome DONE rejections. After
        DONE_REJECTED_FALLBACK_THRESHOLD without a verified flip, the
        dispatcher drops the deterministic verifier and asks the LLM
        judge — rescues an unsatisfiable spec from init_ledger.
      - last_verifier_trace: latest verify_with_trace dict per outcome,
        rendered into force_replan so the planner sees WHAT was checked
        and what didn't match, not just "still pending".
      - file_baseline_size / file_baseline_hash: file_grep target snapshot
        at first observation, compared against the current trace to derive
        "file changed since first observation?" — a strong signal of
        whether the actor's edits are reaching disk.
    """
    states: Dict[str, str] = field(default_factory=dict)
    last_satisfy_step: Dict[str, int] = field(default_factory=dict)
    last_invalidate_step: Dict[str, int] = field(default_factory=dict)
    revert_count: Dict[str, int] = field(default_factory=dict)
    done_rejected_count: Dict[str, int] = field(default_factory=dict)
    last_verifier_trace: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    file_baseline_size: Dict[str, int] = field(default_factory=dict)
    file_baseline_hash: Dict[str, str] = field(default_factory=dict)
    # True once an outcome reached VERIFIED via a deterministic KeyNode.
    # While True, an LLM invalidation is refused — only another
    # deterministic verdict reverts it. Guards the false-negative trap
    # where the verifier transiently can't read just-changed state (e.g.
    # Chrome caches Bookmarks in memory, flushes lazily) and an LLM judge
    # overturns a real achievement on stale visual evidence.
    verified_by_deterministic: Dict[str, bool] = field(default_factory=dict)
    # Diagnosis memoization. signature = cheap hash of the verifier trace
    # at diagnose-time; last_diagnosis = the LLM's text. Re-run the LLM
    # only when the signature changes (file size, per-pattern counts,
    # content hash) — bounds calls to "actor did something new", ~3-5 per
    # task instead of per turn.
    last_diagnosis_signature: Dict[str, str] = field(default_factory=dict)
    last_diagnosis: Dict[str, str] = field(default_factory=dict)
    # Root-cause history. verifier_mismatch_strikes counts CONSECUTIVE
    # FRESH LLM calls classifying this outcome as verifier_mismatch;
    # resets on any planner_action. Gates auto-reauthor (G1: strikes >= 2).
    last_diagnosis_cause: Dict[str, str] = field(default_factory=dict)
    verifier_mismatch_strikes: Dict[str, int] = field(default_factory=dict)
    # Per-outcome reauthor count (G4). Once capped, fall through to the
    # K-fallback path instead of patching again.
    patch_applied: Dict[str, int] = field(default_factory=dict)

    def get_state(self, outcome_id: str) -> str:
        return self.states.get(outcome_id, OUTCOME_PENDING)

    def record_done_rejection(self, bad_outcome_ids: List[str]) -> None:
        """Bump the DONE-rejection counter for each outcome that caused
        a can_accept_done_claim failure."""
        for oid in bad_outcome_ids:
            self.done_rejected_count[oid] = (
                self.done_rejected_count.get(oid, 0) + 1
            )

    def get_done_rejection_count(self, outcome_id: str) -> int:
        return self.done_rejected_count.get(outcome_id, 0)

    def should_fall_back_to_llm(self, outcome_id: str,
                                 threshold: Optional[int] = None) -> bool:
        """True once this outcome has been rejected ``threshold`` times —
        caller drops deterministic verify and hands it to the LLM judge."""
        if threshold is None:
            try:
                import os as _os
                threshold = int(_os.environ.get(
                    "DONE_REJECTED_FALLBACK_THRESHOLD",
                    DONE_REJECTED_FALLBACK_THRESHOLD,
                ))
            except (ValueError, TypeError):
                threshold = DONE_REJECTED_FALLBACK_THRESHOLD
        return self.get_done_rejection_count(outcome_id) >= threshold

    def record_verifier_trace(
        self, outcome_id: str, trace: Dict[str, Any]
    ) -> None:
        """Stash the latest verify_with_trace dict (called by
        key_node_detector after each deterministic run). Rendered into
        force_replan as concrete diagnostic feedback. Also records the
        first-seen file size/hash as the baseline."""
        if not isinstance(trace, dict):
            return
        self.last_verifier_trace[outcome_id] = dict(trace)
        size = trace.get("file_size_bytes")
        if (isinstance(size, int)
                and outcome_id not in self.file_baseline_size):
            self.file_baseline_size[outcome_id] = size
        ch = trace.get("file_content_hash")
        if (isinstance(ch, str)
                and outcome_id not in self.file_baseline_hash):
            self.file_baseline_hash[outcome_id] = ch

    def file_changed_since_baseline(self, outcome_id: str) -> Optional[bool]:
        """Whether the actor has touched the verifier-target file yet.

        Returns True (size or hash differs), False (both match), or None
        (baseline/current data unavailable).
        """
        last = self.last_verifier_trace.get(outcome_id) or {}
        base_size = self.file_baseline_size.get(outcome_id)
        base_hash = self.file_baseline_hash.get(outcome_id)
        cur_size = last.get("file_size_bytes")
        cur_hash = last.get("file_content_hash")
        if base_size is None or cur_size is None:
            return None
        if base_size != cur_size:
            return True
        if base_hash is not None and cur_hash is not None:
            return base_hash != cur_hash
        return False

    def apply_step(self, step: StepRecord, all_outcome_ids: List[str]) -> None:
        """Fold a single StepRecord into the cache."""
        for kn in step.key_nodes:
            if not kn.enters_derivation():
                continue
            if kn.target not in all_outcome_ids:
                continue
            if kn.kind == KN_OUTCOME_SATISFIED:
                self.states[kn.target] = OUTCOME_VERIFIED
                self.last_satisfy_step[kn.target] = step.step_idx
                # Track HOW it was verified: deterministic re-satisfy sets
                # the flag, LLM re-satisfy clears it (so a later LLM judge
                # can't overturn a deterministic confirmation).
                self.verified_by_deterministic[kn.target] = (
                    getattr(kn, "source", "") == "deterministic")
            elif kn.kind == KN_OUTCOME_INVALIDATED:
                if self.states.get(kn.target) != OUTCOME_VERIFIED:
                    continue
                if step.step_idx <= self.last_satisfy_step.get(kn.target, -1):
                    continue
                # Deterministic confirmation overturned only by another
                # deterministic verdict, never by an LLM judge ("can't read
                # it" ≠ "refuted" — same as the DoneAuditor fix).
                if (self.verified_by_deterministic.get(kn.target)
                        and getattr(kn, "source", "") != "deterministic"):
                    continue
                if self.revert_count.get(kn.target, 0) >= MAX_REVERT_COUNT:
                    continue  # flap protection
                self.states[kn.target] = OUTCOME_REVERTED
                self.last_invalidate_step[kn.target] = step.step_idx
                self.revert_count[kn.target] = self.revert_count.get(kn.target, 0) + 1

    def all_verified(self, all_outcome_ids: List[str]) -> bool:
        return all(self.get_state(oid) == OUTCOME_VERIFIED for oid in all_outcome_ids)

    def can_accept_done_claim(
        self, all_outcome_ids: List[str]
    ) -> "tuple[bool, str]":
        """Gate: DONE accepted iff every outcome is in OUTCOME_VERIFIED."""
        if self.all_verified(all_outcome_ids):
            return True, ""
        bad = [oid for oid in all_outcome_ids
               if self.get_state(oid) != OUTCOME_VERIFIED]
        labels = []
        for oid in bad:
            st = self.get_state(oid)
            labels.append(f"{oid}({st})")
        return False, f"{len(bad)} outcome(s) not verified: " + ", ".join(labels)

    @classmethod
    def rebuild_from_timeline(
        cls,
        timeline: List[TimelineEvent],
        all_outcome_ids: List[str],
    ) -> "OutcomeStateCache":
        """Full rebuild for replay tools / integrity checks. Online the
        agent maintains the cache incrementally and never calls this."""
        cache = cls()
        for ev in timeline:
            for s in ev.steps:
                cache.apply_step(s, all_outcome_ids)
        return cache


# --------------------------------------------------------------------------- #
# Timeline maintenance helpers
# --------------------------------------------------------------------------- #


def _last_event(timeline: List[TimelineEvent]) -> Optional[TimelineEvent]:
    return timeline[-1] if timeline else None


def append_step(
    timeline: List[TimelineEvent],
    step: StepRecord,
    current_subgoal: str,
) -> TimelineEvent:
    """Attach a finished StepRecord to its event; return that event.

    New event when: timeline empty, last event already closed, or subgoal
    text changed. Otherwise append to the last event.
    """
    last = _last_event(timeline)
    open_new = (
        last is None
        or last.outcome != EV_ONGOING
        or (last.subgoal or "").strip() != (current_subgoal or "").strip()
    )
    if open_new:
        ev = TimelineEvent(
            event_idx=(last.event_idx + 1) if last else 0,
            subgoal=current_subgoal or "",
            started_at_step=step.step_idx,
            steps=[step],
            outcome=EV_ONGOING,
        )
        timeline.append(ev)
        return ev
    last.steps.append(step)
    return last


def close_current_event(
    timeline: List[TimelineEvent],
    *,
    outcome: str,
    at_step: int,
    reason: Optional[str] = None,
) -> None:
    """Apply a terminal outcome to the currently-ongoing event.

    ``reason`` (optional): planner's <Bottleneck> at close-time, stored
    so renderers can show WHY an abandoned event ended.
    """
    last = _last_event(timeline)
    if last is None or last.outcome != EV_ONGOING:
        return
    last.outcome = outcome
    last.ended_at_step = at_step
    if reason:
        last.closing_reason = reason[:240].strip()


def event_outcome_for_decision(decision: Optional[str], done: Optional[str]) -> str:
    """Map (decision, done) to the PREVIOUS event's outcome.

    `done` = previous subgoal's success; decision governs the next plan:
      YES + CONTINUE/REPLAN → committed
      NO  + CONTINUE        → ongoing (don't close, will retry)
      NO  + REPLAN          → abandoned_replan
      DONE                  → ended_done (prev subgoal assumed done)
    Ambiguous combos (e.g. done=None) default to abandoned_replan only on
    REPLAN, which usually follows a failure even without a clean assessment.
    """
    if decision == "DONE":
        return EV_ENDED_DONE
    if done == "YES":
        return EV_COMMITTED
    if decision == "REPLAN":
        return EV_ABANDONED_REPLAN
    # done=NO + CONTINUE (or unknown + CONTINUE) — leave ongoing
    return EV_ONGOING


# --------------------------------------------------------------------------- #
# Render helpers (consumed by planner prompt + trajectory.html)
# --------------------------------------------------------------------------- #


_OUTCOME_GLYPH = {
    EV_COMMITTED: "✓ committed",
    EV_ABANDONED_REPLAN: "✗ abandoned (REPLAN)",
    EV_ENDED_DONE: "● task DONE",
    EV_ONGOING: "⌛ ONGOING",
}


def _summarize_event_keynodes(ev: TimelineEvent) -> str:
    """One-line summary of all KeyNodes in an event, e.g.
    '+price_25_60 @s8; +on_sale @s4; -price_25_60 @s30 ⚠'."""
    bits: List[str] = []
    for kn in ev.all_key_nodes:
        if kn.kind == KN_OUTCOME_SATISFIED:
            sign, mark = "+", ""
        elif kn.kind == KN_OUTCOME_INVALIDATED:
            sign, mark = "-", " ⚠"
        elif kn.kind == KN_NAVIGATION:
            sign, mark = "→", ""
        elif kn.kind == KN_DIALOG_OPENED:
            sign, mark = "[]", ""
        elif kn.kind == KN_DIALOG_CLOSED:
            sign, mark = "[X]", ""
        elif kn.kind == KN_VALUE_COMMITTED:
            sign, mark = "=", ""
        elif kn.kind == KN_STRATEGY_FAILED:
            sign, mark = "✗", " ⚠"
        else:
            sign, mark = "?", ""
        evidence = (kn.evidence or "")[:60]
        conf_tag = "" if kn.confidence == CONF_HIGH else f"({kn.confidence})"
        bits.append(f"{sign}{kn.target}@s{kn.detected_at_step}{mark}{conf_tag}")
        if evidence:
            bits[-1] += f"[{evidence}]"
    return "; ".join(bits) if bits else "(no key nodes)"


# --------------------------------------------------------------------------- #
# Render-layer grouping (P1) — collapse consecutive REPLAN-in-place events
# whose subgoals are near-rephrasings ("scroll down" vs "scroll down further")
# into one compact group. Render-only; the event list is not mutated, so
# persistence/replay still see the granular sequence.
# --------------------------------------------------------------------------- #

_TIMELINE_GROUP_STOPWORDS = frozenset({
    "the", "and", "for", "with", "into", "from", "this", "that",
    "are", "not", "any", "all", "to", "in", "on", "of", "or",
    "a", "an", "is", "be", "by", "as", "it", "its",
    # Generic UI verbs/nouns that don't discriminate strategy
    "click", "scroll", "type", "press", "select", "hover", "find",
    "reveal", "show", "open", "close", "page", "screen", "view",
    "tab", "section", "panel", "area", "left", "right", "top", "bottom",
    "down", "up", "further", "more",
})


def _timeline_subgoal_signature(subgoal: str) -> set:
    """Token-set for deciding whether two events describe the same
    strategy. Like ledger._review_signature but with a stopword list tuned
    for subgoals (drops 'scroll'/'click'/'reveal' so the discriminating
    tokens are domain nouns like 'sidebar', 'color', 'filter')."""
    if not subgoal:
        return set()
    import re as _re
    words = _re.findall(r"[a-zA-Z]+", subgoal.lower())
    return {w for w in words
            if len(w) >= 3 and w not in _TIMELINE_GROUP_STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _group_consecutive_similar(
    events: List[TimelineEvent],
    *,
    threshold: float = 0.5,
) -> List[List[TimelineEvent]]:
    """Merge each event into the previous group when its subgoal Jaccard
    >= ``threshold`` against the group's last event AND they share an
    outcome class. Same-class only so the group carries one consistent
    outcome label (e.g. an abandoned vs committed "scroll sidebar" stay
    separate; the scroll-loop pattern is many consecutive ABANDONEDs)."""
    groups: List[List[TimelineEvent]] = []
    for ev in events:
        ev_sig = _timeline_subgoal_signature(ev.subgoal)
        if groups:
            last_group = groups[-1]
            tail = last_group[-1]
            tail_sig = _timeline_subgoal_signature(tail.subgoal)
            same_outcome_class = (tail.outcome == ev.outcome)
            sim = _jaccard(ev_sig, tail_sig)
            if same_outcome_class and sim >= threshold:
                last_group.append(ev)
                continue
        groups.append([ev])
    return groups


def _truncate_at_word(text: str, max_len: int) -> str:
    """Truncate to ``max_len`` chars at a word boundary, backing up to the
    last space within the final 16 chars; hard-cut + ellipsis otherwise."""
    if not text or len(text) <= max_len:
        return text or ""
    cut = text[:max_len]
    sp = cut.rfind(" ")
    if sp > max_len - 16:
        return cut[:sp].rstrip() + "…"
    return cut.rstrip() + "…"


def _format_steps_range(group: List[TimelineEvent]) -> str:
    """Step span across a group: 'A-B' or 'A-B+' if the tail is ongoing."""
    first = group[0]
    last = group[-1]
    start = first.started_at_step
    if last.ended_at_step is not None:
        end = last.ended_at_step
        return f"{start}-{end}" if start != end else str(start)
    last_step = last.steps[-1].step_idx if last.steps else last.started_at_step
    return f"{start}-{last_step}+"


def _format_ev_range(group: List[TimelineEvent]) -> str:
    """Event-index span across a group, e.g. 'ev7-12' or 'ev9'."""
    if len(group) == 1:
        return f"{group[0].event_idx}"
    return f"{group[0].event_idx}-{group[-1].event_idx}"


def _group_outcome_label(group: List[TimelineEvent]) -> str:
    """Outcome cell text. Multi-event abandoned groups surface the retry
    count: ``✗ abandoned (6× REPLAN)``."""
    n_events = len(group)
    n_steps = sum(ev.n_steps for ev in group)
    last_outcome = group[-1].outcome
    base = _OUTCOME_GLYPH.get(last_outcome, last_outcome)
    if last_outcome == EV_ABANDONED_REPLAN and n_events >= 2:
        return f"✗ abandoned ({n_events}× REPLAN, {n_steps} steps)"
    if last_outcome == EV_ABANDONED_REPLAN and n_steps > 1:
        return f"{base} ({n_steps}retry)"
    if last_outcome == EV_COMMITTED and n_steps > 1:
        return f"{base} ({n_steps}retry)"
    if last_outcome != EV_ONGOING and n_steps > 1:
        return f"{base} ({n_steps}retry)"
    return base


def _group_keynodes_summary(group: List[TimelineEvent]) -> str:
    """Union of KeyNodes across a group, deduped on (kind, target, step),
    rendered via the single-event summarizer so the column reads the same."""
    seen = set()
    fake_ev = TimelineEvent(
        event_idx=-1, subgoal="", started_at_step=0, steps=[],
    )
    for ev in group:
        for kn in ev.all_key_nodes:
            sig = (kn.kind, kn.target, kn.detected_at_step)
            if sig in seen:
                continue
            seen.add(sig)
            # Feed the single-event summarizer dummy steps of unique KeyNodes.
            fake_step = StepRecord(step_idx=kn.detected_at_step, key_nodes=[kn])
            fake_ev.steps.append(fake_step)
    return _summarize_event_keynodes(fake_ev)


def _group_closing_reasons(group: List[TimelineEvent]) -> str:
    """Distinct closing_reasons across a group, newest first, length-bounded."""
    seen: List[str] = []
    for ev in reversed(group):
        r = (ev.closing_reason or "").strip()
        if not r:
            continue
        # Token-jaccard dedup against already-collected reasons.
        sig = _timeline_subgoal_signature(r)
        is_dup = any(_jaccard(sig, _timeline_subgoal_signature(s)) >= 0.7
                     for s in seen)
        if is_dup:
            continue
        seen.append(r)
        if len(seen) >= 2:  # cap distinct reasons per group
            break
    return " | ".join(seen)


def render_timeline_for_planner(
    timeline: List[TimelineEvent],
    *,
    n_recent_events: int = 8,
    subgoal_col_width: int = 60,
) -> str:
    """Compact event-granularity table for the planner prompt.

    P1 grouping collapses consecutive near-rephrasing events sharing an
    outcome class into one block, so "scroll down / scroll down further /
    scroll down in sidebar" doesn't show as 6 separate ``✗ abandoned``
    rows. Render-only; persistence/replay see raw events. Closing reasons
    are surfaced under each abandoned row (WHY it was abandoned).
    """
    if not timeline:
        return "[Task Event Timeline] (empty — task just started)"

    shown = timeline[-n_recent_events:]
    elided = len(timeline) - len(shown)
    head = f"[Task Event Timeline] ({len(timeline)} event(s) total"
    if elided > 0:
        head += f", showing last {len(shown)}"
    head += ")"
    lines = [head, ""]
    lines.append(
        f"{'ev':>5} | {'steps':<10} | "
        f"{'subgoal':<{subgoal_col_width}} | {'outcome':<28} | key_nodes"
    )
    lines.append("─" * 6 + "┼" + "─" * 12 + "┼" + "─" * (subgoal_col_width + 2)
                 + "┼" + "─" * 30 + "┼" + "─" * 30)

    groups = _group_consecutive_similar(shown)
    for group in groups:
        first = group[0]
        ev_str = _format_ev_range(group)
        steps_str = _format_steps_range(group)
        # Subgoal cell — first event's subgoal, word-truncated; grouped
        # runs append "(+N rephrasings)".
        canonical = first.subgoal or ""
        subgoal_cell = _truncate_at_word(canonical, subgoal_col_width)
        if len(group) > 1:
            extra = f"  (+{len(group)-1} rephrasings)"
            # Keep total within column width.
            if len(subgoal_cell) + len(extra) > subgoal_col_width:
                subgoal_cell = _truncate_at_word(
                    canonical, subgoal_col_width - len(extra))
            subgoal_cell = subgoal_cell + extra
        outcome_cell = _group_outcome_label(group)
        kn_cell = _group_keynodes_summary(group)
        lines.append(
            f"{ev_str:>5} | {steps_str:<10} | "
            f"{subgoal_cell:<{subgoal_col_width}} | {outcome_cell:<28} | "
            f"{kn_cell[:120]}"
        )
        # Reason continuation row for abandoned outcomes with a captured
        # closing_reason; indented under the subgoal column.
        last_outcome = group[-1].outcome
        if last_outcome == EV_ABANDONED_REPLAN:
            reasons = _group_closing_reasons(group)
            if reasons:
                indent = (" " * 5 + " | " + " " * 10 + " |   reason: ")
                wrap_len = subgoal_col_width + 30
                lines.append(indent + _truncate_at_word(reasons, wrap_len))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Strategy-grouped render (groups events by strategy span instead of per-event)
# --------------------------------------------------------------------------- #


def _strategy_spans_from_history(
    strategy_history: List[Any],   # List[StrategyEvent]
    final_step: int,
) -> List[Dict[str, Any]]:
    """Convert a flat StrategyEvent list into [start, end) spans.

    A span runs from when a strategy became active until the next
    ``switch`` (exclusive), or ``final_step + 1`` for the active span.
    ``install``/``switch`` start a span; ``same`` refines within it.

    Each dict carries strategy_text, start_step (incl), end_step (excl),
    is_active, and replan_count (number of ``same`` re-confirmations).
    """
    spans: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for ev in strategy_history:
        kind = getattr(ev, "kind", "")
        if kind in ("install", "switch"):
            if current is not None:
                current["end_step"] = ev.step_idx
                current["is_active"] = False
                spans.append(current)
            current = {
                "strategy_text": getattr(ev, "new_strategy", "") or "",
                "start_step": getattr(ev, "step_idx", 0)
                                if kind == "install"
                                else getattr(ev, "step_idx", 0),
                "end_step": final_step + 1,  # placeholder; set by next switch
                "is_active": True,
                "replan_count": 0,
            }
        elif kind == "same" and current is not None:
            current["replan_count"] += 1
    if current is not None:
        spans.append(current)
    return spans


def _events_in_span(
    timeline: List[Any],   # List[TimelineEvent]
    start_step: int,
    end_step: int,
) -> List[Any]:
    """Events whose started_at_step ∈ [start_step, end_step)."""
    return [
        ev for ev in timeline
        if start_step <= getattr(ev, "started_at_step", 0) < end_step
    ]


def _keynodes_in_events(
    events: List[Any],
) -> List[Tuple[int, str, str, str]]:
    """Flatten KeyNodes to ``[(step_idx, kind, target, evidence_short), ...]``
    sorted by step."""
    kns: List[Tuple[int, str, str, str]] = []
    for ev in events:
        for s in getattr(ev, "steps", []) or []:
            for kn in getattr(s, "key_nodes", []) or []:
                kns.append((
                    getattr(s, "step_idx", 0),
                    getattr(kn, "kind", ""),
                    getattr(kn, "target", ""),
                    (getattr(kn, "evidence", "") or "")[:80],
                ))
    kns.sort(key=lambda t: t[0])
    return kns


def _format_keynode_chain(kns: List[Tuple[int, str, str, str]]) -> str:
    """Render a compact one-line summary of a KeyNode sequence."""
    if not kns:
        return "(none)"
    pieces = []
    for step_idx, kind, target, _ev in kns:
        glyph = "✓" if kind == KN_OUTCOME_SATISFIED else (
            "✗" if kind == KN_OUTCOME_INVALIDATED else "•")
        pieces.append(f"{glyph} {target}@s{step_idx}")
    return " → ".join(pieces)


def _unique_subgoals(events: List[Any], *, cap: int = 5) -> List[str]:
    """Distinct subgoal texts from events, in order, capped."""
    seen = set()
    out: List[str] = []
    for ev in events:
        sg = (getattr(ev, "subgoal", "") or "").strip()
        if not sg or sg.lower() in seen:
            continue
        seen.add(sg.lower())
        out.append(sg)
    return out[:cap], len(out) - cap if len(out) > cap else 0


def _span_outcome_label(
    span: Dict[str, Any], events: List[Any], n_keynodes_satisfied: int,
) -> str:
    """One-line outcome label for a strategy span."""
    if span["is_active"]:
        if n_keynodes_satisfied > 0:
            return f"⏵ ACTIVE (advancing, {n_keynodes_satisfied} outcome verified)"
        return "⏵ ACTIVE — ongoing"
    # Inactive span (switched away from)
    if n_keynodes_satisfied > 0:
        # made progress but planner switched anyway — usually the next
        # strategy is a refinement / sub-step
        return f"✓ committed (advanced {n_keynodes_satisfied}, then refined)"
    return "✗ ABANDONED (no outcome verified during span)"


def _span_close_reason(events: List[Any]) -> Optional[str]:
    """Most-recent non-empty closing_reason in a span (last event first)."""
    for ev in reversed(events):
        r = (getattr(ev, "closing_reason", "") or "").strip()
        if r:
            return r
    return None


def render_timeline_grouped_by_strategy(
    timeline: List[Any],                   # List[TimelineEvent]
    strategy_history: List[Any],           # List[StrategyEvent]
    *,
    n_recent_spans: int = 3,
    bullet_cap_per_span: int = 5,
    curator_strategy_notes: Optional[Dict[str, str]] = None,
) -> str:
    """Render PAST (abandoned / committed-then-refined) strategy spans.

    The ACTIVE span is deliberately skipped — its strategy, subgoal, and
    recent actions already appear in [Current plan]/[Current subgoal]/
    [Recent actions], and re-stating them under a "timeline" header misled
    the model (the echoed focus subgoal got read as "do this next").

    This block uniquely carries each past approach's strategy text, abandon
    reason (+ curator verdict if any), and the KeyNode chain that fired —
    so it's only worth rendering when something was actually abandoned.

    Returns "" when there are no past spans; callers MUST skip the block
    on empty rather than emit a header with no body.

    ``curator_strategy_notes`` (optional): strategy_text → note from the
    dead-end curator; adds a ``curator: <verdict>`` line per span.
    """
    if not strategy_history:
        return ""

    final_step = 0
    if timeline:
        for ev in timeline:
            for s in getattr(ev, "steps", []) or []:
                final_step = max(final_step, getattr(s, "step_idx", 0))

    spans = _strategy_spans_from_history(strategy_history, final_step)
    if not spans:
        return ""

    # Drop the active span — duplicates Current plan / subgoal / actions.
    past_spans = [s for s in spans if not s["is_active"]]
    if not past_spans:
        return ""

    elided = max(0, len(past_spans) - n_recent_spans)
    shown = past_spans[-n_recent_spans:]
    head_parts = [
        f"[Past abandoned strategies] "
        f"({len(past_spans)} past span(s)"
    ]
    if elided > 0:
        head_parts.append(f", showing most recent {len(shown)}")
    head_parts.append(")")

    lines: List[str] = ["".join(head_parts), ""]

    for span in shown:
        span_events = _events_in_span(
            timeline, span["start_step"], span["end_step"])
        kns = _keynodes_in_events(span_events)
        n_satisfied = sum(1 for _, k, *_ in kns if k == KN_OUTCOME_SATISFIED)

        end_str = f"{span['end_step'] - 1}"
        header = (
            f"╭─ Strategy "
            f"(steps {span['start_step']}-{end_str}) — "
            f"{(span['strategy_text'] or '')[:140]}"
        )
        lines.append(header)
        lines.append(
            f"│   {_span_outcome_label(span, span_events, n_satisfied)}"
            f"{', ' + str(span['replan_count']) + ' same-strategy REPLANs' if span['replan_count'] else ''}"
        )

        if kns:
            lines.append(f"│   key nodes: {_format_keynode_chain(kns)}")

        unique, overflow = _unique_subgoals(
            span_events, cap=bullet_cap_per_span)
        if unique:
            lines.append(f"│   subgoals tried under this strategy:")
            for sg in unique:
                lines.append(f"│     • {_truncate_at_word(sg, 100)}")
            if overflow > 0:
                lines.append(f"│     • (+{overflow} more)")

        reason = _span_close_reason(span_events)
        if reason:
            lines.append(
                f"│   abandon reason: "
                f"{_truncate_at_word(reason, 200)}")

        if curator_strategy_notes:
            note = curator_strategy_notes.get(span["strategy_text"] or "")
            if note:
                lines.append(f"│   curator: {_truncate_at_word(note, 180)}")

        lines.append("╰─")
        lines.append("")

    return "\n".join(lines).rstrip()


def _compute_focus_outcome_id(
    required_outcomes: List[Any],
) -> Optional[str]:
    """First pending outcome whose deps are all verified. Mirrors
    ProgressLedger.next_focus_outcome but lives here so the renderer runs
    on duck-typed args without an importable ledger. A1: reads state from
    per-outcome fields, not a separate cache."""
    by_id = {getattr(o, "id", None): o for o in required_outcomes}
    for o in required_outcomes:
        if getattr(o, "state", OUTCOME_PENDING) == OUTCOME_VERIFIED:
            continue
        deps = list(getattr(o, "depends_on", []) or [])
        if all(
            (by_id.get(d) is not None
             and getattr(by_id[d], "state", OUTCOME_PENDING) == OUTCOME_VERIFIED)
            for d in deps
        ):
            return o.id
    return None


def render_verifier_feedback(
    outcome: Any,
    *,
    call_llm: Optional[Callable] = None,
    model: Optional[str] = None,
) -> str:
    """Diagnostic block for ONE outcome's verifier state, fed into
    force_replan: the spec, the latest trace (per-pattern hits, file
    size, modification status), a file-content preview, the DONE-rejection
    count, and a root-cause hint.

    With ``call_llm`` the hint is an LLM diagnosis (reads file content +
    spec, 2-4 sentences); otherwise a heuristic fallback. LLM calls are
    memoized on last_diagnosis_signature so unchanged-state force_replans
    re-use the prior diagnosis.

    Returns "" when no verifier or no trace yet (caller skips the section).
    A1: reads trace / rejection count / diagnosis cache off the Outcome.
    A3: prefers outcome.evidence[-1].verifier_trace (full trace from
    bundle_assembler.build) over the legacy last_verifier_trace, falling
    back when no Evidence exists (legacy traj.jsonl replays)."""
    verify = getattr(outcome, "verify", None)
    if verify is None:
        return ""
    # A3: prefer the structured Evidence trace, fall back to legacy dict.
    trace: Dict[str, Any] = {}
    ev_list = getattr(outcome, "evidence", None) or []
    if ev_list:
        trace = dict(ev_list[-1].verifier_trace or {})
    if not trace:
        trace = dict(getattr(outcome, "last_verifier_trace", {}) or {})
    if not trace:
        return ""
    rej = getattr(outcome, "done_rejected_count", 0)
    kind = getattr(verify, "kind", "?")
    lines: List[str] = [
        f"[Verifier Feedback — outcome `{outcome.id}` "
        f"(rejected {rej} time{'s' if rej != 1 else ''})]",
        "",
        "  Verifier specification:",
        f"    kind:      {kind}",
    ]
    if kind == "file_grep":
        lines.append(f"    file_path: {getattr(verify, 'file_path', '?')}")
        amm = getattr(verify, 'all_must_match', True)
        plist = list(getattr(verify, 'patterns', []) or [])
        lines.append(
            f"    patterns ({'all must match' if amm else 'any may match'}):"
        )
        for i, p in enumerate(plist, 1):
            lines.append(f"      {i}. {p}")
    elif kind == "url_match":
        lines.append(f"    url_pattern: {getattr(verify, 'url_pattern', '?')}")
    elif kind == "a11y_match":
        if getattr(verify, "tag", None):
            lines.append(f"    tag:            {verify.tag}")
        if getattr(verify, "text_contains", None):
            lines.append(f"    text_contains:  {list(verify.text_contains)}")
        if getattr(verify, "state_contains", None):
            lines.append(f"    state_contains: {list(verify.state_contains)}")
        # Visual gates explain "a11y matched but DONE rejected": a failed
        # gate kept the LLM judge at pending. Showing them tells the
        # planner which side to fix (a11y row vs visible UI state).
        vmh = list(getattr(verify, "visual_must_hold", []) or [])
        vmn = list(getattr(verify, "visual_must_not_hold", []) or [])
        if vmh:
            lines.append("    visual_must_hold (LLM judges these "
                         "against the screenshot):")
            for c in vmh:
                lines.append(f"      • {c}")
        if vmn:
            lines.append("    visual_must_not_hold (must be FALSE):")
            for c in vmn:
                lines.append(f"      • {c}")
    elif kind == "shell_command":
        cmd = list(getattr(verify, "command", None) or [])
        exp = list(getattr(verify, "expected_substring", None) or [])
        forb = list(getattr(verify, "forbidden_substring", None) or [])
        lines.append(f"    command:             {cmd}")
        if exp:
            lines.append(f"    expected_substring:  {exp}  (any-of)")
        if forb:
            lines.append(f"    forbidden_substring: {forb}  (none may appear)")
        lines.append(
            f"    expected_exit_code:  "
            f"{getattr(verify, 'expected_exit_code', 0)}"
        )
    lines.append("")
    lines.append("  Last verifier run on the actual file/URL/a11y:")
    if kind == "file_grep":
        rp = trace.get("resolved_path", "?")
        sz = trace.get("file_size_bytes")
        lines.append(f"    resolved path: {rp}")
        if sz is not None:
            lines.append(f"    file size:     {sz} bytes")
        chg = outcome.file_changed_since_baseline()
        chg_str = (
            "CHANGED since first observation"
            if chg is True else (
                "UNCHANGED since first observation "
                "  ← actor's edits may not have committed to disk"
                if chg is False else "(unable to determine)"
            )
        )
        lines.append(f"    modified:      {chg_str}")
        ppm = trace.get("per_pattern_matches", []) or []
        for entry in ppm[:6]:
            pat = entry.get("pattern", "?")
            ml = entry.get("matched_lines", []) or []
            err = entry.get("compile_error")
            if err:
                lines.append(f"    pattern {pat!r}: COMPILE ERROR — {err}")
            else:
                lines.append(
                    f"    pattern {pat!r}: {len(ml)} match(es)"
                )
                if ml:
                    snip = (ml[0].get("snippet") or "")[:120]
                    lines.append(f"        ↳ first hit: {snip!r}")
    elif kind == "a11y_match":
        rs = trace.get("rows_scanned", 0)
        mr = trace.get("matched_row")
        lines.append(f"    rows scanned: {rs}")
        if mr:
            lines.append(f"    matched row:  {mr[:200]}")
        else:
            lines.append("    matched row:  (no row matched all constraints)")
    elif kind == "url_match":
        url = trace.get("url_extracted", "(no URL extracted)")
        lines.append(f"    url extracted: {url}")
    elif kind == "shell_command":
        ec = trace.get("exit_code")
        lines.append(
            f"    exit code:    {ec if ec is not None else '(none — runner failed)'}"
        )
        ms = trace.get("matched_substring")
        if ms:
            lines.append(f"    matched_substring: {ms!r}")
        head = (trace.get("stdout_head") or "").strip()
        if head:
            lines.append("    stdout (first 600 chars, stderr merged):")
            for snip in head.splitlines()[:8]:
                lines.append(f"      | {snip[:140]}")
    vr = trace.get("verdict_reason") or ""
    if vr:
        lines.append(f"    reason:        {vr[:200]}")
    lines.append("")
    lines.append("  Diagnosis hint:")
    diag = _diagnose_outcome(
        outcome, verify, trace,
        call_llm=call_llm, model=model,
    )
    for ln in diag.split("\n"):
        lines.append(f"    {ln}")
    return "\n".join(lines)


def _trace_signature(trace: Dict[str, Any]) -> str:
    """Cheap fingerprint of a verifier trace; detects whether the actor
    produced new state (or the spec was reauthored) since the last
    diagnosis. New signature = fresh diagnose; same = cache hit."""
    parts = [
        str(trace.get("file_size_bytes", "?")),
        str(trace.get("file_content_hash", "?")),
        # shell_command discriminator: without exit_code + stdout, all
        # shell failures collapse to '?|?' and cache-hit forever, starving
        # the in-step healing loop of the new spec's trace.
        f"ec:{trace.get('exit_code', '?')}",
    ]
    sh = (trace.get("stdout_head", "") or "")[:100].strip()
    if sh:
        parts.append(f"sh:{sh}")
    for e in (trace.get("per_pattern_matches") or []):
        parts.append(str(len(e.get("matched_lines") or [])))
    if trace.get("matched_row"):
        parts.append("a11y:hit")
    if trace.get("url_extracted"):
        parts.append(f"url:{str(trace.get('url_extracted', ''))[:40]}")
    return "|".join(parts)


def _diagnose_outcome(
    outcome: Any, verify: Any, trace: Dict[str, Any],
    *,
    call_llm: Optional[Callable], model: Optional[str],
) -> str:
    """2-4 sentence diagnosis of why the verifier keeps returning False,
    plus a root-cause classification.

    With ``call_llm``: ask the LLM (file head consumed internally to keep
    the planner block compact), cached by trace signature. Otherwise a
    heuristic fallback (no root-cause tag).

    Side effects on the Outcome (A1): last_diagnosis_signature,
    last_diagnosis, last_diagnosis_cause, and verifier_mismatch_strikes —
    the strikes counter moves on FRESH LLM calls only, not cache hits.

    Returns the diagnosis text; the agent reads the cause fields off the
    Outcome to gate G1 + reauthor decisions."""
    sig = _trace_signature(trace)
    cached_sig = outcome.last_diagnosis_signature
    if cached_sig is not None and cached_sig == sig:
        cached = outcome.last_diagnosis
        if cached:
            # Cache hit — don't touch strikes (FRESH calls only).
            return cached
    if call_llm is None:
        diag_text = _derive_diagnosis_hint(verify, trace, outcome)
        outcome.last_diagnosis_signature = sig
        outcome.last_diagnosis = diag_text
        # heuristic path doesn't classify → leave cause field as-is
        return diag_text
    try:
        result = _llm_diagnose(outcome, verify, trace, call_llm, model)
        diag_text = result.get("text") or ""
        cause = result.get("root_cause") or ""
        if not diag_text:
            diag_text = _derive_diagnosis_hint(verify, trace, outcome)
            cause = ""
    except Exception:
        diag_text = _derive_diagnosis_hint(verify, trace, outcome)
        cause = ""
    outcome.last_diagnosis_signature = sig
    outcome.last_diagnosis = diag_text
    if cause in ("planner_action", "verifier_mismatch"):
        outcome.last_diagnosis_cause = cause
        # Strikes: bump on verifier_mismatch, reset on planner_action.
        if cause == "verifier_mismatch":
            outcome.verifier_mismatch_strikes = (
                outcome.verifier_mismatch_strikes + 1
            )
        else:
            outcome.verifier_mismatch_strikes = 0
    return diag_text


_DIAGNOSE_SYSTEM_PROMPT = """You are diagnosing why an outcome's
deterministic verifier keeps returning NOT-SATISFIED. The user gives
you the verifier specification, the last verifier run's trace, and a
preview of the current target file content.

Output exactly two things, in this order:

1. A 2-4 sentence root-cause analysis followed by one concrete
   next-step recommendation. Pick ONE of these templates:

   (a) "The required content has not yet been written to the file
        (file shows ... instead). Next: actor should ..."
   (b) "The actor saved the wrong location/key (file has ... but
        verifier expects ...). Next: ..."
   (c) "The verifier patterns appear inconsistent with the task
        intent (patterns expect ... but the file's natural form
        is ...). Next: ..."
   (d) "The required content IS in the file but not in a form the
        verifier matches (e.g. ... vs ...). Next: ..."

2. On the very LAST line, append exactly one root-cause classification
   tag — pick STRICTLY ONE of TWO values:

       [ROOT_CAUSE: planner_action]
         Choose this when the diagnosis is template (a) or (b), or
         whenever the file/UI state shows the actor has not yet
         produced the required result. The planner should keep
         driving the actor to take the right action.

       [ROOT_CAUSE: verifier_mismatch]
         Choose this when the diagnosis is template (c) or (d), or
         whenever you believe the verifier specification itself does
         NOT correctly capture the task's success state — i.e. the
         actor has substantially done the right thing but the spec
         can't recognise it.

Do NOT use any other tag value (no "unknown", no "both"). Do NOT
add markdown headers, no preamble like "Diagnosis:". Plain prose
diagnosis, then the tag on its own final line."""


_ROOT_CAUSE_TAG_RE = re.compile(
    r"\[ROOT_CAUSE\s*:\s*(planner_action|verifier_mismatch)\s*\]",
    re.IGNORECASE,
)


def _llm_diagnose(
    outcome: Any, verify: Any, trace: Dict[str, Any],
    call_llm: Callable, model: Optional[str],
) -> Dict[str, str]:
    """One-shot LLM diagnosis. Returns ``{"text", "root_cause"}`` where
    root_cause is "" when the tag couldn't be parsed.

    The file_content_head feeds this LLM's input but only ``text`` reaches
    the planner-visible prompt, keeping that block compact."""
    kind = getattr(verify, "kind", "?")
    desc = getattr(outcome, "description", "") or ""
    hint = getattr(outcome, "evidence_hint", "") or ""
    user_parts: List[str] = [
        f"Outcome id: {outcome.id}",
        f"Description: {desc}",
        f"Evidence hint: {hint}",
        "",
        "Verifier specification:",
        f"  kind: {kind}",
    ]
    if kind == "file_grep":
        user_parts.append(f"  file_path: {getattr(verify, 'file_path', '?')}")
        amm = getattr(verify, "all_must_match", True)
        user_parts.append(
            f"  patterns ({'all must match' if amm else 'any may match'}):"
        )
        for i, p in enumerate(getattr(verify, "patterns", []) or [], 1):
            user_parts.append(f"    {i}. {p}")
    elif kind == "url_match":
        user_parts.append(f"  url_pattern: {getattr(verify, 'url_pattern', '?')}")
    elif kind == "a11y_match":
        if getattr(verify, "tag", None):
            user_parts.append(f"  tag: {verify.tag}")
        if getattr(verify, "text_contains", None):
            user_parts.append(f"  text_contains: {list(verify.text_contains)}")
        if getattr(verify, "state_contains", None):
            user_parts.append(f"  state_contains: {list(verify.state_contains)}")
    elif kind == "shell_command":
        user_parts.append(f"  command: {list(getattr(verify, 'command', None) or [])}")
        user_parts.append(
            f"  expected_substring (any-of): "
            f"{list(getattr(verify, 'expected_substring', None) or [])}"
        )
        if getattr(verify, "forbidden_substring", None):
            user_parts.append(
                f"  forbidden_substring: {list(verify.forbidden_substring)}"
            )
        user_parts.append(
            f"  expected_exit_code: "
            f"{getattr(verify, 'expected_exit_code', 0)}"
        )
    elif kind in ("calc_verify", "impress_verify"):
        # Structured verifiers: the framework builds the body from a fixed
        # op+kwargs schema, so the "spec" to diagnose is the check list,
        # not a shell string. Without it the LLM sees only the kind and
        # guesses verifier_mismatch — almost always wrong, since these
        # specs can't be "off"; a FAIL means the actor isn't at target yet.
        checks_attr = "calc_checks" if kind == "calc_verify" else "impress_checks"
        checks = list(getattr(verify, checks_attr, None) or [])
        user_parts.append(
            f"  {checks_attr} (ALL must pass — AND across the list):"
        )
        if not checks:
            user_parts.append("    (empty — spec build error)")
        else:
            for i, chk in enumerate(checks, 1):
                user_parts.append(f"    {i}. {chk}")
        user_parts.append(
            "  NOTE: this spec is framework-generated from a fixed "
            "schema (op + kwargs) — the spec itself is not authored "
            "by an LLM. A FAIL almost always means the actor has not "
            "yet driven the document to the target state, not that "
            "the patterns are off. Prefer the ``planner_action`` root "
            "cause unless live state unambiguously shows the property "
            "IS at the target value despite the FAIL."
        )
    user_parts.append("")
    user_parts.append("Last verifier run:")
    if kind == "file_grep":
        resolved = trace.get("resolved_path")
        user_parts.append(f"  resolved_path: {resolved!r}")
        user_parts.append(f"  file_size: {trace.get('file_size_bytes', '?')} bytes")
        vreason = trace.get("verdict_reason") or ""
        if vreason:
            user_parts.append(f"  verdict_reason: {vreason}")
        attempts = trace.get("fallback_attempts") or []
        if attempts:
            user_parts.append("  fallback_attempts:")
            for at in attempts[:6]:
                strat = at.get("strategy", "?")
                ap = at.get("path", "?")
                ar = at.get("result", "?")
                user_parts.append(f"    - {strat:24s} {ap}  →  {ar}")
        chg = outcome.file_changed_since_baseline()
        user_parts.append(
            "  file_changed_since_task_start: "
            f"{'YES' if chg is True else 'NO' if chg is False else 'unknown'}"
        )
        for entry in (trace.get("per_pattern_matches") or []):
            ml = entry.get("matched_lines") or []
            user_parts.append(
                f"  pattern {entry.get('pattern','?')!r}: {len(ml)} match(es)"
            )
        head = trace.get("file_content_head") or ""
        if head:
            user_parts.append("")
            user_parts.append("Current file content (first 800 chars):")
            user_parts.append("```")
            user_parts.append(head[:800])
            user_parts.append("```")
    elif kind == "a11y_match":
        user_parts.append(f"  rows_scanned: {trace.get('rows_scanned', 0)}")
        user_parts.append(f"  matched_row: {trace.get('matched_row') or '(none)'}")
    elif kind == "url_match":
        user_parts.append(f"  url_extracted: {trace.get('url_extracted', '(none)')}")
    elif kind == "shell_command":
        user_parts.append(f"  exit_code: {trace.get('exit_code')}")
        if trace.get("matched_substring"):
            user_parts.append(
                f"  matched_substring: {trace.get('matched_substring')!r}"
            )
        vreason = trace.get("verdict_reason") or ""
        if vreason:
            user_parts.append(f"  verdict_reason: {vreason}")
        head = (trace.get("stdout_head") or "").strip()
        if head:
            user_parts.append("")
            user_parts.append("Command stdout (first 600 chars, stderr merged):")
            user_parts.append("```")
            user_parts.append(head[:600])
            user_parts.append("```")
    elif kind in ("calc_verify", "impress_verify"):
        # Structured verifiers report the failed check in stdout (e.g.
        # ``FAIL: check[0] 'font_props'``). Render the raw exec output so
        # the LLM can locate the offending kwarg.
        user_parts.append(f"  exit_code: {trace.get('exit_code')}")
        vreason = trace.get("verdict_reason") or ""
        if vreason:
            user_parts.append(f"  verdict_reason: {vreason}")
        head = (trace.get("stdout_head") or "").strip()
        if head:
            user_parts.append("")
            user_parts.append("Verifier stdout (first 600 chars):")
            user_parts.append("```")
            user_parts.append(head[:600])
            user_parts.append("```")
    user_parts.append(
        f"\nThis outcome has been the cause of "
        f"{outcome.get_done_rejection_count()} DONE rejection(s).\n"
        "Diagnose the root cause."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _DIAGNOSE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
        ],
        "max_tokens": 250,
        "temperature": 0.0,
        "top_p": 0.9,
        # Disable Qwen3.5's <think> block (as init_ledger): short
        # structured output, thinking just burns tokens.
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    out = call_llm(payload, model)
    if not out:
        return {"text": "", "root_cause": ""}
    out = out.strip()
    for prefix in ("```\n", "Diagnosis:", "DIAGNOSIS:"):
        if out.startswith(prefix):
            out = out[len(prefix):].lstrip()
    if out.endswith("```"):
        out = out[:-3].rstrip()
    # Parse + strip the [ROOT_CAUSE: ...] tag; take the LAST occurrence
    # in case the model echoes it earlier in prose.
    matches = list(_ROOT_CAUSE_TAG_RE.finditer(out))
    cause = ""
    if matches:
        cause = matches[-1].group(1).lower()
    text = _ROOT_CAUSE_TAG_RE.sub("", out).strip()
    return {"text": text, "root_cause": cause}


def _derive_diagnosis_hint(
    verify: Any, trace: Dict[str, Any],
    outcome: Any,
) -> str:
    """Heuristic hint from verifier kind + pattern hit/miss + file-change
    status. Pure function (cheap, testable). A1: file-change reads off the
    Outcome."""
    kind = getattr(verify, "kind", "?")
    if kind == "file_grep":
        chg = outcome.file_changed_since_baseline()
        ppm = trace.get("per_pattern_matches", []) or []
        n_total = len(ppm)
        n_hit = sum(1 for e in ppm if (e.get("matched_lines") or []))
        if not n_hit and chg is False:
            return (
                "All patterns missed AND the target file is UNCHANGED "
                "since task start. Most likely your last edit was not "
                "actually committed to disk — check whether the "
                "modified-tab dot is still showing, or whether a save "
                "dialog is still open."
            )
        if not n_hit and chg is True:
            return (
                "All patterns missed BUT the file HAS been modified. "
                "The actor wrote SOMETHING; the patterns may not match "
                "the actual file structure (wrong syntax, wrong key "
                "names, wrong glob form, missing quotes, etc.). "
                "Reread the verifier patterns above and compare to a "
                "known sample of the file's intended final form."
            )
        if 0 < n_hit < n_total:
            return (
                f"Partial match ({n_hit}/{n_total} patterns hit). "
                "The actor wrote part of the required content but "
                "missed at least one field. Identify which pattern is "
                "still missing and emit one targeted action to add it."
            )
        if n_hit == n_total:
            return (
                "All patterns matched (this should not have rejected — "
                "may be a transient race; next turn should accept)."
            )
        return "(no patterns recorded in trace — cannot infer)"
    if kind == "a11y_match":
        if not trace.get("matched_row"):
            return (
                "No a11y row satisfied all of (tag, text_contains, "
                "state_contains). The required UI element may be "
                "off-screen, occluded, or in a different state than "
                "expected. Scroll / activate / re-focus before retrying."
            )
        return (
            "An a11y row matched, but DONE was still rejected — likely "
            "the visual_must_hold / visual_must_not_hold gates failed "
            "(LLM judged the screenshot and saw the action did not "
            "actually commit, e.g. a modal dialog is still visible)."
        )
    if kind == "url_match":
        if not trace.get("url_extracted"):
            return (
                "No active-tab URL extracted from the a11y. Either "
                "the browser tab is unfocused or the URL bar is hidden."
            )
        return (
            "URL extracted but did not match the pattern. Check the "
            "url_pattern above against the extracted URL."
        )
    return "(no auto-diagnosis available for this verifier kind)"


def render_outcome_view(
    required_outcomes: List[Any],   # List[Outcome] from ledger.py — duck-typed
    failed_paths: Optional[List[Any]] = None,
    completed_strategies: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Render derived outcome state with a single-focus marker.

    The first pending outcome with all deps verified gets ``▶ FOCUS ◀``;
    downstream pending outcomes show ``[blocked]`` so the planner attacks
    one milestone at a time instead of skipping ahead.

    ``completed_strategies``: optional dicts from the curator's
    strategy_outcome_satisfied verdict ({"strategy", "satisfied_outcomes"});
    rendered so the planner won't re-propose an already-produced strategy.
    """
    focus_id = _compute_focus_outcome_id(required_outcomes)
    by_id = {getattr(o, "id", None): o for o in required_outcomes}
    lines = ["[Required Outcomes — state derived from event timeline]"]
    n_verified = 0
    n_reverted = 0
    n_pending_blocked = 0
    for o in required_outcomes:
        st = getattr(o, "state", OUTCOME_PENDING)
        deps = list(getattr(o, "depends_on", []) or [])
        deps_str = ""
        if deps:
            # Glyph each dep's state so the planner sees what's blocking.
            dep_glyphs = []
            for d in deps:
                d_o = by_id.get(d)
                d_st = getattr(d_o, "state", OUTCOME_PENDING) if d_o is not None else OUTCOME_PENDING
                glyph = "✓" if d_st == OUTCOME_VERIFIED else (
                    "✗" if d_st == OUTCOME_REVERTED else "·")
                dep_glyphs.append(f"{glyph}{d}")
            deps_str = f"  [needs: {', '.join(dep_glyphs)}]"

        # Multi-app: tag each outcome's app so the planner sees per-app
        # structure. Empty for single-app tasks (o.app is None).
        _app = getattr(o, "app", None)
        if _app:
            deps_str = f"  [app={_app}]" + deps_str

        if st == OUTCOME_VERIFIED:
            since = getattr(o, "last_satisfy_step", None)
            since_str = since if since is not None else "?"
            lines.append(
                f"  [verified] {o.id:<28} — since step {since_str}, evidence intact{deps_str}"
            )
            n_verified += 1
        elif st == OUTCOME_REVERTED:
            sat = getattr(o, "last_satisfy_step", None)
            inv = getattr(o, "last_invalidate_step", None)
            sat_str = sat if sat is not None else "?"
            inv_str = inv if inv is not None else "?"
            lines.append(
                f"  [REVERTED] {o.id:<28} — verified step {sat_str}, "
                f"INVALIDATED step {inv_str} ← MUST RE-DO{deps_str}"
            )
            n_reverted += 1
        else:
            # Pending: FOCUS if active, else blocked when deps unmet.
            if o.id == focus_id:
                lines.append(
                    f"  ▶ FOCUS ◀  {o.id:<28} — hint: "
                    f"{(o.evidence_hint or '')[:120]}{deps_str}"
                )
                # Surface the verify FAIL reason every turn, not just on
                # force_replan, so the planner sees the concrete failure
                # mode on regular turns. A3: prefer the Evidence trace,
                # fall back to the legacy field.
                trace: Dict[str, Any] = {}
                ev_list = getattr(o, "evidence", None) or []
                if ev_list:
                    trace = dict(ev_list[-1].verifier_trace or {})
                if not trace:
                    trace = dict(getattr(o, "last_verifier_trace", {}) or {})
                stdout_head = (trace.get("stdout_head") or "").strip()
                if stdout_head:
                    first_line = stdout_head.splitlines()[0][:160]
                    lines.append(f"      last verify stdout: {first_line}")
                diag = (getattr(o, "last_diagnosis", None) or "").strip()
                if diag:
                    diag_short = diag.replace("\n", " ")[:200]
                    lines.append(f"      diagnosis: {diag_short}")
            else:
                # Pending, after focus in source order. [blocked] when
                # deps actually gate it, [pending] otherwise.
                deps_blocked = any(
                    (by_id.get(d) is None
                     or getattr(by_id[d], "state", OUTCOME_PENDING) != OUTCOME_VERIFIED)
                    for d in deps
                )
                tag = "[blocked]" if deps_blocked else "[pending] "
                if deps_blocked:
                    n_pending_blocked += 1
                lines.append(
                    f"  {tag} {o.id:<28} — hint: "
                    f"{(o.evidence_hint or '')[:120]}{deps_str}"
                )

    n_total = len(required_outcomes)
    n_failed = len(failed_paths or [])
    n_pending = n_total - n_verified - n_reverted
    lines.append("")
    summary = (
        f"Summary: {n_verified}/{n_total} verified; "
        f"{n_reverted} REVERTED; {n_pending} pending"
    )
    if n_pending_blocked:
        summary += f" ({n_pending_blocked} blocked by unmet deps)"
    n_strategy = sum(1 for fp in (failed_paths or [])
                     if getattr(fp, "level", "action") == "strategy")
    n_action = (n_failed - n_strategy)
    if n_failed:
        summary += (
            f"; dead-ends: {n_strategy} strategy, {n_action} tactical-step."
        )
    else:
        summary += "; 0 dead-ends recorded."
    lines.append(summary)
    if focus_id:
        lines.append(
            f"➜ Active focus: '{focus_id}'. Plan ONLY for this one milestone "
            f"this turn; later pending outcomes are blocked until it verifies."
        )
    # Completed strategies, from the curator's strategy_outcome_satisfied
    # verdicts. Without this the planner re-proposed strategies whose
    # outcome was already [verified] (e.g. "create Sheet2 via Sheet menu"
    # after sheet2_created): nothing retired them, the old plan stayed in
    # context, and the stale strategy got reused verbatim.
    if completed_strategies:
        lines.append("")
        lines.append(
            f"✓ COMPLETED STRATEGIES ({len(completed_strategies)}) — "
            "high-level approaches whose primary outcomes are now "
            "[verified]. DO NOT re-propose these in a new <strategy>; "
            "they are done. Pick a fresh strategy targeting the "
            "▶ FOCUS ◀ outcome's evidence."
        )
        for i, cs in enumerate(completed_strategies, 1):
            text = (cs.get("strategy") or "")[:200]
            outs = cs.get("satisfied_outcomes") or []
            outs_tag = (", ".join(outs)) if outs else "?"
            lines.append(f"  ✓ #{i}. \"{text}\"  → produced: {outs_tag}")
    if failed_paths:
        # A4: filter failures by target_outcome_id so replanning toward
        # focus_id only sees dead-ends against THIS outcome. target=None
        # failures stay visible everywhere (cross-cutting patterns the
        # curator couldn't pin to one outcome).
        def _is_relevant(fp) -> bool:
            tid = getattr(fp, "target_outcome_id", None)
            if tid is None:
                return True              # cross-cutting
            if focus_id is None:
                return True              # no focus
            return tid == focus_id

        relevant_fps = [fp for fp in failed_paths if _is_relevant(fp)]

        # Two-layer split: strategy vs action.
        strategy_fps = [fp for fp in relevant_fps
                        if getattr(fp, "level", "action") == "strategy"]
        action_fps = [fp for fp in relevant_fps
                      if getattr(fp, "level", "action") == "action"]

        def _render_fp(i, fp, kind_label):
            path = getattr(fp, "path", "")
            why = getattr(fp, "why", "")
            n_obs = getattr(fp, "observation_count", 1)
            actions = getattr(fp, "action_sequence", []) or []
            obs_tag = f" ×{n_obs}" if n_obs > 1 else ""
            tgt = getattr(fp, "target_outcome_id", None)
            tgt_tag = f"  [→{tgt}]" if tgt else "  [cross-cutting]"
            block = [f"  ⚠ #{i}{obs_tag}. {kind_label}:  \"{path[:200]}\"{tgt_tag}"]
            if actions:
                shown = actions[:5]
                actions_str = "; ".join(a[:60] for a in shown)
                more_tag = (f" (+{len(actions)-5} more)"
                            if len(actions) > 5 else "")
                block.append(
                    f"           Tried:     {actions_str[:240]}{more_tag}"
                )
            block.append(f"           Outcome:   {why[:240]}")
            return block

        # Headline reflects the per-outcome scope when focus is set.
        focus_label = (f" for outcome `{focus_id}`" if focus_id else "")

        if strategy_fps:
            lines.append("")
            lines.append(
                f"⚠  DEAD-END STRATEGIES{focus_label} "
                f"({len(strategy_fps)}) — "
                f"DO NOT PROPOSE THESE HIGH-LEVEL APPROACHES AGAIN:"
            )
            lines.append(
                "   These are high-level approaches that the dead-end "
                "curator has marked as known-failing for this task. The "
                "new <strategy> in your next <plan> MUST be structurally "
                "different from every entry below (different entry point, "
                "different interaction pattern, different page/route)."
            )
            for i, fp in enumerate(strategy_fps[:6], 1):
                lines.extend(_render_fp(i, fp, "Strategy"))

        if action_fps:
            lines.append("")
            lines.append(
                f"⚠  PROBLEMATIC ACTION PATTERNS{focus_label} "
                f"({len(action_fps)}) — "
                f"DO NOT REPEAT THESE TACTICAL MISTAKES:"
            )
            lines.append(
                "   These are SPECIFIC actor-execution patterns that the "
                "curator flagged as wasted effort (e.g. \"repeatedly "
                "scrolled without clicking\", \"clicked the same coord "
                "3+ times with no UI change\", \"typed a wrong URL\"). "
                "Within your CURRENT <strategy>, your next step must use "
                "a different tactic — alternate element, keyboard "
                "shortcut, alt coords, or scroll-into-view first — "
                "rather than repeat any pattern below."
            )
            for i, fp in enumerate(action_fps[:6], 1):
                lines.extend(_render_fp(i, fp, "Pattern"))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Serialization (JSON-friendly via asdict/from_dict, with forgiving loader)
# --------------------------------------------------------------------------- #


def timeline_to_dict(timeline: List[TimelineEvent]) -> List[Dict[str, Any]]:
    return [asdict(ev) for ev in timeline]


def timeline_from_dict(data: List[Dict[str, Any]]) -> List[TimelineEvent]:
    out: List[TimelineEvent] = []
    for ev_d in data or []:
        steps = []
        for s_d in ev_d.get("steps") or []:
            kns = [KeyNode(**kn) for kn in s_d.get("key_nodes") or []]
            s_dd = {**s_d, "key_nodes": kns}
            steps.append(StepRecord(**s_dd))
        out.append(TimelineEvent(
            event_idx=ev_d.get("event_idx", len(out)),
            subgoal=ev_d.get("subgoal", ""),
            started_at_step=ev_d.get("started_at_step", 0),
            ended_at_step=ev_d.get("ended_at_step"),
            steps=steps,
            outcome=ev_d.get("outcome", EV_ONGOING),
        ))
    return out


def cache_to_dict(cache: OutcomeStateCache) -> Dict[str, Any]:
    return asdict(cache)


def cache_from_dict(data: Dict[str, Any]) -> OutcomeStateCache:
    return OutcomeStateCache(
        states=dict(data.get("states") or {}),
        last_satisfy_step=dict(data.get("last_satisfy_step") or {}),
        last_invalidate_step=dict(data.get("last_invalidate_step") or {}),
        revert_count=dict(data.get("revert_count") or {}),
    )

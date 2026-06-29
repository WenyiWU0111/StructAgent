"""Ledger summarizer: one LLM call per subgoal boundary.

Fires when the agent finishes (or abandons) a subgoal. Consumes the
*whole sequence* of screenshots + actions accumulated during that
subgoal unit, together with the previous ledger, and emits a JSON patch:

  { "done_add":   [{"outcome_id": ..., "evidence": ...}, ...],
    "failed_add": [{"path": ..., "why": ...}, ...] }

The patch is applied to the ledger via ``ledger.add_done`` /
``ledger.add_failed_path``; nothing else mutates the ledger. The reason
this is a separate LLM call from the planner is exactly to prevent the
observed "narrative drift" phenomenon: observed facts produced here are
frozen before the planner's next call reads them, so the planner cannot
silently replace an observed value with its desired target inside its
own response body.

Model routing note: when the caller passes a ``vllm_qwen35-vl``-style
alias and uses the planner's ``self.call_llm``, the underlying
``_call_llm_vllm`` already disables qwen3.5 thinking mode via
``chat_template_kwargs={"enable_thinking": False}``. Callers using a
different LLM path should make sure thinking is disabled themselves, or
the summarizer may burn all tokens on reasoning and return empty
``content``.
"""
from __future__ import annotations
import base64
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from mm_agents.structagent.ledger.core.ledger import ProgressLedger

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Observable delta — hard-tier signal over the firing window
# --------------------------------------------------------------------------- #
# These are computed deterministically from env-reported state (URL + a11y
# tree) before the summarizer LLM runs, and injected into its prompt as
# objective facts. The LLM reasons FROM the delta, not from freely reading
# screenshots. This is the "soft tier + hard tier" architecture from the
# proprioception_layer paper frame: observation and belief disentangled.

@dataclass
class WindowDelta:
    """Machine-computed delta between window-start and window-end state.
    Feeds into the summarizer prompt as objective ground truth.
    """
    url_before: Optional[str]
    url_after: Optional[str]
    url_changed: bool
    nodes_added: List[Tuple[str, str, str]]            # (tag, text, state)
    nodes_removed: List[Tuple[str, str, str]]          # (tag, text, state)
    state_flips: List[Tuple[str, str, str, str]]       # (tag, text, state_before, state_after)

    @property
    def is_empty(self) -> bool:
        return (not self.url_changed
                and not self.nodes_added
                and not self.nodes_removed
                and not self.state_flips)

    def to_prompt_block(self, max_items: int = 12) -> str:
        """Compact, LLM-readable rendering. Caps each list at ``max_items``
        to keep the prompt bounded."""
        parts = ["[Observable Delta — computed from env, not from your interpretation]"]
        if self.url_before is None and self.url_after is None:
            parts.append("  URL:            (unknown / not tracked)")
        elif self.url_changed:
            parts.append(f"  URL:            {self.url_before or '?'} → {self.url_after or '?'}  (CHANGED)")
        else:
            parts.append(f"  URL:            {self.url_after or '?'}  (unchanged during this window)")

        def _fmt_node(t: Tuple[str, str, str]) -> str:
            tag, txt, st = t
            txt_s = (txt or "")[:60]
            return f"    {tag} '{txt_s}' [{st}]"

        if self.nodes_added:
            parts.append(f"  New UI elements ({len(self.nodes_added)}):")
            for n in self.nodes_added[:max_items]:
                parts.append(_fmt_node(n))
            if len(self.nodes_added) > max_items:
                parts.append(f"    … (+{len(self.nodes_added) - max_items} more)")
        else:
            parts.append("  New UI elements: NONE")

        if self.nodes_removed:
            parts.append(f"  Removed UI elements ({len(self.nodes_removed)}):")
            for n in self.nodes_removed[:max_items]:
                parts.append(_fmt_node(n))
            if len(self.nodes_removed) > max_items:
                parts.append(f"    … (+{len(self.nodes_removed) - max_items} more)")
        else:
            parts.append("  Removed UI elements: NONE")

        if self.state_flips:
            parts.append(f"  State flips on existing elements ({len(self.state_flips)}):")
            for (tag, txt, sb, sa) in self.state_flips[:max_items]:
                parts.append(f"    {tag} '{(txt or '')[:60]}': [{sb}] → [{sa}]")
            if len(self.state_flips) > max_items:
                parts.append(f"    … (+{len(self.state_flips) - max_items} more)")
        else:
            parts.append("  State flips: NONE")

        if self.is_empty:
            parts.append("  ⚠ Window is completely empty — NO URL change, NO UI change, NO state flips. "
                         "Whatever actions were executed produced no observable effect.")
        return "\n".join(parts)


# Tags considered "interactive" for the purposes of diff filtering.
# Tag names come from AT-SPI / pyatspi (Ubuntu linearized a11y output);
# verified by inspecting trajectory.html a11y dumps. We also include a
# few synonyms from other backends (e.g. ATK vs IAccessible2) so the
# same code works across platforms without reconfiguration.
_INTERACTIVE_TAGS: Set[str] = {
    # main set (AT-SPI on Ubuntu, which is what our linearize emits)
    "push-button", "toggle-button", "check-box", "radio-button",
    "link", "menu-item", "menu", "menu-bar", "tab", "tab-list",
    "combo-box", "list-box", "list-item", "entry", "text",
    "slider", "spin-button", "tree-item", "document-web",
    # synonyms from other a11y backends
    "button", "switch", "checkbox", "radiobutton",
}


def _parse_a11y_lines(a11y_text: Optional[str]) -> Dict[Tuple[str, str], str]:
    """Parse a tab-separated a11y dump into ``{(tag, text): state}``.

    Keeps only rows whose ``tag`` is in ``_INTERACTIVE_TAGS`` so the diff
    reflects clickable/inputable UI changes, not layout noise. Anonymous
    rows (empty text) are kept but keyed by tag alone so multiple empty
    buttons don't collapse to one.
    """
    out: Dict[Tuple[str, str], str] = {}
    if not a11y_text:
        return out
    for ln in a11y_text.splitlines():
        parts = ln.split("\t")
        if len(parts) < 2:
            continue
        tag = parts[0].strip().lower()
        txt = parts[1].strip() if len(parts) > 1 else ""
        st  = parts[2].strip() if len(parts) > 2 else ""
        if tag not in _INTERACTIVE_TAGS:
            continue
        key = (tag, txt)
        # Collision handling: if tag+text already present with a different
        # state, keep the last one (best-effort; a11y can have duplicates).
        out[key] = st
    return out


def compute_window_delta(
    *,
    url_before: Optional[str],
    url_after: Optional[str],
    a11y_before: Optional[str],
    a11y_after: Optional[str],
) -> WindowDelta:
    """Compute a deterministic delta between two observation snapshots.
    Pure function, no LLM.

    Treats (tag, text) as element identity; ``state`` is the value for
    flip detection. Missing before/after a11y yields an empty diff
    (not a false-signal).
    """
    url_changed = bool(url_before) and bool(url_after) and url_before != url_after
    # If one side is None (a11y unavailable), we can't honestly diff —
    # leave node-lists empty so the LLM is not misled.
    if not a11y_before or not a11y_after:
        return WindowDelta(
            url_before=url_before, url_after=url_after,
            url_changed=url_changed,
            nodes_added=[], nodes_removed=[], state_flips=[],
        )
    before = _parse_a11y_lines(a11y_before)
    after  = _parse_a11y_lines(a11y_after)
    before_keys = set(before)
    after_keys  = set(after)
    added_keys   = after_keys  - before_keys
    removed_keys = before_keys - after_keys
    common_keys  = before_keys & after_keys
    nodes_added   = [(t, x, after[(t, x)])  for (t, x) in sorted(added_keys)]
    nodes_removed = [(t, x, before[(t, x)]) for (t, x) in sorted(removed_keys)]
    state_flips: List[Tuple[str, str, str, str]] = []
    for (t, x) in sorted(common_keys):
        sb, sa = before[(t, x)], after[(t, x)]
        if sb != sa:
            state_flips.append((t, x, sb, sa))
    return WindowDelta(
        url_before=url_before, url_after=url_after,
        url_changed=url_changed,
        nodes_added=nodes_added,
        nodes_removed=nodes_removed,
        state_flips=state_flips,
    )


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

_SYSTEM_PROMPT = """You are the strategy-failure summarizer for a GUI
automation agent. Every N actor turns (and at task-end / boundary)
you are called to scan the recent window of actions and decide
whether a STRATEGIC dead-end was just visited that the planner should
remember and avoid re-entering.

(Per-turn outcome state-tracking is handled by a separate KeyNode
detector. You do NOT decide whether outcomes are satisfied / verified
— that is not your job in v2. Focus only on `failed_paths`.)

You are given:
  - the CURRENT LEDGER (verified outcomes, pending outcomes, known
    `failed_paths`) — for context, and to deduplicate
  - an OBSERVABLE DELTA block (URL + a11y diff over the window)
  - the screenshots and actor actions that ran during the window
  - the latest-frame a11y tree + active URL
  - (when available) a `[Timeline signals]` block summarizing the
    recent event timeline: how many outcome KeyNodes fired, the
    distribution of event outcomes (committed / abandoned), and any
    SUBGOAL that has been attempted ≥2 times in a row without
    progress. A signal like "0 outcome KeyNodes despite N steps" or
    "recurring subgoal × 3" is a strong dead-end indicator.

Decide failed_add based on the ACTION-DELTA relationship over the
window:
  Record a failed_path when a SEQUENCE of 2+ actor actions in this
  window collectively failed to produce delta relevant to the
  subgoal's intent — i.e. the agent went down a route and the route
  led nowhere useful.

  "path" MUST be a short sequence of the key actions (2+ steps)
  joined by arrows → plus the terminal state reached. Examples:
    - path: "Help menu → FAQ sub-link → scrolled full page →
             used Ctrl+F 'export' — 0 results"
      why:  "FAQ page does not contain the export option;
             wrong route"
    - path: "Typed /products/lamp in address bar → 404"
      why:  "catalog detail pages require an internal numeric id;
             must reach via site search"

  DE-DUPE: if your candidate failed_path is a re-wording or
  continuation of an existing entry in `failed_paths`, leave
  failed_add empty for that item.

  COUNTER-examples (do NOT record):
    - single-action misclick ("clicked at (733, 271)")   ✗
    - single hop of an unfinished multi-step plan        ✗
    - ongoing exploration where outcomes were verified — that route
      worked, not failed                                 ✗

OUTPUT FORMAT — strictly two blocks in this order, no other text:

<reasoning>
Brief one-paragraph delta transcription, then ONE bullet per failed-
path candidate:

  Failed-path candidate:
      action_arc:  "<2+ step sequence with arrows>"
      delta_evidence: "<delta showed empty / unrelated change despite
                       these actions>"
      conclusion:  RECORD | SKIP (skip if duplicate of existing failed
                                   entries, or single misclick)
</reasoning>

<patch>
{
  "failed_add": [{"path": "<arc>", "why": "<why dead-end>"}, ...]
}
</patch>

Be conservative. False positives in failed_add block the agent from
retrying a viable path. Leave failed_add empty if the window's delta
does not clearly support a strategic-failure conclusion — that is the
correct answer when nothing strategically meaningful happened.
"""


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

def update_ledger(
    ledger: ProgressLedger,
    *,
    subgoal_screenshots: List[bytes],
    subgoal_actions: List[str],
    subgoal_plan_step: str,
    boundary_type: str,                  # "periodic" | "done"
    latest_a11y: Optional[str],
    latest_url: Optional[str],
    step_idx: int,
    call_llm: Callable,
    model: Optional[str] = None,
    window_delta: Optional[WindowDelta] = None,
    timeline_window: Optional[List[Any]] = None,   # List[TimelineEvent]
) -> ProgressLedger:
    """Apply one summarizer pass to the ledger. Returns the same ledger
    (mutated in-place for convenience; also returned so callers can chain).

    The caller is expected to fire this ONLY at a subgoal boundary. See
    the plan for boundary definitions. If the segment is empty (no
    screenshots and no actions), this call is a no-op.
    """
    if not subgoal_screenshots and not subgoal_actions:
        return ledger

    # Nothing to do if we have no outcomes defined — the ledger is just
    # initial_context in that degenerate case.
    if not ledger.required_outcomes:
        return ledger

    timeline_signals = _summarize_timeline_signals(timeline_window) if timeline_window else None

    messages = _build_messages(
        ledger=ledger,
        subgoal_screenshots=subgoal_screenshots,
        subgoal_actions=subgoal_actions,
        subgoal_plan_step=subgoal_plan_step,
        boundary_type=boundary_type,
        latest_a11y=latest_a11y,
        latest_url=latest_url,
        window_delta=window_delta,
        timeline_signals=timeline_signals,
    )

    raw: Optional[str] = None
    try:
        raw = call_llm(
            {"model": model, "messages": messages,
             "max_tokens": 800, "temperature": 0.0, "top_p": 0.9},
            model,
        )
    except Exception as e:
        logger.warning("[LedgerSum] LLM call failed: %s", e)

    patch = _parse_patch_json(raw) if raw else None
    if patch is None:
        return ledger

    n_done, n_failed = _apply_patch(
        ledger,
        patch=patch,
        step_idx=step_idx,
        subgoal_text=subgoal_plan_step,
    )
    if n_done or n_failed:
        logger.info(
            "[LedgerSum] boundary=%s step=%d applied: +%d done, +%d failed",
            boundary_type, step_idx, n_done, n_failed,
        )
    return ledger


# --------------------------------------------------------------------------- #
# Message construction
# --------------------------------------------------------------------------- #

_MAX_IMAGES = 6  # cap to keep the prompt bounded
_MAX_ACTIONS = 12


def _summarize_timeline_signals(timeline_window: List[Any]) -> str:
    """Derive a compact ``[Timeline signals]`` block from the recent
    TimelineEvent list. Pure-Python aggregation — no LLM. Intended as
    a structured input to the failed-path summarizer.

    Provides counts that strongly indicate strategic dead-ends:
      • how many actor steps in this window
      • how many outcome_satisfied / outcome_invalidated KeyNodes
      • subgoal recurrence (same text across multiple events with
        outcome=abandoned_replan → likely loop)
      • event-outcome distribution (committed / abandoned / ongoing)
    """
    if not timeline_window:
        return ""
    # Avoid hard import dependency on timeline at module level (tests
    # may stub TimelineEvent); use duck-typed access.
    from collections import Counter

    n_events = len(timeline_window)
    n_steps = sum(len(getattr(ev, "steps", []) or []) for ev in timeline_window)
    outcome_dist: Counter = Counter(
        getattr(ev, "outcome", "ongoing") for ev in timeline_window
    )

    n_satisfied = 0
    n_invalidated = 0
    other_kn: Counter = Counter()
    for ev in timeline_window:
        for s in getattr(ev, "steps", []) or []:
            for kn in getattr(s, "key_nodes", []) or []:
                kind = getattr(kn, "kind", "")
                if kind == "outcome_satisfied":
                    n_satisfied += 1
                elif kind == "outcome_invalidated":
                    n_invalidated += 1
                elif kind:
                    other_kn[kind] += 1

    subgoal_counts: Counter = Counter(
        (getattr(ev, "subgoal", "") or "").strip()[:200]
        for ev in timeline_window if getattr(ev, "subgoal", None)
    )
    recurring = [
        (sg, c) for sg, c in subgoal_counts.most_common(6)
        if c >= 2
    ]

    lines: List[str] = ["[Timeline signals — last %d event(s), %d step(s)]" % (n_events, n_steps)]
    lines.append(
        f"  outcome KeyNodes: +{n_satisfied} satisfied, -{n_invalidated} invalidated"
    )
    if other_kn:
        ks = ", ".join(f"{k}={v}" for k, v in other_kn.most_common(4))
        lines.append(f"  other key_nodes: {ks}")
    if outcome_dist:
        ks = ", ".join(f"{k}={v}" for k, v in outcome_dist.most_common(5))
        lines.append(f"  event outcome counts: {ks}")
    if recurring:
        lines.append("  recurring subgoals (≥2 events with same text):")
        for sg, c in recurring:
            lines.append(f"    × {c}: {sg[:120]}")
    if n_steps >= 3 and n_satisfied == 0 and n_invalidated == 0:
        lines.append(
            "  ⚠ no outcome KeyNodes despite ≥3 steps — strong signal "
            "this window is a strategic dead-end candidate"
        )
    return "\n".join(lines)


def _build_messages(
    *,
    ledger: ProgressLedger,
    subgoal_screenshots: List[bytes],
    subgoal_actions: List[str],
    subgoal_plan_step: str,
    boundary_type: str,
    latest_a11y: Optional[str],
    latest_url: Optional[str],
    window_delta: Optional[WindowDelta] = None,
    timeline_signals: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []

    imgs = _select_images(subgoal_screenshots, _MAX_IMAGES)
    for b in imgs:
        try:
            b64 = base64.b64encode(b).decode("ascii")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })
        except Exception as e:
            logger.warning("[LedgerSum] failed to encode screenshot: %s", e)

    # Ledger snapshot (pre-update) so summarizer sees what's already done
    text_parts: List[str] = [
        "Current ledger (before this update):",
        ledger.to_prompt_block(),
        "",
        "Subgoal just finished:",
        f"  plan step text:  {subgoal_plan_step or '(none)'}",
        f"  boundary reason: {boundary_type}",
        f"  screenshots:     {len(subgoal_screenshots)} total "
        f"(showing {len(imgs)} attached, oldest to latest)",
    ]

    actions_shown = subgoal_actions[-_MAX_ACTIONS:]
    text_parts.append("  actions executed:")
    if actions_shown:
        for i, a in enumerate(actions_shown, 1):
            text_parts.append(f"    {i}. {(a or '').strip()[:200]}")
    else:
        text_parts.append("    (none recorded)")

    # Hard-tier signal: machine-computed delta over this window. This is
    # the authoritative "what changed" fact sheet the LLM must reason
    # from in Step 1 of the procedural reasoning below.
    if window_delta is not None:
        text_parts.append("")
        text_parts.append(window_delta.to_prompt_block())

    # Structured timeline signals — derived from the agent's event
    # timeline + KeyNode detector. Tells the failed-path summarizer
    # things like "0 outcomes satisfied across N actions" or "subgoal
    # X attempted in 3 separate events all REPLAN'd" — strong evidence
    # of strategic dead-end without re-interpreting screenshots.
    if timeline_signals:
        text_parts.append("")
        text_parts.append(timeline_signals)

    if latest_a11y:
        trimmed = latest_a11y[:6000]
        text_parts.append("")
        text_parts.append("Accessibility tree (latest screen, aux):")
        text_parts.append(trimmed)
    if latest_url:
        text_parts.append("")
        text_parts.append(f"Active tab URL (latest, aux): {latest_url}")

    text_parts.append("")
    text_parts.append("Return the JSON patch object and nothing else.")

    user_content.append({"type": "text", "text": "\n".join(text_parts)})

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _select_images(imgs: List[bytes], cap: int) -> List[bytes]:
    """Pick up to ``cap`` images preserving chronological order. Prefer the
    first frame (before the attempt) and the last few (after). If there
    are more than ``cap`` frames, keep 1 early + last ``cap-1``."""
    if not imgs:
        return []
    if len(imgs) <= cap:
        return list(imgs)
    return [imgs[0]] + imgs[-(cap - 1):]


# --------------------------------------------------------------------------- #
# Patch parsing / application
# --------------------------------------------------------------------------- #

def _parse_patch_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    # Prefer <patch>...</patch> body if present (Phase-3 reasoning-first
    # output format). Falls back to the whole string for legacy outputs.
    m = re.search(r"<patch>\s*([\s\S]*?)\s*</patch>", s, re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blob = s[start:i + 1]
                try:
                    return json.loads(blob)
                except Exception as e:
                    logger.warning("[LedgerSum] JSON parse error: %s", e)
                    return None
    return None


def _apply_patch(
    ledger: ProgressLedger,
    *,
    patch: Dict[str, Any],
    step_idx: int,
    subgoal_text: str,
) -> Tuple[int, int]:
    """Apply a v2 summarizer patch. Both ``done_add`` AND ``failed_add``
    are now IGNORED — outcome state is owned by the KeyNode detector,
    and dead-end paths are owned by the new write sources:

      - heuristic triggers: REPLAN cadence (rule #6), KeyNode
        invalidation causality (planned), same-coord stuck (planned)
      - planner-driven: ``<dead_end_review>`` with the ref:N protocol,
        merged via ``ledger.merge_planner_review`` after each planner
        call.

    The summarizer's free-form ``failed_add`` was producing low-signal,
    duplicate-prone entries that confused the planner; removing it
    leaves only deterministic + planner-vetted dead-ends in the ledger.

    Returns ``(0, 0)`` for caller signature compatibility."""
    if patch.get("done_add"):
        logger.debug(
            "[LedgerSum] ignoring %d done_add entries — v2 uses KeyNode detector for outcomes",
            len(patch["done_add"]),
        )
    if patch.get("failed_add"):
        logger.debug(
            "[LedgerSum] ignoring %d failed_add entries — dead-ends now come from "
            "REPLAN cadence + planner <dead_end_review>, not the summarizer",
            len(patch["failed_add"]),
        )
    return 0, 0

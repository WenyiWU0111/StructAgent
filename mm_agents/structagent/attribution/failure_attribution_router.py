"""Per-role intervention router — Phase A of the failure-attribution stack.

The diagnose call (`failure_attribution.diagnose_failure`) attributes a
stuck state to one of five roles (planner / actor / verifier / env /
unclear). This router consumes that attribution and decides what the
NEXT action of the agent loop should be — not always a force_replan.

Five intervention kinds (mapped from role + extra signals):

  planner_replan      — inject diagnosis block into next planner call
                        (current behaviour for v1 stuck_diagnosis).
                        Used for role=planner and the unclear-fallback.

  actor_redo          — keep the current subgoal, skip the planner,
                        and feed the actor PROSE-LEVEL guidance only
                        (root_cause summary + memory's canonical
                        approach + divergence). Crucially: NEVER pass
                        the diagnose-emitted concrete action hint
                        (e.g. click(start_box='(570,160)')) — the
                        actor will copy it verbatim, and the diagnose
                        model's grounding for those coords isn't
                        reliable. The actor has its own grounding
                        (decomposer inline grounding) and must
                        do the visual lookup itself. We give the actor
                        WHAT to achieve and WHY the previous tries
                        failed; HOW (which pixel) is the actor's job.

  verifier_override   — programmatically mark the focus outcome as
                        satisfied, citing the LLM's
                        verifier_override_evidence quotes. SAFETY:
                        requires (a) role=verifier, (b) confidence=high,
                        (c) ≥1 verifier_override_evidence quote,
                        (d) ≥N prior verifier-attributed rejects on
                        the same outcome (default 2). Otherwise falls
                        back to planner_replan.

  env_intervention    — non-LLM environment action (wait / focus
                        widget / run recovery script / retry last
                        action). Used for role=env. Falls back to
                        planner_replan when env_recovery_kind isn't
                        populated.

  fallback_planner    — any safety-gate failure routes here so the
                        agent stays on the existing well-understood
                        path.

The router is a PURE FUNCTION: it inspects the FailureAttribution + a
small subset of agent state (timeline for dedup, prior diagnoses count
for verifier safety gate) and returns the decision. Actual execution
of the decision lives in the agent loop's recovery dispatcher (see
``structagent.core.planner.planner`` and ``structagent.core.actor``).

Intentionally LIGHT on imports so the same module can be unit-tested
and replayed without standing up the full agent.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from mm_agents.structagent.attribution.failure_attribution import (
    FailureAttribution,
)

logger = logging.getLogger(__name__)


# ─── InterventionDecision dataclass ──────────────────────────────────


_VALID_INTERVENTION_KINDS = {
    "planner_replan",
    "actor_redo",
    "verifier_override",
    "env_intervention",
    "fallback_planner",
}


@dataclass
class InterventionDecision:
    kind: str                              # one of _VALID_INTERVENTION_KINDS
    params: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""                       # 1-sentence explanation
    fallback_from: Optional[str] = None    # set when safety gate redirected
                                           # us to fallback_planner

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "params": dict(self.params),
            "reason": self.reason,
            "fallback_from": self.fallback_from,
        }


# ─── Helpers ─────────────────────────────────────────────────────────


_COORD_RE = re.compile(
    r"start_box\s*=\s*'\(\s*(\d+)\s*,\s*(\d+)\s*\)'")
_SIGNATURE_RE = re.compile(r"([a-z_][a-z0-9_]*)\s*\(([^)]{0,200})\)")


def _hint_signature(hint: str) -> Optional[str]:
    """Extract a normalised signature from an arbitrary action snippet
    (hint OR timeline action_text). Handles "Action: " prefix.

    Examples:
      "click(start_box='(394,141)')"               → "click(394,141)"
      "Action: click(start_box='(394,141)')"       → "click(394,141)"
      "Thought:... Action: hotkey(key='ctrl+l')"   → "hotkey(ctrl+l)"
      "cli_run('pkill -f chrome')"                 → "cli_run(pkill_-f_chrome)"
    """
    if not hint:
        return None
    cm = _COORD_RE.search(hint)
    if cm:
        # Find function name nearest to the coord match (search backward
        # so we pick "click" not the start-of-string "Action").
        # Match the LAST function-call signature before the coords.
        prefix = hint[: cm.start()]
        sig_m = list(re.finditer(r"([a-z_][a-z0-9_]*)\s*\(", prefix))
        if sig_m:
            return f"{sig_m[-1].group(1)}({cm.group(1)},{cm.group(2)})"
        return f"_({cm.group(1)},{cm.group(2)})"
    # No coords — extract the last function-call signature in the text.
    matches = list(_SIGNATURE_RE.finditer(hint))
    if matches:
        m = matches[-1]
        args = re.sub(r"\s+", "_", m.group(2)).strip("_")
        return f"{m.group(1)}({args[:60]})"
    return hint.strip()[:120]


def _extract_recent_action_signatures(
    timeline: Optional[List[Any]],
    n_recent: int = 8,
) -> List[str]:
    """Pull the last N action signatures from timeline / rollup.

    The router is called from both production (timeline = list of step
    objects) and replay (timeline = list of dicts). We tolerate both
    shapes: accept any sequence of objects that either have
    ``action_text`` attribute or are dicts with an ``action`` key."""
    if not timeline:
        return []
    sigs: List[str] = []
    for step in timeline[-n_recent:]:
        txt = ""
        if hasattr(step, "action_text"):
            txt = str(getattr(step, "action_text", "") or "")
        elif isinstance(step, dict):
            txt = str(step.get("action") or step.get("action_text") or "")
        else:
            txt = str(step)
        sig = _hint_signature(txt)
        if sig:
            sigs.append(sig)
    return sigs


def _hint_is_repeat(hint: str, recent_sigs: List[str]) -> bool:
    """True when the proposed hint's signature exactly matches one of
    the recent action signatures — i.e. we'd be re-trying a tried-
    and-failed action."""
    h_sig = _hint_signature(hint)
    if not h_sig:
        return False
    return h_sig in recent_sigs


def _count_verifier_attributions(focus_outcome: Any,
                                  current_role: str) -> int:
    """How many prior failure_attribution calls on this focus outcome
    have also attributed to 'verifier'? Used as a safety gate so a
    single false-positive verifier vote cannot override.

    Accepts the cached attribution history we save on
    focus_outcome.failure_attribution_history (set by
    diagnose_failure when it appends each result)."""
    history = getattr(focus_outcome, "failure_attribution_history", None)
    if not history:
        return 0
    return sum(1 for h in history
               if (h or {}).get("attributed_to_role") == "verifier")


# ─── Router safety-gate constants ───────────────────────────────────


VERIFIER_OVERRIDE_MIN_PRIOR = 2     # need at least N prior verifier-
                                     # attributed rejects on this outcome
                                     # before we trust the override
# Dry-run control merged into the USE_FAILURE_ATTRIBUTION switch (read
# by agent.py): =1 → dry-run, =2 → live. The router itself is pure
# (no env reads); the executor receives ``dry_run`` from the caller.


# ─── The router ─────────────────────────────────────────────────────


def route_failure_attribution(
    diag: FailureAttribution,
    *,
    focus_outcome: Any,
    timeline: Optional[List[Any]] = None,
) -> InterventionDecision:
    """Pick an intervention from a single FailureAttribution.

    Safety gates redirect to ``fallback_planner`` whenever a role-
    specific route can't satisfy its preconditions; the agent never
    "loses" a force_replan opportunity because of an unreliable
    diagnosis.
    """
    role = (diag.attributed_to_role or "unclear").lower()
    conf = (diag.confidence or "low").lower()

    # ── planner / unclear → existing force_replan path ──
    if role in ("planner", "unclear"):
        return InterventionDecision(
            kind="planner_replan",
            params={"diagnosis": diag.to_dict()},
            reason=f"role={role} → existing replan path",
        )

    # ── actor → skip planner, re-call actor with PROSE guidance only ──
    # The actor has its own grounding (decomposer inline grounding). We
    # explicitly DO NOT pass the diagnose-emitted coord-level hint —
    # the actor would copy it verbatim, and the diagnose model's pixel
    # grounding isn't reliable. We give it WHAT + WHY; HOW is its job.
    if role == "actor":
        rcs = (diag.root_cause_summary or "").strip()
        canonical = (diag.memory_recall_canonical or "").strip()
        divergence = (diag.divergence_description or "").strip()
        # Gate: actor_redo only fires when the diagnosis has substantive
        # prose to hand the actor. An empty root_cause + no memory
        # canonical means we'd be calling the actor with nothing new —
        # better to fall back to the planner.
        if not rcs and not canonical:
            return InterventionDecision(
                kind="fallback_planner",
                params={"diagnosis": diag.to_dict()},
                reason=("role=actor but neither root_cause_summary nor "
                        "memory_recall_canonical have substantive content"),
                fallback_from="actor_redo",
            )
        if canonical == "<no memory>":
            canonical = ""
        return InterventionDecision(
            kind="actor_redo",
            params={
                "subgoal_id": getattr(focus_outcome, "id", None),
                # Prose-level guidance handed to the actor's prompt:
                "root_cause_summary":      rcs,
                "memory_canonical_approach": canonical,
                "divergence_description":  divergence,
                "missing_or_wrong":        list(diag.missing_or_wrong or []),
                # Full diagnosis retained for audit dumps — NOT for the
                # actor's prompt. Caller renders only the four prose
                # fields above into the actor input.
                "diagnosis": diag.to_dict(),
            },
            reason=(f"role=actor with prose guidance "
                    f"(rcs={len(rcs)} chars, "
                    f"canonical={len(canonical)} chars)"),
        )

    # ── verifier → spec override, gated on confidence + evidence + history ──
    if role == "verifier":
        prior_n = _count_verifier_attributions(focus_outcome, role)
        evidence = list(diag.verifier_override_evidence or [])
        if conf != "high":
            return InterventionDecision(
                kind="fallback_planner",
                params={"diagnosis": diag.to_dict()},
                reason=(f"role=verifier but confidence={conf} (need high "
                        f"to override)"),
                fallback_from="verifier_override",
            )
        if not evidence:
            return InterventionDecision(
                kind="fallback_planner",
                params={"diagnosis": diag.to_dict()},
                reason=("role=verifier but verifier_override_evidence is "
                        "empty (need ≥1 verbatim snapshot quote)"),
                fallback_from="verifier_override",
            )
        if prior_n < VERIFIER_OVERRIDE_MIN_PRIOR:
            return InterventionDecision(
                kind="fallback_planner",
                params={"diagnosis": diag.to_dict(),
                        "prior_verifier_attributions": prior_n,
                        "min_required": VERIFIER_OVERRIDE_MIN_PRIOR},
                reason=(f"role=verifier with {prior_n} prior verifier "
                        f"attribution(s); need ≥"
                        f"{VERIFIER_OVERRIDE_MIN_PRIOR} before overriding"),
                fallback_from="verifier_override",
            )
        return InterventionDecision(
            kind="verifier_override",
            params={
                "outcome_id": getattr(focus_outcome, "id", None),
                "evidence_quotes": evidence,
                "rationale": diag.root_cause_summary,
                # dry_run decision lives at the caller (agent.py reads
                # USE_FAILURE_ATTRIBUTION and passes dry_run to the
                # executor). The router stays pure.
            },
            reason=(f"role=verifier confidence=high with {len(evidence)} "
                    f"override evidence quote(s); "
                    f"{prior_n} prior verifier attributions on this "
                    f"outcome"),
        )

    # ── env → non-LLM env-side recovery (wait/focus/script/retry) ──
    if role == "env":
        kind = diag.env_recovery_kind
        if not kind:
            return InterventionDecision(
                kind="fallback_planner",
                params={"diagnosis": diag.to_dict()},
                reason="role=env but env_recovery_kind not populated",
                fallback_from="env_intervention",
            )
        return InterventionDecision(
            kind="env_intervention",
            params={
                "recovery_kind": kind,
                "recovery_params": dict(diag.env_recovery_params or {}),
                "diagnosis": diag.to_dict(),
            },
            reason=f"role=env recovery_kind={kind}",
        )

    # Catch-all (shouldn't reach here given enum validation upstream).
    return InterventionDecision(
        kind="fallback_planner",
        params={"diagnosis": diag.to_dict()},
        reason=f"unrecognised role={role!r}; defaulting to planner",
        fallback_from=None,
    )


# ─── Voting helper (for unstable model outputs) ─────────────────────


def vote_attributions(diags: List[FailureAttribution]
                      ) -> Tuple[FailureAttribution, Dict[str, Any]]:
    """Majority-vote across N FailureAttribution samples.

    Tie-break order: (1) confidence high > medium > low; (2) presence
    of verifier_override_evidence or env_recovery_kind (more committed
    diagnosis wins); (3) first sample.

    Returns (winning_diag, voting_meta) so the caller can log e.g.
    "voting: planner=2/3 actor=1/3 → planner won, low_agreement=False".
    """
    if not diags:
        raise ValueError("vote_attributions: no inputs")
    if len(diags) == 1:
        return diags[0], {"votes": 1, "agreement": 1.0,
                          "role_counts": {diags[0].attributed_to_role: 1}}

    from collections import Counter
    role_counts = Counter(d.attributed_to_role for d in diags)
    cat_counts = Counter(d.category for d in diags)
    top_role, top_role_n = role_counts.most_common(1)[0]
    top_cat, top_cat_n = cat_counts.most_common(1)[0]
    candidates = [d for d in diags if d.attributed_to_role == top_role]

    # Tie-break within candidates by confidence then commitment
    _conf_rank = {"high": 3, "medium": 2, "low": 1}

    def commit_score(d: FailureAttribution) -> int:
        s = _conf_rank.get(d.confidence, 0)
        if d.verifier_override_evidence:
            s += 1
        if d.env_recovery_kind:
            s += 1
        if d.evidence_quotes:
            s += 1
        return s

    winner = max(candidates, key=commit_score)
    meta = {
        "votes": len(diags),
        "agreement": top_role_n / len(diags),
        "role_counts": dict(role_counts),
        "category_counts": dict(cat_counts),
        "winner_role": top_role,
        "winner_category": top_cat,
        "low_agreement": top_role_n < len(diags) - 1,
    }
    return winner, meta

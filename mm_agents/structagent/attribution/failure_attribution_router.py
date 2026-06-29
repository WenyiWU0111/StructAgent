"""Per-role intervention router — Phase A of the failure-attribution stack.

``failure_attribution.diagnose_failure`` attributes a stuck state to one
of five roles (planner / actor / verifier / env / unclear). This router
maps that attribution to the loop's next action — not always a replan.

Five intervention kinds:

  planner_replan      — inject diagnosis into the next planner call.
                        Used for role=planner and the unclear fallback.

  actor_redo          — keep the subgoal, skip the planner, feed the
                        actor PROSE-ONLY guidance (root_cause + memory
                        canonical approach + divergence). Never pass the
                        diagnose-emitted coord hint: the actor copies it
                        verbatim and the diagnose model's grounding is
                        unreliable. The actor grounds itself — we give it
                        WHAT + WHY, not which pixel.

  verifier_override   — mark the focus outcome satisfied, citing the LLM's
                        verifier_override_evidence. Gated on role=verifier,
                        confidence=high, ≥1 evidence quote, and ≥N prior
                        verifier-attributed rejects on the outcome
                        (default 2); else planner_replan.

  env_intervention    — non-LLM env action (wait / focus / recovery
                        script / retry). role=env; falls back to
                        planner_replan when env_recovery_kind is empty.

  fallback_planner    — any safety-gate miss lands here so the agent stays
                        on the well-understood path.

PURE FUNCTION: inspects the FailureAttribution + a little agent state
(timeline for dedup, prior-diagnosis count for the verifier gate) and
returns a decision. Execution lives in the loop's recovery dispatcher.
Kept light on imports so it can be unit-tested / replayed standalone.
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
    """Normalise an action snippet (hint or timeline action_text) into a
    signature. Handles the "Action: " prefix.

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
        # Last function call before the coords (so we pick "click", not
        # the leading "Action").
        prefix = hint[: cm.start()]
        sig_m = list(re.finditer(r"([a-z_][a-z0-9_]*)\s*\(", prefix))
        if sig_m:
            return f"{sig_m[-1].group(1)}({cm.group(1)},{cm.group(2)})"
        return f"_({cm.group(1)},{cm.group(2)})"
    # No coords — last function call in the text.
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
    """Last N action signatures from the timeline.

    Tolerates both production (step objects with ``action_text``) and
    replay (dicts with an ``action`` key)."""
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
    """True when the hint's signature matches a recent one — i.e. we'd be
    re-trying a tried-and-failed action."""
    h_sig = _hint_signature(hint)
    if not h_sig:
        return False
    return h_sig in recent_sigs


def _count_verifier_attributions(focus_outcome: Any,
                                  current_role: str) -> int:
    """Count prior verifier-attributed diagnoses on this outcome (from the
    cached focus_outcome.failure_attribution_history). Safety gate so one
    false-positive verifier vote can't override."""
    history = getattr(focus_outcome, "failure_attribution_history", None)
    if not history:
        return 0
    return sum(1 for h in history
               if (h or {}).get("attributed_to_role") == "verifier")


# ─── Router safety-gate constants ───────────────────────────────────


VERIFIER_OVERRIDE_MIN_PRIOR = 2     # prior verifier-attributed rejects
                                     # required before trusting an override
# Dry-run lives in the USE_FAILURE_ATTRIBUTION switch (agent.py): 1 →
# dry-run, 2 → live. The router is pure; the executor gets dry_run from
# the caller.


# ─── The router ─────────────────────────────────────────────────────


def route_failure_attribution(
    diag: FailureAttribution,
    *,
    focus_outcome: Any,
    timeline: Optional[List[Any]] = None,
) -> InterventionDecision:
    """Pick an intervention from a single FailureAttribution.

    Safety gates redirect to ``fallback_planner`` whenever a role-specific
    route can't meet its preconditions, so an unreliable diagnosis never
    costs a replan opportunity.
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
    # The actor grounds itself; we deliberately drop the diagnose coord
    # hint (it'd be copied verbatim and the diagnose model's pixels are
    # unreliable). Give it WHAT + WHY; HOW is its job.
    if role == "actor":
        rcs = (diag.root_cause_summary or "").strip()
        canonical = (diag.memory_recall_canonical or "").strip()
        divergence = (diag.divergence_description or "").strip()
        # actor_redo needs substantive prose to hand over; empty root_cause
        # + no canonical = nothing new for the actor, so fall back.
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
                # Prose guidance for the actor's prompt:
                "root_cause_summary":      rcs,
                "memory_canonical_approach": canonical,
                "divergence_description":  divergence,
                "missing_or_wrong":        list(diag.missing_or_wrong or []),
                # Full diagnosis kept for audit dumps only; the caller
                # renders just the four prose fields into the actor input.
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
                # dry_run decided by the caller (agent.py); router stays pure.
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

    Tie-break: (1) confidence high > medium > low; (2) more committed
    diagnosis (has verifier_override_evidence / env_recovery_kind);
    (3) first sample. Returns (winner, voting_meta) for logging.
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

    # Tie-break candidates by confidence then commitment.
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

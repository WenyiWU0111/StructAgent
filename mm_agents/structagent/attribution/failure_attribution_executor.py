"""Side-effect executors for the failure-attribution router decisions.

The router (failure_attribution_router) is a PURE function that picks an
``InterventionDecision``. The executors here actually carry out two of
those decisions in the live agent loop:

  apply_verifier_override(...)   — synthesize a high-confidence
                                    ``KN_OUTCOME_SATISFIED`` KeyNode and
                                    flip the focus outcome to VERIFIED.
                                    Caller decides ``dry_run`` (default
                                    True — never flips state unless the
                                    caller explicitly passes False).

  apply_env_intervention(...)    — perform a non-LLM env action (only
                                    ``wait`` is wired today; other kinds
                                    log + return False, leaving the
                                    caller to fall back).

Both functions are deliberately defensive and return a small dict
capturing what they did (or would have done) so the agent can log + dump
it without conditional duplication at the callsite.

actor_redo + planner_replan / fallback_planner do NOT live here — those
decisions affect the planner / actor prompt-build flow (handled in
agent.py + planner.py + actor.py), not env / ledger state.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


def apply_verifier_override(
    *,
    focus_outcome: Any,
    decision_params: Dict[str, Any],
    step_idx: int,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Mark the focus outcome as VERIFIED via a synthetic KeyNode.

    Returns an audit dict the caller logs / dumps. Idempotent on already-
    verified outcomes (returns ``status=already_verified``).

    Safety:
      • Default behaviour is DRY-RUN — the caller must explicitly pass
        ``dry_run=False`` to actually flip the outcome. agent.py derives
        this from ``USE_FAILURE_ATTRIBUTION``: =1 → dry-run (safe
        default); =2 → LIVE override.
      • The KeyNode is emitted at ``confidence=CONF_HIGH`` so that
        ``enters_derivation()`` is True and OutcomeStateCache picks it
        up; we intentionally bypass low-confidence filtering because the
        router has already imposed safety gates (high conf + N prior
        verifier rejects + ≥1 verbatim evidence quote).
    """
    from mm_agents.structagent.ledger.core.timeline import (
        KeyNode, KN_OUTCOME_SATISFIED, CONF_HIGH,
    )

    dry = bool(dry_run)
    outcome_id = decision_params.get("outcome_id") or getattr(
        focus_outcome, "id", "<unknown>")
    evidence_quotes = list(decision_params.get("evidence_quotes") or [])
    rationale = (decision_params.get("rationale") or "").strip()

    audit: Dict[str, Any] = {
        "outcome_id": outcome_id,
        "step_idx": step_idx,
        "dry_run": dry,
        "evidence_quotes": evidence_quotes,
        "rationale": rationale[:240],
        "status": "pending",
    }

    if focus_outcome is None:
        audit["status"] = "no_focus_outcome"
        logger.warning("[verifier_override] no focus outcome — skipping")
        return audit

    if getattr(focus_outcome, "is_verified", lambda: False)():
        audit["status"] = "already_verified"
        logger.info("[verifier_override] outcome %s already VERIFIED — "
                    "no-op", outcome_id)
        return audit

    ev_joined = " | ".join(q[:120] for q in evidence_quotes) or rationale
    ev_joined = f"[verifier_override] {ev_joined}"[:240]
    audit["synthetic_keynode_evidence"] = ev_joined

    if dry:
        audit["status"] = "dry_run_logged"
        logger.info(
            "[verifier_override DRY-RUN] would mark %s VERIFIED "
            "(evidence=%r)", outcome_id, ev_joined[:160])
        return audit

    kn = KeyNode(
        kind=KN_OUTCOME_SATISFIED,
        target=outcome_id,
        evidence=ev_joined,
        confidence=CONF_HIGH,
        detected_at_step=step_idx,
    )
    try:
        focus_outcome.apply_keynode(kn, step_idx)
    except Exception as e:  # noqa: BLE001
        audit["status"] = "apply_keynode_failed"
        audit["error"] = repr(e)
        logger.warning("[verifier_override] apply_keynode failed for %s: %s",
                       outcome_id, e)
        return audit

    audit["status"] = "applied"
    audit["outcome_state_after"] = getattr(focus_outcome, "state", None)
    logger.info("[verifier_override] APPLIED — %s → VERIFIED (evidence=%r)",
                outcome_id, ev_joined[:160])
    return audit


def apply_env_intervention(
    *,
    env: Any,
    decision_params: Dict[str, Any],
    step_idx: int,
) -> Dict[str, Any]:
    """Perform a non-LLM env-side recovery action.

    Only ``recovery_kind="wait"`` is wired today — env.step("WAIT") is
    a benign existing API. Other kinds (``focus`` / ``recovery_script``
    / ``retry``) require deciding what counts as safe to execute without
    the actor in the loop; for now we log and return ``status=
    unsupported_kind`` so the caller can fall back.

    Returns an audit dict with the new observation (if any) on success.
    """
    kind = decision_params.get("recovery_kind")
    params = decision_params.get("recovery_params") or {}
    audit: Dict[str, Any] = {
        "recovery_kind": kind,
        "recovery_params": dict(params),
        "step_idx": step_idx,
        "status": "pending",
    }

    if env is None:
        audit["status"] = "no_env"
        logger.warning("[env_intervention] env is None — skipping")
        return audit

    if kind == "wait":
        ms = int(params.get("ms") or 2000)
        ms = max(100, min(ms, 30_000))   # clamp 0.1s … 30s
        audit["wait_ms_clamped"] = ms
        try:
            time.sleep(ms / 1000.0)
            obs, _, done, _ = env.step("WAIT")
            audit["env_step_done"] = bool(done)
            audit["obs_returned"] = obs is not None
            audit["status"] = "applied"
            logger.info("[env_intervention] WAIT applied (slept %dms, "
                        "env.step done=%s)", ms, done)
        except Exception as e:  # noqa: BLE001
            audit["status"] = "env_step_failed"
            audit["error"] = repr(e)
            logger.warning("[env_intervention] env.step failed: %s", e)
        return audit

    # Other recovery kinds are intentionally NOT auto-executed yet.
    # ``recovery_script`` (shell command), ``retry`` (re-emit prior
    # action), ``focus`` (click a a11y target) all have non-trivial
    # blast radius without the actor in the loop. Log the desired
    # action and let the caller fall back to planner_replan.
    audit["status"] = "unsupported_kind"
    logger.info("[env_intervention] kind=%r not auto-executed yet — "
                "caller should fall back to planner_replan", kind)
    return audit

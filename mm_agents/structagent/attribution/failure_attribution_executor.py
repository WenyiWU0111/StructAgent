"""Side-effect executors for failure-attribution router decisions.

The router is a pure function that picks an ``InterventionDecision``; these
executors carry out two of them in the live loop:

  apply_verifier_override  — synthesize a high-confidence KN_OUTCOME_SATISFIED
                             KeyNode and flip the focus outcome to VERIFIED
                             (dry_run defaults True; only flips when passed False).
  apply_env_intervention   — non-LLM env action (only ``wait`` wired today;
                             other kinds log and let the caller fall back).

Both return an audit dict so the caller can log/dump uniformly. actor_redo /
planner_replan / fallback_planner live elsewhere (agent.py + planner.py +
actor.py) since they touch prompt-build flow, not env/ledger state.
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
    """Mark the focus outcome VERIFIED via a synthetic KeyNode. Idempotent
    (already-verified → ``status=already_verified``). Returns an audit dict.

    DRY-RUN by default; caller must pass ``dry_run=False`` to flip. agent.py
    derives this from USE_FAILURE_ATTRIBUTION (=1 dry-run, =2 LIVE).

    KeyNode is emitted at CONF_HIGH so ``enters_derivation()`` is True and
    OutcomeStateCache picks it up — bypassing low-confidence filtering is safe
    because the router already gated on high conf + N prior verifier rejects +
    >=1 verbatim evidence quote.
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
    """Perform a non-LLM env-side recovery action. Returns an audit dict.

    Only ``recovery_kind="wait"`` is wired (env.step("WAIT") is benign). Other
    kinds (``focus`` / ``recovery_script`` / ``retry``) have non-trivial blast
    radius without the actor in the loop, so they log and return
    ``status=unsupported_kind`` for the caller to fall back.
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
        ms = max(100, min(ms, 30_000))   # clamp 0.1s..30s
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

    # recovery_script / retry / focus have non-trivial blast radius without the
    # actor in the loop; log the intent and fall back to planner_replan.
    audit["status"] = "unsupported_kind"
    logger.info("[env_intervention] kind=%r not auto-executed yet — "
                "caller should fall back to planner_replan", kind)
    return audit

"""DoneAuditSnapshot + builders for both offline replay and live agent.

The snapshot is the **fresh-context input** to the Done-Auditor LLM. It
contains the task instruction, screenshots, a11y dump, outcome proofs,
and perceiver focus at the DONE moment — but NEVER the planner's chain
of thought, message history, current subgoal text, or strategy notes
(adversarial-purity invariant from plan file Part I.2).

Two builders:

  - ``build_snapshot_from_run_dir(run_dir)`` — offline replay path. Reads
    finished trajectory artifacts. Used by ``tests/replay_done_audit.py``
    for Phase-A validation.

  - ``build_snapshot_from_live_state(agent)`` — runtime path. Pulls from
    the agent's in-memory state at the moment ``_pa_predict`` is about
    to accept a DONE. Used by ``done_auditor.audit_done`` from
    ``agent.py``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# === Snapshot dataclass ========================================== #


@dataclass(frozen=True)
class DoneAuditSnapshot:
    """What the auditor LLM sees. NO planner CoT / messages / strategy.

    Same shape whether built from offline run-dir or live agent state.
    The auditor cannot tell the difference — and that's the point.
    """

    task_instruction: str                       # raw user task text
    initial_context: Dict[str, Any]             # active_app/url, window, tabs
    verifier_specs: List[Dict[str, Any]]        # [{id, description, verify}]
    initial_screenshot_b64: Optional[str]       # task-start frame
    recent_screenshots_b64: List[str]           # last K sub-action frames
    current_a11y: str                           # a11y at DONE step
    current_url: Optional[str]                  # URL at DONE step
    outcome_proofs: List[Dict[str, Any]]        # latest per-outcome proof
    perceiver_focus_block: Dict[str, Any]       # perceiver snapshot at DONE


@dataclass
class SnapshotSourceMeta:
    """Diagnostic — where each field came from + sizes. Useful for the
    offline replay path (lets the validation report explain
    reconstruction feasibility per run). For live-state builds, only
    ``label`` / ``task_id`` may be set; the rest is offline-only."""

    task_id: str = ""
    run_dir: str = ""
    done_step: Optional[int] = None
    final_score: Optional[float] = None
    label: str = ""  # false-DONE / true-DONE / didnt-DONE / borderline-DONE / unknown / live
    sources: Dict[str, str] = field(default_factory=dict)
    snapshot_chars: Dict[str, int] = field(default_factory=dict)
    n_screenshots: int = 0


# === Run-dir builder (offline replay) ============================ #


def _read_done_step(run_dir: Path) -> Optional[int]:
    p = run_dir / "traj.jsonl"
    if not p.exists():
        return None
    with p.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("action") == "DONE (terminate)":
                return row.get("step_num")
    return None


def _read_final_score(run_dir: Path) -> Optional[float]:
    p = run_dir / "result.txt"
    if not p.exists():
        return None
    try:
        return float(p.read_text().strip())
    except ValueError:
        return None


def _label_from(done_step: Optional[int],
                final_score: Optional[float]) -> str:
    if done_step is None:
        return "didnt-DONE"
    if final_score is None:
        return "unknown"
    if final_score >= 0.99:
        return "true-DONE"
    if final_score <= 0.01:
        return "false-DONE"
    return "borderline-DONE"


def _extract_task_instruction(run_dir: Path) -> str:
    p = run_dir / "perceiver_debug" / "step_001_planner_messages.json"
    if not p.exists():
        return ""
    try:
        with p.open() as f:
            msgs = json.load(f)
    except (json.JSONDecodeError, OSError):
        return ""
    for m in msgs:
        if m.get("role") != "user":
            continue
        for part in (m.get("parts") or []):
            text = part.get("text") or ""
            if "Current Task:" not in text:
                continue
            mm = re.search(
                r"Current Task:\s*\n(.+?)(?:\n\n|\Z)", text, re.DOTALL)
            if mm:
                return mm.group(1).strip()
    return ""


def _extract_initial_context_and_specs(
    run_dir: Path,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    p = run_dir / "keynode_debug" / "init_ledger_prompt_attempt_1.json"
    if not p.exists():
        return {}, []
    try:
        with p.open() as f:
            d = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}, []
    raw = d.get("raw_response", "")
    if not isinstance(raw, str):
        return {}, []
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}, []
    ctx = parsed.get("initial_context", {}) or {}
    outs = parsed.get("required_outcomes", []) or []
    specs = [
        {
            "id": o.get("id"),
            "description": o.get("description"),
            "verify": o.get("verify"),
        }
        for o in outs
    ]
    return ctx, specs


def _extract_perceiver(
    run_dir: Path, step: int,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Return (a11y_filtered_text, current_url, perceiver_focus_block)."""
    pi = run_dir / "perceiver_debug" / f"step_{step:03d}_perceiver_input.json"
    ps = run_dir / "perceiver_debug" / f"step_{step:03d}_perceiver_snapshot.json"

    a11y = ""
    url: Optional[str] = None
    focus: Dict[str, Any] = {}

    if pi.exists():
        try:
            with pi.open() as f:
                d = json.load(f)
            a11y = d.get("a11y_filtered_text", "") or ""
            tb = d.get("transition_block", "") or ""
            m = re.search(r"URL:\s*\S+\s*→\s*(\S+?)(?:\s|$)", tb)
            if m:
                url = m.group(1).strip()
            else:
                m = re.search(r"URL:\s*(\S+)", tb)
                if m:
                    url = m.group(1).strip()
        except (json.JSONDecodeError, OSError):
            pass

    if ps.exists():
        try:
            with ps.open() as f:
                snap = json.load(f)
            focus = {k: v for k, v in snap.items() if k != "slot_bindings"}
        except (json.JSONDecodeError, OSError):
            pass

    return a11y, url, focus


def _fold_outcome_proofs(run_dir: Path) -> List[Dict[str, Any]]:
    """Walk step_*_keynode_verdicts.json, fold to latest emitted_keynode
    per outcome (satisfied or invalidated)."""
    kd = run_dir / "keynode_debug"
    if not kd.exists():
        return []
    by_outcome: Dict[str, Dict[str, Any]] = {}
    files = sorted(kd.glob("step_*_keynode_verdicts.json"))
    for jp in files:
        try:
            with jp.open() as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        step_idx = d.get("step_idx", -1)
        step_num = step_idx + 1 if isinstance(step_idx, int) else -1
        for po in (d.get("per_outcome") or []):
            oid = po.get("outcome_id")
            ek = po.get("emitted_keynode") or {}
            kind = ek.get("kind")
            if not oid or not kind:
                continue
            if kind == "outcome_satisfied":
                by_outcome[oid] = {
                    "outcome_id": oid,
                    "state": "verified",
                    "verified_at_step": step_num,
                    "evidence": ek.get("evidence", ""),
                    "confidence": ek.get("confidence", ""),
                }
            elif kind == "outcome_invalidated":
                by_outcome[oid] = {
                    "outcome_id": oid,
                    "state": "reverted",
                    "reverted_at_step": step_num,
                    "evidence": ek.get("evidence", ""),
                    "confidence": ek.get("confidence", ""),
                }
    return list(by_outcome.values())


_B64_RE = re.compile(r"data:image/png;base64,([A-Za-z0-9+/=]{1000,})")


def _extract_screenshots(
    run_dir: Path, k: int = 3,
) -> Tuple[Optional[str], List[str]]:
    th = run_dir / "trajectory.html"
    if not th.exists():
        return None, []
    try:
        txt = th.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, []
    matches = _B64_RE.findall(txt)
    if not matches:
        return None, []
    initial = matches[0]
    last_k = matches[-k:] if len(matches) >= k else list(matches)
    return initial, last_k


def build_snapshot_from_run_dir(
    run_dir: str,
) -> Tuple[Optional[DoneAuditSnapshot], SnapshotSourceMeta]:
    """Reconstruct a snapshot from a finished OSWorld run directory.

    Returns ``(snapshot, meta)``. ``snapshot`` is None when the run
    never said DONE (label='didnt-DONE') — caller should SKIP these.
    ``meta`` is always populated.
    """
    run_path = Path(run_dir)
    done_step = _read_done_step(run_path)
    final_score = _read_final_score(run_path)
    label = _label_from(done_step, final_score)

    meta = SnapshotSourceMeta(
        task_id=run_path.name,
        run_dir=str(run_path),
        done_step=done_step,
        final_score=final_score,
        label=label,
    )

    if done_step is None:
        return None, meta

    task = _extract_task_instruction(run_path)
    init_ctx, specs = _extract_initial_context_and_specs(run_path)
    a11y, url, focus = _extract_perceiver(run_path, done_step)
    proofs = _fold_outcome_proofs(run_path)
    initial_ss, recent_ss = _extract_screenshots(run_path, k=3)

    snap = DoneAuditSnapshot(
        task_instruction=task,
        initial_context=init_ctx,
        verifier_specs=specs,
        initial_screenshot_b64=initial_ss,
        recent_screenshots_b64=recent_ss,
        current_a11y=a11y,
        current_url=url,
        outcome_proofs=proofs,
        perceiver_focus_block=focus,
    )

    meta.sources = {
        "task_instruction": "perceiver_debug/step_001_planner_messages.json (regex)",
        "initial_context": "keynode_debug/init_ledger_prompt_attempt_1.json raw_response.initial_context",
        "verifier_specs": "keynode_debug/init_ledger_prompt_attempt_1.json raw_response.required_outcomes",
        "screenshots": "trajectory.html (regex data:image/png;base64)",
        "current_a11y+url+focus": f"perceiver_debug/step_{done_step:03d}_*.json",
        "outcome_proofs": "keynode_debug/step_*_keynode_verdicts.json (walked)",
    }
    meta.snapshot_chars = {
        "task_instruction": len(task),
        "current_a11y": len(a11y),
        "outcome_proofs": sum(len(p.get("evidence", "")) for p in proofs),
    }
    meta.n_screenshots = (1 if initial_ss else 0) + len(recent_ss)
    return snap, meta


# === Live-state builder (runtime) ================================ #


def _project_outcome_proofs_from_ledger(ledger: Any) -> List[Dict[str, Any]]:
    """Render one line per VERIFIED-or-REVERTED outcome from
    ``ledger.required_outcomes[*].last_verifier_trace``.

    Outcomes that never reached a verified or reverted state are
    omitted — the auditor can spot coverage gaps by diffing against
    ``verifier_specs`` (which lists ALL required outcomes).
    """
    if ledger is None:
        return []
    proofs: List[Dict[str, Any]] = []
    for o in getattr(ledger, "required_outcomes", []) or []:
        state = getattr(o, "state", "pending")
        trace = getattr(o, "last_verifier_trace", None) or {}
        evidence = ""
        if isinstance(trace, dict):
            evidence = (
                trace.get("evidence")
                or (trace.get("emitted_keynode") or {}).get("evidence")
                or ""
            )
        if state == "verified":
            proofs.append({
                "outcome_id": getattr(o, "id", ""),
                "state": "verified",
                "verified_at_step": getattr(o, "last_satisfy_step", None),
                "evidence": evidence,
                "confidence": (
                    trace.get("confidence")
                    if isinstance(trace, dict) else ""
                ) or "",
            })
        elif state == "reverted":
            proofs.append({
                "outcome_id": getattr(o, "id", ""),
                "state": "reverted",
                "reverted_at_step": getattr(o, "last_invalidate_step", None),
                "evidence": evidence,
                "confidence": (
                    trace.get("confidence")
                    if isinstance(trace, dict) else ""
                ) or "",
            })
    return proofs


def _verify_spec_to_dict(v: Any) -> Optional[Dict[str, Any]]:
    """Best-effort serialize a VerifySpec to dict for prompt rendering."""
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if hasattr(v, "to_dict"):
        try:
            return v.to_dict()
        except Exception:  # noqa: BLE001
            pass
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(v):
            return asdict(v)
    except Exception:  # noqa: BLE001
        pass
    try:
        return {
            k: getattr(v, k) for k in dir(v)
            if not k.startswith("_") and not callable(getattr(v, k))
        }
    except Exception:  # noqa: BLE001
        return None


def _project_verifier_specs_from_ledger(ledger: Any) -> List[Dict[str, Any]]:
    if ledger is None:
        return []
    out: List[Dict[str, Any]] = []
    for o in getattr(ledger, "required_outcomes", []) or []:
        out.append({
            "id": getattr(o, "id", ""),
            "description": getattr(o, "description", ""),
            "verify": _verify_spec_to_dict(getattr(o, "verify", None)),
        })
    return out


def _project_initial_context_from_ledger(ledger: Any) -> Dict[str, Any]:
    """Pull ``open_tabs / active_url / active_app / window_title`` from
    ``ledger.initial_context``. Defensive against dict-vs-dataclass shape.
    """
    if ledger is None:
        return {}
    ic = getattr(ledger, "initial_context", None)
    if ic is None:
        return {}
    keys = ("open_tabs", "active_url", "active_app", "window_title")
    if isinstance(ic, dict):
        return {k: ic.get(k) for k in keys if k in ic}
    return {k: getattr(ic, k, None) for k in keys if hasattr(ic, k)}


def _project_perceiver_focus_from_agent(agent: Any) -> Dict[str, Any]:
    """Best-effort dict from ``agent._current_snapshot`` — small set of
    high-signal fields for the auditor prompt."""
    snap = getattr(agent, "_current_snapshot", None)
    if snap is None:
        return {}
    if isinstance(snap, dict):
        return {k: v for k, v in snap.items() if k != "slot_bindings"}
    keys = (
        "subgoal_text", "expected_post_state", "transition_summary",
        "coverage", "unexpected_blockers",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if hasattr(snap, k):
            try:
                out[k] = getattr(snap, k)
            except Exception:  # noqa: BLE001
                pass
    sc = getattr(snap, "subgoal_check", None)
    if sc is not None:
        try:
            out["subgoal_check"] = {"overall": getattr(sc, "overall", "?")}
        except Exception:  # noqa: BLE001
            pass
    return out


def build_snapshot_from_live_state(agent: Any) -> DoneAuditSnapshot:
    """Build the snapshot from an in-flight ``StructAgent`` instance.

    Called from ``agent._pa_predict`` at the DONE-acceptance site (see
    plan Part I.2). Pulls task text + ledger state + recent screenshots
    + perceiver focus. Excludes ALL planner CoT / message history /
    current subgoal text (adversarial purity).
    """
    # Task instruction — prefer the raw user task over any augmented
    # version the planner saw.
    task = (
        getattr(agent, "_raw_instruction", None)
        or getattr(agent, "instruction", "")
        or ""
    )

    ledger = getattr(agent, "ledger", None)
    initial_context = _project_initial_context_from_ledger(ledger)
    verifier_specs = _project_verifier_specs_from_ledger(ledger)
    outcome_proofs = _project_outcome_proofs_from_ledger(ledger)

    # Initial screenshot (task-start anchor)
    initial_ss: Optional[str] = (
        getattr(ledger, "initial_screenshot_b64", None) if ledger else None
    )

    # Recent screenshots — prefer the last K from agent.timeline; fall
    # back to the current candidate-step screenshot.
    # Last 5 frames ENDING at the DONE frame, so the auditor sees the workflow
    # leading to DONE (e.g. a save/print flow), not just the terminal screen.
    # NOTE: ``agent.timeline`` is a list of TimelineEvents — screenshots live on
    # each event's ``.steps`` (StepRecords), NOT on the event. An earlier
    # version read ``event.screenshot_b64`` and so always got nothing, leaving
    # the auditor with a single frame.
    recent_ss: List[str] = []
    timeline = getattr(agent, "timeline", None)
    try:
        all_steps = [s for ev in (timeline or [])
                     for s in (getattr(ev, "steps", None) or [])]
        frames = [getattr(s, "screenshot_b64", None) for s in all_steps]
        cand = getattr(agent, "_candidate_step", None)  # the DONE frame
        if cand is not None:
            frames.append(getattr(cand, "screenshot_b64", None))
        recent_ss = [f for f in frames if f][-5:]
    except Exception:  # noqa: BLE001
        recent_ss = []
    if not recent_ss:
        cand = getattr(agent, "_candidate_step", None)
        cand_ss = getattr(cand, "screenshot_b64", None) if cand is not None else None
        if cand_ss:
            recent_ss = [cand_ss]

    # Current a11y + URL from the candidate step (the frame the planner
    # was looking at when it decided DONE).
    current_a11y = ""
    current_url: Optional[str] = None
    cand = getattr(agent, "_candidate_step", None)
    if cand is not None:
        current_a11y = getattr(cand, "a11y", "") or ""
        current_url = getattr(cand, "url", None)

    perceiver_focus = _project_perceiver_focus_from_agent(agent)

    return DoneAuditSnapshot(
        task_instruction=task,
        initial_context=initial_context,
        verifier_specs=verifier_specs,
        initial_screenshot_b64=initial_ss,
        recent_screenshots_b64=recent_ss,
        current_a11y=current_a11y,
        current_url=current_url,
        outcome_proofs=outcome_proofs,
        perceiver_focus_block=perceiver_focus,
    )

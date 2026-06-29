"""Phase Board — the per-app phase layer for multi-application tasks.

A multi-app task is fundamentally a chain of app phases:

    Phase 1 (app A: produce artifact X) → handoff(X)
        → Phase 2 (app B: consume X, produce Y) → handoff(Y) → ...

Before this module the agent had three loosely-coupled "progress"
systems that disagreed with each other:

  * ``subgoal_queue`` / ``completed_subgoals`` — the plan execution
    pointer (planner-authored text, frozen at plan time).
  * the ledger ``Outcome`` set — deliverable verification state.
  * ``_current_domain`` — per-turn-derived execution context routing.

None of them was the *phase* — the coarse "we are working in app X
toward goal G, and app X hands Y to the next app". The PhaseBoard
adds exactly that missing layer. It is:

  * an **app-aggregated view** of the ledger (a phase is "done" iff
    every outcome assigned to it is verified) — NOT a new verifier.
  * the **committed** execution-context source: ``_current_domain``
    follows the active phase's ``app`` instead of being re-derived
    (and thrashed) every turn.
  * the home of the explicit **handoff** record — what each phase
    produced for the next, summarized by an LLM at the phase boundary
    (see ``Phase.actual_handoff_out``).

Multi-app only. Single-app tasks never build a PhaseBoard; every
call site is gated so single-domain behaviour is byte-for-byte
unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set


PHASE_PENDING = "pending"
PHASE_ACTIVE = "active"
PHASE_DONE = "done"


@dataclass
class Phase:
    """One app phase of a multi-app task.

    ``goal`` is SEMANTIC — it names what to accomplish in this app
    without concrete anchors (cell ranges, indices, counts, literal
    values), because the phase is authored at task start before the
    app is observed (same rule as plan-quality rule 10).

    Handoff fields, two layers:
      * ``planned_handoff_in`` / ``planned_handoff_out`` — authored by
        init_ledger, the SEMANTIC contract ("this phase receives the
        requested rows on the clipboard / produces a saved PDF").
      * ``actual_handoff_out`` — filled at the phase→done boundary by
        an LLM that reads the phase's real trajectory + filesystem
        diff + clipboard probe. This is what the downstream phase
        actually sees ("clipboard holds 工作表1!A1:L2, 2 rows × 12
        cols" / "saved /home/user/Desktop/receipt.pdf").
    """
    index: int                       # 1-based execution order
    app: str                         # domain, e.g. "libreoffice_calc" / "chrome" / "os"
    goal: str                        # semantic goal — NO concrete anchors
    status: str = PHASE_PENDING      # pending | active | done
    outcome_ids: List[str] = field(default_factory=list)
    planned_handoff_in: Optional[str] = None
    planned_handoff_out: Optional[str] = None
    actual_handoff_out: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Phase":
        return cls(
            index=int(d.get("index", 0)),
            app=str(d.get("app", "") or ""),
            goal=str(d.get("goal", "") or ""),
            status=str(d.get("status", PHASE_PENDING) or PHASE_PENDING),
            outcome_ids=list(d.get("outcome_ids", []) or []),
            planned_handoff_in=d.get("planned_handoff_in"),
            planned_handoff_out=d.get("planned_handoff_out"),
            actual_handoff_out=d.get("actual_handoff_out"),
        )


@dataclass
class PhaseBoard:
    """Ordered list of phases for one multi-app task.

    Exactly one phase is ``active`` at a time (the first non-done
    phase). ``advance()`` is called by the agent once every outcome
    assigned to the active phase is verified.
    """
    phases: List[Phase] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_init_payload(cls, phases_json: Any) -> Optional["PhaseBoard"]:
        """Build from the ``phases`` array of the init_ledger LLM output.

        Returns None when the payload is missing / malformed / has <2
        phases (a <2-phase board is not a multi-app handoff chain and
        adds only overhead — fall back to the flat subgoal model).
        """
        if not isinstance(phases_json, list) or len(phases_json) < 2:
            return None
        phases: List[Phase] = []
        for i, p in enumerate(phases_json):
            if not isinstance(p, dict):
                return None
            app = str(p.get("app", "") or "").strip().lower().replace(" ", "_")
            goal = str(p.get("goal", "") or "").strip()
            if not app or not goal:
                return None
            phases.append(Phase(
                index=i + 1,
                app=app,
                goal=goal,
                status=PHASE_ACTIVE if i == 0 else PHASE_PENDING,
                outcome_ids=[str(x) for x in (p.get("outcome_ids", []) or [])],
                planned_handoff_in=(p.get("planned_handoff_in") or None),
                planned_handoff_out=(p.get("planned_handoff_out") or None),
            ))
        return cls(phases=phases)

    def to_dict(self) -> Dict[str, Any]:
        return {"phases": [p.to_dict() for p in self.phases]}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PhaseBoard":
        return cls(phases=[Phase.from_dict(p) for p in d.get("phases", [])])

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def active(self) -> Optional[Phase]:
        for p in self.phases:
            if p.status == PHASE_ACTIVE:
                return p
        return None

    def previous_done(self) -> Optional[Phase]:
        """The most-recently-completed phase (the one whose handoff the
        active phase consumes). None when the active phase is first."""
        done = [p for p in self.phases if p.status == PHASE_DONE]
        return done[-1] if done else None

    def all_done(self) -> bool:
        return bool(self.phases) and all(
            p.status == PHASE_DONE for p in self.phases)

    def outcome_ids_for_active(self) -> List[str]:
        a = self.active()
        return list(a.outcome_ids) if a else []

    def summary_line(self) -> str:
        """Compact one-line status for runtime.log (greppable per turn)."""
        if not self.phases:
            return "PhaseBoard: empty"
        active = self.active()
        cells = []
        for p in self.phases:
            mark = {PHASE_DONE: "✓", PHASE_ACTIVE: "▶",
                    PHASE_PENDING: "·"}.get(p.status, "·")
            cells.append(f"{mark}{p.index}:{p.app}")
        cur = (f"phase {active.index}/{len(self.phases)} ({active.app})"
               if active else "ALL DONE")
        return f"active={cur} | " + " ".join(cells)

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #
    def advance(self, actual_handoff_out: Optional[str] = None
                ) -> Optional[Phase]:
        """Mark the active phase done (storing its LLM-summarized
        handoff) and promote the next pending phase to active.

        Returns the new active phase, or None if the board is now
        fully done.
        """
        cur = self.active()
        if cur is None:
            return None
        cur.status = PHASE_DONE
        if actual_handoff_out:
            cur.actual_handoff_out = actual_handoff_out.strip()
        for p in self.phases:
            if p.status == PHASE_PENDING:
                p.status = PHASE_ACTIVE
                return p
        return None

    # ------------------------------------------------------------------ #
    # Rendering — injected into the planner prompt
    # ------------------------------------------------------------------ #
    def render_block(self) -> str:
        """Human-readable PHASE BOARD block for the planner prompt.

        Shows every phase's status, the active phase's incoming
        handoff (what the previous phase actually produced), and which
        phase the planner should plan for right now.
        """
        if not self.phases:
            return ""
        active = self.active()
        pos = (f"{active.index}/{len(self.phases)}"
               if active else f"all {len(self.phases)} done")
        lines = [f"PHASE BOARD  (you are in phase {pos} — plan ONLY for "
                 f"the [active] phase)"]
        for p in self.phases:
            tag = {PHASE_DONE: "[done]  ",
                   PHASE_ACTIVE: "[active]",
                   PHASE_PENDING: "[pending]"}.get(p.status, "[pending]")
            lines.append(f"{tag} {p.index}. {p.app} — {p.goal}")
            if p.status == PHASE_DONE and p.actual_handoff_out:
                lines.append(f"         └ produced: {p.actual_handoff_out}")
            elif p.status == PHASE_ACTIVE:
                prev = self.previous_done()
                # What the active phase receives: prefer the upstream
                # phase's ACTUAL produced summary; fall back to this
                # phase's planned_handoff_in semantic description.
                recv = None
                if prev is not None and prev.actual_handoff_out:
                    recv = prev.actual_handoff_out
                elif p.planned_handoff_in:
                    recv = p.planned_handoff_in
                if recv:
                    lines.append(f"         └ receives: {recv}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Orchestration — phase-boundary handoff summary + advance
#
# These are module-level functions (not agent methods) so the agent
# file stays thin: it only holds the PhaseBoard reference + the
# per-phase action-index cursor, and delegates the logic here.
# ---------------------------------------------------------------------- #

def _strip_think(text: str) -> str:
    """Drop Qwen-style <think>…</think> reasoning from an LLM reply."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    cut = t.rfind("</think>")
    return t[cut + len("</think>"):].lstrip() if cut >= 0 else t


def summarize_phase_handoff(
    phase: Phase,
    phase_actions: List[str],
    *,
    call_llm: Callable,
    model: str,
    logger: Any = None,
) -> Optional[str]:
    """LLM-summarize what a just-completed phase produced for the next
    phase. ``phase_actions`` is the slice of the agent's action log
    taken while this phase was active. Returns a 1-2 sentence handoff
    description (WHAT was produced + WHERE it lives), or the phase's
    ``planned_handoff_out`` on empty trajectory / LLM failure.
    """
    fallback = phase.planned_handoff_out
    acts = [a for a in (phase_actions or []) if a]
    if not acts:
        return fallback
    actions_text = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(acts[-40:]))
    prompt = (
        "A multi-application task just finished one application PHASE. "
        "Summarize, in ONE or TWO sentences, the concrete artifact this "
        "phase produced for the NEXT phase to consume — name WHAT it is "
        "and WHERE it now lives: a file path, the system clipboard, a "
        "URL, or — when it is a value the agent must simply carry "
        "forward — state the value itself. Be specific and factual; if "
        "the trajectory shows the phase did NOT actually produce its "
        "intended artifact, say so plainly.\n\n"
        "EVIDENCE RULE: each concrete claim MUST be supported by a "
        "specific action in the trajectory below. Cite by step number "
        "(e.g. ``(step 14)``) or quote the verbatim action token. "
        "Claims you cannot cite to a specific action MUST be tagged "
        "``(unverified — planned only)`` so the downstream phase knows "
        "not to rely on them. Do NOT extrapolate beyond the action "
        "log: if the trajectory only shows the agent CLICKED a save "
        "dialog but does not confirm the file landed on disk, do not "
        "claim the file exists — say so plainly or tag unverified.\n\n"
        f"Phase goal: {phase.goal}\n"
        f"Phase app: {phase.app}\n"
        f"Planned to produce: {fallback or '(unspecified)'}\n\n"
        f"Actions executed in this phase:\n{actions_text}\n\n"
        "Reply with ONLY the 1-2 sentence handoff summary, no preamble."
    )
    try:
        resp = call_llm(
            {"model": model,
             "messages": [{"role": "user", "content": prompt}],
             "max_tokens": 220, "temperature": 0.0},
            model,
        )
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.info("[Phase] handoff summary LLM failed: %s", e)
        return fallback
    return _strip_think(resp or "") or fallback


_PHASE_AUDIT_SYSTEM_PROMPT = """\
You are the Phase-Boundary Auditor for an OSWorld multi-app agent.
The agent has just finished one application PHASE in a multi-phase
task. A summarizer LLM wrote a 1-2 sentence claim of WHAT this phase
produced for the next phase to consume. Your job: adversarially
check that claim against the actual evidence.

You do NOT see the next phase's plan, the task's final criteria, or
any planner reasoning. Your sole question: did this phase actually
produce the artifact the claim names?

Evidence you receive:
  - HANDOFF CLAIM (the summarizer's 1-2 sentence claim)
  - PHASE ACTION TRAJECTORY (last 40 actions, text)
  - CURRENT SCREENSHOT (the screen at phase end — when available)
  - CURRENT ACCESSIBILITY TREE (capped — when available)
  - DOCUMENT STATE (UNO probe of any open libreoffice doc — when
    available; AUTHORITATIVE for office structural questions like
    cell value / paragraph style / shape props — supersedes the
    screenshot for those)

You will feel the urge to rubber-stamp a plausible claim. Resist it:
  - The summarizer is an LLM that extrapolated from action text
    without verifying outcomes. Trajectory looking right is not the
    same as artifact existing.
  - The screenshot can mislead — dialogs close on cancel, on
    auto-rename, on disk-full. A closed dialog is NOT proof a file
    was written.
  - For office tasks, the DOCUMENT STATE is ground truth for cell /
    shape / paragraph claims; cross-check the claim against it.

Output structure:
  CLAIM PARSE:
    <one bullet per concrete artifact named in the handoff claim>
  EVIDENCE CHECK:
    <one bullet per claim, citing the SPECIFIC action (step number),
     a11y row, screenshot region, or document-state line that
     supports OR refutes it. If a claim is unsupported, say so
     explicitly.>
  GAP:
    <if any claim is unsupported, name it; if all supported, "(none)">

Final line MUST be exactly one of:
  VERDICT: PASS    (every concrete claim supported by at least one
                    specific cited piece of evidence)
  VERDICT: FAIL    (at least one claim is unsupported or refuted)

Do NOT use any other verdict tokens. Do NOT add preamble.
"""


def audit_phase_handoff(
    phase: Phase,
    phase_actions: List[str],
    *,
    call_llm: Callable,
    model: str,
    logger: Any = None,
    current_screenshot_b64: Optional[str] = None,
    current_a11y: Optional[str] = None,
    doc_inspect_block: Optional[str] = None,
) -> str:
    """Adversarially verify the just-summarized ``actual_handoff_out``
    against the action trajectory + current observable state. Returns
    ``"PASS"`` / ``"FAIL"``.

    Fail-open semantics: on LLM error, empty inputs, or unparseable
    verdict, returns ``"PASS"`` (don't block phase advance on
    framework hiccup). The caller checks ``"FAIL"`` explicitly.
    """
    if not phase_actions or not phase.actual_handoff_out:
        return "PASS"
    acts = [a for a in phase_actions if a]
    if not acts:
        return "PASS"
    actions_text = "\n".join(
        f"  {i+1}. {a}" for i, a in enumerate(acts[-40:]))
    user_parts: List[str] = [
        f"Phase {phase.index} ({phase.app}) just completed.",
        f"Phase goal: {phase.goal}",
        "",
        "HANDOFF CLAIM (LLM-summarized, awaiting your audit):",
        f"  {phase.actual_handoff_out}",
        "",
        "PHASE ACTION TRAJECTORY (last 40 actions, 1-based):",
        actions_text,
    ]
    if current_a11y:
        a11y_trim = current_a11y if len(current_a11y) < 6000 else current_a11y[-6000:]
        user_parts.extend([
            "",
            "CURRENT ACCESSIBILITY TREE (capped to last 6000 chars):",
            a11y_trim,
        ])
    if doc_inspect_block:
        doc_trim = (doc_inspect_block if len(doc_inspect_block) < 4000
                    else doc_inspect_block[:4000]
                    + "\n  ... (truncated, earlier sections preserved)")
        user_parts.extend([
            "",
            "DOCUMENT STATE (UNO probe — AUTHORITATIVE for cell / "
            "shape / paragraph claims; trust over screenshot for "
            "structural office props):",
            doc_trim,
        ])
    user_parts.extend([
        "",
        "Run CLAIM PARSE / EVIDENCE CHECK / GAP and emit a verdict.",
    ])
    user_text = "\n".join(user_parts)
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if current_screenshot_b64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{current_screenshot_b64}"
            },
        })
    try:
        raw = call_llm(
            {"model": model,
             "messages": [
                 {"role": "system",
                  "content": _PHASE_AUDIT_SYSTEM_PROMPT},
                 {"role": "user", "content": user_content},
             ],
             "max_tokens": 400, "temperature": 0.0},
            model,
        )
    except Exception as e:  # noqa: BLE001
        if logger:
            logger.info("[Phase] audit LLM failed (fail-open): %s", e)
        return "PASS"
    if not raw:
        return "PASS"
    last = (raw or "").strip().splitlines()[-1].strip()
    if last == "VERDICT: FAIL":
        return "FAIL"
    if last == "VERDICT: PASS":
        return "PASS"
    if logger:
        logger.info(
            "[Phase] audit verdict unparseable (fail-open): %r",
            last[:120])
    return "PASS"


def advance_phase_if_complete(
    board: Optional[PhaseBoard],
    *,
    verified_ids: Set[str],
    current_subgoal_app: Optional[str],
    phase_actions: List[str],
    call_llm: Callable,
    model: str,
    logger: Any = None,
    enable_audit: bool = False,
    current_screenshot_b64: Optional[str] = None,
    current_a11y: Optional[str] = None,
    doc_inspect_block: Optional[str] = None,
) -> bool:
    """If the active phase is complete, summarize its handoff and
    advance ``board``. Returns True iff a phase advanced (the caller
    then resets its per-phase action cursor).

    Phase-complete test:
      * outcome-bearing phase → every assigned outcome_id is in
        ``verified_ids``.
      * zero-outcome phase    → ``current_subgoal_app`` has moved to
        the NEXT phase's app (nothing to verify; the planner's own
        progress is the only available signal).

    No-op (returns False) when ``board`` is None — single-app tasks.
    """
    if board is None:
        return False
    active = board.active()
    if active is None:
        return False
    if active.outcome_ids:
        if not all(oid in verified_ids for oid in active.outcome_ids):
            return False
    else:
        nxt = next((p for p in board.phases
                    if p.index == active.index + 1), None)
        sg = (current_subgoal_app or "").strip().lower().replace(
            " ", "_").replace("-", "_")
        sg = {"terminal": "os", "vscode": "vs_code"}.get(sg, sg)
        if not (nxt is not None and sg and sg == nxt.app):
            return False
    summary = summarize_phase_handoff(
        active, phase_actions, call_llm=call_llm, model=model, logger=logger)
    # Stage the summary on the phase so the auditor can read
    # ``phase.actual_handoff_out`` even before advance() commits.
    if summary:
        active.actual_handoff_out = summary.strip()
    if enable_audit:
        verdict = audit_phase_handoff(
            active, phase_actions,
            call_llm=call_llm, model=model, logger=logger,
            current_screenshot_b64=current_screenshot_b64,
            current_a11y=current_a11y,
            doc_inspect_block=doc_inspect_block,
        )
        if verdict == "FAIL":
            if logger:
                logger.info(
                    "[Phase] phase %d (%s) audit FAILED — staying "
                    "active; handoff claim refuted by trajectory + "
                    "snapshot. Claim was: %s",
                    active.index, active.app,
                    (summary or "(none)")[:200],
                )
            # Roll back the staged claim so a future retry's summary
            # call writes fresh, not appended to the rejected one.
            active.actual_handoff_out = None
            return False
    new_active = board.advance(summary)
    if logger:
        logger.info(
            "[Phase] phase %d (%s) DONE — handoff: %s%s",
            active.index, active.app, (summary or "(none)")[:200],
            (f" | now phase {new_active.index} ({new_active.app})"
             if new_active else " | ALL phases done"),
        )
    return True

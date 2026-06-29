"""A8-V0: LLM-based stuck-state diagnosis for the planner's force_replan path.

When the planner has already issued N replans against the same focus
outcome without progress, the existing 6-rule recovery cascade has done
its job — it fired force_replan, surfaced a stuck_reason, and let the
planner try again. But the failure mode this addresses is the case where
*the planner keeps trying* and *still can't recognize the root cause*
on its own — typically because the relevant pieces (what O4 requires,
what working memory holds, what's currently observable, what the recent
actions actually changed) are scattered across separate parts of the
prompt.

This module's job is to be a high-fidelity information collector that
hands all of those structured pieces to a small LLM call which
synthesizes:
  • a 2-3 sentence root-cause summary
  • a category tag (info_gap / wrong_info / wrong_target / wrong_env /
    strategy_unfit / unclear)
  • a concrete "missing or wrong" list
  • a short next-action hint

The result is rendered into the force_replan prompt under
[Stuck diagnosis ...]. Cache memoizes by (focus_id, action-summary digest,
observation digest) so unchanged state re-uses the prior diagnosis
without re-billing.

For debug: every call's full prompt messages + raw response + parsed
result get persisted under ``{results_dir}/stuck_diagnosis/`` keyed by
step + focus outcome. Cache hits also dump (with cache_status="hit") so
the trace is complete.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# How many replans against the same focus outcome must have happened
# before we fire the diagnose call. 0 = fire every force_replan; 2 =
# fire only after the planner has already replanned ≥2 times here.
# Conservative default avoids LLM calls on first-replan turns where the
# 6-rule cascade has not yet shown the planner can't self-recover.
STUCK_DIAGNOSIS_MIN_REPLANS = 2

# Cap on per-turn LLM input size (rough char budget).
_MAX_OBS_A11Y_LINES = 12
_MAX_RECENT_STEPS = 5
_MAX_FACT_LINES = 12
_TRUNC_TEXT = 240


_SYSTEM_PROMPT = """You are a root-cause diagnostician for a GUI agent that has gotten stuck on a single goal.

You receive a structured snapshot:
  • GOAL the agent is trying to verify (an outcome + its verify spec)
  • RECOVERY TRIGGER (which 6-rule recovery condition fired, plus a one-line reason)
  • PRIOR DIAGNOSES (the last 3 diagnoses YOU produced on this same focus — read these first; if you flip your category without new evidence, you're hallucinating)
  • ACTION ROLLUP (recent actions grouped by literal signature with counts and observable deltas — the SAME signature tried ≥3 times with no positive delta is a STRATEGIC EXHAUSTION signal)
  • AVAILABLE ACTIONS (the catalogue of actions the actor can emit — your hint should name one of these by its exact signature)
  • LAST CLI_RUN / OPEN_APP RESULT (returncode + stdout + stderr from the most recent shell/script the actor ran — your direct signal for coding / CLI failures: script crashes, returncode != 0, missing files, etc.)
  • CURRENT SCREENSHOT (the live screen the agent is looking at — your most direct evidence of GUI state)
  • CURRENT OBSERVATION (a11y text signals to complement the screenshot)
  • RECENT ATTEMPTS (last few actions and what they appeared to produce — RAW form, useful for read-back; ROLLUP above is the structured view)
  • LAST VERIFIER TRACE (why the current outcome's verify last returned False)

Prefer the screenshot for visual state judgments. The a11y text section is supplementary — it can be empty, malformed, or stale when a window is unfocused or a transient modal is up; in those cases trust the screenshot first.

Your job: pinpoint the single most-likely root cause of why the agent cannot reach the goal *right now*, classify it, and suggest one concrete next move.

Use RECOVERY TRIGGER to scope your diagnosis:
  • ``actor_failure``     — the actor returned IMPOSSIBLE / FAIL last turn. Most often tactical (a click target / a script command failed).
  • ``done_rejected``     — most outcomes are verified; ONE specific outcome is still pending. Root cause is almost always inside that outcome's verifier trace + LAST CLI_RUN result.
  • ``budget_exhausted``  — the same subgoal burned through its step budget without verified progress. Read RECENT ATTEMPTS for the dominant failure mode.
  • ``replan_cadence``    — same strategy across N consecutive replans without progress. Default to tactical; only pick strategic when the SAME tactic has been tried with the same args ≥3 times without observable change OR LAST CLI_RUN keeps returning the same error after fixes.
  • ``queue_empty_no_done`` — subgoal queue drained but task end-state unverified. Look at GOAL for what's still missing.

Four root-cause categories (PICK ONE for ``category``):
  tactical   — the current strategy is correct, but the EXECUTING ACTION is going wrong (wrong UI element, element missing/disabled, modal blocking, focus lost, script crashed, returncode != 0, command not found, …). Retry with a DIFFERENT TACTIC under the same strategy.
  strategic  — the strategy itself cannot satisfy the verify spec. Switch to a different approach category (GUI ↔ CLI, different feature path, different L2 skill).
  info       — the agent lacks a concrete value the spec needs (verifier explicitly flags an unresolved slot, OR the GOAL's verify spec requires a value the snapshot/CLI output cannot supply).
  unclear    — insufficient signal to localize; explain in root_cause_summary what additional signal you need.

When ``category=tactical`` you ALSO emit ``tactical_subkind`` (one of) — these cover both GUI and code/CLI failure modes:
  wrong_target     — operating on a wrong element / file / identifier (right kind of thing, wrong instance)
  target_missing   — the named element / file / symbol / command does not exist in the current observation
  target_blocked   — exists but inaccessible (UI greyed/disabled, file locked, permission denied, port busy)
  blocking_state   — environment is in a state that prevents the action (modal/banner over UI, unresolved import, missing env var, shell waiting for input)
  focus_lost       — wrong context has focus (wrong window for GUI, wrong cwd / venv / shell for CLI)
  execution_error  — action ran but errored or had no effect (GUI click no-effect on dead pixel, script crashed with traceback, returncode != 0 with stderr)
``tactical_subkind`` is OMITTED (or null) for non-tactical categories.

Output JSON ONLY (no prose, no code fences):
{
  "root_cause_summary": "<2-3 sentences pinpointing why the agent is stuck right now>",
  "category": "tactical" | "strategic" | "info" | "unclear",
  "tactical_subkind": "wrong_target" | "target_missing" | "target_blocked" | "blocking_state" | "focus_lost" | "execution_error" | null,
  "missing_or_wrong": ["<concrete item 1>", "<item 2>", ...],
  "next_action_hint": "<one short concrete suggestion the planner can use as the seed of its next plan>"
}

Rules:
  • Anchor every claim in the snapshot. Do NOT invent UI state the snapshot does not contain.
  • Keep root_cause_summary tight (2-3 sentences max).
  • Keep next_action_hint short (one sentence, action-oriented).
  • next_action_hint MUST name a specific action from AVAILABLE ACTIONS by its exact signature (e.g. ``hotkey(key='escape')``, ``cli_run('pkill -f chrome; sleep 2; ...')``, ``impress_set_font(slide=1, shape='Title', font_name='Arial')``). Vague phrases ("switch to GUI", "try a different approach", "use the right tool") are NOT acceptable — the planner cannot translate them into runnable code without you. If you genuinely cannot name an action, set category=unclear and explain in root_cause_summary what additional signal you need.
  • When PRIOR DIAGNOSES is non-empty, you MUST either (a) build on the previous diagnosis (cite which prior call's hint failed and explain WHY — quote new evidence from RECENT ATTEMPTS / LAST CLI_RUN / SCREENSHOT) OR (b) explicitly explain in root_cause_summary why the category is changing. A pure category flip across consecutive calls without new observation evidence is hallucination — pick category=unclear instead and ask for the missing signal.
  • When ACTION ROLLUP shows ANY signature with the ``⚠ STRATEGIC CANDIDATE`` flag (≥3 tries, no positive delta), strongly prefer category=strategic — the same tactic is exhausted and the planner needs a different approach. Picking category=tactical here REQUIRES citing concrete new evidence in root_cause_summary that the next tactic under the same strategy will produce a different result (e.g. a freshly-visible UI element you didn't see before, a different stderr from LAST CLI_RUN that fixing addresses, etc.)."""


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class StuckDiagnosis:
    root_cause_summary: str
    category: str  # "tactical" | "strategic" | "info" | "unclear"
    missing_or_wrong: List[str] = field(default_factory=list)
    next_action_hint: str = ""
    tactical_subkind: Optional[str] = None  # None for non-tactical categories

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root_cause_summary": self.root_cause_summary,
            "category": self.category,
            "tactical_subkind": self.tactical_subkind,
            "missing_or_wrong": list(self.missing_or_wrong),
            "next_action_hint": self.next_action_hint,
        }


# Legacy 6-category → new 4-category mapping. Used when reading cached
# diagnoses written by older diagnose runs (the cache lives on
# focus_outcome.last_stuck_diagnosis). Keeps cross-version replay sane.
_LEGACY_CATEGORY_MAP = {
    "info_gap":       ("info",      None),
    "wrong_info":     ("info",      None),
    "wrong_target":   ("tactical",  "wrong_target"),
    "wrong_env":      ("tactical",  "blocking_state"),
    "strategy_unfit": ("strategic", None),
    "unclear":        ("unclear",   None),
}

_NEW_CATEGORIES = {"tactical", "strategic", "info", "unclear"}
_VALID_SUBKINDS = {
    "wrong_target", "target_missing", "target_blocked",
    "blocking_state", "focus_lost", "execution_error",
}


def _normalize_category(
    raw_category: str, raw_subkind: Optional[str],
) -> tuple:
    """Accept either the new 4-cat enum or the old 6-cat enum (from
    cached diagnoses). Returns (category, tactical_subkind).
    Unknown values map to ('unclear', None)."""
    c = (raw_category or "").strip().lower()
    s = (raw_subkind or "").strip().lower() if raw_subkind else None
    if c in _NEW_CATEGORIES:
        # New schema: validate subkind only when tactical.
        if c == "tactical":
            return ("tactical", s if s in _VALID_SUBKINDS else None)
        return (c, None)
    if c in _LEGACY_CATEGORY_MAP:
        return _LEGACY_CATEGORY_MAP[c]
    return ("unclear", None)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def replans_on_focus(strategy_history: List[Any], focus_id: Optional[str]) -> int:
    """Count REPLAN events whose ``focus_outcome_id`` matches ``focus_id``.
    'install' events are excluded (no prior strategy to replan from);
    'same' and 'switch' both count as "the planner felt this wasn't
    working".
    """
    if not focus_id:
        return 0
    return sum(
        1 for ev in (strategy_history or [])
        if getattr(ev, "focus_outcome_id", None) == focus_id
        and getattr(ev, "kind", "") in ("same", "switch")
    )


def _truncate(text: Any, n: int = _TRUNC_TEXT) -> str:
    s = str(text)
    return s if len(s) <= n else (s[:n] + " …")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, dict):
        return d.get(key, default)
    return default


# ──────────────────────────────────────────────────────────────────────
# Action-catalogue compactor — feeds diagnose's AVAILABLE ACTIONS section
# ──────────────────────────────────────────────────────────────────────


# Base GUI/CLI actions that are always available regardless of domain.
_BASE_ACTIONS = [
    ("click",         "click(start_box='(x,y)') — left-click at pixel (x,y)"),
    ("right_single",  "right_single(start_box='(x,y)') — right-click at pixel"),
    ("left_double",   "left_double(start_box='(x,y)') — double-left-click"),
    ("drag",          "drag(start_box='(x1,y1)', end_box='(x2,y2)') — drag mouse"),
    ("hotkey",        "hotkey(key='ctrl+l') — keyboard shortcut (e.g. 'escape', 'tab', 'alt+f4')"),
    ("type",          "type(content='text') — type literal text at focused field; append \\n to submit"),
    ("scroll",        "scroll(start_box='(x,y)', direction='down|up|left|right')"),
    ("wait",          "wait() — 5s sleep + fresh screenshot (use after page-load / animation)"),
    ("cli_run",       "cli_run('<bash command>') — one-shot shell on the VM (sed/gsettings/pip/file ops/launch app)"),
    ("open_app",      "open_app(app='<name>') — launch a named app (chrome, vlc, gimp, libreoffice_calc, ...)"),
]


_ACTION_LINE_RE = re.compile(r"^\*\s+([a-z][a-z0-9_]+)\s+[—–-]\s+(.+)$", re.MULTILINE)


def _extract_action_one_liners(schema_text: str) -> List[tuple]:
    """Pull (action_name, one-line-description) pairs from an
    ``action_schema_block()`` text. The schema follows the pattern
    ``* <name> — <description>`` with optional JSON-template lines
    indented below. We grab the first description line only — diagnose
    just needs the name + purpose, not the full param signature."""
    out: List[tuple] = []
    seen = set()
    for m in _ACTION_LINE_RE.finditer(schema_text or ""):
        name = (m.group(1) or "").strip()
        desc = (m.group(2) or "").strip().rstrip(",;:")
        # Some descriptions start with continuation prefix like "—" or "-"
        if not name or name in seen:
            continue
        seen.add(name)
        # Truncate to one short line (description sometimes runs long).
        out.append((name, desc[:160]))
    return out


def _render_available_actions_section(domain: Optional[str]) -> str:
    """Render the AVAILABLE ACTIONS catalogue for diagnose.

    Lists base GUI/CLI actions always present, then the domain-specific
    structured ops when ``domain`` is one of the LibreOffice variants.
    Tries to keep total length manageable (target ≤2k chars) so the
    diagnostician's prompt stays focused.
    """
    lines: List[str] = ["  # Always available (GUI + shell):"]
    for name, desc in _BASE_ACTIONS:
        lines.append(f"  - {name:<15} : {desc}")

    dom_lc = (domain or "").lower()
    try:
        if dom_lc == "libreoffice_calc":
            from mm_agents.structagent.actions.calc.actions import (
                action_schema_block,
            )
            ops = _extract_action_one_liners(action_schema_block())
            lines.append("")
            lines.append("  # Calc structured actions (JSON kwargs; "
                         "framework builds the Python):")
            for name, desc in ops:
                lines.append(f"  - {name:<32} : {desc}")
        elif dom_lc == "libreoffice_impress":
            from mm_agents.structagent.actions.impress.actions import (
                action_schema_block,
            )
            ops = _extract_action_one_liners(action_schema_block())
            lines.append("")
            lines.append("  # Impress structured actions (JSON kwargs):")
            for name, desc in ops:
                lines.append(f"  - {name:<32} : {desc}")
        elif dom_lc == "libreoffice_writer":
            from mm_agents.structagent.actions.writer.actions import (
                action_schema_block,
            )
            ops = _extract_action_one_liners(action_schema_block())
            lines.append("")
            lines.append("  # Writer structured actions (JSON kwargs):")
            for name, desc in ops:
                lines.append(f"  - {name:<32} : {desc}")
    except Exception:
        # Any import / extract failure is non-fatal; the base block above
        # is still useful and the LLM will still produce a hint.
        pass

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Snapshot collectors — each returns text suitable for the LLM prompt.
# ──────────────────────────────────────────────────────────────────────


def _render_goal_section(focus_outcome: Any) -> str:
    lines: List[str] = [
        f"  id:            {focus_outcome.id}",
        f"  description:   {_truncate(focus_outcome.description, 240)}",
        f"  evidence_hint: {_truncate(focus_outcome.evidence_hint, 240)}",
    ]
    verify = getattr(focus_outcome, "verify", None)
    if verify is None:
        lines.append("  verify spec:   (none — LLM-judged outcome)")
        return "\n".join(lines)
    kind = getattr(verify, "kind", "?")
    lines.append(f"  verify spec:   kind={kind}")
    if kind == "file_grep":
        lines.append(f"    file_path: {getattr(verify, 'file_path', '?')}")
        amm = getattr(verify, "all_must_match", True)
        plist = list(getattr(verify, "patterns", []) or [])
        lines.append(f"    patterns ({'all must match' if amm else 'any may match'}):")
        for i, p in enumerate(plist[:6], 1):
            lines.append(f"      {i}. {p}")
    elif kind == "url_match":
        lines.append(f"    url_pattern: {getattr(verify, 'url_pattern', '?')}")
    elif kind == "a11y_match":
        if getattr(verify, "tag", None):
            lines.append(f"    tag:            {verify.tag}")
        if getattr(verify, "text_contains", None):
            lines.append(f"    text_contains:  {list(verify.text_contains)[:6]}")
        if getattr(verify, "state_contains", None):
            lines.append(f"    state_contains: {list(verify.state_contains)[:6]}")
    elif kind == "shell_command":
        cmd = list(getattr(verify, "command", None) or [])
        lines.append(f"    command: {cmd[:6]}")
        exp = list(getattr(verify, "expected_substring", None) or [])
        if exp:
            lines.append(f"    expected_substring: {exp[:6]}")
    elif kind in ("calc_verify", "impress_verify", "writer_verify"):
        checks_attr = (
            "calc_checks" if kind == "calc_verify"
            else "impress_checks" if kind == "impress_verify"
            else "writer_checks"
        )
        checks = list(getattr(verify, checks_attr, None) or [])
        lines.append(f"    {checks_attr} ({len(checks)} checks):")
        for i, c in enumerate(checks[:5], 1):
            lines.append(f"      {i}. {_truncate(c, 160)}")
    return "\n".join(lines)


# Match literal ``Action: <verb>(<args>)`` text in an actor response.
# The first capture group is the action verb; the second is the
# parens-balanced arg blob (truncated to a reasonable size).
_ACTION_SIGN_RE = re.compile(r"Action:\s*([a-z_][a-z0-9_]*)\(([^)]{0,300})\)",
                              re.IGNORECASE)

# Keyword arg names whose VALUE is content/data rather than target
# identity. When generating a rollup signature we drop these arg's
# values (replacing them with ``…``) so calls that hit the same target
# with different payloads cluster together — that's the pattern we
# need to catch ("5x impress_set_text on the SAME shape with different
# text" → 1 group of 5 in rollup, not 5 groups of 1).
_VALUE_ARG_NAMES = frozenset({
    "text", "content", "value", "values", "new_text", "formula",
    "formula_template", "header", "body", "message", "name",
    "font_name", "color", "url", "path", "filename", "command",
})

# Within the args blob, find ``<name>=<value>`` pairs. The value can
# be a quoted string, a number, or a bracketed list — we just want to
# blank out content-like values, so a tolerant pattern is fine.
_KWARG_RE = re.compile(
    r"(?P<name>[a-z_][a-z0-9_]*)\s*=\s*"
    r"(?P<value>(?:'[^']*'|\"[^\"]*\"|\[[^\]]*\]|[^,)]+))",
    re.IGNORECASE,
)


def _extract_action_signature(action_text: str) -> Optional[str]:
    """Extract a canonical ``<verb>(<args>)`` signature from an actor
    response. Returns None if the response has no ``Action: ...``
    payload.

    Normalization: blank out kwarg values for content/data args
    (text, content, value, formula, etc.) so the SAME target with
    different payloads clusters into one rollup row. Keep target-
    identifying args (slide, shape, sheet, column, a1_ref, key,
    coords, etc.) verbatim so distinct targets stay separate.
    """
    if not action_text:
        return None
    m = _ACTION_SIGN_RE.search(action_text)
    if not m:
        return None
    verb = m.group(1).strip().lower()
    args = m.group(2).strip()
    # Drop kwarg values for content-like args.
    def _sub(m2):
        name = m2.group("name").lower()
        if name in _VALUE_ARG_NAMES:
            return f"{m2.group('name')}=…"
        return m2.group(0)
    args = _KWARG_RE.sub(_sub, args)
    # Normalize whitespace.
    args = re.sub(r"\s+", " ", args)
    return f"{verb}({args[:80]})"


def _render_action_rollup_section(
    timeline: List[Any], focus_id: str, *, max_n: int = 12,
) -> str:
    """Group the last N actions on this focus by their literal
    ``<verb>(<args>)`` signature and report the count + observable
    deltas (perceiver's ``overall`` verdict).

    Without this, RECENT ATTEMPTS surfaces 5 raw action strings and the
    LLM has to manually count "tried this 5x". The v_p audit showed one
    case where the planner emitted ``impress_set_text`` with the SAME
    shape arg 27 times — invisible in the raw list. This rollup makes
    it impossible to miss.
    """
    if not timeline:
        return "  (no actions on record this strategy)"
    # Walk newest → oldest, collect last ``max_n`` step records.
    flat: List[Any] = []
    for ev in reversed(timeline):
        for s in reversed(getattr(ev, "steps", []) or []):
            flat.append(s)
            if len(flat) >= max_n:
                break
        if len(flat) >= max_n:
            break
    flat.reverse()
    # Group by signature.
    groups: Dict[str, Dict[str, Any]] = {}
    for s in flat:
        action_text = (getattr(s, "actor_action_summary", "") or "").strip()
        sig = _extract_action_signature(action_text)
        if not sig:
            continue
        snap = getattr(s, "snapshot", None)
        overall = (getattr(snap, "overall", None)
                   or getattr(snap, "subgoal_done", None)) if snap else None
        overall_s = str(overall).strip().lower() if overall else "unknown"
        g = groups.setdefault(sig, {
            "count": 0,
            "first_step": getattr(s, "step_idx", "?"),
            "last_step": getattr(s, "step_idx", "?"),
            "outcomes": {"yes": 0, "no": 0, "partial": 0, "unknown": 0},
        })
        g["count"] += 1
        g["last_step"] = getattr(s, "step_idx", g["last_step"])
        if overall_s in g["outcomes"]:
            g["outcomes"][overall_s] += 1
        else:
            g["outcomes"]["unknown"] += 1
    if not groups:
        return "  (no parseable Action: lines in recent attempts)"
    # Sort by count desc — most-tried signatures first (those are the
    # exhaustion candidates the diagnostician must flag as strategic).
    ranked = sorted(groups.items(),
                     key=lambda kv: (-kv[1]["count"], kv[1]["last_step"]))
    lines: List[str] = []
    for sig, g in ranked:
        cnt = g["count"]
        first = g["first_step"]
        last = g["last_step"]
        out = g["outcomes"]
        # Compact delta summary: only mention non-zero buckets.
        delta_bits = []
        for k in ("yes", "partial", "no", "unknown"):
            if out[k] > 0:
                delta_bits.append(f"{k}={out[k]}")
        delta_str = ", ".join(delta_bits)
        # Flag exhaustion: 3+ tries with no positive delta = strategic candidate.
        flag = ""
        if cnt >= 3 and out["yes"] == 0 and out["partial"] == 0:
            flag = "  ⚠ STRATEGIC CANDIDATE (≥3 tries, no positive delta)"
        lines.append(
            f"  ▸ {sig}"
        )
        lines.append(
            f"      tried {cnt}× (steps {first}..{last}); deltas: {delta_str}{flag}"
        )
    return "\n".join(lines)


def _render_prior_diagnoses_section(focus_outcome: Any) -> str:
    """Render the last 3 diagnoses on the SAME focus outcome. Without
    this, the diagnostician cannot tell that it just said the opposite
    thing 2 steps ago — and v_p data showed 11-call foci flip-flopping
    between wrong_target and wrong_env on adjacent calls.
    """
    history = list(getattr(focus_outcome, "recent_stuck_diagnoses", []) or [])
    if not history:
        return "  (no prior diagnoses on this focus — this is the first call)"
    lines: List[str] = []
    for i, d in enumerate(history, 1):
        step = d.get("step_idx", "?")
        cat = d.get("category", "?")
        sub = d.get("tactical_subkind")
        cat_str = f"{cat}/{sub}" if sub else cat
        hint = (d.get("next_action_hint") or "").strip()
        lines.append(
            f"  #{i} @ step {step}: category={cat_str}"
        )
        if hint:
            lines.append(f"      hint: {_truncate(hint, 160)}")
    return "\n".join(lines)


def _render_cli_run_section(cli_run_output: Optional[Dict[str, Any]]) -> str:
    """Render the most recent cli_run/open_app result so diagnose can
    see script crashes, returncodes, stdout / stderr from the actor's
    own commands. WITHOUT this, diagnose cannot distinguish a script
    that crashed at line 5 from one that ran but produced wrong output.

    ``cli_run_output`` is the dict persisted to /tmp/_last_cli_run.json
    by the cli_run wrappers (see actions/handlers/cli_run_helpers.py:71). Shape:
        {command, mode, returncode, stdout, stderr, timestamp}
    Truncated heads/tails so the LLM sees the salient ends.
    """
    if not isinstance(cli_run_output, dict) or not cli_run_output:
        return "  (no cli_run / open_app executed since last reset)"
    cmd = (cli_run_output.get("command") or "").strip()
    mode = cli_run_output.get("mode") or "?"
    rc = cli_run_output.get("returncode")
    stdout = (cli_run_output.get("stdout") or "").strip()
    stderr = (cli_run_output.get("stderr") or "").strip()
    lines: List[str] = [
        f"  command:    {_truncate(cmd, 240)!r}",
        f"  mode:       {mode}",
        f"  returncode: {rc if rc is not None else '(none — backgrounded)'}",
    ]
    # stdout: show head + tail if long (script may print progress at
    # head and the error at the tail).
    if stdout:
        if len(stdout) > 800:
            lines.append(f"  stdout (head, {len(stdout)} chars total):")
            lines.append("    " + stdout[:600].replace("\n", "\n    "))
            lines.append(f"  stdout (tail):")
            lines.append("    " + stdout[-200:].replace("\n", "\n    "))
        else:
            lines.append(f"  stdout:")
            lines.append("    " + stdout.replace("\n", "\n    "))
    else:
        lines.append("  stdout:     (empty)")
    if stderr:
        # stderr is usually short; show all.
        lines.append(f"  stderr ({len(stderr)} chars):")
        lines.append("    " + stderr[:1000].replace("\n", "\n    "))
    else:
        lines.append("  stderr:     (empty)")
    return "\n".join(lines)


def _render_observation_section(observation: Dict[str, Any]) -> str:
    """Render a11y-derived signals as supplementary text. Screenshot
    is passed separately as an image_url message — this is the textual
    complement, not the primary observation.

    Uses ``linearize_obs_a11y`` to turn the raw AT-SPI XML into
    tab-separated rows (``tag\\ttext\\tstate``) — the same format the
    perceiver consumes. We then filter to interactive-widget tags with
    non-empty text. The prior implementation looked for
    ``<node role=...>`` regex against the raw XML — but OSWorld AT-SPI
    XML uses the tag itself as the role (``<push-button name="...">``)
    with no ``role=`` attribute, so that regex matched 0 / 681 calls in
    the v_p audit.
    """
    if not isinstance(observation, dict):
        return "  (no observation dict)"
    # Lazy import: utils.a11y depends on autoglm which may not load in
    # some replay contexts. If linearization fails we still fall back
    # below to the raw a11y splitlines.
    linearized: Optional[str] = None
    try:
        from mm_agents.structagent.utils.a11y import linearize_obs_a11y
        linearized = linearize_obs_a11y(observation, max_items=300)
    except Exception:
        linearized = None
    a11y_raw = (observation.get("accessibility_tree")
                 or observation.get("a11y_tree") or "")
    if not isinstance(a11y_raw, str):
        a11y_raw = ""
    if not (linearized and linearized.strip()) and not a11y_raw.strip():
        return "  (no a11y tree captured this turn — rely on the screenshot above)"

    # Interactive AT-SPI widget tags. These are roles a planner / actor
    # can click, type into, toggle, or navigate to. Structural tags
    # (panel, section, filler, separator) are EXCLUDED — they add noise
    # without helping the diagnostician understand what the user can
    # interact with. ``label`` and ``heading`` are included because they
    # often carry the visible target text (e.g. "Time range: All time")
    # even when the actionable control is anonymous.
    _IFACE_TAGS = {
        "push-button", "toggle-button", "menu-item", "menu",
        "entry", "combo-box", "link", "check-box", "radio-button",
        "tab", "page-tab", "dialog", "label", "heading", "text",
    }
    lines: List[str] = []
    extracted: List[str] = []
    seen = set()
    if linearized:
        for ln in linearized.splitlines():
            parts = ln.split("\t")
            if len(parts) < 2:
                continue
            tag = parts[0].strip().lower()
            name = parts[1].strip()
            state = parts[2].strip() if len(parts) > 2 else ""
            if tag not in _IFACE_TAGS:
                continue
            if not name:
                continue
            key = (tag, name)
            if key in seen:
                continue
            seen.add(key)
            # State is informative — keep "enabled" / "focused" / etc.;
            # treat "-" as no-state-info.
            state_str = f"  [{state}]" if (state and state != "-") else ""
            extracted.append(f"    {tag:<14}  \"{name[:80]}\"{state_str}")
            if len(extracted) >= _MAX_OBS_A11Y_LINES:
                break

    if extracted:
        lines.append("  a11y interactive widgets (tag / name / state):")
        lines.extend(extracted)
    else:
        # Fall back to first ~8 non-empty content-bearing lines, raw,
        # so the LLM at least sees the tree shape; the screenshot
        # carries the real signal.
        src = linearized or a11y_raw
        snippet: List[str] = []
        for ln in src.splitlines():
            s = ln.strip()
            if not s or s.startswith("<?xml") or "xmlns" in s[:30]:
                continue
            snippet.append("    " + s[:200])
            if len(snippet) >= 8:
                break
        if snippet:
            lines.append("  a11y (no interactive widgets matched; raw head):")
            lines.extend(snippet)
        else:
            lines.append("  (no interactive widgets found in a11y — likely "
                         "the active window's tree is empty or transient; "
                         "use the screenshot above as ground truth)")
    return "\n".join(lines)


def _render_recent_attempts_section(
    timeline: List[Any], focus_outcome_id: str, *, max_n: int = _MAX_RECENT_STEPS,
) -> str:
    """Walk the timeline back-to-front, collect the last ``max_n`` step
    records (regardless of which event they belong to). For each step
    surface: step_idx, actor_action_summary, and the perceiver overall
    verdict + relevant_controls count when present (this is the cheapest
    "what observation did this action produce?" signal we have).
    """
    rows: List[Dict[str, Any]] = []
    if not timeline:
        return "  (no recent attempts on record)"
    # Flatten step records in reverse chronological order.
    flat: List[Any] = []
    for ev in reversed(timeline):
        for s in reversed(getattr(ev, "steps", []) or []):
            flat.append(s)
            if len(flat) >= max_n:
                break
        if len(flat) >= max_n:
            break
    # Re-sort ascending by step_idx for natural read.
    flat.sort(key=lambda s: getattr(s, "step_idx", 0))
    for s in flat:
        rows.append({
            "step_idx": getattr(s, "step_idx", "?"),
            "action": (getattr(s, "actor_action_summary", "") or "").strip(),
            "snapshot": getattr(s, "snapshot", None),
            "planner_decision": getattr(s, "planner_decision", None),
        })
    if not rows:
        return "  (no recent attempts on record)"
    lines: List[str] = []
    for r in rows:
        lines.append(
            f"  step {r['step_idx']}: action = {_truncate(r['action'], 200)!r}"
        )
        snap = r["snapshot"]
        if snap is not None:
            overall = getattr(snap, "overall", None) or getattr(snap, "subgoal_done", None)
            ctrls = getattr(snap, "relevant_controls_count", None)
            blockers = getattr(snap, "blockers_count", None)
            bits: List[str] = []
            if overall is not None:
                bits.append(f"perceiver overall = {overall}")
            if ctrls is not None:
                bits.append(f"relevant_controls = {ctrls}")
            if blockers is not None:
                bits.append(f"blockers = {blockers}")
            if bits:
                lines.append(f"           observation: {'; '.join(bits)}")
        if r["planner_decision"]:
            lines.append(f"           planner_decision: {r['planner_decision']}")
    return "\n".join(lines)


def _screenshot_data_url_from_observation(
    observation: Dict[str, Any],
) -> Optional[str]:
    """Pull the current screenshot bytes from an OSWorld observation
    dict and return a ``data:image/png;base64,...`` URL ready to embed
    in an OpenAI-style multimodal message. Returns None when no usable
    screenshot field is present so the caller can fall back to text.
    """
    if not isinstance(observation, dict):
        return None
    raw = observation.get("screenshot")
    if raw is None:
        # Some replay paths key the field differently.
        raw = observation.get("screenshot_bytes")
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("data:image"):
            return s
        # Bare base64 string — wrap as PNG data URL.
        return f"data:image/png;base64,{s}"
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            b64 = base64.b64encode(bytes(raw)).decode("ascii")
            return f"data:image/png;base64,{b64}"
        except Exception:
            return None
    return None


def _screenshot_digest_from_observation(
    observation: Dict[str, Any],
) -> str:
    """Short hash of the current screenshot bytes for the cache
    signature. Same screen → same digest → cache hit. Returns
    ``"no_screenshot"`` when the observation lacks bytes."""
    if not isinstance(observation, dict):
        return "no_screenshot"
    raw = observation.get("screenshot") or observation.get("screenshot_bytes")
    if raw is None:
        return "no_screenshot"
    if isinstance(raw, str):
        # base64 or data URL — hash the string itself
        return _digest(raw)
    if isinstance(raw, (bytes, bytearray, memoryview)):
        try:
            return hashlib.sha256(bytes(raw)).hexdigest()[:16]
        except Exception:
            return "no_screenshot"
    return "no_screenshot"


def _render_recovery_trigger_section(
    stuck_category: Optional[str], stuck_reason: Optional[str],
) -> str:
    """One-line summary of which 6-rule recovery condition fired this
    turn. Empty marker when the caller didn't pass these (back-compat
    for any test fixture; LLM still works without it)."""
    if not stuck_category and not stuck_reason:
        return "  (no recovery-trigger info supplied — treat as generic stuck)"
    lines: List[str] = []
    if stuck_category:
        lines.append(f"  category: {stuck_category}")
    if stuck_reason:
        lines.append(f"  reason:   {_truncate(stuck_reason, 360)}")
    return "\n".join(lines)


def _render_verifier_trace_section(focus_outcome: Any) -> str:
    trace = dict(getattr(focus_outcome, "last_verifier_trace", {}) or {})
    if not trace:
        return "  (no verifier trace recorded yet)"
    lines: List[str] = []
    vr = trace.get("verdict_reason") or ""
    if vr:
        lines.append(f"  verdict_reason: {_truncate(vr, 260)}")
    if "matched_row" in trace:
        mr = trace.get("matched_row")
        lines.append(f"  matched_row: {_truncate(mr or '(none)', 240)}")
    if "url_extracted" in trace:
        lines.append(f"  url_extracted: {_truncate(trace.get('url_extracted'), 240)}")
    if "exit_code" in trace:
        lines.append(f"  exit_code: {trace.get('exit_code')}")
    ppm = trace.get("per_pattern_matches") or []
    if ppm:
        lines.append("  per_pattern_matches:")
        for entry in ppm[:5]:
            pat = entry.get("pattern", "?")
            n = len(entry.get("matched_lines") or [])
            lines.append(f"    {pat!r}: {n} match(es)")
    head = (trace.get("stdout_head") or "").strip()
    if head:
        first_line = head.splitlines()[0][:200]
        lines.append(f"  stdout_head: {first_line}")
    return "\n".join(lines) or "  (trace present but no salient fields)"


# ──────────────────────────────────────────────────────────────────────
# Cache signature + parse
# ──────────────────────────────────────────────────────────────────────


def _signature(
    *, focus_outcome_id: str,
    observation_text: str, recent_attempts_text: str,
    verifier_trace_text: str, recovery_trigger_text: str,
    screenshot_digest: str,
) -> str:
    """Hash the LLM input slice. Identical state → same signature → cache hit.

    A different ``stuck_category`` (or reason) OR a different screenshot
    produces a different signature: each is part of what the LLM is asked
    to reason over, so a fresh diagnosis is warranted.
    """
    blob = "|".join([
        focus_outcome_id,
        _digest(observation_text),
        _digest(recent_attempts_text),
        _digest(verifier_trace_text),
        _digest(recovery_trigger_text),
        screenshot_digest,
    ])
    return _digest(blob)


def _parse_response(raw: str) -> Optional[StuckDiagnosis]:
    """Tolerant JSON parse with fence/prose stripping."""
    if not raw:
        return None
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    summary = str(obj.get("root_cause_summary") or "").strip()
    raw_category = str(obj.get("category") or "unclear").strip().lower()
    raw_subkind = obj.get("tactical_subkind")
    missing = obj.get("missing_or_wrong") or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    hint = str(obj.get("next_action_hint") or "").strip()
    if not summary or not raw_category:
        return None
    # Map legacy 6-cat → new 4-cat enum + validate subkind.
    category, subkind = _normalize_category(raw_category, raw_subkind)
    # Bound the field sizes so prompt rendering stays compact.
    return StuckDiagnosis(
        root_cause_summary=summary[:600],
        category=category,
        tactical_subkind=subkind,
        missing_or_wrong=[str(item)[:200] for item in missing[:5]],
        next_action_hint=hint[:300],
    )


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────


def diagnose_stuck(
    *,
    focus_outcome: Any,
    ledger: Any,
    observation: Dict[str, Any],
    timeline: List[Any],
    call_llm: Callable,
    model: Optional[str],
    stuck_category: Optional[str] = None,
    stuck_reason: Optional[str] = None,
    results_dir: Optional[str] = None,
    step_idx: int = 0,
    current_domain: Optional[str] = None,
    last_cli_run_output: Optional[Dict[str, Any]] = None,
) -> Optional[StuckDiagnosis]:
    """Run the LLM diagnose, cache the result on the focus outcome, and
    persist prompt + response to disk for debug. Returns None when the
    call fails or output can't be parsed (caller should skip rendering
    the block in that case).

    ``stuck_category`` + ``stuck_reason`` are which 6-rule recovery
    condition fired (actor_failure / done_rejected / budget_exhausted /
    replan_cadence / queue_empty_no_done) and its one-line reason. Fed
    to the LLM so the diagnosis can be scoped tactically — e.g.
    replan_cadence pulls focus to wrong_target / wrong_env not
    strategy_unfit.
    """
    if focus_outcome is None:
        return None

    # 1. Build prompt sections.
    goal_section = _render_goal_section(focus_outcome)
    actions_section = _render_available_actions_section(current_domain)
    cli_section = _render_cli_run_section(last_cli_run_output)
    prior_diag_section = _render_prior_diagnoses_section(focus_outcome)
    rollup_section = _render_action_rollup_section(
        timeline or [], focus_outcome.id,
    )
    obs_section = _render_observation_section(observation or {})
    attempts_section = _render_recent_attempts_section(
        timeline or [], focus_outcome.id,
    )
    trace_section = _render_verifier_trace_section(focus_outcome)
    recovery_section = _render_recovery_trigger_section(
        stuck_category, stuck_reason,
    )

    # Two text blocks bracketing the screenshot. The screenshot lands
    # in between so the LLM sees: goal/trigger/memory → look at
    # screen → recent attempts/verifier trace. Multimodal Qwen3.5-VL
    # processes images interleaved with text.
    text_pre = (
        "GOAL (the outcome currently being attempted):\n"
        f"{goal_section}\n\n"
        "RECOVERY TRIGGER (which 6-rule condition fired this turn):\n"
        f"{recovery_section}\n\n"
        "PRIOR DIAGNOSES ON THIS FOCUS (last 3, oldest first — read "
        "BEFORE classifying so you don't flip-flop):\n"
        f"{prior_diag_section}\n\n"
        "ACTION ROLLUP (recent actions on this focus grouped by literal "
        "signature; ≥3 tries with no positive delta = STRATEGIC "
        "EXHAUSTION):\n"
        f"{rollup_section}\n\n"
        "AVAILABLE ACTIONS (catalogue — your hint must name one of these "
        "by its exact signature):\n"
        f"{actions_section}\n\n"
        "LAST CLI_RUN / OPEN_APP RESULT (most recent shell/script the "
        "actor executed — read this BEFORE the screenshot when the "
        "GOAL involves code/CLI work):\n"
        f"{cli_section}\n\n"
        "CURRENT SCREENSHOT — primary visual evidence of where the agent "
        "is right now:"
    )
    text_post = (
        "CURRENT OBSERVATION (a11y signals; supplementary to the screenshot):\n"
        f"{obs_section}\n\n"
        f"RECENT ATTEMPTS (last {_MAX_RECENT_STEPS} steps):\n"
        f"{attempts_section}\n\n"
        "LAST VERIFIER TRACE on this outcome (why verify currently returns False):\n"
        f"{trace_section}\n"
    )

    # Extract current screenshot, if available, and emit as image_url.
    # Falls back to a placeholder when missing — keeps the prompt
    # working even if the caller passed a screenshot-less observation
    # (tests, replay tools).
    user_content: List[Dict[str, Any]] = [
        {"type": "text", "text": text_pre},
    ]
    screenshot_data_url = _screenshot_data_url_from_observation(observation)
    if screenshot_data_url is not None:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": screenshot_data_url},
        })
    else:
        user_content.append({
            "type": "text",
            "text": "  (no screenshot bytes in this observation — diagnose from text signals only)",
        })
    user_content.append({"type": "text", "text": text_post})

    # 2. Cache check on the focus outcome. Screenshot bytes are folded
    # into the signature so the same prompt text but a different screen
    # still produces a fresh diagnosis (the screen IS the primary
    # signal now).
    sig = _signature(
        focus_outcome_id=focus_outcome.id,
        observation_text=obs_section,
        recent_attempts_text=attempts_section,
        verifier_trace_text=trace_section,
        recovery_trigger_text=recovery_section,
        screenshot_digest=_screenshot_digest_from_observation(observation or {}),
    )
    cached_sig = getattr(focus_outcome, "last_stuck_diagnosis_signature", None)
    cached_dict = getattr(focus_outcome, "last_stuck_diagnosis", None)
    cache_status = "fresh"
    diag: Optional[StuckDiagnosis] = None
    raw: Optional[str] = None

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if cached_sig == sig and isinstance(cached_dict, dict):
        try:
            diag = StuckDiagnosis(**{
                k: v for k, v in cached_dict.items()
                if k in {"root_cause_summary", "category",
                         "missing_or_wrong", "next_action_hint"}
            })
            cache_status = "hit"
        except Exception:
            diag = None

    # 3. Fresh LLM call when cache misses.
    if diag is None:
        try:
            raw = call_llm({
                "model": model,
                "messages": messages,
                "max_tokens": 400,
                "temperature": 0.0,
                "top_p": 0.9,
            }, model)
        except Exception as e:  # noqa: BLE001
            logger.warning("[StuckDiagnosis] LLM call failed: %s", e)
            _persist_to_disk(
                results_dir=results_dir, step_idx=step_idx,
                focus_id=focus_outcome.id, messages=messages,
                raw_response=None, parsed=None,
                cache_status="llm_failed", signature=sig,
                stuck_category=stuck_category,
            )
            return None
        diag = _parse_response(raw or "")
        if diag is None:
            logger.info("[StuckDiagnosis] could not parse LLM response — skipping")
            _persist_to_disk(
                results_dir=results_dir, step_idx=step_idx,
                focus_id=focus_outcome.id, messages=messages,
                raw_response=raw, parsed=None,
                cache_status="parse_failed", signature=sig,
                stuck_category=stuck_category,
            )
            return None
        # Cache the fresh diagnosis on the outcome.
        focus_outcome.last_stuck_diagnosis = diag.to_dict()
        focus_outcome.last_stuck_diagnosis_signature = sig
        # B1.2: maintain a rolling history (last 3) of diagnoses on this
        # focus so the NEXT diagnose can see what we said before. This
        # is what kills the same-focus flip-flop pattern
        # (wrong_target ↔ wrong_env alternation on consecutive steps in
        # the v_p audit) — the model now has to either build on its
        # prior diagnosis or explain why it's flipping.
        recent = list(getattr(focus_outcome, "recent_stuck_diagnoses", []) or [])
        recent.append({
            "step_idx": step_idx,
            "category": diag.category,
            "tactical_subkind": diag.tactical_subkind,
            "next_action_hint": diag.next_action_hint[:180],
        })
        focus_outcome.recent_stuck_diagnoses = recent[-3:]

    # 4. Persist for debug — every successful resolution (fresh OR hit).
    _persist_to_disk(
        results_dir=results_dir, step_idx=step_idx,
        focus_id=focus_outcome.id, messages=messages,
        raw_response=raw, parsed=diag.to_dict(),
        cache_status=cache_status, signature=sig,
        stuck_category=stuck_category,
    )
    logger.info(
        "[StuckDiagnosis] step=%d outcome=%s category=%s (%s)",
        step_idx, focus_outcome.id, diag.category, cache_status,
    )
    return diag


# ──────────────────────────────────────────────────────────────────────
# Rendering for the force_replan prompt
# ──────────────────────────────────────────────────────────────────────


def render_stuck_diagnosis(diag: StuckDiagnosis, *, focus_id: str) -> str:
    """Format the LLM's diagnosis as a single block for the force_replan
    prompt. Compact (so it doesn't displace other prompt sections) but
    structured (so the planner can latch onto each part)."""
    category_str = diag.category
    if diag.category == "tactical" and diag.tactical_subkind:
        category_str = f"{diag.category} / {diag.tactical_subkind}"
    parts: List[str] = [
        f"[Stuck diagnosis — outcome `{focus_id}` (root-cause analysis from a framework-side LLM)]",
        "",
        "Root cause:",
        f"  {diag.root_cause_summary}",
        "",
        f"Category: {category_str}",
    ]
    if diag.missing_or_wrong:
        parts.append("")
        parts.append("Missing or wrong (concrete blockers):")
        for item in diag.missing_or_wrong:
            parts.append(f"  • {item}")
    if diag.next_action_hint:
        parts.append("")
        parts.append("Suggested next move:")
        parts.append(f"  {diag.next_action_hint}")
    parts.append("")
    parts.append(
        "(This is a framework hint. Your next plan should treat the "
        "above 'Missing or wrong' list as concrete blockers to address; "
        "the suggested move is a starting point, not an obligation.)"
    )
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Disk persistence (prompt + response logging for debug / replay)
# ──────────────────────────────────────────────────────────────────────


def _sanitize_messages_for_disk(
    messages: List[Dict[str, Any]],
    *, dir_path: Path, base_fname: str,
) -> List[Dict[str, Any]]:
    """Strip large base64 image_url payloads from messages before JSON
    dump (to keep stuck_diagnosis JSON files in the KB range instead
    of MB). The screenshot bytes are saved as a sidecar PNG so the
    debug viewer can show them; JSON references the sidecar path +
    a sha256 digest.
    """
    out: List[Dict[str, Any]] = []
    img_count = 0
    for msg in messages or []:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            new_content: List[Any] = []
            for item in content:
                if (isinstance(item, dict)
                        and item.get("type") == "image_url"):
                    url = (item.get("image_url") or {}).get("url") or ""
                    # Extract base64 payload + save sidecar PNG.
                    sidecar_name = None
                    digest = None
                    nbytes = None
                    if url.startswith("data:image"):
                        m = re.match(r"data:image/[a-zA-Z]+;base64,(.*)", url)
                        if m:
                            b64 = m.group(1)
                            try:
                                blob = base64.b64decode(b64)
                                digest = hashlib.sha256(blob).hexdigest()[:16]
                                nbytes = len(blob)
                                sidecar_name = (
                                    f"{base_fname}_img{img_count}.png"
                                )
                                (dir_path / sidecar_name).write_bytes(blob)
                                img_count += 1
                            except Exception:
                                pass
                    new_content.append({
                        "type": "image_url_stripped",
                        "sidecar_png": sidecar_name,
                        "sha256_short": digest,
                        "bytes": nbytes,
                        "note": (
                            "image_url base64 payload was stripped from this "
                            "JSON dump; see sidecar PNG for the actual screen"
                        ),
                    })
                else:
                    new_content.append(item)
            out.append({"role": role, "content": new_content})
        else:
            out.append(msg)
    return out


def _persist_to_disk(
    *, results_dir: Optional[str], step_idx: int, focus_id: str,
    messages: List[Dict[str, Any]], raw_response: Optional[str],
    parsed: Optional[Dict[str, Any]], cache_status: str, signature: str,
    stuck_category: Optional[str] = None,
) -> None:
    """Dump the full LLM-call context for offline debug. Layout:

        {results_dir}/stuck_diagnosis/
            step_{N:04d}[_{category}]_{focus_id}.json
            step_{N:04d}[_{category}]_{focus_id}_img0.png   (sidecar)

    The category tag is included so two force_replan triggers at the
    same step (e.g. actor_failure + budget_exhausted in-turn retry)
    each get their own dump instead of overwriting each other.

    Each JSON contains the full text prompt messages, raw LLM response
    (or None on LLM failure), parsed structured result (or None on
    parse failure), cache_status, and the cache signature.
    image_url base64 payloads are stripped from the JSON and saved
    as sidecar PNG files instead, so the dump stays in the KB range.
    """
    if not results_dir:
        return
    try:
        dir_path = Path(results_dir) / "stuck_diagnosis"
        dir_path.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", focus_id)[:80]
        if stuck_category:
            cat_tag = "_" + re.sub(r"[^A-Za-z0-9_-]", "_", stuck_category)[:24]
        else:
            cat_tag = ""
        base_fname = f"step_{step_idx:04d}{cat_tag}_{safe_id}"
        json_fname = f"{base_fname}.json"
        sanitized = _sanitize_messages_for_disk(
            messages, dir_path=dir_path, base_fname=base_fname,
        )
        payload = {
            "step_idx": step_idx,
            "focus_outcome_id": focus_id,
            "signature": signature,
            "cache_status": cache_status,
            "prompt_messages": sanitized,
            "raw_response": raw_response,
            "parsed": parsed,
        }
        (dir_path / json_fname).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        )
    except Exception as e:  # noqa: BLE001
        logger.info("[StuckDiagnosis] persist failed: %s", e)

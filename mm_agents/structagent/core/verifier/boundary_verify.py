"""boundary_verify.py — context-aware milestone verification at boundaries.

Phase 1 of the boundary-verify architecture (the planned replacement for the
per-step KeyNode poll in ``key_node_detector.py``). PURE module — nothing here
touches the loop yet.

WHY THIS EXISTS
  The legacy design freezes a ``VerifySpec`` for every milestone at t=0 (when
  the agent has only the first screenshot) and then re-checks every milestone
  on EVERY step (often a multimodal LLM call per step). Two problems: t=0
  specs are authored with the least information available, and per-step
  checking is expensive and mostly fires mid-subgoal when nothing is expected
  to be done yet.

  This module instead verifies a milestone ONCE, at the boundary where the
  planner itself declares the targeting subgoal done — the moment the most
  context is available. It authors the check ON THE SPOT.

THE "JUDGE-OR-PROBE" CALL (v1)
  One LLM call per boundary, covering all of that subgoal's target outcomes.
  For each outcome the verifier either:
    * judges directly from the current observation (screenshot + a11y +
      perceiver prior + office document-structure block), or
    * requests ONE deterministic PROBE to settle a LATENT fact a screenshot
      cannot show (file content, saved state, document model, active tab URL).
      Probes reuse the proven executors in ``verifiers.py`` via
      ``verify_with_trace``.

  Probe kinds: file_grep, url_match, a11y_match, calc_verify, impress_verify,
  writer_verify (all structurally read-only — the LLM supplies only structured
  args, the framework builds the read), plus ``shell_command`` behind a
  READ-ONLY guard (see ``_shell_probe_is_readonly``). A verifier must never
  change the state it is checking.

  The authored probe is returned on the verdict (``authored_check``) so a
  later DONE-gate re-check can reuse it without spending another LLM call.

CONTRACT
  ``verify_milestone(outcomes, ctx, call_llm, model, env=None, logger=None)
      -> List[MilestoneVerdict]``

  Conservative by design: anything the verifier cannot positively confirm is
  reported ``verified=False`` (keep working) — never rubber-stamp the
  planner's claim. A verifier-infrastructure failure is flagged
  (``error=True``) so the caller can distinguish "not done" from "could not
  check" and pick a fallback policy.

  Duck-typed ``outcomes``: each only needs ``.id``, ``.description``,
  ``.evidence_hint``, ``.app`` (read via ``getattr``) — so tests can pass
  lightweight stubs without constructing a full ``Outcome``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mm_agents.structagent.ledger.core.ledger import VerifySpec
# Module-global so tests can monkeypatch the executor to isolate the fold logic.
from mm_agents.structagent.core.verifier.verifiers import verify_with_trace


# Probe kinds the boundary verifier may request.
PROBE_KINDS: frozenset = frozenset({
    "file_grep", "url_match", "a11y_match",
    "calc_verify", "impress_verify", "writer_verify",
    "shell_command",
})

# ── Per-domain trust map ──────────────────────────────────────────────────────
# Single source for (a) which probe kinds the verifier may author at a boundary in
# this domain and (b) the "trust X / don't trust Y" guidance shown to the LLM.
# Reflects REAL probe capability: shell_command is read-only env/existence only
# (which / pgrep / test / gsettings get / dpkg -l / snap list — NO cat/grep/ls/
# sqlite/unzip/pipes); office docs are ZIP so file_grep can't read them → use
# *_verify (UNO). Excluded kinds are NEVER shown to the model (silent restriction,
# mirroring the actor-side coding gate). Unknown domain → all probes, no guidance.
DOMAIN_TRUST_MAP: Dict[str, Dict[str, Any]] = {
    "chrome": {
        "probes": ("a11y_match", "url_match", "file_grep"),
        "trust": "the visible page / toggle / list state (a11y_match) and the current tab URL (url_match); Bookmarks live in a plaintext JSON file, so file_grep works for bookmark milestones",
        "avoid": "Preferences / Local State (settings) cache in memory and flush to disk LAZILY — a file_grep on them is frequently stale and misleading; History and Cookies are SQLite (not inspectable here) — verify those via the history / settings UI (a11y_match)",
    },
    "libreoffice_calc": {
        "probes": ("calc_verify", "file_grep"),
        "trust": "calc_verify (reads cell values / formulas via UNO — the ground truth)",
        "avoid": "a cell's displayed value is NOT its formula; the .xlsx/.ods file is a ZIP so a plain file_grep finds nothing — use file_grep only for plaintext exports (.csv/.txt)",
    },
    "libreoffice_writer": {
        "probes": ("writer_verify", "file_grep"),
        "trust": "writer_verify (reads paragraph / character properties via UNO)",
        "avoid": "strike-through / bold / style and partial-run application are not reliable in a screenshot; .docx/.odt is a ZIP — file_grep only for plaintext exports",
    },
    "libreoffice_impress": {
        "probes": ("impress_verify", "file_grep"),
        "trust": "impress_verify (reads shape / text properties via UNO)",
        "avoid": "shape names and paragraph indices are not visible on the slide; .pptx/.odp is a ZIP — file_grep only for plaintext exports",
    },
    "gimp": {
        "probes": ("a11y_match", "shell_command"),
        "trust": "the in-app state (a11y_match); for an export / save milestone, shell_command `test -s <path>` confirms the deliverable file exists and is non-empty",
        "avoid": "the canvas shows in-memory edits that may not be exported/flattened yet; the exported image is binary, so its dimensions / format / pixels cannot be inspected here — confirm the file exists and trust the export",
    },
    "vlc": {
        "probes": ("a11y_match", "file_grep", "shell_command"),
        "trust": "the visible playback / UI state (a11y_match); shell_command `pgrep -a vlc` confirms VLC is running with the expected file",
        "avoid": "vlcrc is only written on a clean exit, so a setting toggled in the running app may not be on disk yet — prefer the visible UI",
    },
    "thunderbird": {
        "probes": ("a11y_match",),
        "trust": "the visible UI state (a11y_match): folders, the open message, compose fields",
        "avoid": "the local mail store and config files are not updated in real time and are complex to parse — read the UI, not the files",
    },
    "vs_code": {
        "probes": ("file_grep", "a11y_match", "shell_command"),
        "trust": "the on-disk file (file_grep) and shell_command `test -s` for existence; a11y_match for editor / UI state",
        "avoid": "an open or edited buffer is NOT the same as a saved file; settings.json can lag the UI toggle — confirm the saved file",
    },
    "os": {
        "probes": ("file_grep", "shell_command"),
        "trust": "file_grep on plaintext files and shell_command `test -f/-s/-d` (existence), `gsettings get` (system settings), `pgrep` (process)",
        "avoid": "a file-manager view can be stale — confirm via the filesystem; shell here is read-only env / existence only (no ls/find counting, no cat/sqlite)",
    },
    "multi_apps": {
        "probes": ("file_grep", "a11y_match", "url_match", "calc_verify", "writer_verify", "impress_verify", "shell_command"),
        "trust": "whichever probe matches the END deliverable: file_grep / test for files, *_verify for office docs, a11y_match / url_match for live app state",
        "avoid": "verify the final artifact, not an intermediate app's screen",
    },
}


def allowed_probes_for_domain(domain: str) -> frozenset:
    """Probe kinds the verifier may author in this domain (per DOMAIN_TRUST_MAP).
    Unknown domain → all probes. Excluded kinds are never shown to the model."""
    entry = DOMAIN_TRUST_MAP.get((domain or "").strip())
    return frozenset(entry["probes"]) if entry else PROBE_KINDS


def domain_trust_guidance(domain: str) -> str:
    """The 'trust X / don't trust Y' block for this domain, or '' if unknown."""
    entry = DOMAIN_TRUST_MAP.get((domain or "").strip())
    if not entry:
        return ""
    return (f"GROUND TRUTH for domain '{domain}':\n"
            f"- TRUST: {entry['trust']}.\n"
            f"- DO NOT TRUST (misleading): {entry['avoid']}.")

# Per-kind schema doc rendered into the prompt's probe catalog. build_messages
# emits only the lines for the kinds ALLOWED at this boundary (domain-dependent),
# so an excluded kind never appears as an option to the model.
_PROBE_DOC = {
    "url_match": 'url_match      {"kind":"url_match","url_pattern":"regex against the active tab URL"}',
    "a11y_match": 'a11y_match     {"kind":"a11y_match","text_contains":["..."],"state_contains":["..."],"tag":"..."}',
    "file_grep": 'file_grep      {"kind":"file_grep","file_path":"/abs/path","patterns":["regex",...],"all_must_match":true}',
    "calc_verify": 'calc_verify    {"kind":"calc_verify","calc_checks":[{"op":"cell_value","sheet":"Sheet1","cell":"A1","expected":"..."}]}',
    "impress_verify": 'impress_verify {"kind":"impress_verify","impress_checks":[{...}]}',
    "writer_verify": 'writer_verify  {"kind":"writer_verify","writer_checks":[{...}]}',
    "shell_command": ('shell_command  {"kind":"shell_command","command":["gsettings","get","org.gnome.desktop...","key"],"expected_substring":["value"]}\n'
                      '                 READ-ONLY argv only: which / pgrep / test / "gsettings get" / "dpkg -l" / "snap list". '
                      'No pipes, redirects, interpreters (bash -c / python -c), or state-changing commands.'),
}
_PROBE_ORDER = ["url_match", "a11y_match", "file_grep", "calc_verify",
                "impress_verify", "writer_verify", "shell_command"]

# Cap the a11y text sent to the verifier. Content-heavy pages (esp. chrome)
# produce a11y trees of ~1M chars / 260k+ tokens that blow the model's context
# window (observed: BadRequestError 400, 260645 > 262144). KeyNode caps at
# 6000; the boundary verifier is the SOLE verifier so we keep more (30k chars
# ≈ 7.5k tokens) — generous but safely under the limit alongside the image.
_A11Y_MAX_CHARS = 30000

# Whitelist of VerifySpec fields read from an LLM probe request, per kind.
# Anything else the model emits is dropped — it cannot invent spec fields.
_PROBE_FIELDS: Dict[str, tuple] = {
    "file_grep": ("file_path", "patterns", "all_must_match", "proximity_lines"),
    "url_match": ("url_pattern",),
    "a11y_match": ("tag", "text_contains", "state_contains", "visual_must_hold"),
    "calc_verify": ("calc_checks",),
    "impress_verify": ("impress_checks",),
    "writer_verify": ("writer_checks",),
    "shell_command": ("command", "expected_substring", "forbidden_substring",
                      "expected_exit_code"),
}

# ── shell_command read-only guard ──
# ``VerifySpec.is_valid()`` already hard-whitelists argv[0] to
# {which, dpkg, snap, pgrep, test, gsettings, python3}. Because that set is
# tiny, we guard with an ALLOWLIST refinement (tighter than a denylist — no
# leak): the inherently-read commands pass; the dual-use tools must use a read
# subcommand; the free-form interpreter (python3) is rejected outright (a
# verify probe must not run arbitrary code — document checks belong in
# calc/impress/writer_verify).
_SHELL_ALWAYS_READ: frozenset = frozenset({"which", "pgrep", "test"})
_SHELL_DUAL_READ: Dict[str, frozenset] = {
    "gsettings": frozenset({
        "get", "list-schemas", "list-keys", "list-children",
        "list-recursively", "range", "describe", "help"}),
    "dpkg": frozenset({
        "-l", "-L", "-s", "-S", "--list", "--status", "--listfiles",
        "--search", "--get-selections"}),
    "snap": frozenset({"list", "info", "find", "version", "connections",
                       "services"}),
}


def _shell_probe_is_readonly(command: Any) -> bool:
    """Best-effort read-only guard for a ``shell_command`` PROBE.

    Returns True only when the command provably reads state. ``is_valid``
    already bounds argv[0] to a 7-command whitelist; here ``which``/``pgrep``/
    ``test`` always pass, the dual-use tools (gsettings/dpkg/snap) must invoke
    a read subcommand, and everything else (notably ``python3``) is rejected.
    """
    if not isinstance(command, list) or not command:
        return False
    exe = str(command[0])
    if exe in _SHELL_ALWAYS_READ:
        return True
    if exe in _SHELL_DUAL_READ:
        sub = str(command[1]) if len(command) > 1 else ""
        return sub in _SHELL_DUAL_READ[exe]
    return False                                   # python3 / anything else


# --------------------------------------------------------------------------- #
# Data contracts
# --------------------------------------------------------------------------- #
@dataclass
class VerifyContext:
    """Everything the boundary verifier needs about the current situation.

    A single current screenshot is NOT enough on its own: milestone
    verification is a transition question (did THIS subgoal achieve the
    change?) and often depends on latent state. So we carry the delta and the
    full a11y, and lean on the probe for what no screenshot can show.
    """
    instruction: str = ""                    # the task instruction
    domain: str = ""                         # OSWorld domain (context; GUI-only gating is via gui_only)
    subgoal_text: str = ""                   # the subgoal the planner declared done
    expected_post_state: str = ""            # the subgoal's expected visible post-state
    actor_history: List[str] = field(default_factory=list)  # recent actions this subgoal
    screenshot_b64: Optional[str] = None     # current (most recent) screenshot (base64 PNG)
    prior_screenshots_b64: List[str] = field(default_factory=list)  # earlier frames oldest→newest (workflow context); the current frame is screenshot_b64
    a11y_text: str = ""                      # current full a11y (intent-prefiltered)
    delta_summary: str = ""                  # what changed during this subgoal
    snapshot: Any = None                     # SubgoalSnapshot — opportunistic perceiver prior
    doc_inspect_block: str = ""              # office UNO structure block (authoritative)
    check_recipe_block: str = ""             # retrieved check recipes (verifier memory)


@dataclass
class MilestoneVerdict:
    outcome_id: str
    verified: bool
    reason: str = ""
    evidence: str = ""
    method: str = "obs"                       # "obs" | "probe:<kind>" | "probe-inconclusive->obs"
    authored_check: Optional[VerifySpec] = None  # the probe used (cache on the outcome)
    error: bool = False                       # True iff the verifier itself could not run


# --------------------------------------------------------------------------- #
# System prompt
# --------------------------------------------------------------------------- #
SYSTEM = """\
You are the MILESTONE VERIFIER for a GUI computer-use agent. The planner just \
declared a subgoal complete. For each listed milestone, decide whether it is \
OBJECTIVELY achieved on the CURRENT screen — reason from evidence; never \
rubber-stamp the planner's claim.

THE TASK INSTRUCTION IS THE GROUND TRUTH — NOT THE MILESTONE LABEL.
Milestone labels are often shallow ("flight results displayed", "forms list \
visible", "dates displayed"). Achieving the right TYPE of thing is NOT enough: \
you must verify the SPECIFIC constraints the instruction demands. Traps:
  - "flights ... only those purchasable with miles" → a generic flight list is \
NOT done; the miles-only constraint must be observable.
  - "flights to ORD for tomorrow" → flights for the WRONG date is NOT done.
  - "Next Available dates for Diamond" → availability for some OTHER item is NOT done.
First extract the instruction's specific constraints for the milestone, then \
check EACH constraint against the evidence.

DERIVE FROM THE CONTROL / DATA ROW — NOT FROM LABELS, SEARCH RESULTS, OR BADGES.
A setting / selection / list-state milestone must be judged from the actual \
control or data row, NOT from search-result counts ("Showing N matches"), the \
text typed into a search box, or count badges. A filtered or "0 results" view \
is NOT proof the underlying data changed — the same view appears for an empty \
initial state.

A11Y STATE COLUMN (rows are <tag>\\t<text>\\t<state>; state lists only TRUE flags):
  `checked` / `selected` → the toggle / option is ON / chosen → may satisfy.
  `enabled` alone → the control is merely CLICKABLE, NOT on. `enabled` NEVER \
means a feature is on or a value is set — do not accept it as satisfaction.

INTENT ≠ POST-STATE. "The user just clicked X" is not evidence X took effect; \
require the observable resulting state.

LATENT STATE → PROBE. A screenshot cannot prove a file's saved content, a \
document's model (cell values / fonts / shapes), or the active URL. When a \
milestone hinges on latent state, request ONE read-only check rather than \
guessing from pixels.

NO EVIDENCE → NOT VERIFIED. If you cannot quote concrete current evidence that \
EVERY constraint holds, the milestone is not verified. Be conservative by rule: \
a false "verified" stops the agent too early; a false "not_verified" only makes \
it keep working — prefer the latter.

For EACH milestone output one JSON object:
{
  "outcome_id": "<the id>",
  "constraints": "<the SPECIFIC requirements the instruction demands for this milestone>",
  "observation": "<what the current a11y / screen actually shows for those constraints — quote the row/text>",
  "obs_verdict": "verified" | "not_verified" | "uncertain",
  "reason": "<one sentence tying the observation to the constraints>",
  "evidence": "<short literal quote from a11y / screen / doc block>",
  "probe": null | { "kind": "...", ... }
}
- "obs_verdict" is your judgment from the visible observation ALONE; use \
"uncertain" when you cannot tell from what is visible.
- "probe" is OPTIONAL. To request a deterministic check, pick a low-level probe \
kind from the catalog at the end of the user message. Its result OVERRIDES your \
obs_verdict; obs_verdict is the fallback when the check is inconclusive.

Output ONLY the JSON array, one object per milestone, and nothing else.
"""


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _oid(o: Any) -> str:
    return str(getattr(o, "id", "") or "")


def _b64_to_url(b64: str) -> str:
    if b64.startswith("data:image"):
        return b64
    return f"data:image/png;base64,{b64}"


def _render_snapshot_prior(snap: Any) -> str:
    """Compact perceiver prior (opportunistic — only when PERCEIVER is on).

    Reads a couple of high-signal fields off a SubgoalSnapshot via getattr so
    this module stays decoupled from the perceiver dataclass.
    """
    if snap is None:
        return ""
    parts: List[str] = []
    sc = getattr(snap, "subgoal_check", None)
    overall = getattr(sc, "overall", None) if sc is not None else None
    if overall:
        parts.append(f"perceiver subgoal_check overall: {overall}")
    ts = getattr(snap, "transition_summary", None)
    if ts:
        parts.append(f"perceiver transition: {ts}")
    if not parts:
        return ""
    return (
        "[PERCEIVER pre-distilled prior — another model already looked at this "
        "screen; use it as a starting point and override only with contrary "
        "evidence]\n" + "\n".join(parts)
    )


def spec_from_probe(probe: Any,
                    allowed_kinds: frozenset = PROBE_KINDS) -> Optional[VerifySpec]:
    """Build a validated ``VerifySpec`` from an LLM probe-request dict.

    Returns ``None`` when the kind is disallowed, required fields are missing,
    the model invented unknown fields, the spec fails ``VerifySpec.is_valid()``,
    or (for shell_command) the command is not provably read-only. Defensive by
    construction: only whitelisted fields for the kind are read.
    """
    if not isinstance(probe, dict):
        return None
    kind = probe.get("kind")
    if kind not in PROBE_KINDS or kind not in allowed_kinds:
        return None
    fields = {k: probe.get(k) for k in _PROBE_FIELDS[kind]
              if probe.get(k) is not None}
    try:
        spec = VerifySpec(kind=kind, **fields)
    except TypeError:
        return None
    try:
        ok = spec.is_valid()
    except Exception:
        ok = False
    if not ok:
        return None
    if kind == "shell_command" and not _shell_probe_is_readonly(spec.command):
        return None
    return spec


def _extract_json_array(raw: str) -> Any:
    """Tolerant extraction of the verifier's JSON from an LLM response: strips
    code fences, tries a direct parse, then scans for the first ``[`` (array)
    or ``{`` (single object) and ``raw_decode``s from there (handles prose
    wrappers).

    A bare object is wrapped into a one-element list: when there is a SINGLE
    milestone the model frequently emits ``{...}`` instead of ``[{...}]``,
    which would otherwise be dropped by the ``isinstance(data, list)`` check in
    ``parse_verifier_response`` and silently yield an empty verdict.
    """
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        obj = json.loads(s)
        return [obj] if isinstance(obj, dict) else obj
    except Exception:
        pass
    dec = json.JSONDecoder()
    for opener in ("[", "{"):
        i = s.find(opener)
        while i != -1:
            try:
                obj, _ = dec.raw_decode(s[i:])
                return [obj] if isinstance(obj, dict) else obj
            except Exception:
                i = s.find(opener, i + 1)
    return None


def parse_verifier_response(raw: str, outcomes: List[Any]) -> Dict[str, dict]:
    """Parse the verifier's JSON array into ``{outcome_id: item}``. Items for
    ids not in ``outcomes`` are dropped; malformed input yields ``{}``."""
    out: Dict[str, dict] = {}
    data = _extract_json_array(raw)
    if not isinstance(data, list):
        return out
    valid_ids = {_oid(o) for o in outcomes}
    for item in data:
        if not isinstance(item, dict):
            continue
        oid = str(item.get("outcome_id") or item.get("id") or "")
        if oid and oid in valid_ids:
            out[oid] = item
    return out


def build_messages(outcomes: List[Any], ctx: VerifyContext
                   ) -> List[Dict[str, Any]]:
    """Build the multimodal chat messages for the judge-or-probe call."""
    lines: List[str] = []
    if ctx.instruction:
        lines.append(f"[TASK]\n{ctx.instruction}")
    if ctx.subgoal_text:
        lines.append(f"[SUBGOAL just declared done]\n{ctx.subgoal_text}")
    if ctx.expected_post_state:
        lines.append(f"[EXPECTED post-state]\n{ctx.expected_post_state}")
    if ctx.delta_summary:
        lines.append(f"[WHAT CHANGED during this subgoal]\n{ctx.delta_summary}")
    if ctx.actor_history:
        hist = "\n".join(f"  - {a}" for a in ctx.actor_history[-8:])
        lines.append(f"[RECENT actor actions this subgoal]\n{hist}")
    ol: List[str] = []
    for o in outcomes:
        row = f"  - id={_oid(o)} | {getattr(o, 'description', '')}"
        hint = getattr(o, "evidence_hint", "") or ""
        if hint:
            row += f" | where to look: {hint}"
        app = getattr(o, "app", None)
        if app:
            row += f" | app: {app}"
        ol.append(row)
    lines.append("[MILESTONES to verify]\n" + "\n".join(ol))

    blocks: List[Dict[str, Any]] = [{"type": "text", "text": "\n\n".join(lines)}]
    # Recent screenshots oldest→newest (workflow context, like the KeyNode
    # judge): earlier frames first, then the current. Each is decoded as an
    # IMAGE (~2k tokens), so up to ~5 frames ≈ 10k tokens — safe.
    for _i, _b in enumerate(ctx.prior_screenshots_b64 or []):
        if not _b:
            continue
        blocks.append({"type": "text", "text": (
            f"[earlier screenshot {_i + 1} — before the current frame, "
            "for workflow context]")})
        blocks.append({"type": "image_url", "image_url": {"url": _b64_to_url(_b)}})
    if ctx.screenshot_b64:
        blocks.append({"type": "text", "text": "[CURRENT screenshot]"})
        blocks.append({"type": "image_url",
                       "image_url": {"url": _b64_to_url(ctx.screenshot_b64)}})
    if ctx.a11y_text:
        # Defensive crash-guard ONLY. The caller is expected to pass a
        # subgoal-relevant, intent-prefiltered a11y (see the loop gate); this
        # cap is the last-resort net because the prefilter bounds row COUNT but
        # not per-row text size, and returns input unchanged when rows<=top_n —
        # so a few huge rows could still blow the context window (observed 400).
        a11y = ctx.a11y_text
        if len(a11y) > _A11Y_MAX_CHARS:
            a11y = (a11y[:_A11Y_MAX_CHARS]
                    + f"\n…[a11y truncated, {len(ctx.a11y_text)} total chars]")
        blocks.append({"type": "text", "text": f"[CURRENT a11y tree]\n{a11y}"})
    prior = _render_snapshot_prior(ctx.snapshot)
    if prior:
        blocks.append({"type": "text", "text": prior})
    if ctx.doc_inspect_block:
        blocks.append({"type": "text", "text": (
            "[Document structure block — AUTHORITATIVE for font / color / "
            "size / cell values; supersedes the screenshot for structural "
            "properties]\n" + ctx.doc_inspect_block)})
    if ctx.check_recipe_block:
        blocks.append({"type": "text", "text": ctx.check_recipe_block})
    # Per-domain ground-truth guidance (trust X / don't trust Y) + the low-level
    # probe catalog. Excluded kinds (per DOMAIN_TRUST_MAP) are never shown.
    guidance = domain_trust_guidance(ctx.domain)
    if guidance:
        blocks.append({"type": "text", "text": guidance})
    allowed = allowed_probes_for_domain(ctx.domain)
    probe_lines = [_PROBE_DOC[k] for k in _PROBE_ORDER if k in allowed]
    blocks.append({"type": "text", "text": (
        "Low-level probe kinds (use ONLY if a deterministic check fits; "
        "every probe must READ state, never change it):\n  "
        + "\n  ".join(probe_lines))})
    blocks.append({"type": "text", "text": "Output the JSON array now."})
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": blocks},
    ]


def _run_probe(spec: VerifySpec, ctx: VerifyContext, env: Any,
               obs_verdict: str, obs_reason: str, obs_evidence: str,
               logger: Any = None) -> tuple:
    """Run one probe; fold the (True/False/None) verdict.

    Returns ``(verified_bool, method, reason, evidence)``. ``None``
    (inconclusive) falls back to the LLM's observation judgment.
    """
    try:
        verdict, trace = verify_with_trace(
            spec, env, a11y_text=ctx.a11y_text, instruction=ctx.instruction)
    except Exception as e:                                    # probe must never crash the verify
        if logger:
            logger.warning("[BoundaryVerify] probe %s raised: %s", spec.kind, e)
        verdict, trace = None, {"verdict_reason": f"probe raised: {e}"}
    tr = (trace or {}).get("verdict_reason", "") if isinstance(trace, dict) else ""
    if verdict is True:
        return True, f"probe:{spec.kind}", f"deterministic probe confirmed. {tr}".strip(), obs_evidence
    if verdict is False:
        return False, f"probe:{spec.kind}", f"deterministic probe refuted. {tr}".strip(), obs_evidence
    # None → inconclusive → fall back to the observation judgment.
    return (
        obs_verdict == "verified",
        "probe-inconclusive->obs",
        f"probe inconclusive ({tr}); fell back to observation: {obs_reason}".strip(),
        obs_evidence,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def verify_milestone(outcomes: List[Any], ctx: VerifyContext,
                     call_llm: Any, model: str,
                     env: Any = None, logger: Any = None) -> List[MilestoneVerdict]:
    """Verify the given target outcomes at a planner-declared boundary.

    One LLM call (judge-or-probe) covering all ``outcomes``; per outcome, an
    optional deterministic probe overrides the observation judgment. See the
    module docstring for the full contract. Always returns one verdict per
    outcome (never raises); on verifier-infra failure every verdict is
    ``verified=False, error=True``.
    """
    outcomes = list(outcomes or [])
    if not outcomes:
        return []

    messages = build_messages(outcomes, ctx)
    try:
        raw = call_llm(
            {"model": model, "messages": messages,
             "max_tokens": 1500, "temperature": 0.0, "top_p": 0.9},
            model,
        )
    except Exception as e:
        if logger:
            logger.warning("[BoundaryVerify] LLM call failed: %s", e)
        return [MilestoneVerdict(_oid(o), verified=False, error=True,
                                 reason="verifier LLM call failed")
                for o in outcomes]

    if not raw:
        return [MilestoneVerdict(_oid(o), verified=False, error=True,
                                 reason="verifier returned empty response")
                for o in outcomes]

    parsed = parse_verifier_response(raw, outcomes)
    allowed_kinds = allowed_probes_for_domain(ctx.domain)
    verdicts: List[MilestoneVerdict] = []
    for o in outcomes:
        oid = _oid(o)
        item = parsed.get(oid) or {}
        obs = str(item.get("obs_verdict") or "uncertain").lower()
        reason = str(item.get("reason") or "")
        evidence = str(item.get("evidence") or "")
        probe = item.get("probe")
        spec = spec_from_probe(probe, allowed_kinds=allowed_kinds)
        if spec is not None:
            verified, method, reason2, evidence2 = _run_probe(
                spec, ctx, env, obs, reason, evidence, logger)
            verdicts.append(MilestoneVerdict(
                oid, verified, reason2, evidence2, method, authored_check=spec))
        else:
            # No (valid) probe → trust the observation judgment, conservatively:
            # only an explicit "verified" counts; "uncertain"/anything else → not done.
            verdicts.append(MilestoneVerdict(
                oid, verified=(obs == "verified"),
                reason=reason, evidence=evidence, method="obs"))
    return verdicts

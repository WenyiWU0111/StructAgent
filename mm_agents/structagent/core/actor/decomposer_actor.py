"""DecomposerActor — decomposer LLM with inline grounding, a drop-in
replacement for the UI-TARS actor call.

Contract
--------
``call_decomposer_actor`` takes the same args shape as
``qwen25vl_agent_planner.Qwen25VLAgentPlanner._pa_call_actor`` and returns
a plain string in UI-TARS's output format:

    Thought: <one-line reasoning>
    Action: click(start_box='(994,445)')

    type(content='coffee makers')

    hotkey(key='enter')

The upstream parser ``_pa_action_to_pyautogui_codes`` (line 1180 in
qwen25vl_agent_planner) only cares about three things:

  - The literal token ``Action:`` as a split marker.
  - Actions separated by blank lines (``\\n\\n``).
  - Each action being one of ``click(start_box=...)``,
    ``type(content=...)``, ``hotkey(key=...)``, ``scroll(start_box=...,
    direction=...)``, ``wait()``, ``finished()``, or the bare token
    ``<Impossible>``.

We emit exactly that format so downstream code needs no changes.

Pipeline
--------
1. Call the **decomposer** (Qwen3.5-VL @ :8010, reused planner server)
   with the A2 system prompt + screenshot + subgoal. It returns a JSON
   list of typed step dicts, each carrying its own 0-1000 grid coordinates
   (inline grounding — the decomposer reads the location from the
   screenshot itself; see ``INLINE_GROUNDING_INSTRUCTION``).
2. Convert each step's normalized coordinates to pixels via
   ``_point_to_pixels`` (multiply by the physical screen size).
3. Compose the final UI-TARS-shaped string.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import openai


logger = logging.getLogger("mm_agents.decomposer_actor")


# INLINE grounding directive — appended to the decomposer system prompt by
# actor.py. The decomposer reads each element's location from the screenshot
# itself and emits 0-1000 grid coordinates. Single source of truth: edit
# only here.
INLINE_GROUNDING_INSTRUCTION = (
    "[INLINE GROUNDING MODE]\n"
    "For EVERY action that targets a UI element, in ADDITION to "
    "the existing \"target\" text field you MUST include the "
    "element's location read directly from the screenshot, as "
    "coordinates normalized to a 0-1000 grid (top-left = 0,0; "
    "bottom-right = 1000,1000):\n"
    "  - click / left_double / right_single: add \"point\": [x, y]\n"
    "  - drag: add \"start_point\": [x, y] and \"end_point\": [x, y]\n"
    "  - scroll: add \"anchor_point\": [x, y]\n"
    "Keep the \"target\" text field too (used for logging). Do NOT "
    "add coordinates to non-spatial actions (type / hotkey / wait "
    "/ navigate / cli_run / edit_json / calc_* / impress_* / "
    "writer_* / note_write / extract_info / open_app).\n"
    "Example: {\"action\": \"click\", \"target\": \"the Submit "
    "button\", \"point\": [840, 560]}"
)


# Actions the decomposer is allowed to emit. Keep this list mirrored with
# the seed A2_actor_decomposer_points.txt template AND the UI-TARS action
# space declared in mm_agents/qwen25vl_agent_planner.py so the downstream
# parser (``_pa_action_to_pyautogui_codes``) can digest every variant.
_ALLOWED_ACTIONS = {
    "click",          # single left click
    "left_double",    # double-click (open folder / file)
    "right_single",   # right-click (context menu)
    "drag",           # click-and-drag from start_target to end_target
    "type",           # keyboard typewrite
    "hotkey",         # single key or combo (e.g. "ctrl+a", "enter")
    "scroll",         # scroll in a direction, anchored near anchor_target
    "wait",           # sleep 5s then re-observe
    "impossible",     # current subgoal cannot be achieved here
    "navigate",       # Chrome page navigation via CDP HTTP endpoint.
                      # Schema: {"action":"navigate", "url": "https://...",
                      # "new_tab": false}. One reliable step replacing
                      # the (hotkey ctrl+l → type URL\\n) GUI dance —
                      # needed for cross-site multi-hop tasks.
    "finished",       # Terminal sentinel for text-answer tasks
                      # (WebVoyager / Mind2Web / MMInA). Schema:
                      # {"action":"finished", "answer":"<text or URL>"}.
                      # Detected by agent.py's text-answer sentinel
                      # block in _pa_predict — terminates the episode
                      # and hands the answer to the runner / judge.
    "edit_json",      # safely modify a JSON config file via direct file IO
                      # (json.load → mutate dict/list → json.dump). Replaces
                      # GUI ctrl+a + type sequences which corrupt JSON via
                      # partial-content overwrite. See edit_json_helpers.py.
    "cli_run",        # one-shot bash command via subprocess on the VM
                      # (NOT inside any GUI terminal). Use to launch GUI
                      # apps on a path (`code <path>`, `firefox <url>`),
                      # do non-JSON file edits via sed/cp, or perform
                      # nested JSON mutations edit_json's flat ops can't
                      # express. See cli_run_helpers.py.
    "extract_info",   # batch-read N files from the VM filesystem and
                      # ask the planner LLM to extract structured info
                      # per file (sequential, one LLM call per file).
                      # Bypasses GUI gymnastics for "open file → read
                      # → record in spreadsheet" patterns: agent
                      # receives a JSON aggregate of per-file extracted
                      # values which the planner can then write to the
                      # target sheet/doc via calc_*/writer_*. Use for
                      # receipts, invoices, papers, emails — any local
                      # files whose CONTENT the agent must read to
                      # populate a structured deliverable.
    # ``cli_run_uno`` (raw Python against the OPEN LibreOffice document)
    # was REMOVED — all three LibreOffice domains now go through the
    # structured calc_* / impress_* / writer_* action sets, which the
    # framework deterministically lowers to UNO Python. A raw-UNO escape
    # hatch only mislead the 9B actor into hand-writing UNO.
    "open_app",       # deterministically open / focus an application
                      # (optionally with a file): activates an existing
                      # window via wmctrl, or launches it. Replaces the
                      # brittle GUI dock-click / Alt+Tab dance. See
                      # open_app.py.
    "note_write",     # persist (key, value) to the agent-side Notebook
                      # (see actions/handlers/notebook.py). The notebook is
                      # rendered into BOTH planner and decomposer
                      # prompts every turn. Use to pin any observed
                      # concrete value (URL, name, affiliation, count)
                      # so it survives app switches, outcome REVERTs,
                      # and perceiver misses. Wire form:
                      # ``note_write(key="<k>", value="<v>")``. Value
                      # may be a JSON string for structured data.
    # "finished" was previously allowed, meaning "this subgoal is done".
    # Removed in Phase 2: the planner's <last_subgoal_assessment> is now
    # the single source of truth for done-ness. Keeping "finished" as a
    # back-compat escape hatch below so a model that ignores the prompt
    # and emits it doesn't crash the parser.
}

# Calc structured actions (calc_fill_column, calc_set_cell, …) — actor
# emits FLAT JSON kwargs that the dispatcher in
# ``_step_to_uitars_line`` translates to a calc_xxx(…) wire line.
# These names MUST be in the parser's allow-list or
# ``_parse_decomposer_response`` silently drops them (one of the
# subtler regressions: actor produces valid JSON, parser rejects it,
# actor returns <Impossible> every step, ``traj.jsonl`` never appended).
from mm_agents.structagent.actions.calc.actions import CALC_ACTION_NAMES as _CALC_ACTION_NAMES
_ALLOWED_ACTIONS = _ALLOWED_ACTIONS | set(_CALC_ACTION_NAMES)
# Impress structured actions — same allow-list role as calc_*.
from mm_agents.structagent.actions.impress.actions import IMPRESS_ACTION_NAMES as _IMPRESS_ACTION_NAMES
_ALLOWED_ACTIONS = _ALLOWED_ACTIONS | set(_IMPRESS_ACTION_NAMES)
# Writer structured actions — same allow-list role as calc_* / impress_*.
from mm_agents.structagent.actions.writer.actions import WRITER_ACTION_NAMES as _WRITER_ACTION_NAMES
_ALLOWED_ACTIONS = _ALLOWED_ACTIONS | set(_WRITER_ACTION_NAMES)


def _screenshot_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Base64-encode a PNG as a data URL so we don't depend on file://."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _point_to_pixels(
    pt: Any, screen_w: int, screen_h: int
) -> Optional[Tuple[int, int]]:
    """INLINE-mode only: convert a decomposer-emitted ``[x, y]`` point to
    pixel coords. Coordinate-system auto-detect (same lesson as the offline
    A/B — models emit different grids and ignore prompt instructions):

      - both <= 1.0     -> already normalized [0, 1]
      - both <= 1000.0  -> 0-1000 integer grid (Qwen-VL native)
      - otherwise       -> absolute pixels (divide by screen size)

    Returns None if ``pt`` is missing or malformed, so the caller skips the
    action (never blind-clicks a fallback) — mirroring grounding-fail
    handling in ``_ground_to_pixels``.
    """
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None
    if x <= 1.0 and y <= 1.0:
        xn, yn = x, y
    elif x <= 1000.0 and y <= 1000.0:
        xn, yn = x / 1000.0, y / 1000.0
    else:
        xn, yn = x / screen_w, y / screen_h
    xn = max(0.0, min(1.0, xn))
    yn = max(0.0, min(1.0, yn))
    return int(round(xn * screen_w)), int(round(yn * screen_h))


_CODING_ACTIONS = frozenset({"cli_run", "edit_json"})


def _allowed_actions_for_domain(task_domain: Optional[str]) -> set:
    """``_ALLOWED_ACTIONS`` minus the coding actions (cli_run / edit_json)
    when the task domain is GUI-only (chrome/gimp/thunderbird) and
    ENABLE_DOMAIN_CODING_GATE is on. Defense-in-depth: the A2 prompt already
    omits the coding route for those domains, but this also REJECTS a coding
    action if the decomposer emits one anyway (e.g. nudged by residual memory
    or a stray cross-reference) → surfaces as <Impossible> 'unknown action',
    forcing a GUI retry. Gate off / non-GUI-only domain → full set (current
    behaviour)."""
    try:
        from mm_agents.structagent.config import CAConfig
        from mm_agents.structagent.domain import is_gui_only
        if CAConfig.from_env().domain_coding_gate and is_gui_only(task_domain):
            return _ALLOWED_ACTIONS - _CODING_ACTIONS
    except Exception:
        pass
    return _ALLOWED_ACTIONS


def _parse_decomposer_response_with_diagnostics(
    text: str,
    *,
    task_domain: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Extract step dicts AND collect the names of any actions that
    were rejected because they aren't in ``_ALLOWED_ACTIONS``.

    Returns ``(accepted_steps, rejected_action_names)``. The rejected
    list lets the caller surface a specific failure reason ("actor
    rejected unknown action 'X'") to the planner instead of an opaque
    ``<Impossible>`` — without that signal the planner cannot tell a
    hallucinated action name apart from a UI-grounding miss, and
    typically replans into the same hallucination.
    """
    if not text:
        return [], []
    stripped = text.strip()
    fence = re.match(r"```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    # Prefer a JSON array; tolerate a single bare object {...} (the model
    # sometimes emits a lone action object instead of a one-element array) by
    # wrapping it — same lesson as boundary_verify._extract_json_array.
    start = stripped.find("[")
    end = stripped.rfind("]")
    arr = None
    if start != -1 and end != -1 and end > start:
        try:
            arr = json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            arr = None
    if not isinstance(arr, list):
        ostart = stripped.find("{")
        oend = stripped.rfind("}")
        if ostart != -1 and oend != -1 and oend > ostart:
            try:
                obj = json.loads(stripped[ostart:oend + 1])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                arr = [obj]
    if not isinstance(arr, list):
        return [], []
    synth_names: set = set()
    _allowed = _allowed_actions_for_domain(task_domain)
    out: List[Dict[str, Any]] = []
    rejected_unknown: List[str] = []
    for step in arr:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        if action not in _allowed and action not in synth_names:
            if isinstance(action, str) and action.strip():
                rejected_unknown.append(action.strip())
            continue
        out.append(step)
    return out, rejected_unknown


def _parse_decomposer_response(
    text: str,
    *,
    task_domain: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Backward-compatible wrapper — returns just the accepted steps.
    Use :func:`_parse_decomposer_response_with_diagnostics` when you
    need the rejected-unknowns sidechannel."""
    steps, _ = _parse_decomposer_response_with_diagnostics(
        text, task_domain=task_domain)
    return steps


def _valid_action_names_for_domain(task_domain: Optional[str]) -> List[str]:
    """Return the impress_* / calc_* action set most relevant to the
    current task domain. Used in the <Impossible> reason so the
    planner sees a focused list rather than ALL allowed actions."""
    dom = (task_domain or "").lower()
    if dom == "libreoffice_impress":
        return sorted(_IMPRESS_ACTION_NAMES)
    if dom == "libreoffice_calc":
        return sorted(_CALC_ACTION_NAMES)
    return []


def _impossible_with_reason(rejected_unknown: List[str],
                              task_domain: Optional[str]) -> str:
    """Build an ``<Impossible>`` response that carries a specific
    failure reason when the decomposer emitted unknown action names.

    Output format keeps ``<Impossible>`` as the first token so the
    upstream parser still detects it via the existing case-insensitive
    substring check, with a ``Reason:`` line below that the planner
    surfaces in next-turn ``<stuck_analysis>``.
    """
    if not rejected_unknown:
        return "<Impossible>"
    # Dedupe while preserving order of first occurrence.
    seen: set = set()
    unique = []
    for a in rejected_unknown:
        if a not in seen:
            seen.add(a); unique.append(a)
    rejected_str = ", ".join(repr(a) for a in unique[:5])
    valid = _valid_action_names_for_domain(task_domain)
    valid_hint = ""
    if valid:
        valid_hint = (
            f" Valid actions for {task_domain}: "
            f"{', '.join(valid[:24])}"
            + (" …" if len(valid) > 24 else "")
        )
    return (
        f"<Impossible>\n"
        f"Reason: actor rejected unknown action name(s) {rejected_str} "
        f"— not in the action catalogue.{valid_hint}"
    )


def _escape_single_quotes(text: str) -> str:
    """Mirror qwen25vl_agent_planner._pa_escape_single_quotes so the
    emitted type(content='...') string survives ast.parse downstream."""
    # Replace single quotes with escaped single quotes (backslash + quote)
    return text.replace("\\", "\\\\").replace("'", "\\'")


def _call_decomposer(
    *,
    client: openai.OpenAI,
    model: str,
    system_prompt: str,
    subgoal: str,
    screenshot_b64: str,
    user_instruction: str,
    actor_history: List[Dict[str, Any]],
    doc_inspect_block: Optional[str] = None,
    last_cli_run_output: Optional[Dict[str, Any]] = None,
    useful_paths: Optional[List[str]] = None,
    working_memory_text: Optional[str] = None,
    max_tokens: int = 1024,
) -> str:
    """Single LLM call to the decomposer. Returns raw response text.

    A6: ``working_memory_text`` is a pre-rendered ``[Working memory]``
    block (built by the caller from ``ledger.working_memory()`` via
    ``render_working_memory``). Replaces the legacy ``notebook`` dict
    parameter — the decomposer no longer pulls its own renderer.
    """
    # Compact history: last 2 turns' FULL responses (Thought + Action lines).
    # Text-only, no screenshot replay, so the cost stays small — but we
    # MUST retain the concrete ``Action: click(start_box='(x,y)')`` lines so
    # the decomposer can tell that e.g. (470, 269) was already clicked and
    # SEA was already typed. Truncating to the first line (the Thought)
    # loses the action and causes decomposer loops on no-progress screens.
    history_blocks: List[str] = []
    for i, turn in enumerate((actor_history or [])[-4:]):
        resp = (turn.get("response", "") or "").strip()
        if not resp:
            continue
        # Hard-cap each turn at ~2000 chars to keep the prompt bounded when
        # multi-step responses are long; this still retains all action lines
        # for typical 2-4 action sequences.
        if len(resp) > 2000:
            resp = resp[:2000] + "  …[truncated]"
        history_blocks.append(f"Turn -{len(actor_history[-4:]) - i} ago:\n{resp}")
    history_block = (
        "\n\nPrevious actor outputs (most recent last). Use these to avoid "
        "repeating a click/type that already fired in the last turn:\n\n"
        + "\n\n".join(history_blocks)
        if history_blocks else ""
    )
    # Live workbook structure (SP block) goes RIGHT NEXT TO the subgoal so
    # the model picks up real column letters / end-rows when composing
    # calc_* action ranges. Putting this in the system prompt didn't work —
    # the system prompt is 40K chars and the model anchors on the
    # placeholder examples (e.g. `Sheet1.A2:A20`) instead of reading the
    # actual layout.
    doc_block_section = (
        f"\n\nLive document structure (use these EXACT column letters / "
        f"end-row N from `used=A1:<col><N>` when building calc_* action "
        f"ranges; do NOT carry over column letters from system-prompt "
        f"examples):\n{doc_inspect_block}"
        if doc_inspect_block else ""
    )

    # When the previous cli_run turn FAILED (returncode != 0),
    # the planner already saw the truncated stderr — but its <actor_hint>
    # paraphrase tends to lose the specific error (e.g. it becomes "check
    # brackets" instead of "closing parenthesis ')' does not match opening
    # parenthesis '['"). So the actor needs the raw stderr too, head+tail
    # truncated to keep the actual ExceptionClass + message visible.
    last_run_section = ""
    if last_cli_run_output and last_cli_run_output.get("returncode") not in (None, 0):
        _err = (last_cli_run_output.get("stderr") or "").strip()
        _out = (last_cli_run_output.get("stdout") or "").strip()
        # Head+tail truncate (mirrors qwen25vl_agent_planner's helper). For the
        # bracket-mismatch failure mode we MUST keep the tail (where Python
        # prints the actual error message); naive head-only truncation
        # silently drops it.
        def _ht(s, budget=600, head_frac=0.25):
            if not s or len(s) <= budget: return s
            h = max(80, int(budget * head_frac))
            marker = f"\n  ... [{len(s) - budget} chars omitted] ...\n"
            t = budget - h - len(marker)
            if t < 80: return s[:budget]
            return s[:h] + marker + s[-t:]
        _err_t = _ht(_err)
        _out_t = _ht(_out, budget=300)
        last_run_section = (
            "\n\n[Previous cli_run output — the LAST turn's command FAILED. "
            "Read the stderr carefully; the ExceptionClass + message at the "
            "BOTTOM is the precise reason it errored, NOT the planner's "
            "paraphrase. Fix the actual error before retrying.]\n"
            f"  returncode: {last_cli_run_output.get('returncode')}\n"
            f"  stdout:     {_out_t or '(empty)'}\n"
            f"  stderr:     {_err_t or '(empty)'}"
        )

    # [Trusted file paths inventory] — absolute paths the env-perceiver
    # observed on disk at task start. Used by extract_info Pattern 14b
    # as the authoritative source for `paths` when the subgoal text is
    # vague. The decomposer must pick from this list verbatim, not
    # invent filenames from a pattern.
    inventory_section = ""
    if useful_paths:
        _lines = "\n".join(f"  {p}" for p in useful_paths)
        inventory_section = (
            "\n\n[Trusted file paths inventory — observed on disk at "
            "task start. For extract_info (Pattern 14b), source every "
            "`paths` entry from this list VERBATIM; do not normalize "
            "extensions, do not assume numbering, do not invent.]\n"
            f"{_lines}"
        )

    # [Notebook] — the agent's persistent KV memory (free-form
    # note_write entries + auto-mirrored extract_info per-file results).
    # The planner often gives a vague subgoal like "use calc_set_range
    # to write the extracted data into Sheet1 starting at A9" without
    # listing the actual values; the decomposer must source the literal
    # values from this block, NOT re-issue extract_info, NOT make up
    # numbers. Same block format as the planner sees, so both sides
    # have a single shared view of captured state.
    notebook_section = ""
    if working_memory_text:
        notebook_section = (
            "\n\n" + working_memory_text +
            "\n\n(Decomposer rule: when the subgoal references any "
            "value already in [Working memory], copy it VERBATIM into "
            "your structured action — do NOT re-issue extract_info, do "
            "NOT re-observe the GUI, do NOT round numbers or "
            "paraphrase strings.)"
        )

    user_text = (
        f"{user_instruction}\n"
        f"Subgoal: {subgoal}"
        f"{doc_block_section}"
        f"{last_run_section}"
        f"{inventory_section}"
        f"{notebook_section}"
        f"{history_block}\n\n"
        "Return ONLY a JSON array as specified."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": user_text},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
        ]},
    ]
    # Qwen3.5-VL thinking toggle. Default OFF (decomposer's 1024-token
    # budget isn't wasted on chain-of-thought). Flip on via env
    # QWEN35_THINKING=1; max_tokens auto-bumps to QWEN35_THINKING_MAX_TOKENS
    # (default 15000) so the <think> block has headroom. Detect any
    # alias containing "qwen3.5"/"qwen35" case-insensitively.
    #
    # NB: thinking-off wire format DIFFERS between OpenRouter and local
    # vLLM — OpenRouter expects `reasoning.enabled`, vLLM expects
    # `chat_template_kwargs.enable_thinking`. Branch on the client's
    # base_url. Setting the wrong key on OpenRouter silently fails
    # (no error, but thinking stays on, blowing past max_tokens).
    extra_kwargs: Dict[str, Any] = {}
    _model_lc = (model or "").lower()
    if "qwen3.7" in _model_lc:
        # qwen3.7-plus (OpenRouter): thinking ON by default (best quality).
        # Its reasoning trace is ~2600 tokens; the decomposer's default 1024
        # budget would truncate the answer, so bump max_tokens to a high floor.
        # Turn off via QWEN37_THINKING=0.
        thinking_on = os.environ.get(
            "QWEN37_THINKING", "1").strip().lower() in ("1", "true", "yes", "on")
        _base_url = str(getattr(client, "base_url", "") or "")
        if "openrouter.ai" in _base_url:
            extra_kwargs["extra_body"] = {"reasoning": {"enabled": thinking_on}}
        else:
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking_on}}
        if thinking_on:
            try:
                floor = int(os.environ.get("QWEN37_THINKING_MAX_TOKENS", "16000"))
            except ValueError:
                floor = 16000
            if max_tokens < floor:
                max_tokens = floor
    elif "qwen3.5" in _model_lc or "qwen35" in _model_lc:
        thinking_on = os.environ.get(
            "QWEN35_THINKING", "").strip().lower() in ("1", "true", "yes", "on")
        _base_url = str(getattr(client, "base_url", "") or "")
        if "openrouter.ai" in _base_url:
            extra_kwargs["extra_body"] = {
                "reasoning": {"enabled": thinking_on}
            }
        else:
            extra_kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": thinking_on}
            }
        if thinking_on:
            try:
                floor = int(os.environ.get("QWEN35_THINKING_MAX_TOKENS", "15000"))
            except ValueError:
                floor = 15000
            if max_tokens < floor:
                max_tokens = floor
    # Vision-provider lock: kimi-k2.6 / qwen3.5-397b serve VISION on ONE OpenRouter
    # provider only; default routing drops the image (text-only/empty completion).
    # Keep this id->provider map in sync with llm_client._VISION_PROVIDER (which is
    # keyed by alias; here we key by the resolved OpenRouter model id). Merges into
    # any extra_body already set by the thinking branches above.
    _base_url = str(getattr(client, "base_url", "") or "")
    if "openrouter.ai" in _base_url:
        _vp = {"moonshotai/kimi-k2.6": "SiliconFlow",
               "qwen/qwen3.5-397b-a17b": "Alibaba"}.get(model)
        if _vp:
            extra_kwargs.setdefault("extra_body", {})["provider"] = {"only": [_vp]}
    resp = client.chat.completions.create(
        model=model, messages=messages,
        max_tokens=max_tokens, temperature=0.0,
        **extra_kwargs,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Strip Qwen3.5 thinking trailer (free-form reasoning + </think>
    # before the actual answer). No-op when no </think> present.
    _thk = raw.rfind("</think>")
    if _thk >= 0:
        raw = raw[_thk + len("</think>"):].lstrip()
    return raw


# ---------------------------------------------------------------------------
# Composition — build the UI-TARS-compatible output string from steps
# ---------------------------------------------------------------------------

def _compose_action_line(
    step: Dict[str, Any],
    *,
    screen_w: int,
    screen_h: int,
) -> Optional[str]:
    """Translate one decomposer step into a UI-TARS action line.

    Coordinates come from the decomposer's own per-action ``point`` (inline
    grounding). ``_resolve`` returns None to skip the action when the point is
    missing/bad, so the caller never blind-clicks a fallback.
    """
    action = step.get("action")

    def _resolve(target_text: str,
                 point_val: Any) -> Optional[Tuple[int, int]]:
        px = _point_to_pixels(point_val, screen_w, screen_h)
        if px is None:
            logger.warning("inline: missing/bad point %r (target %r)",
                           point_val, (target_text or "")[:60])
        return px

    if action == "click":
        target = (step.get("target") or "").strip()
        if not target:
            return None
        coords = _resolve(target, step.get("point"))
        if coords is None:
            logger.warning("click grounding failed for %r; skipping action",
                           target[:60])
            return None
        px, py = coords
        return f"click(start_box='({px},{py})')"

    if action == "left_double":
        target = (step.get("target") or "").strip()
        if not target:
            return None
        coords = _resolve(target, step.get("point"))
        if coords is None:
            logger.warning("left_double grounding failed for %r; skipping action",
                           target[:60])
            return None
        px, py = coords
        return f"left_double(start_box='({px},{py})')"

    if action == "right_single":
        target = (step.get("target") or "").strip()
        if not target:
            return None
        coords = _resolve(target, step.get("point"))
        if coords is None:
            logger.warning("right_single grounding failed for %r; skipping action",
                           target[:60])
            return None
        px, py = coords
        return f"right_single(start_box='({px},{py})')"

    if action == "drag":
        start_target = (step.get("start_target") or step.get("target") or "").strip()
        end_target = (step.get("end_target") or "").strip()
        if not start_target or not end_target:
            logger.warning("drag missing start_target/end_target: %r", step)
            return None
        start_coords = _resolve(start_target, step.get("start_point"))
        end_coords = _resolve(end_target, step.get("end_point"))
        if start_coords is None or end_coords is None:
            logger.warning("drag grounding failed (start=%r, end=%r); skipping action",
                           start_target[:60], end_target[:60])
            return None
        sx, sy = start_coords
        ex, ey = end_coords
        return (f"drag(start_box='({sx},{sy})', "
                f"end_box='({ex},{ey})')")

    if action == "type":
        text = step.get("text", "")
        # Explicit destructive/submit modifiers (default False = safe
        # additive typing at cursor; no Ctrl+A, no Enter). The actor
        # opts in for single-line replace (wipe_then_type=True, e.g.
        # address bar) or command commit (submit=True, e.g. terminal
        # line or search box). See ACTOR_ACTION_SPACE doc in
        # qwen25vl_agent_planner.py for usage. Old steps without these
        # keys remain backward-compatible: both default False — never
        # destructive (which is the SAFE default we want by design).
        params = [f"content='{_escape_single_quotes(text)}'"]
        if bool(step.get("wipe_then_type", False)):
            # Audit guard: wipe_then_type does a Ctrl+A then overwrite —
            # potentially destructive if the focused field is not the one
            # intended. Log so post-hoc review can spot bad wipes.
            logger.warning("wipe_then_type=True (destructive replace) for text %r",
                           text[:80])
            params.append("wipe_then_type=True")
        if bool(step.get("submit", False)):
            params.append("submit=True")
        return f"type({', '.join(params)})"

    if action == "hotkey":
        key = (step.get("key") or "").strip()
        if not key:
            return None
        return f"hotkey(key='{key}')"

    if action == "scroll":
        direction = step.get("direction", "down")
        anchor = (step.get("anchor_target") or "").strip()
        _anchor_pt = step.get("anchor_point")
        coords = (_resolve(anchor, _anchor_pt)
                  if (anchor or _anchor_pt) else None)
        if coords is not None:
            px, py = coords
        else:
            # No anchor (or anchor grounding failed) → scroll at screen
            # center. Scrolling at the center is harmless, so unlike
            # click/drag we fall back instead of skipping the action.
            px, py = screen_w // 2, screen_h // 2
        return f"scroll(start_box='({px},{py})', direction='{direction}')"

    if action == "wait":
        return "wait()"

    if action == "navigate":
        url = (step.get("url") or "").strip()
        if not url:
            logger.warning("navigate missing 'url': %r", step)
            return None
        new_tab = bool(step.get("new_tab", False))
        # Same wire format as cli_run/edit_json: function-call string
        # the upstream AST parser decodes; the in-VM runtime script
        # comes from navigate_helpers.build_navigate_script.
        return f"navigate(url={url!r}, new_tab={new_tab})"

    if action == "finished":
        # Text-answer terminal sentinel. Emit as ``finished(answer='...')``
        # text; agent.py's text-answer detection block in _pa_predict
        # catches this verbatim, stashes ``self._text_answer``, and
        # terminates the episode (no pyautogui code generated — the
        # detection short-circuits before _to_pyautogui).
        answer = (step.get("answer") or "").strip()
        if not answer:
            logger.warning("finished emitted with empty answer: %r", step)
        safe = _escape_single_quotes(answer)
        return f"finished(answer='{safe}')"

    if action == "cli_run":
        cmd = (step.get("command") or "").strip()
        bg = bool(step.get("background", False))
        wait_s = step.get("wait_seconds", 1)
        try:
            wait_s = int(wait_s)
        except (TypeError, ValueError):
            wait_s = 1
        if not cmd:
            logger.warning("cli_run missing 'command' field: %r", step)
            return None
        # Wire format mirrors edit_json: function-call string the upstream
        # AST parser can decode. ``command`` is a Python string literal
        # (repr-style) — _pa_to_pyautogui will pass it through to the
        # runtime script which json.dumps it again before embedding.
        return f"cli_run(command={cmd!r}, background={bg}, wait_seconds={wait_s})"

    if action == "extract_info":
        # Host-side action — agent (not VM) reads files via
        # controller.get_file, dispatches by extension to the planner
        # LLM (with vision for images / PDFs), aggregates structured
        # results. NO pyautogui code is generated; result is stored on
        # the agent and rendered in the next planner prompt.
        paths = step.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        paths = [str(p) for p in paths if p]
        query = (step.get("query") or "").strip()
        schema = step.get("output_schema") or step.get("schema") or {}
        if not paths or not query:
            logger.warning("extract_info missing paths or query: %r", step)
            return None
        # JSON-encode payloads so they survive the round-trip through
        # repr() + ast.literal_eval (no shell escape risk — runs in-process).
        import json as _json
        return (
            f"extract_info(paths={_json.dumps(paths)!r}, "
            f"query={query!r}, "
            f"output_schema={_json.dumps(schema)!r})"
        )

    if action == "note_write":
        # Host-side action — write a (key, value) into the agent's
        # Notebook. The notebook is rendered into both planner and
        # decomposer prompts every turn so neither side ever has to
        # re-observe what was already captured.
        key = (step.get("key") or "").strip()
        value = step.get("value")
        if value is None:
            value = step.get("val") or ""
        if not key:
            logger.warning("note_write missing 'key': %r", step)
            return None
        # JSON-encode dict/list values so the wire form survives the
        # repr→literal_eval round-trip; plain strings/numbers pass
        # through as-is. handle_note_write decodes on the other side.
        if isinstance(value, (dict, list)):
            import json as _json
            value = _json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        return f"note_write(key={key!r}, value={value!r})"

    if action == "open_app":
        # No grounding needed — deterministic window activation / launch
        # on the VM. Wire format mirrors cli_run: a function-call string
        # the AST parser at _pa_parse_action_ast can decode.
        app = (step.get("app") or "").strip()
        if not app:
            logger.warning("open_app missing 'app' field: %r", step)
            return None
        file_path = (step.get("file_path") or step.get("path") or "").strip()
        if file_path:
            return f"open_app(app={app!r}, file_path={file_path!r})"
        return f"open_app(app={app!r})"

    # Structured Calc actions (calc_fill_column, … — full list in
    # calc_actions.CALC_ACTION_NAMES). Actor emits FLAT JSON kwargs;
    # framework owns the Python-source generation. Keeps all calc_*
    # dispatch in one module instead of growing the elif chain here.
    from mm_agents.structagent.actions.calc.actions import CALC_ACTION_NAMES, step_to_wire
    if action in CALC_ACTION_NAMES:
        wire = step_to_wire(step)
        if wire is None:
            logger.warning("%s validation failed: %r", action, step)
        return wire

    # Structured Impress actions (impress_set_font, impress_set_text_color,
    # … — full list in impress_actions.IMPRESS_ACTION_NAMES). Same
    # contract as calc_*: flat JSON in, ``impress_xxx(k=v, …)`` wire
    # string out; runtime script generation lives in impress_actions.
    from mm_agents.structagent.actions.impress.actions import (
        IMPRESS_ACTION_NAMES as _IMPRESS_ACTION_NAMES_LOCAL,
        step_to_wire as _impress_step_to_wire,
    )
    if action in _IMPRESS_ACTION_NAMES_LOCAL:
        wire = _impress_step_to_wire(step)
        if wire is None:
            logger.warning("%s validation failed: %r", action, step)
        return wire

    # Structured Writer actions (writer_paste_at, … — full list in
    # writer_actions.WRITER_ACTION_NAMES). Same contract as calc_* /
    # impress_*: flat JSON in, ``writer_xxx(k=v, …)`` wire string out.
    from mm_agents.structagent.actions.writer.actions import (
        WRITER_ACTION_NAMES as _WRITER_ACTION_NAMES_LOCAL,
        step_to_wire as _writer_step_to_wire,
    )
    if action in _WRITER_ACTION_NAMES_LOCAL:
        wire = _writer_step_to_wire(step)
        if wire is None:
            logger.warning("%s validation failed: %r", action, step)
        return wire

    if action == "edit_json":
        # No grounding needed — file IO action.
        path = (step.get("path") or "").strip()
        op = (step.get("op") or "").strip()
        data = step.get("data")
        if not path:
            logger.warning("edit_json missing 'path' field: %r", step)
            return None
        if op not in ("set_key", "append_entry", "remove_entry_by_key"):
            logger.warning("edit_json unknown op %r in %r", op, step)
            return None
        if data is None:
            logger.warning("edit_json missing 'data' field: %r", step)
            return None
        # Wire format: ``edit_json(path='...', op='...', data='<json_str>')``.
        # The upstream AST parser (_pa_parse_action_ast) extracts each
        # kwarg as a Python string. Downstream _pa_to_pyautogui will
        # ``json.loads(data)`` to recover the dict/list before generating
        # the runtime script. This double-encoding (Python string holding
        # JSON string) is necessary because ast.Constant only handles
        # primitive literals, not nested dict/list literals.
        import json as _json
        data_json = _json.dumps(data)
        return f"edit_json(path={path!r}, op={op!r}, data={data_json!r})"

    # "impossible" is handled at the caller level.
    return None


def call_decomposer_actor(
    instruction: str,
    subgoal: str,
    screenshot_bytes: bytes,
    screen_size: Tuple[int, int],
    *,
    decomposer_client: openai.OpenAI,
    decomposer_model: str,
    decomposer_system_template: str,
    actor_history: Optional[List[Dict[str, Any]]] = None,
    doc_inspect_block: Optional[str] = None,
    last_cli_run_output: Optional[Dict[str, Any]] = None,
    useful_paths: Optional[List[str]] = None,
    working_memory_text: Optional[str] = None,
    task_domain: Optional[str] = None,
) -> str:
    """Decompose-then-ground actor. Return string in UI-TARS's output shape.

    Parameters
    ----------
    instruction : str
        The full original task instruction.
    subgoal : str
        The current subgoal to execute NOW.
    screenshot_bytes : bytes
        Current screen PNG bytes (the physical screen, not resized).
    screen_size : (int, int)
        Physical screen (width, height) in pixels. Used to de-normalize the
        decomposer's own 0-1000 inline coords into pyautogui pixel coords.
    decomposer_client, decomposer_model
        OpenAI client + model name for the decomposer LLM (typically the
        Qwen3.5-VL planner server at :8010).
    decomposer_system_template : str
        The A2 system prompt, already rendered with any substitutions the
        caller wants (subgoal / completed / remaining). Typically produced
        by ``PromptSet.render("A2_actor_decomposer_points", ...)``.
    actor_history : list, optional
        Recent actor turns for decomposer context. Same shape as the
        UI-TARS path (``[{"screenshot_b64": ..., "response": ...}]``).
    """
    screen_w, screen_h = int(screen_size[0]), int(screen_size[1])
    screenshot_b64 = base64.b64encode(screenshot_bytes).decode("ascii")

    # 1) Decomposer call
    try:
        decomposer_text = _call_decomposer(
            client=decomposer_client,
            model=decomposer_model,
            system_prompt=decomposer_system_template,
            subgoal=subgoal,
            screenshot_b64=screenshot_b64,
            user_instruction=instruction,
            actor_history=actor_history or [],
            doc_inspect_block=doc_inspect_block,
            last_cli_run_output=last_cli_run_output,
            useful_paths=useful_paths,
            working_memory_text=working_memory_text,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("decomposer call failed: %s", e)
        return "<Impossible>"

    # Diagnostic: log a head of the raw decomposer text so post-hoc
    # debugging can see WHY <Impossible> fires (empty? markdown? bare
    # 'impossible' step? non-array prose?). Truncate to keep the log
    # readable but long enough to show the action shape.
    logger.info("[decomposer] raw response head (%d chars): %s",
                len(decomposer_text), decomposer_text[:600].replace("\n", " ⏎ "))

    steps, rejected_unknown = _parse_decomposer_response_with_diagnostics(
        decomposer_text, task_domain=task_domain)
    if not steps:
        # Retry once with a terser enforcement note.
        logger.warning("decomposer returned unparseable output; retrying once")
        try:
            decomposer_text = _call_decomposer(
                client=decomposer_client,
                model=decomposer_model,
                system_prompt=decomposer_system_template,
                subgoal=subgoal,
                screenshot_b64=screenshot_b64,
                user_instruction=(instruction +
                                    "\n\nIMPORTANT: Respond with ONE JSON array and nothing else."),
                actor_history=actor_history or [],
                doc_inspect_block=doc_inspect_block,
                last_cli_run_output=last_cli_run_output,
                useful_paths=useful_paths,
                working_memory_text=working_memory_text,
            )
            steps, retry_rejected = (
                _parse_decomposer_response_with_diagnostics(
                    decomposer_text, task_domain=task_domain))
            # Surface unknowns from either attempt — caller wants to
            # know what the model actually emitted, not just that we
            # failed.
            rejected_unknown = retry_rejected or rejected_unknown
        except Exception as e:  # noqa: BLE001
            logger.exception("decomposer retry failed: %s", e)
            return _impossible_with_reason(rejected_unknown, task_domain)

    if not steps:
        logger.warning("decomposer retry still empty; raw=%r",
                        decomposer_text[:200])
        return _impossible_with_reason(rejected_unknown, task_domain)

    # 2) Handle immediate terminators
    for step in steps:
        if step.get("action") == "impossible":
            return "<Impossible>"

    # 2b) Guard against multi-click batches.
    #
    # POINTS-G grounds every click in this response against the SAME
    # pre-action screenshot. A second click whose target only becomes
    # visible after the first click (dropdown items after opening,
    # calendar days after month nav, checkbox states after toggle,
    # CAPTCHA cells after the first selection) is grounded against
    # thin air. We keep the 1st click and drop everything from the
    # 2nd click onwards — next actor invocation will re-ground the
    # follow-up against the updated screenshot.
    _click_kinds = {"click", "left_double", "right_single"}
    _second_click_idx = None
    _first_seen = False
    for _i, _st in enumerate(steps):
        if (_st.get("action") or "") in _click_kinds:
            if _first_seen:
                _second_click_idx = _i
                break
            _first_seen = True
    if _second_click_idx is not None:
        _n_clicks = sum(1 for s in steps
                         if (s.get("action") or "") in _click_kinds)
        logger.warning(
            "Decomposer emitted %d click action(s); truncating at 2nd "
            "click (index %d). Kept %d step(s); dropped %d. Reason: "
            "later clicks would be grounded against a stale screenshot.",
            _n_clicks, _second_click_idx, _second_click_idx,
            len(steps) - _second_click_idx,
        )
        steps = steps[:_second_click_idx]
        if not steps:
            return "<Impossible>"

    # 2c) Guard against multiple structured actions per response.
    #
    # Structured actions (impress_* / calc_* / writer_* / cli_run /
    # edit_json) are the granularity at which the planner already
    # decomposes — multiple of them in a single decomposer batch is
    # the model carpet-bombing rather than reasoning about which op
    # to run (the observed failure mode is 25× impress_set_font, one
    # per shape on the slide). Keep the FIRST structured action and
    # drop every step AFTER it. Subsequent structured ops belong in
    # their own turn so the planner can verify each diff; subsequent
    # clicks would also ground against a stale screenshot because
    # the structured op may have mutated doc state visible in the
    # next screen capture.
    _structured_kinds = (set(_CALC_ACTION_NAMES)
                          | set(_IMPRESS_ACTION_NAMES)
                          | set(_WRITER_ACTION_NAMES)
                          | {"cli_run", "edit_json"})
    _first_struct_idx = None
    for _i, _st in enumerate(steps):
        if (_st.get("action") or "") in _structured_kinds:
            _first_struct_idx = _i
            break
    if (_first_struct_idx is not None
            and _first_struct_idx + 1 < len(steps)):
        _n_struct = sum(1 for s in steps
                         if (s.get("action") or "") in _structured_kinds)
        _dropped = len(steps) - (_first_struct_idx + 1)
        logger.warning(
            "Decomposer emitted %d structured action(s) (impress_*/"
            "calc_*/writer_*/cli_run/edit_json); truncating after "
            "the first at index %d. Kept %d step(s); dropped %d. "
            "Reason: each structured op belongs in its own turn so "
            "the planner can verify the diff; following clicks would "
            "also ground against a stale screenshot.",
            _n_struct, _first_struct_idx, _first_struct_idx + 1, _dropped,
        )
        steps = steps[:_first_struct_idx + 1]

    # 3) Compose each action line from the decomposer's own inline coordinates.
    action_lines: List[str] = []
    for step in steps:
        line = _compose_action_line(
            step, screen_w=screen_w, screen_h=screen_h,
        )
        if line:
            action_lines.append(line)
        else:
            # Diagnostic: emit the step that failed to compose so the
            # runtime.log shows whether the decomposer produced an
            # unknown action, an empty/missing required field, etc.
            logger.warning("step produced no action line: %r", step)

    if not action_lines:
        logger.warning("no valid action lines produced from %d step(s); steps=%r",
                       len(steps), steps[:3])
        return "<Impossible>"

    # 4) Assemble UI-TARS-compatible output
    thought_bits: List[str] = []
    for step in steps[:3]:  # brief summary of first few steps
        a = step.get("action", "?")
        if a in ("click", "left_double", "right_single"):
            thought_bits.append(f"{a} {step.get('target', '')[:200]}")
        elif a == "drag":
            thought_bits.append(
                f"drag {step.get('start_target','')[:25]} → "
                f"{step.get('end_target','')[:25]}"
            )
        elif a == "type":
            thought_bits.append(f"type {step.get('text', '')[:30]!r}")
        elif a == "hotkey":
            thought_bits.append(f"hotkey {step.get('key', '')}")
        elif a == "scroll":
            thought_bits.append(f"scroll {step.get('direction', '')}")
        else:
            thought_bits.append(a)
    thought = "Plan: " + " → ".join(thought_bits)

    # Upstream splits by "Action:" then by \n\n — match that layout exactly.
    action_block = action_lines[0]
    if len(action_lines) > 1:
        action_block = action_block + "\n\n" + "\n\n".join(action_lines[1:])
    return f"Thought: {thought}\nAction: {action_block}"

"""Decomposer actor with inline grounding — drop-in replacement for the
UI-TARS actor call.

``call_decomposer_actor`` takes the same args as
``Qwen25VLAgentPlanner._pa_call_actor`` and returns a UI-TARS-format string:

    Thought: <one-line reasoning>
    Action: click(start_box='(994,445)')

    type(content='coffee makers')

The upstream parser ``_pa_action_to_pyautogui_codes`` only needs the
``Action:`` split marker, actions separated by blank lines, and each action
being one of the known forms (click/type/hotkey/scroll/wait/finished or the
bare ``<Impossible>``). We emit exactly that so downstream needs no changes.

Pipeline:
1. Call the decomposer (Qwen3.5-VL @ :8010, the planner server) with the A2
   prompt + screenshot + subgoal. Returns a JSON list of typed steps, each
   carrying its own 0-1000 inline coords (see INLINE_GROUNDING_INSTRUCTION).
2. De-normalize coords to pixels via ``_point_to_pixels``.
3. Compose the UI-TARS string.
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


# Inline grounding directive — appended to the decomposer system prompt by
# actor.py. Single source of truth; edit only here.
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


# Actions the decomposer may emit. Keep mirrored with the seed
# A2_actor_decomposer_points.txt template AND the UI-TARS action space in
# qwen25vl_agent_planner.py so the downstream parser accepts every variant.
_ALLOWED_ACTIONS = {
    "click",          # single left click
    "left_double",    # double-click (open folder / file)
    "right_single",   # right-click (context menu)
    "drag",           # click-and-drag from start_target to end_target
    "type",           # keyboard typewrite
    "hotkey",         # single key or combo
    "scroll",         # scroll in a direction, anchored near anchor_target
    "wait",           # sleep 5s then re-observe
    "impossible",     # subgoal can't be achieved here
    "navigate",       # Chrome page navigation via CDP HTTP endpoint.
                      # {"action":"navigate", "url":..., "new_tab":false}.
                      # One reliable step vs the (ctrl+l → type URL\\n) GUI
                      # dance — needed for cross-site multi-hop tasks.
    "finished",       # Terminal sentinel for text-answer tasks (WebVoyager
                      # / Mind2Web / MMInA). {"action":"finished",
                      # "answer":...}. Caught by agent.py's text-answer
                      # block in _pa_predict, which ends the episode and
                      # hands the answer to the runner / judge.
    "edit_json",      # modify a JSON config via file IO (load → mutate →
                      # dump). Avoids GUI ctrl+a + type, which corrupts JSON
                      # via partial-content overwrite. See edit_json_helpers.
    "cli_run",        # one-shot bash via subprocess on the VM (NOT a GUI
                      # terminal). Launch apps (`code <path>`), non-JSON file
                      # edits (sed/cp), or nested JSON mutations edit_json's
                      # flat ops can't express. See cli_run_helpers.py.
    "extract_info",   # batch-read N VM files and ask the planner LLM to
                      # extract structured info per file (one LLM call each).
                      # Bypasses "open file → read → record in spreadsheet"
                      # GUI gymnastics: returns a JSON aggregate the planner
                      # writes out via calc_*/writer_*. Use for receipts,
                      # invoices, papers, emails — any local file whose
                      # CONTENT must populate a structured deliverable.
    # ``cli_run_uno`` (raw Python against the open LibreOffice doc) was
    # REMOVED — all three LibreOffice domains go through the structured
    # calc_* / impress_* / writer_* sets, which the framework lowers to UNO
    # Python. A raw-UNO escape hatch only misled the 9B actor into hand-
    # writing UNO.
    "open_app",       # deterministic window activate (wmctrl) or launch,
                      # optionally with a file. Replaces the brittle GUI
                      # dock-click / Alt+Tab dance. See open_app.py.
    "note_write",     # persist (key, value) to the agent Notebook (see
                      # actions/handlers/notebook.py), rendered into both
                      # planner and decomposer prompts every turn. Pin any
                      # observed concrete value (URL, name, count) so it
                      # survives app switches, outcome REVERTs, and perceiver
                      # misses. Value may be a JSON string.
    # "finished" used to mean "this subgoal is done"; removed in Phase 2 —
    # the planner's <last_subgoal_assessment> is now the sole done-ness
    # source. Kept as a back-compat escape hatch below so a model that
    # ignores the prompt and emits it doesn't crash the parser.
}

# Calc structured actions (calc_fill_column, …). Actor emits flat JSON kwargs
# that the dispatcher translates to a calc_xxx(…) wire line. These names MUST
# be in the allow-list or _parse_decomposer_response silently drops them — a
# subtle regression (valid JSON in, parser rejects, <Impossible> every step,
# traj.jsonl never appended).
from mm_agents.structagent.actions.calc.actions import CALC_ACTION_NAMES as _CALC_ACTION_NAMES
_ALLOWED_ACTIONS = _ALLOWED_ACTIONS | set(_CALC_ACTION_NAMES)
# Impress structured actions — same allow-list role as calc_*.
from mm_agents.structagent.actions.impress.actions import IMPRESS_ACTION_NAMES as _IMPRESS_ACTION_NAMES
_ALLOWED_ACTIONS = _ALLOWED_ACTIONS | set(_IMPRESS_ACTION_NAMES)
# Writer structured actions — same allow-list role.
from mm_agents.structagent.actions.writer.actions import WRITER_ACTION_NAMES as _WRITER_ACTION_NAMES
_ALLOWED_ACTIONS = _ALLOWED_ACTIONS | set(_WRITER_ACTION_NAMES)


def _screenshot_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    """Base64-encode a PNG as a data URL so we don't depend on file://."""
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _point_to_pixels(
    pt: Any, screen_w: int, screen_h: int
) -> Optional[Tuple[int, int]]:
    """Convert a decomposer ``[x, y]`` point to pixels. Auto-detects the
    coordinate system since models emit different grids and ignore the prompt:

      - both <= 1.0     -> normalized [0, 1]
      - both <= 1000.0  -> 0-1000 grid (Qwen-VL native)
      - otherwise       -> absolute pixels

    Returns None on missing/malformed ``pt`` so the caller skips the action
    rather than blind-clicking a fallback.
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
    """``_ALLOWED_ACTIONS`` minus cli_run / edit_json for GUI-only domains
    (chrome/gimp/thunderbird) when domain_coding_gate is on. Defense-in-depth:
    the A2 prompt already omits the coding route, but this also rejects a
    coding action if the decomposer emits one anyway → surfaces as <Impossible>
    'unknown action', forcing a GUI retry. Gate off / non-GUI-only → full set."""
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
    """Extract step dicts; also collect action names rejected as not in
    ``_ALLOWED_ACTIONS``.

    Returns ``(accepted_steps, rejected_action_names)``. The rejected list lets
    the caller tell the planner "actor rejected unknown action 'X'" instead of
    an opaque <Impossible> — without it the planner can't distinguish a
    hallucinated action name from a grounding miss, and replans the same way.
    """
    if not text:
        return [], []
    stripped = text.strip()
    fence = re.match(r"```(?:json)?\s*\n?(.*?)\n?```\s*$", stripped, re.DOTALL)
    if fence:
        stripped = fence.group(1)
    # Prefer a JSON array; tolerate a lone {...} object (the model sometimes
    # emits one instead of a one-element array) by wrapping it — same as
    # boundary_verify._extract_json_array.
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
    """Back-compat wrapper returning just the accepted steps. Use
    :func:`_parse_decomposer_response_with_diagnostics` for the
    rejected-unknowns sidechannel."""
    steps, _ = _parse_decomposer_response_with_diagnostics(
        text, task_domain=task_domain)
    return steps


def _valid_action_names_for_domain(task_domain: Optional[str]) -> List[str]:
    """The impress_* / calc_* action set for the current domain. Used in the
    <Impossible> reason so the planner sees a focused list, not all actions."""
    dom = (task_domain or "").lower()
    if dom == "libreoffice_impress":
        return sorted(_IMPRESS_ACTION_NAMES)
    if dom == "libreoffice_calc":
        return sorted(_CALC_ACTION_NAMES)
    return []


def _impossible_with_reason(rejected_unknown: List[str],
                              task_domain: Optional[str]) -> str:
    """``<Impossible>`` carrying a specific reason for unknown action names.

    Keeps ``<Impossible>`` as the first token so the upstream substring check
    still detects it, with a ``Reason:`` line the planner surfaces in
    next-turn <stuck_analysis>.
    """
    if not rejected_unknown:
        return "<Impossible>"
    # Dedupe, preserving first-seen order.
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
    """Mirror _pa_escape_single_quotes so the emitted type(content='...')
    string survives ast.parse downstream."""
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

    ``working_memory_text`` is a pre-rendered ``[Working memory]`` block (built
    by the caller from ``ledger.working_memory()``), replacing the legacy
    ``notebook`` dict — the decomposer no longer renders its own.
    """
    # Compact history: last few turns' FULL responses (text-only, no
    # screenshot replay). MUST retain the concrete Action: lines so the
    # decomposer can tell what was already clicked/typed; truncating to just
    # the Thought drops the action and causes loops on no-progress screens.
    history_blocks: List[str] = []
    for i, turn in enumerate((actor_history or [])[-4:]):
        resp = (turn.get("response", "") or "").strip()
        if not resp:
            continue
        # Cap each turn at ~2000 chars to bound the prompt; still keeps all
        # action lines for typical 2-4 action sequences.
        if len(resp) > 2000:
            resp = resp[:2000] + "  …[truncated]"
        history_blocks.append(f"Turn -{len(actor_history[-4:]) - i} ago:\n{resp}")
    history_block = (
        "\n\nPrevious actor outputs (most recent last). Use these to avoid "
        "repeating a click/type that already fired in the last turn:\n\n"
        + "\n\n".join(history_blocks)
        if history_blocks else ""
    )
    # Live workbook structure goes right next to the subgoal so the model
    # picks up real column letters / end-rows for calc_* ranges. In the 40K-
    # char system prompt it doesn't work — the model anchors on the
    # placeholder examples (e.g. `Sheet1.A2:A20`) instead of the real layout.
    doc_block_section = (
        f"\n\nLive document structure (use these EXACT column letters / "
        f"end-row N from `used=A1:<col><N>` when building calc_* action "
        f"ranges; do NOT carry over column letters from system-prompt "
        f"examples):\n{doc_inspect_block}"
        if doc_inspect_block else ""
    )

    # When the previous cli_run FAILED, the planner's <actor_hint> paraphrase
    # tends to lose the specific error (e.g. "check brackets" instead of the
    # exact "closing parenthesis ')' does not match ..."). Give the actor the
    # raw stderr too, head+tail truncated to keep the ExceptionClass + message.
    last_run_section = ""
    if last_cli_run_output and last_cli_run_output.get("returncode") not in (None, 0):
        _err = (last_cli_run_output.get("stderr") or "").strip()
        _out = (last_cli_run_output.get("stdout") or "").strip()
        # Head+tail truncate. MUST keep the tail — Python prints the actual
        # error message there; head-only truncation would drop it.
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

    # Trusted file paths the env-perceiver observed on disk at task start.
    # Authoritative source for extract_info's `paths` (Pattern 14b) when the
    # subgoal is vague — pick verbatim, don't invent filenames.
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

    # Notebook — the agent's persistent KV memory (note_write entries +
    # mirrored extract_info results). The planner often gives a vague subgoal
    # ("write the extracted data into Sheet1 starting at A9") without the
    # actual values; the decomposer must source literals from here, not
    # re-issue extract_info or invent numbers. Same format the planner sees.
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
    # Qwen3.5-VL thinking toggle. Default OFF so the 1024-token budget isn't
    # spent on chain-of-thought; flip on via QWEN35_THINKING=1 (max_tokens
    # auto-bumps to QWEN35_THINKING_MAX_TOKENS, default 15000).
    #
    # NB: the wire format DIFFERS between OpenRouter (`reasoning.enabled`) and
    # local vLLM (`chat_template_kwargs.enable_thinking`); branch on base_url.
    # Wrong key on OpenRouter fails silently — thinking stays on, blows past
    # max_tokens.
    extra_kwargs: Dict[str, Any] = {}
    _model_lc = (model or "").lower()
    if "qwen3.7" in _model_lc:
        # qwen3.7-plus (OpenRouter): thinking ON by default. Its ~2600-token
        # trace would truncate the answer under the 1024 budget, so bump
        # max_tokens to a high floor. Off via QWEN37_THINKING=0.
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
    # Vision-provider lock: kimi-k2.6 / qwen3.5-397b serve vision on ONE
    # OpenRouter provider; default routing drops the image. Keep this
    # id->provider map in sync with llm_client._VISION_PROVIDER (keyed by alias
    # there, by resolved model id here). Merges into any extra_body above.
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
    # Strip the Qwen3.5 thinking trailer up to </think>. No-op when absent.
    _thk = raw.rfind("</think>")
    if _thk >= 0:
        raw = raw[_thk + len("</think>"):].lstrip()
    return raw


# ---------------------------------------------------------------------------
# Composition — build the UI-TARS output string from steps
# ---------------------------------------------------------------------------

def _compose_action_line(
    step: Dict[str, Any],
    *,
    screen_w: int,
    screen_h: int,
) -> Optional[str]:
    """Translate one decomposer step into a UI-TARS action line.

    Coordinates come from the step's own ``point`` (inline grounding).
    ``_resolve`` returns None to skip the action on a missing/bad point, so the
    caller never blind-clicks a fallback.
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
        # Destructive/submit modifiers, both default False = safe additive
        # typing at cursor (no Ctrl+A, no Enter). Opt in for single-line
        # replace (wipe_then_type, e.g. address bar) or commit (submit, e.g.
        # terminal/search). Old steps without these keys stay back-compatible.
        params = [f"content='{_escape_single_quotes(text)}'"]
        if bool(step.get("wipe_then_type", False)):
            # wipe_then_type does Ctrl+A then overwrite — destructive if the
            # wrong field is focused. Log so review can spot bad wipes.
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
            # No/failed anchor → scroll at center. Harmless, so unlike
            # click/drag we fall back instead of skipping.
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
        # Wire format like cli_run/edit_json: a function-call string the AST
        # parser decodes; runtime script from navigate_helpers.
        return f"navigate(url={url!r}, new_tab={new_tab})"

    if action == "finished":
        # Text-answer terminal sentinel. agent.py's text-answer block in
        # _pa_predict catches finished(answer='...') verbatim, stashes
        # _text_answer, and ends the episode (short-circuits before
        # _to_pyautogui, so no pyautogui code is generated).
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
        # Wire format like edit_json. ``command`` is a repr'd Python string
        # literal; _pa_to_pyautogui passes it to the runtime script, which
        # json.dumps it again before embedding.
        return f"cli_run(command={cmd!r}, background={bg}, wait_seconds={wait_s})"

    if action == "extract_info":
        # Host-side action: the agent reads files via controller.get_file,
        # dispatches by extension to the planner LLM (vision for images/PDFs),
        # aggregates structured results. No pyautogui code — result is stored
        # on the agent and rendered in the next planner prompt.
        paths = step.get("paths") or []
        if isinstance(paths, str):
            paths = [paths]
        paths = [str(p) for p in paths if p]
        query = (step.get("query") or "").strip()
        schema = step.get("output_schema") or step.get("schema") or {}
        if not paths or not query:
            logger.warning("extract_info missing paths or query: %r", step)
            return None
        # JSON-encode payloads so they survive the repr() + ast.literal_eval
        # round-trip (no shell-escape risk — runs in-process).
        import json as _json
        return (
            f"extract_info(paths={_json.dumps(paths)!r}, "
            f"query={query!r}, "
            f"output_schema={_json.dumps(schema)!r})"
        )

    if action == "note_write":
        # Host-side action: write (key, value) into the agent Notebook, which
        # is rendered into both prompts every turn so neither side re-observes
        # what was already captured.
        key = (step.get("key") or "").strip()
        value = step.get("value")
        if value is None:
            value = step.get("val") or ""
        if not key:
            logger.warning("note_write missing 'key': %r", step)
            return None
        # JSON-encode dict/list values to survive the repr→literal_eval
        # round-trip; scalars pass through. handle_note_write decodes.
        if isinstance(value, (dict, list)):
            import json as _json
            value = _json.dumps(value, ensure_ascii=False)
        else:
            value = str(value)
        return f"note_write(key={key!r}, value={value!r})"

    if action == "open_app":
        # No grounding — deterministic window activate/launch. Wire format
        # like cli_run, decoded by _pa_parse_action_ast.
        app = (step.get("app") or "").strip()
        if not app:
            logger.warning("open_app missing 'app' field: %r", step)
            return None
        file_path = (step.get("file_path") or step.get("path") or "").strip()
        if file_path:
            return f"open_app(app={app!r}, file_path={file_path!r})"
        return f"open_app(app={app!r})"

    # Structured Calc actions (full list in CALC_ACTION_NAMES). Actor emits
    # flat JSON kwargs; the framework owns Python-source generation, keeping
    # calc_* dispatch in one module instead of growing the elif chain.
    from mm_agents.structagent.actions.calc.actions import CALC_ACTION_NAMES, step_to_wire
    if action in CALC_ACTION_NAMES:
        wire = step_to_wire(step)
        if wire is None:
            logger.warning("%s validation failed: %r", action, step)
        return wire

    # Structured Impress actions (full list in IMPRESS_ACTION_NAMES). Same
    # contract as calc_*: flat JSON in, impress_xxx(k=v, …) wire string out.
    from mm_agents.structagent.actions.impress.actions import (
        IMPRESS_ACTION_NAMES as _IMPRESS_ACTION_NAMES_LOCAL,
        step_to_wire as _impress_step_to_wire,
    )
    if action in _IMPRESS_ACTION_NAMES_LOCAL:
        wire = _impress_step_to_wire(step)
        if wire is None:
            logger.warning("%s validation failed: %r", action, step)
        return wire

    # Structured Writer actions (full list in WRITER_ACTION_NAMES). Same
    # contract as calc_* / impress_*: flat JSON in, writer_xxx(k=v, …) out.
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
        # No grounding — file IO action.
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
        # Wire: edit_json(path='...', op='...', data='<json_str>'). The AST
        # parser extracts each kwarg as a string; _pa_to_pyautogui json.loads
        # data back to a dict/list. The double-encoding (Python string holding
        # JSON) is needed because ast.Constant only handles primitive literals,
        # not nested dict/list.
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
    """Decompose-then-ground actor. Returns a UI-TARS-shaped string.

    screen_size is the physical screen (w, h) in pixels, used to de-normalize
    the decomposer's 0-1000 inline coords. decomposer_client/model point at the
    decomposer LLM (typically the Qwen3.5-VL planner server at :8010).
    decomposer_system_template is the already-rendered A2 prompt. actor_history
    is recent turns, same shape as the UI-TARS path
    (``[{"screenshot_b64": ..., "response": ...}]``).
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

    # Log a head of the raw response so debugging can see WHY <Impossible>
    # fires (empty? markdown? bare 'impossible'? non-array prose?).
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
            # Surface unknowns from either attempt — the caller wants what the
            # model actually emitted, not just that we failed.
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
    # Every click in this response is grounded against the SAME pre-action
    # screenshot. A second click whose target only appears after the first
    # (dropdown items, post-nav calendar days, toggled checkboxes, CAPTCHA
    # cells) grounds against thin air. Keep the 1st click, drop from the 2nd
    # on; the next invocation re-grounds against the updated screenshot.
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
    # Structured actions (impress_* / calc_* / writer_* / cli_run / edit_json)
    # are the granularity the planner already decomposes at; several in one
    # batch is carpet-bombing, not reasoning (observed: 25x impress_set_font,
    # one per shape). Keep the FIRST and drop the rest — each structured op
    # belongs in its own turn so the planner can verify its diff, and a
    # following click would ground against a stale screenshot after the op
    # mutates doc state.
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
            # Log the step that failed to compose (unknown action, missing
            # required field, etc).
            logger.warning("step produced no action line: %r", step)

    if not action_lines:
        logger.warning("no valid action lines produced from %d step(s); steps=%r",
                       len(steps), steps[:3])
        return "<Impossible>"

    # 4) Assemble UI-TARS-compatible output
    thought_bits: List[str] = []
    for step in steps[:3]:
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

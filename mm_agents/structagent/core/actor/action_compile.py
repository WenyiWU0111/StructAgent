"""Action compile: parse actor responses & emit pyautogui code.

Pure helpers (``_parse_action_ast``, ``_escape_single_quotes``,
``_fix_click_output``, ``_fix_drag_output``, ``_to_pyautogui``) live at
module top. ``_pa_parse_actor_response`` is the glue: dispatches
host-side actions (extract_info, note_write, open_app) and delegates the
rest to ``_to_pyautogui``.

``ActionCompile`` is mixed into ``StructAgent`` via inheritance. The
legacy ``_pa_`` prefix is kept so callers in ``agent.py`` don't move.
"""

import ast
import json
import logging
import re
from typing import List, Optional


_logger = logging.getLogger("desktopenv.qwen25vl_agent_planner")


def _parse_action_ast(action_str):
    """Parse ``click(start_box='(x,y)')`` style strings into func + kwargs.

    Kwargs are ``ast.literal_eval``'d so list/dict/tuple values (e.g.
    ``row_fields=['Promotion']``) survive; falls back to
    ``ast.Constant`` / ``ast.Str`` for non-literal nodes.
    """
    try:
        node = ast.parse(action_str, mode="eval")
        if not isinstance(node, ast.Expression):
            return None
        call = node.body
        if not isinstance(call, ast.Call):
            return None
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            func_name = None
        kwargs = {}
        for kw in call.keywords:
            key = kw.arg
            try:
                value = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError, TypeError):
                if isinstance(kw.value, ast.Constant):
                    value = kw.value.value
                elif isinstance(kw.value, ast.Str):
                    value = kw.value.s
                else:
                    value = None
            kwargs[key] = value
        return {"function": func_name, "args": kwargs}
    except Exception as e:
        print(f"[PA-Actor] Failed to parse action '{action_str}': {e}")
        return None


def _escape_single_quotes(text):
    return re.sub(r"(?<!\\)'", r"\\'", text)


def _fix_click_output(output: str, action_type: str = "click") -> Optional[str]:
    matches = re.findall(r"(\d+)\s*,\s*(\d+)", output)
    if matches:
        x, y = matches[-1]
        return f"{action_type}(start_box='({x},{y})')"
    return None


def _fix_drag_output(output: str) -> Optional[str]:
    matches = re.findall(r"(\d+)\s*,\s*(\d+)", output)
    if matches and len(matches) >= 2:
        x1, y1 = matches[-2]
        x2, y2 = matches[-1]
        return f"drag(start_box='({x1},{y1})', end_box='({x2},{y2})')"
    return None


def _to_pyautogui(action_type: str, action_inputs: dict,
                  image_width: int, image_height: int,
                  skip_select_all: bool = False,
                  skip_auto_enter: bool = False) -> Optional[str]:
    """Convert a single parsed action to pyautogui code string."""
    code = "import pyautogui\nimport time\n"

    if action_type == "hotkey":
        hotkey = action_inputs.get("key", "")
        for old, new in [("arrowleft", "left"), ("arrowright", "right"), ("arrowup", "up"), ("arrowdown", "down")]:
            if hotkey == old:
                hotkey = new
        if hotkey:
            keys = hotkey.replace("+", " ").split()
            convert_keys = [" " if k == "space" else k for k in keys]
            if convert_keys == ["backspace"] or convert_keys == ["delete"]:
                code += f"\npyautogui.hotkey('ctrl', 'a')"
                code += f"\ntime.sleep(0.1)"
            code += f"\npyautogui.hotkey({', '.join([repr(k) for k in convert_keys])})"
            return code

    elif action_type == "type":
        content = action_inputs.get("content", "")
        stripped = content
        while stripped.endswith("\\n"):
            stripped = stripped[:-2]
        while stripped.endswith("\n"):
            stripped = stripped[:-1]
        if not stripped:
            return code

        content_for_write = re.sub(r"(?<!\\)'", r"\\'", stripped)
        if not skip_select_all:
            code += f"\npyautogui.hotkey('ctrl', 'a')"
            code += f"\ntime.sleep(0.1)"
        safe_str = repr(content_for_write)
        code += f"\npyautogui.write({safe_str}, interval=0.05)"
        code += f"\ntime.sleep(0.5)"
        if not skip_auto_enter:
            code += f"\npyautogui.press('enter')"
        return code

    elif action_type in ["drag", "select"]:
        start_box = action_inputs.get("start_box")
        end_box = action_inputs.get("end_box")
        if start_box and end_box:
            x1, y1, x2, y2 = ast.literal_eval(start_box)
            sx = round(float((x1 + x2) / 2) * image_width, 3)
            sy = round(float((y1 + y2) / 2) * image_height, 3)
            x1, y1, x2, y2 = ast.literal_eval(end_box)
            ex = round(float((x1 + x2) / 2) * image_width, 3)
            ey = round(float((y1 + y2) / 2) * image_height, 3)
            code += f"\npyautogui.moveTo({sx}, {sy})"
            code += f"\npyautogui.dragTo({ex}, {ey}, duration=1.0)"
            return code

    elif action_type == "scroll":
        start_box = action_inputs.get("start_box")
        direction = action_inputs.get("direction", "")
        x, y = None, None
        if start_box:
            x1, y1, x2, y2 = ast.literal_eval(start_box)
            x = round(float((x1 + x2) / 2) * image_width, 3)
            y = round(float((y1 + y2) / 2) * image_height, 3)
        if x is None:
            if "up" in direction.lower():
                code += f"\npyautogui.scroll(10)"
            elif "down" in direction.lower():
                code += f"\npyautogui.scroll(-10)"
        else:
            if "up" in direction.lower():
                code += f"\npyautogui.scroll(10, x={x}, y={y})"
            elif "down" in direction.lower():
                code += f"\npyautogui.scroll(-10, x={x}, y={y})"
        return code

    elif action_type == "cli_run":
        # _pa_parse_actor_response stringifies scalar kwargs, so "False"
        # arrives truthy. Recover the bool explicitly.
        from mm_agents.structagent.actions.handlers.cli_run_helpers import build_runtime_script as _bcli
        cmd = action_inputs.get("command", "")
        bg_raw = action_inputs.get("background", False)
        if isinstance(bg_raw, str):
            bg = bg_raw.strip().lower() in ("true", "1", "yes", "on")
        else:
            bg = bool(bg_raw)
        ws = action_inputs.get("wait_seconds", 1)
        try:
            ws = int(ws)
        except (TypeError, ValueError):
            ws = 1
        try:
            return _bcli(cmd, background=bg, wait_seconds=ws)
        except ValueError as e:
            _logger.warning("[cli_run] %s", e)
            return None

    elif action_type == "open_app":
        from mm_agents.structagent.actions.app.open_app import build_open_app_script
        app = action_inputs.get("app", "")
        file_path = action_inputs.get("file_path") or None
        return build_open_app_script(app, file_path, logger=_logger)

    elif action_type == "navigate":
        # Chrome navigation via the DevTools HTTP endpoint on port 1337 in
        # the VM (socat'd to 9222 outside). One action, no grounding —
        # beats mimicking ctrl+L → type → enter. Used by cross-site
        # multi-hop tasks (MMInA compare / multi567 / multipro / normal).
        from mm_agents.structagent.actions.handlers.navigate_helpers import (
            build_navigate_script,
        )
        url = action_inputs.get("url", "")
        new_tab_raw = action_inputs.get("new_tab", False)
        if isinstance(new_tab_raw, str):
            new_tab = new_tab_raw.strip().lower() in ("true", "1", "yes", "on")
        else:
            new_tab = bool(new_tab_raw)
        try:
            return build_navigate_script(url, new_tab=new_tab)
        except ValueError as e:
            _logger.warning("[navigate] %s", e)
            return None

    elif action_type.startswith("calc_"):
        from mm_agents.structagent.actions.calc.actions import (
            CALC_ACTION_NAMES, action_to_runtime_script,
        )
        if action_type not in CALC_ACTION_NAMES:
            _logger.warning("[%s] unknown calc_* action — supported: %s",
                            action_type, sorted(CALC_ACTION_NAMES))
            return None
        return action_to_runtime_script(action_type, action_inputs, logger=_logger)

    elif action_type.startswith("impress_"):
        from mm_agents.structagent.actions.impress.actions import (
            IMPRESS_ACTION_NAMES,
            action_to_runtime_script as _impress_action_to_runtime,
        )
        if action_type not in IMPRESS_ACTION_NAMES:
            _logger.warning("[%s] unknown impress_* action — supported: %s",
                            action_type, sorted(IMPRESS_ACTION_NAMES))
            return None
        return _impress_action_to_runtime(action_type, action_inputs, logger=_logger)

    elif action_type.startswith("writer_"):
        from mm_agents.structagent.actions.writer.actions import (
            WRITER_ACTION_NAMES,
            action_to_runtime_script as _writer_action_to_runtime,
        )
        if action_type not in WRITER_ACTION_NAMES:
            _logger.warning("[%s] unknown writer_* action — supported: %s",
                            action_type, sorted(WRITER_ACTION_NAMES))
            return None
        return _writer_action_to_runtime(action_type, action_inputs, logger=_logger)

    elif action_type == "edit_json":
        from mm_agents.structagent.actions.handlers.edit_json_helpers import build_runtime_script
        path = action_inputs.get("path", "")
        op = action_inputs.get("op", "")
        data_raw = action_inputs.get("data", "null")
        try:
            data = json.loads(data_raw) if isinstance(data_raw, str) else data_raw
        except Exception:
            _logger.warning("[edit_json] bad data JSON: %r", data_raw)
            return None
        try:
            return build_runtime_script(path=path, op=op, data=data)
        except ValueError as e:
            _logger.warning("[edit_json] %s", e)
            return None

    elif action_type in ["click", "left_single", "left_double", "right_single", "hover"]:
        start_box = action_inputs.get("start_box")
        if start_box:
            box = ast.literal_eval(str(start_box))
            if len(box) == 4:
                x1, y1, x2, y2 = box
            elif len(box) == 2:
                x1, y1 = box
                x2, y2 = x1, y1
            else:
                return None
            x = round(float((x1 + x2) / 2) * image_width, 3)
            y = round(float((y1 + y2) / 2) * image_height, 3)
            if action_type in ("left_single", "click"):
                code += f"\npyautogui.click({x}, {y}, button='left')"
            elif action_type == "left_double":
                code += f"\npyautogui.doubleClick({x}, {y}, button='left')"
            elif action_type == "right_single":
                code += f"\npyautogui.click({x}, {y}, button='right')"
            elif action_type == "hover":
                code += f"\npyautogui.moveTo({x}, {y})"
            return code

    return None


class ActionCompile:
    """Actor response → list of pyautogui code strings."""

    # Expose the pure helpers as static methods so existing
    # ``self._pa_*`` call sites keep working.
    _pa_parse_action_ast = staticmethod(_parse_action_ast)
    _pa_escape_single_quotes = staticmethod(_escape_single_quotes)
    _pa_fix_click_output = staticmethod(_fix_click_output)
    _pa_fix_drag_output = staticmethod(_fix_drag_output)
    _pa_to_pyautogui = staticmethod(_to_pyautogui)

    def _pa_parse_actor_response(self, response: str, model_image_width: int, model_image_height: int,
                                  screen_width: int, screen_height: int,
                                  env=None) -> List[str]:
        """Parse actor response into pyautogui code strings.

        ``["import pyautogui\\n..."]`` for runnable actions, sentinels
        ``["DONE"]`` / ``["WAIT"]`` / ``["IMPOSSIBLE"]``, or ``[]`` when
        malformed.
        """
        response = response.strip()
        if "<impossible>" in response.lower():
            return ["IMPOSSIBLE"]
        if not response or "Action:" not in response:
            return []

        action_str = response.split("Action:")[-1]

        # If the Thought mentions right-click but the Action uses click(), upgrade to right_single()
        thought_str = response.split("Action:")[0] if "Action:" in response else ""
        if re.search(r"right[\s-]*click", thought_str, re.IGNORECASE):
            action_str = re.sub(r"\bclick\(start_box=", "right_single(start_box=", action_str)

        tmp_all_action = action_str.split("\n\n")
        all_action = []
        for a in tmp_all_action:
            if "type(content" in a:
                # Multi-line bodies (heredocs / scripts the planner asks
                # the actor to type) span newlines — DOTALL required.
                pattern = r"type\(content='(.*?)'\)"
                m = re.search(pattern, a, flags=re.DOTALL)
                if m:
                    content = _escape_single_quotes(m.group(1))
                    a = "type(content='" + content + "')"
            elif "right_single(start_box" in a or "right_single(" in a:
                fixed = _fix_click_output(a, action_type="right_single")
                if fixed:
                    a = fixed
            elif "left_double(start_box" in a or "left_double(" in a:
                fixed = _fix_click_output(a, action_type="left_double")
                if fixed:
                    a = fixed
            elif "click(start_box" in a:
                fixed = _fix_click_output(a)
                if fixed:
                    a = fixed
            elif "drag(start_box" in a:
                fixed = _fix_drag_output(a)
                if fixed:
                    a = fixed
            all_action.append(a)

        parsed_actions = [_parse_action_ast(a.replace("\n", "\\n").lstrip()) for a in all_action]
        valid_actions = [a for a in parsed_actions if a is not None]

        pyautogui_codes = []
        prev_action_type = None
        for idx, action_instance in enumerate(valid_actions):
            action_type = action_instance["function"]
            params = action_instance["args"]

            if action_type == "finished":
                return ["DONE"]
            if action_type == "wait":
                # Queue wait() as a no-op step, not a mid-batch early
                # return. Old ``return ["WAIT"]`` dropped every real action
                # earlier in the batch — e1e75309 steps 4-6 emitted
                # ``click(...)\n\nwait()`` and the click coords vanished.
                # env.step("WAIT") is a valid runtime action, so append.
                pyautogui_codes.append("WAIT")
                continue

            # Normalize box coords. Only stringify scalars; container
            # kwargs (list/dict/tuple) must round-trip natively to the
            # ``_build_*`` validators.
            action_inputs = {}
            for param_name, param in params.items():
                if param == "" or param is None:
                    continue
                if isinstance(param, str):
                    stripped = param.lstrip()
                    if stripped:
                        param = stripped
                action_inputs[param_name.strip()] = param
                if (("start_box" in param_name or "end_box" in param_name)
                        and isinstance(param, str)):
                    ori_box = param
                    numbers = ori_box.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
                    numbers = [float(n) for n in numbers if n.strip()]
                    if model_image_width > 0 and model_image_height > 0:
                        float_numbers = []
                        for i in range(0, len(numbers), 2):
                            float_numbers.append(numbers[i] / model_image_width)
                            if i + 1 < len(numbers):
                                float_numbers.append(numbers[i + 1] / model_image_height)
                    else:
                        float_numbers = [num / 1000 for num in numbers]
                    if len(float_numbers) == 2:
                        float_numbers = [float_numbers[0], float_numbers[1], float_numbers[0], float_numbers[1]]
                    action_inputs[param_name.strip()] = str(float_numbers)

            # ``type``: Ctrl+A and Enter are explicit params, both
            # defaulting off. The old auto-Ctrl+A heuristic caused data
            # loss in code editors.
            skip_select = (action_type == "type"
                           and not bool(action_inputs.get("wipe_then_type", False)))
            _c = action_inputs.get("content", "") or ""
            _wants_submit = (
                bool(action_inputs.get("submit", False))
                or _c.endswith(("\n", "\\n"))
            )
            skip_enter = action_type == "type" and not _wants_submit
            # _to_pyautogui is static — smuggle current domain via the
            # reserved ``__current_domain`` key for the runtime boilerplate.
            action_inputs.setdefault("__current_domain",
                                      getattr(self, "_current_domain", None))

            if action_type == "extract_info":
                from mm_agents.structagent.actions.handlers.extract_info import handle as _handle_ei
                from mm_agents.structagent.actions.handlers.notebook import write_extract_info_batch
                _ei_batch = _handle_ei(
                    env=env, action_inputs=action_inputs,
                    call_llm=self.call_llm, model=self.model,
                    logger=_logger,
                )
                if _ei_batch:
                    # owning_outcome defaults to current focus; routes to
                    # ledger.global_facts when no focus is set.
                    _focus = None
                    try:
                        _f = self.ledger.next_focus_outcome() if self.ledger else None
                        _focus = _f.id if _f is not None else None
                    except Exception:
                        _focus = None
                    n_w = write_extract_info_batch(
                        _ei_batch,
                        ledger=self.ledger,
                        owning_outcome=_focus,
                        step_idx=getattr(self, "_wall_step_idx", 0) or 0,
                    )
                    _logger.info("[working memory] extract_info wrote %d entr%s",
                                 n_w, "y" if n_w == 1 else "ies")
                prev_action_type = action_type
                continue

            if action_type == "note_write":
                from mm_agents.structagent.actions.handlers.notebook import handle_note_write
                _focus = None
                try:
                    _f = self.ledger.next_focus_outcome() if self.ledger else None
                    _focus = _f.id if _f is not None else None
                except Exception:
                    _focus = None
                handle_note_write(
                    action_inputs,
                    ledger=self.ledger,
                    logger=_logger,
                    owning_outcome=_focus,
                    step_idx=getattr(self, "_wall_step_idx", 0) or 0,
                )
                prev_action_type = action_type
                continue

            # open_app with no file_path: fall back to the initial-file
            # registry from task start.
            if action_type == "open_app" and not action_inputs.get("file_path"):
                try:
                    from mm_agents.structagent.actions.app.open_app import normalize_app
                    _reg = getattr(self, "_initial_open_files", None) or {}
                    _capp = normalize_app(action_inputs.get("app"))
                    if _capp and _capp in _reg:
                        action_inputs["file_path"] = _reg[_capp]
                        _logger.info(
                            "[open_app] file_path filled from initial "
                            "registry: %s -> %s",
                            _capp, _reg[_capp])
                except Exception:
                    pass
            code = _to_pyautogui(action_type, action_inputs, screen_width, screen_height,
                                  skip_select_all=skip_select, skip_auto_enter=skip_enter)
            if code:
                pyautogui_codes.append(code)
            prev_action_type = action_type

        return pyautogui_codes

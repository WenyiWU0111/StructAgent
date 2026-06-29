"""UNO-runtime-script builder for the structured LibreOffice action sets
(calc_* / impress_* / writer_*) and their verify specs.

``build_runtime_script`` wraps a UNO-Python body with connect+setup
boilerplate against a live LibreOffice via the UNO bridge (socket port
2002). One socket serves Writer/Calc/Impress; the domain setup block
re-binds ``doc`` by service type and exposes the right handles
(text/cursor/paragraphs, sheets, slides). The combined block is exec()'d
with stdout/stderr captured and written to ``/tmp/_last_cli_run.json``
(the same sentinel cli_run uses, so ``_pa_predict`` picks it up).

Not a standalone actor action: the actor emits structured calc_*/impress_*/
writer_* JSON, which the framework lowers to a UNO body and routes here.
(The raw-UNO ``cli_run_uno`` action was removed — it just lured the actor
into hand-writing UNO.)

Assumed on OSWorld VMs (true for any passing libreoffice_* task): python
has the ``uno`` module; LibreOffice is running with the document open.
Listener may be off — we retry-with-soffice on first connect failure.
"""
from __future__ import annotations
import json
from typing import Optional


# Line-1 marker on every script build_runtime_script produces. Callers that
# might wrap twice test for it and skip — see
# verifiers._maybe_wrap_calc_verify_command.
WRAPPED_SENTINEL = "# __cli_run_uno_wrapped__"


# Connect-or-launch block. First _connect() works if LO is already
# listening on 2002; on the OSWorld VM it usually isn't, so we Popen
# ``soffice --accept=...`` to enable the listener and retry once.
# Idempotent — if already listening, the second soffice no-ops on the
# existing instance's lockfile.
_CONNECT_BLOCK = """import uno, subprocess, time
def _connect():
    ctx = uno.getComponentContext()
    res = ctx.ServiceManager.createInstanceWithContext(
        'com.sun.star.bridge.UnoUrlResolver', ctx)
    return res.resolve(
        'uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')

try:
    remote_ctx = _connect()
except Exception:
    # Enable UNO listener on the already-running LO instance, then retry.
    subprocess.Popen([
        'soffice',
        '--accept=socket,host=localhost,port=2002;urp;StarOffice.Service',
    ])
    time.sleep(2)
    remote_ctx = _connect()

desktop = remote_ctx.ServiceManager.createInstanceWithContext(
    'com.sun.star.frame.Desktop', remote_ctx)
doc = desktop.getCurrentComponent()
"""


# Per-domain SETUP block, prepended after the connect block. Exposes the
# top-level handles the code templates assume. Unknown domains get
# connect-only — actors can still pull handles off `doc`.
_DOMAIN_SETUP_BLOCKS = {
    "libreoffice_writer": (
        # ``doc`` from connect is getCurrentComponent() = whatever window
        # is focused — often Calc/Impress at verify time in a multi-app
        # task, so ``doc.Text`` raises AttributeError. Re-bind to the open
        # Writer document by service type.
        "if doc is None or not doc.supportsService("
        "'com.sun.star.text.TextDocument'):\n"
        "    _we = desktop.getComponents().createEnumeration()\n"
        "    doc = None\n"
        "    while _we.hasMoreElements():\n"
        "        _wc = _we.nextElement()\n"
        "        try:\n"
        "            if _wc.supportsService('com.sun.star.text.TextDocument'):\n"
        "                doc = _wc\n"
        "                break\n"
        "        except Exception:\n"
        "            pass\n"
        "if doc is None or not doc.supportsService("
        "'com.sun.star.text.TextDocument'):\n"
        "    raise RuntimeError("
        "'No LibreOffice Writer document is open. '\n"
        "        'FIX: open the target .docx/.odt file first.')\n"
        "text = doc.Text\n"
        "cursor = text.createTextCursor()\n"
        # view_cursor = the user's visible cursor. For insert-at-cursor ops
        # build an insert cursor from view_cursor.getStart(); for
        # whole-document ops use `cursor` or iterate paragraphs instead.
        "view_cursor = doc.getCurrentController().getViewCursor()\n"
        "paragraphs = [p for p in text.createEnumeration()\n"
        "              if p.supportsService('com.sun.star.text.Paragraph')]\n"
    ),
    "libreoffice_calc": (
        # Calc handles: doc.Sheets (index + name access); controller is the
        # live view — active_sheet/active_selection follow the user.
        #
        # ``doc`` from connect is getCurrentComponent() = the focused window,
        # often Writer/Impress at verify time. Re-bind to the open Calc doc
        # by service type. If none is open (e.g. LO sitting on the Start
        # Center, where doc has no Sheets attr), raise an actionable error
        # instead of a cryptic AttributeError — so the planner's next-turn
        # [Last cli_run output] tells it to open the file first.
        "def _calc_is(_d):\n"
        "    try:\n"
        "        return bool(_d is not None and _d.supportsService('com.sun.star.sheet.SpreadsheetDocument'))\n"
        "    except Exception:\n"
        "        return hasattr(_d, 'Sheets')\n"
        "if not _calc_is(doc):\n"
        "    _ce = desktop.getComponents().createEnumeration()\n"
        "    doc = None\n"
        "    while _ce.hasMoreElements():\n"
        "        _cc = _ce.nextElement()\n"
        "        if _calc_is(_cc):\n"
        "            doc = _cc\n"
        "            break\n"
        "if not _calc_is(doc):\n"
        "    raise RuntimeError(\n"
        "        'No LibreOffice Calc document is currently open. '\n"
        "        'FIX: open the target .xlsx/.ods file via the GUI first '\n"
        "        '(File menu > Open, Ctrl+O, or click a file in '\n"
        "        'Recent Documents). Re-issue this cli_run_uno step '\n"
        "        'only AFTER the file is loaded.'\n"
        "    )\n"
        "sheets = doc.Sheets\n"
        "controller = doc.getCurrentController()\n"
        "active_sheet = controller.getActiveSheet()\n"
        "active_selection = controller.getSelection()\n"
        "sheet_names = list(sheets.getElementNames())\n"
        # Snapshot each sheet's used-range bounds before the actor's code.
        # The post-block diffs against it to find the mutated sheet and
        # auto-switch the active view there, so the next screen capture
        # shows the change without the actor calling setActiveSheet.
        "_pre_sheet_names = list(sheet_names)\n"
        "_pre_sheet_bounds = {}\n"
        "for _n in sheet_names:\n"
        "    try:\n"
        "        _c = sheets.getByName(_n).createCursor()\n"
        "        _c.gotoEndOfUsedArea(True)\n"
        "        _ra = _c.getRangeAddress()\n"
        "        _pre_sheet_bounds[_n] = (int(_ra.EndColumn), int(_ra.EndRow))\n"
        "    except Exception:\n"
        "        _pre_sheet_bounds[_n] = (-1, -1)\n"
    ),
    "libreoffice_impress": (
        # ``doc`` from connect is getCurrentComponent() = the focused window,
        # often Writer/Calc at verify time (then ``doc.DrawPages`` raises
        # AttributeError). Re-bind to the open Impress doc by service type.
        "if doc is None or not doc.supportsService("
        "'com.sun.star.presentation.PresentationDocument'):\n"
        "    _ie = desktop.getComponents().createEnumeration()\n"
        "    doc = None\n"
        "    while _ie.hasMoreElements():\n"
        "        _ic = _ie.nextElement()\n"
        "        try:\n"
        "            if _ic.supportsService('com.sun.star.presentation.PresentationDocument'):\n"
        "                doc = _ic\n"
        "                break\n"
        "        except Exception:\n"
        "            pass\n"
        "if doc is None or not doc.supportsService("
        "'com.sun.star.presentation.PresentationDocument'):\n"
        "    raise RuntimeError("
        "'No LibreOffice Impress document is open. '\n"
        "        'FIX: open the target .pptx/.odp file first.')\n"
        # Impress: slides accessible via doc.DrawPages
        "slides = doc.DrawPages\n"
        "controller = doc.getCurrentController()\n"
        "active_slide = controller.getCurrentPage()\n"
    ),
}


def _ops_library_for(domain: Optional[str]) -> str:
    """Per-domain pre-built ops library, concatenated between the setup
    blocks and the actor's code so the actor can call op_*/verify_*
    directly without writing UNO.

    Fast path: if _ops_lib_remote already uploaded it to /tmp/<mod>.py
    (once per process at predict() entry), return a tiny ``import *`` shim
    (~80 bytes vs ~120KB inlined). Fallback: inline the full library text
    (cold start, env without run_python_script). Unknown domains: none.
    """
    dom = (domain or "").lower().strip().replace("-", "_")
    if dom not in ("libreoffice_calc", "libreoffice_impress",
                   "libreoffice_writer"):
        return ""
    try:
        from mm_agents.structagent.actions.uno._ops_lib_remote import get_uploaded_module
        mod = get_uploaded_module(dom)
    except Exception:
        mod = None
    if mod is not None:
        # Already uploaded to /tmp — emit the import shim.
        #
        # Namespace bridge. The library's op_*/verify_* fns were written to
        # be INLINED, referencing doc/slides/controller/sheets/... as bare
        # globals. Imported as a module, their __globals__ point at the
        # module namespace where those handles aren't bound (NameError).
        # Fix: copy every runtime handle into the module's __dict__ after
        # import. Skip underscore names (internal scaffolding the lib never
        # needs). Runs after connect+setup, so globals() already has them.
        return (
            "import sys as _sys\n"
            "if '/tmp' not in _sys.path:\n"
            "    _sys.path.insert(0, '/tmp')\n"
            f"import {mod} as _ops_mod\n"
            # (1) Bridge runtime handles into the module namespace so its
            # functions (whose __globals__ point at the module) resolve them.
            "for _k, _v in list(globals().items()):\n"
            "    if not _k.startswith('_'):\n"
            "        setattr(_ops_mod, _k, _v)\n"
            # (2) Pull EVERY module name into the runtime namespace, not
            # just the ``import *`` subset — that skips underscore names, but
            # the post-block calls ``_impress_get_slide`` directly. Without
            # it the post-block NameErrors and silently no-ops, so the view
            # never follows the edit.
            "for _k in dir(_ops_mod):\n"
            "    if not (_k.startswith('__') and _k.endswith('__')):\n"
            "        globals()[_k] = getattr(_ops_mod, _k)\n"
        )
    # Inline fallback — before the first successful upload (e.g. cold-start
    # tests that don't go through predict()).
    if dom == "libreoffice_calc":
        try:
            from mm_agents.structagent.actions.calc.ops_lib import CALC_OPS_LIBRARY
        except Exception:
            return ""
        return CALC_OPS_LIBRARY
    if dom == "libreoffice_impress":
        try:
            from mm_agents.structagent.actions.impress.ops_lib import IMPRESS_OPS_LIBRARY
        except Exception:
            return ""
        return IMPRESS_OPS_LIBRARY
    if dom == "libreoffice_writer":
        try:
            from mm_agents.structagent.actions.writer.ops_lib import WRITER_OPS_LIBRARY
        except Exception:
            return ""
        return WRITER_OPS_LIBRARY
    return ""


def _boilerplate_for(domain: Optional[str]) -> str:
    """Full prepended boilerplate (connect + domain setup + ops library).
    Unknown domains get connect-only."""
    setup = _DOMAIN_SETUP_BLOCKS.get(
        (domain or "").lower().strip().replace("-", "_"),
        "",
    )
    return _CONNECT_BLOCK + setup + _ops_library_for(domain)


# Per-domain POST-block appended after the actor's code, to normalize state
# the actor shouldn't have to manage. Wrapped in its own try/except so a
# failure here can't mask the actor's print.
_DOMAIN_POST_BLOCKS = {
    "libreoffice_calc": (
        # Diff each sheet's used-range bounds vs the pre-snapshot and switch
        # the active view to the mutated sheet, so the next capture shows the
        # change. Newly-added sheets win over expanded existing ones
        # ('create then populate' wants the new sheet visible).
        "try:\n"
        "    _post_sheet_names = list(sheets.getElementNames())\n"
        "    _modified = None\n"
        # Newly-added sheet — last one wins if >1.
        "    for _n in _post_sheet_names:\n"
        "        if _n not in _pre_sheet_names:\n"
        "            _modified = _n\n"
        # Else: existing sheet whose used range changed.
        "    if _modified is None:\n"
        "        for _n in _post_sheet_names:\n"
        "            try:\n"
        "                _c = sheets.getByName(_n).createCursor()\n"
        "                _c.gotoEndOfUsedArea(True)\n"
        "                _ra = _c.getRangeAddress()\n"
        "                _now = (int(_ra.EndColumn), int(_ra.EndRow))\n"
        "                if _now != _pre_sheet_bounds.get(_n, (-1, -1)):\n"
        "                    _modified = _n\n"
        "            except Exception: pass\n"
        # Switch view if the modified sheet isn't already active.
        "    if _modified is not None:\n"
        "        try:\n"
        "            _cur_active = controller.getActiveSheet().getName()\n"
        "        except Exception:\n"
        "            _cur_active = None\n"
        "        if _modified != _cur_active:\n"
        "            try:\n"
        "                controller.setActiveSheet(sheets.getByName(_modified))\n"
        "                doc.store()\n"
        "            except Exception: pass\n"
        "except Exception: pass\n"
    ),
    "libreoffice_impress": (
        # Navigate the view to the slide the action targeted. impress_*
        # actions (impress_actions._build_generic) prepend
        # ``_pa_target_slide = <slide>`` so this fires even when the actor's
        # code doesn't move the view. Without it, set_font(slide=5) mutates
        # slide 5 but the view stays on slide 1 — the next screenshot shows
        # no change and the planner thinks the action failed.
        "try:\n"
        "    _tgt = globals().get('_pa_target_slide')\n"
        "    if _tgt is not None:\n"
        "        try:\n"
        "            _tgt_sl = _impress_get_slide(_tgt)\n"
        "        except Exception:\n"
        "            _tgt_sl = None\n"
        "        if _tgt_sl is not None:\n"
        "            try:\n"
        "                _cur_sl = controller.getCurrentPage()\n"
        "            except Exception:\n"
        "                _cur_sl = None\n"
        "            if _cur_sl is None or _cur_sl is not _tgt_sl:\n"
        "                try:\n"
        "                    controller.setCurrentPage(_tgt_sl)\n"
        "                except Exception: pass\n"
        "except Exception: pass\n"
    ),
}


def _post_block_for(domain: Optional[str]) -> str:
    return _DOMAIN_POST_BLOCKS.get(
        (domain or "").lower().strip().replace("-", "_"),
        "",
    )


def build_runtime_script(code: str, domain: Optional[str] = None) -> str:
    """Build the Python runtime script the VM runs (via ``python -c``) for a
    cli_run_uno action.

    Prepends the UNO connect boilerplate, exec()s the combined script with
    stdout/stderr captured, echoes a one-line status to the trajectory log,
    and persists results to ``/tmp/_last_cli_run.json`` (the path cli_run
    uses, read by the planner next turn).
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("cli_run_uno requires a non-empty `code` string")

    # The decomposer sometimes double-escapes newlines in its JSON `code`,
    # sending literal ``\n`` (2 chars) where a real newline was meant —
    # which the tokenizer rejects. If `code` fails to compile but compiles
    # after literal-\n → real-newline, apply the rewrite. Only when it both
    # fails before and compiles after — never on valid code.
    try:
        compile(code, "<cli_run_uno>", "exec")
    except SyntaxError:
        candidate = code.replace("\\n", "\n")
        try:
            compile(candidate, "<cli_run_uno-normalized>", "exec")
        except SyntaxError:
            pass  # leave original; runtime traceback informs the planner
        else:
            code = candidate

    # Bracket-mismatch repair — small Calc decomposers (qwen3.5-9b) reliably
    # drop a closing `]` on long single-column lists (``values=[[v1],…,[v11])``
    # should be ``…[v11]])``), giving "closing parenthesis ')' does not match
    # opening parenthesis '['". Try inserting 1 then 2 `]` before the
    # offending `)`; use the first that compiles. Compile-time only, never
    # touches code that already compiles or a non-bracket SyntaxError.
    try:
        compile(code, "<cli_run_uno-bracket-check>", "exec")
    except SyntaxError as _e:
        _msg = str(getattr(_e, "msg", "") or "")
        _ln = getattr(_e, "lineno", None) or 0
        _off = getattr(_e, "offset", None) or 0
        if (
            "does not match opening parenthesis '['" in _msg
            and 0 < _ln
            and 0 < _off
        ):
            _lines = code.split("\n")
            if _ln <= len(_lines):
                _line = _lines[_ln - 1]
                # offset is 1-based, points at the offending `)`
                if 0 < _off <= len(_line) and _line[_off - 1] == ")":
                    for _n_inserts in (1, 2):
                        _fixed_line = (
                            _line[: _off - 1] + ("]" * _n_inserts) + _line[_off - 1 :]
                        )
                        _trial = "\n".join(
                            _lines[: _ln - 1] + [_fixed_line] + _lines[_ln :]
                        )
                        try:
                            compile(_trial, "<cli_run_uno-bracket-fix>", "exec")
                        except SyntaxError:
                            continue
                        else:
                            code = _trial
                            break

    inner_parts = [
        _boilerplate_for(domain),
        "\n# === Actor operation code ===\n",
        code,
    ]
    post = _post_block_for(domain)
    if post:
        # Newline so the post-block doesn't fuse with the actor's last line.
        inner_parts.append("\n# === Post: auto-normalize view state ===\n")
        inner_parts.append(post)
    inner = "".join(inner_parts)
    inner_repr = json.dumps(inner)

    # Sentinel first line — lets callers (e.g.
    # _maybe_wrap_calc_verify_command) detect an already-wrapped script and
    # skip re-wrapping. A harmless comment everywhere else.
    return f"""{WRAPPED_SENTINEL}
import io, json, os, sys, time, traceback

_script = {inner_repr}

_stdout_buf = io.StringIO()
_stderr_buf = io.StringIO()
_returncode = 0
_orig_out, _orig_err = sys.stdout, sys.stderr
sys.stdout = _stdout_buf
sys.stderr = _stderr_buf
try:
    exec(_script, {{'__name__': '__main__'}})
except SystemExit as _exc:
    # actor's script can sys.exit(0|1); honor the code as our returncode
    try:
        _returncode = int(_exc.code) if _exc.code is not None else 0
    except (TypeError, ValueError):
        _returncode = 1
except Exception:
    _returncode = 1
    traceback.print_exc(file=_stderr_buf)
finally:
    sys.stdout = _orig_out
    sys.stderr = _orig_err

_stdout = _stdout_buf.getvalue()
_stderr = _stderr_buf.getvalue()

def _head_tail_trunc(s, budget, head_frac=0.3):
    # Head + tail truncation. Naive s[:budget] cuts the END of Python
    # tracebacks, which is where the actual error class+message live —
    # without that line the planner has no idea WHY the command failed
    # and falls back to guessing. Keep head + tail joined by an
    # "...omitted..." marker so the SyntaxError / TypeError / etc.
    # remains visible.
    if not s or len(s) <= budget:
        return s
    head_n = max(80, int(budget * head_frac))
    marker = f"\\n  ... [{{len(s) - budget}} chars omitted] ...\\n"
    tail_n = budget - head_n - len(marker)
    if tail_n < 80:
        return s[:budget]
    return s[:head_n] + marker + s[-tail_n:]

# Echo to trajectory log (visible in runtime.log)
print(f'cli_run_uno [fg]: returncode={{_returncode}}')
if _stdout:
    print(f'cli_run_uno stdout:\\n{{_head_tail_trunc(_stdout, 2000)}}')
if _stderr:
    print(f'cli_run_uno stderr:\\n{{_head_tail_trunc(_stderr, 2000)}}')

# Persist to sentinel — _pa_predict picks this up on next turn and
# inlines a [Last cli_run output] block into the planner prompt.
try:
    os.makedirs('/tmp', exist_ok=True)
    _snip = _script[:400]
    if len(_script) > 400:
        _snip += '... (truncated)'
    with open('/tmp/_last_cli_run.json', 'w') as _f:
        json.dump({{
            'command': '[cli_run_uno] ' + _snip,
            'mode': 'uno',
            'returncode': _returncode,
            'stdout': _head_tail_trunc(_stdout, 4000),
            'stderr': _head_tail_trunc(_stderr, 2000),
            'timestamp': time.time(),
        }}, _f)
except Exception:
    pass
"""

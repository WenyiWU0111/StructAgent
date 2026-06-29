"""Shared UNO-runtime-script builder for the structured LibreOffice
action sets (calc_* / impress_* / writer_*) and their verify specs.

``build_runtime_script`` wraps a body of UNO Python with a connect-
and-setup boilerplate against a LIVE LibreOffice instance via the UNO
bridge (socket port 2002). Works for ALL of LibreOffice — Writer (text
docs), Calc (spreadsheets), Impress (presentations) — because UNO uses
a single unified socket. The domain-specific setup block (paragraphs
for Writer, sheets for Calc, slides for Impress) resolves ``doc`` by
matching the requested service type among the open components, and
binds the right top-level handles (``doc``, plus ``text/cursor/
paragraphs`` for Writer, ``sheets`` for Calc, etc.). The combined
block is exec()'d with stdout/stderr captured and persisted to
``/tmp/_last_cli_run.json`` — the same sentinel file cli_run uses, so
the planner-side fetch in ``_pa_predict`` picks it up unmodified.

This module is NOT a standalone actor action. The actor only ever
emits the structured calc_* / impress_* / writer_* JSON actions; the
framework deterministically lowers each to a UNO body and routes it
through ``build_runtime_script`` here. (The former raw-UNO
``cli_run_uno`` actor action was removed — it only mislead the actor
into hand-writing UNO.)

Prerequisites assumed (already true on OSWorld VMs that pass any
libreoffice_* task):
  * /usr/bin/python (or python3) has the ``uno`` module
  * LibreOffice is running with the task's document open
  * (Listener may or may not be enabled — we retry-with-soffice on
    first connect failure to cover both cases.)
"""
from __future__ import annotations
import json
from typing import Optional


# Marker on line 1 of every script ``build_runtime_script`` produces.
# Callers that might wrap a script a second time test for this and
# skip — see ``verifiers._maybe_wrap_calc_verify_command``.
WRAPPED_SENTINEL = "# __cli_run_uno_wrapped__"


# Shared connect-or-launch block. The first ``_connect()`` succeeds
# when LO is already listening on port 2002; on the OSWorld VM it
# usually isn't by default, so we Popen ``soffice --accept=...`` to
# enable the listener on the running LO and retry once. This is
# idempotent — when the listener is already up the soffice call is a
# no-op (the second soffice exits on the existing instance's lockfile).
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


# Per-domain SETUP block, prepended AFTER the connect block. Exposes
# the natural top-level handles the schema's code templates assume.
# Unknown domains get the connect-only flavor — actors can still pull
# handles off `doc` themselves.
_DOMAIN_SETUP_BLOCKS = {
    "libreoffice_writer": (
        # ``doc`` from the connect block is desktop.getCurrentComponent()
        # — whatever window is FOCUSED. In a multi-app task that is
        # often Calc/Impress at verify time, not the Writer document,
        # and ``doc.Text`` then raises ``AttributeError: Text``.
        # Re-bind ``doc`` to the open Writer document by service type.
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
        # view_cursor tracks the USER's visible cursor (where they
        # would type next). For insert-at-cursor ops (insert table /
        # image / page break at the current location), build an insert
        # cursor via text.createTextCursorByRange(view_cursor.getStart()).
        # For whole-document ops (set font on all paragraphs, etc.),
        # ignore view_cursor and use `cursor` or iterate paragraphs.
        "view_cursor = doc.getCurrentController().getViewCursor()\n"
        "paragraphs = [p for p in text.createEnumeration()\n"
        "              if p.supportsService('com.sun.star.text.Paragraph')]\n"
    ),
    "libreoffice_calc": (
        # Calc: sheets accessible via doc.Sheets (XSpreadsheets — supports
        # both index and name access). The controller is the live view —
        # active_sheet is the sheet the user is currently on, and
        # active_selection is whatever range the user has selected (could
        # be a single Cell, a CellRange, or a MultiSelection).
        #
        # FAIL-FAST guard: when LibreOffice is sitting on the Start Center
        # (no .xlsx/.ods loaded), `desktop.getCurrentComponent()` returns
        # the start-center component which has NO Sheets attribute. The
        # natural `doc.Sheets` line below would then raise
        # `AttributeError: Sheets` — a cryptic trace that doesn't tell
        # the planner the real fix (open the file first). Detect this
        # state up front and raise a clear RuntimeError instead, so the
        # planner's next-turn [Last cli_run output] block surfaces the
        # actionable diagnosis.
        # ``doc`` from the connect block is desktop.getCurrentComponent()
        # — whatever window is FOCUSED. In a multi-app task that is
        # often Writer/Impress at verify time, not the Calc document.
        # Re-bind ``doc`` to the open Calc document by service type;
        # only when NONE is open do we raise the actionable error.
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
        # Snapshot of every sheet's used-range bounds BEFORE the actor's
        # code runs. The post-block (appended after the actor's code)
        # diffs against this snapshot to detect which sheet the actor
        # mutated and automatically switches the active view to it. Lets
        # the next-step screen capture reflect the change without the
        # actor having to remember to call setActiveSheet.
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
        # ``doc`` from the connect block is desktop.getCurrentComponent()
        # — whatever window is FOCUSED. In a multi-app task that is
        # often Writer/Calc at verify time, not the Impress document
        # (then ``doc.DrawPages`` raises AttributeError). Re-bind
        # ``doc`` to the open Impress document by service type.
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
    """Per-domain pre-built operations library. Concatenated AFTER the
    connect+setup blocks and BEFORE the actor's code, so the actor's
    snippet can call ``op_*(...)`` / ``verify_*(...)`` functions
    directly without writing any UNO API code itself.

    Fast path — if ``mm_agents.actions.uno._ops_lib_remote.maybe_upload_ops_lib``
    has uploaded the library to ``/tmp/<modname>.py`` already (called
    once per process at agent ``predict()`` entry), this returns a
    tiny ``from <modname> import *`` shim. Inline cost drops from
    ~120KB to ~80 bytes per cli_run_uno dispatch.

    Slow path / fallback — when the upload hasn't happened (cold
    start, env without ``run_python_script``, etc.), returns the
    full library text inlined as before.

    Unknown domains get no library.
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
        # Already uploaded to /tmp on the VM — emit the lightweight
        # import shim.
        #
        # CRITICAL — namespace bridge. The op_*/verify_* functions in
        # the library were authored to be INLINED: they reference
        # ``doc`` / ``slides`` / ``controller`` / ``active_slide`` /
        # ``remote_ctx`` / ``sheets`` / ... as bare globals, expecting
        # to share the runtime script's namespace. Imported as a
        # separate module, each function's ``__globals__`` points to
        # the MODULE's namespace instead — where those handles were
        # never bound (``NameError: name 'slides' is not defined``).
        # Fix: after import, copy every runtime handle (the names the
        # connect + domain-setup blocks produced) into the module's
        # ``__dict__`` so the functions resolve them. Underscore-
        # prefixed names (``_connect``, ``_pre_sheet_bounds``, …) are
        # internal scaffolding the ops library never needs — skipped.
        # The shim runs AFTER the connect + setup blocks, so by here
        # ``globals()`` already holds doc/slides/controller/etc.
        return (
            "import sys as _sys\n"
            "if '/tmp' not in _sys.path:\n"
            "    _sys.path.insert(0, '/tmp')\n"
            f"import {mod} as _ops_mod\n"
            # (1) Bridge runtime UNO handles INTO the module namespace
            # so the ops library's functions (whose __globals__ point
            # at the module) resolve doc/slides/controller/etc.
            "for _k, _v in list(globals().items()):\n"
            "    if not _k.startswith('_'):\n"
            "        setattr(_ops_mod, _k, _v)\n"
            # (2) Pull EVERY module name into the runtime namespace —
            # not just the ``import *`` subset. ``import *`` skips
            # underscore-prefixed names, but the cli_run_uno post-block
            # (auto-switch the view to the edited slide) calls
            # ``_impress_get_slide`` DIRECTLY in the runtime namespace.
            # Without underscore helpers here that call NameErrors, the
            # post-block silently no-ops and the on-screen view never
            # follows the edit.
            "for _k in dir(_ops_mod):\n"
            "    if not (_k.startswith('__') and _k.endswith('__')):\n"
            "        globals()[_k] = getattr(_ops_mod, _k)\n"
        )
    # Fallback to inline. Used before the first successful upload
    # (e.g. in cold-start tests that don't go through predict()).
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
    """Return the full prepended boilerplate (connect + domain setup +
    pre-built ops library) for the given domain. Unknown domains get
    connect-only."""
    setup = _DOMAIN_SETUP_BLOCKS.get(
        (domain or "").lower().strip().replace("-", "_"),
        "",
    )
    return _CONNECT_BLOCK + setup + _ops_library_for(domain)


# Per-domain POST-block appended AFTER the actor's code. Use this to
# normalize state in ways the actor shouldn't have to remember. The
# block runs inside the same try/except that wraps the actor's code,
# so any errors here are isolated and won't mask the actor's print.
_DOMAIN_POST_BLOCKS = {
    "libreoffice_calc": (
        # After the actor's code, diff each sheet's used-range bounds
        # against the pre-snapshot. The MOST-MUTATED sheet (or, failing
        # any bound diff, the LAST-ADDED sheet) becomes the new active
        # sheet — so the next screen capture shows the change instead
        # of whichever sheet was active before. Newly-added sheets
        # always win over expanded existing sheets (a 'create sheet
        # then populate' plan typically wants the new sheet visible).
        "try:\n"
        "    _post_sheet_names = list(sheets.getElementNames())\n"
        "    _modified = None\n"
        # Newly-added sheet — pick the LAST new one in case >1.
        "    for _n in _post_sheet_names:\n"
        "        if _n not in _pre_sheet_names:\n"
        "            _modified = _n\n"
        # Otherwise: existing sheet whose used range changed.
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
        # Switch view if we found a modified sheet and it isn't active.
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
        # After the actor's code, navigate the Impress view to whichever
        # slide the action targeted. The structured impress_* actions
        # (built in impress_actions._build_generic) prepend
        # ``_pa_target_slide = <slide_value>`` before the op call, so
        # this runs even when the actor's code doesn't manage the view
        # itself. Without this, ``impress_set_font(slide=5, ...)``
        # mutates slide 5 but the on-screen view stays on slide 1 — the
        # next-step screenshot then shows no change and the planner
        # concludes the action "didn't work" when it actually did.
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
    """Build the full Python runtime script the VM will execute for a
    cli_run_uno action.

    The returned string is what ``execute_python_command`` runs via
    ``python -c``. It:
      1. Prepends the UNO connect-or-launch boilerplate
      2. exec()s the combined script with stdout/stderr captured
      3. Echoes a one-line status to the trajectory log so debug runs
         are readable without opening the sentinel JSON
      4. Persists results to ``/tmp/_last_cli_run.json`` so the
         planner reads them on the next turn (same path cli_run uses)
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("cli_run_uno requires a non-empty `code` string")

    # Defensive: the decomposer sometimes double-escapes newlines in its
    # JSON `code` value, sending us literal backslash-n sequences where a
    # real newline was meant. These survive ast.parse + repr() and end up
    # inside the runtime _script as the 2 chars ``\n``, which Python's
    # tokenizer rejects (``import foo\nbar`` → SyntaxError). If the
    # supplied ``code`` doesn't compile but a literal-\n → real-newline
    # rewrite WOULD compile, apply the rewrite and continue. We only
    # touch the string when it both (a) currently fails to parse and (b)
    # parses after the rewrite — never on valid code.
    try:
        compile(code, "<cli_run_uno>", "exec")
    except SyntaxError:
        candidate = code.replace("\\n", "\n")
        try:
            compile(candidate, "<cli_run_uno-normalized>", "exec")
        except SyntaxError:
            pass  # leave original; runtime traceback will tell the planner
        else:
            code = candidate

    # Bracket-mismatch repair — small Calc decomposers (qwen3.5-9b) reliably
    # drop one closing `]` when emitting a long single-column-fill list like
    # ``values=[[v1], [v2], …, [v11])`` (should be ``…[v11]])``). The result
    # is a SyntaxError of the exact form "closing parenthesis ')' does not
    # match opening parenthesis '['". We try inserting 1, then 2 missing
    # `]` before the offending `)` — if that yields a compileable script,
    # use it. Pure compile-time check, no shell-out, never touches code
    # that already compiles or whose SyntaxError isn't bracket-shaped.
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
        # Newline before the post-block so it doesn't fuse with the actor's
        # last line. The post-block uses a top-level try/except so any
        # failure here cannot mask the actor's PASS/FAIL print.
        inner_parts.append("\n# === Post: auto-normalize view state ===\n")
        inner_parts.append(post)
    inner = "".join(inner_parts)
    inner_repr = json.dumps(inner)

    # Sentinel first line — lets callers (e.g. the verify-path wrapper
    # ``_maybe_wrap_calc_verify_command``) detect an ALREADY-wrapped
    # script and skip re-wrapping it. Harmless comment everywhere else.
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

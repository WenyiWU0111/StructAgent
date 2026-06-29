"""Disk-content probe for files surfaced in ``[Possibly useful files]``.

Gives init_ledger / the verify-author a structural view of the target
document BEFORE anything is opened. Without it the verify-author guesses
cell addresses (unmatchable placeholders like ``a1_ref: "new_day_cell"``)
and the planner guesses the task shape ("insert a new row" on a fixed-grid
timetable whose target was just an empty cell).

Why disk-based: init_ledger runs once at task start, before the file opens
in LibreOffice; we read it directly via the VM's soffice/libs. The live UNO
perceivers (sheet/doc/slide) take over once the document is open. Both paths
produce a parsed dict in the SAME schema, so the SAME ``render_block`` turns
either into text — init-time and per-turn blocks look identical.

Consumed by init_ledger ONLY. Attached to
:class:`InitialContext.useful_file_contents` for replay parity.
"""
from __future__ import annotations
import json
import logging
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)

# Sentinel printed by the VM-side script; host parses by it.
_SENTINEL = "USEFUL_FILES_RESULT:"

# Caps: per-file output budget drives preview truncation; file count caps
# the per-task budget.
_MAX_FILES = 6
_PER_FILE_BUDGET_DEFAULT = 1500


# --------------------------------------------------------------------------- #
# Path extraction from the digested environment probe
# --------------------------------------------------------------------------- #

def extract_useful_paths(environment_probe_text: Optional[str]) -> List[str]:
    """Pull absolute paths from the ``[Possibly useful files]`` section of
    the env-perceiver digest. Up to ``_MAX_FILES`` in source order; ``[]``
    if the section/digest is missing or empty.

    The perceiver never invents paths, so anything here starting with ``/``
    is real on disk. Skip the ``▸`` open_app hint, ``──`` rules, "(none)".
    """
    if not environment_probe_text:
        return []
    paths: List[str] = []
    in_section = False
    for ln in environment_probe_text.splitlines():
        stripped = ln.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped.lower() == "[possibly useful files]"
            continue
        if not in_section:
            continue
        if not stripped.startswith("/"):
            continue
        # Whole line is the path; paths can contain spaces ("Course
        # Timetable.xlsx"). Strip a trailing perceiver note, e.g. " (existed: yes)".
        path = stripped
        if path.endswith(")") and " (" in path:
            path = path.rsplit(" (", 1)[0].rstrip()
        paths.append(path)
        if len(paths) >= _MAX_FILES:
            break
    return paths


# --------------------------------------------------------------------------- #
# VM-side script — emits parsed dicts in the existing perceiver schemas
# --------------------------------------------------------------------------- #

def _build_script(path: str) -> str:
    """Build the self-contained Python script the VM runs for ONE file.

    One ``/run_python`` call per file — small payloads keep the controller
    responsive instead of one long call blocking screenshot/perceiver traffic
    (mirrors ``_ops_lib_remote._chunked_upload``).

    Dispatches on extension and emits a parsed dict matching the matching
    live perceiver's schema, on a single ``USEFUL_FILES_RESULT:<json>``
    sentinel line keyed by path.
    """
    paths_json = json.dumps([path])
    return r'''
import json, os, sys, csv, subprocess, tempfile, traceback

_PATHS = ''' + paths_json + r'''
_SENTINEL = "''' + _SENTINEL + r'''"

# ----- helpers ----------------------------------------------------------- #

def _soffice_convert(src, fmt, tmpdir):
    """Run soffice headless to convert ``src`` into ``fmt`` (e.g. "csv"
    or "txt"), isolated by a temp user-profile so it never collides
    with the running LO instance. Returns the output absolute path or
    None on any failure. ~3-6s per call (cold start dominates)."""
    try:
        r = subprocess.run([
            "soffice", "--headless", "--nologo", "--nofirststartwizard",
            "-env:UserInstallation=file://" + tmpdir + "/profile",
            "--convert-to", fmt, src, "--outdir", tmpdir,
        ], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return None
    except Exception:
        return None
    base = os.path.splitext(os.path.basename(src))[0]
    out = os.path.join(tmpdir, base + "." + fmt.split(":")[0])
    return out if os.path.exists(out) else None


def _coerce(s):
    """Best-effort numeric coerce so the renderer's column-type
    detection sees N instead of S for numeric columns (soffice csv
    emits everything as strings, so without this every column reads
    as STRING and `type:` ends up uniformly `?`)."""
    if s is None or s == "":
        return None
    # quick filter: only attempt coerce if it could plausibly be a number
    if not (s[0].isdigit() or s[0] in "+-." and len(s) > 1):
        return s
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def _read_csv_rows(path, max_rows=120):
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        for r in csv.reader(f):
            rows.append([_coerce(c) for c in r])
            if len(rows) >= max_rows:
                break
    return rows


def _csv_to_sheet(rows, name):
    """Turn a CSV row list into a SheetPerceiver-shape sheet record.
    Density-adaptive: sparse / short → all rows; dense → head + tail."""
    if not rows:
        return {"name": name, "empty": True, "n_cols": 0, "n_rows": 0}
    n_cols = max(len(r) for r in rows)
    rows = [r + [None] * (n_cols - len(r)) for r in rows]
    n_rows = len(rows)
    # header row heuristic: row 1..5 with >=2 string-y cells
    header_row, best = 0, 0
    for ri in range(min(5, n_rows)):
        strs = sum(1 for c in rows[ri] if isinstance(c, str) and c)
        if strs > best and strs >= 2:
            best, header_row = strs, ri
    col_types = []
    for ci in range(n_cols):
        hdr = rows[header_row][ci]
        types = set()
        # Sample below the header for type signal.
        for ri in range(header_row + 1, min(header_row + 1 + 50, n_rows)):
            v = rows[ri][ci]
            if v is None:
                continue
            if isinstance(v, bool):
                types.add("STRING")
            elif isinstance(v, (int, float)):
                types.add("NUMERIC")
            else:
                types.add("STRING")
        if not types:
            ct_type = "EMPTY"
        elif len(types) == 1:
            ct_type = next(iter(types))
        else:
            ct_type = "MIXED"
        ct = {"type": ct_type}
        if isinstance(hdr, str) and hdr:
            ct["header"] = hdr
        col_types.append(ct)
    non_empty = sum(1 for r in rows for c in r if c is not None)
    N_HEAD, N_TAIL = 10, 8
    sheet = {"name": name, "n_cols": n_cols, "n_rows": n_rows,
             "header_row": header_row, "col_types": col_types,
             "merged_regions": [], "n_charts": 0}
    if non_empty <= 30 or n_rows <= N_HEAD + N_TAIL:
        cap = min(n_rows, N_HEAD + N_TAIL + 12)
        sheet["rows_head"] = rows[:cap]
        sheet["rows_head_first_idx"] = 0
        sheet["rows_tail"] = []
        if n_rows > cap:
            sheet["rows_tail_first_idx"] = cap
    else:
        tail_start = max(N_HEAD, n_rows - N_TAIL)
        sheet["rows_head"] = rows[:N_HEAD]
        sheet["rows_head_first_idx"] = 0
        sheet["rows_tail"] = rows[tail_start:]
        sheet["rows_tail_first_idx"] = tail_start
    return sheet


# ----- per-format probes ------------------------------------------------- #

def _probe_spreadsheet(path):
    """xlsx / xls / ods / csv → SheetPerceiver-shape parsed dict.
    Non-csv files go through ``soffice --convert-to csv``. CSV
    conversion loses formulas, types, merges, multi-sheet structure
    (soffice exports only the active sheet) — for init purposes the
    cell *values* are what matter most, and that's what survives."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        try:
            rows = _read_csv_rows(path)
        except Exception as e:
            return {"error": "csv read failed: %s" % e}
        sheet = _csv_to_sheet(rows, "csv")
        return {"globals": {"n_sheets": 1}, "sheets": [sheet],
                "sheets_truncated": []}
    with tempfile.TemporaryDirectory(prefix="_fileprobe_") as td:
        out = _soffice_convert(path, "csv", td)
        if not out:
            return {"error": "soffice --convert-to csv failed"}
        try:
            rows = _read_csv_rows(out)
        except Exception as e:
            return {"error": "post-convert csv read failed: %s" % e}
    base = os.path.splitext(os.path.basename(path))[0]
    sheet = _csv_to_sheet(rows, base)
    return {"globals": {"n_sheets": 1}, "sheets": [sheet],
            "sheets_truncated": []}


def _probe_doc(path):
    """docx / doc / odt → flat text via soffice. The DocPerceiver
    schema needs paragraph styles (Heading vs Body) which a plain-text
    conversion loses — so we emit a ``_text`` block (rendered directly,
    not through DocPerceiver.render_block). Good enough for init: the
    LLM sees what the document is about and can name target paragraphs
    by text content."""
    with tempfile.TemporaryDirectory(prefix="_fileprobe_") as td:
        out = _soffice_convert(path, "txt", td)
        if not out:
            return {"_text": "(soffice --convert-to txt failed)"}
        try:
            with open(out, encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except Exception as e:
            return {"_text": "(read of converted txt failed: %s)" % e}
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    head = lines[:25]
    body = "\n".join("   " + ln[:160] for ln in head)
    summary = "   non-empty lines=%d  (showing first %d)" % (
        len(lines), len(head))
    return {"_text": summary + "\n" + body}


def _probe_pptx(path):
    """pptx / ppt / odp → flat text via soffice. Same trade-off as
    _probe_doc: SlidePerceiver shape needs shape-kind / placeholder
    info that txt conversion loses, so we emit a ``_text`` summary."""
    with tempfile.TemporaryDirectory(prefix="_fileprobe_") as td:
        out = _soffice_convert(path, "txt", td)
        if not out:
            return {"_text": "(soffice --convert-to txt failed)"}
        try:
            with open(out, encoding="utf-8", errors="replace") as f:
                txt = f.read()
        except Exception as e:
            return {"_text": "(read of converted txt failed: %s)" % e}
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    head = lines[:30]
    body = "\n".join("   " + ln[:120] for ln in head)
    return {"_text": "   non-empty lines=%d  (showing first %d)\n%s" % (
        len(lines), len(head), body)}


def _probe_dir(path):
    try:
        entries = sorted(os.listdir(path))
    except Exception as e:
        return {"_text": "(could not list dir: %s)" % e}
    lines = ["   directory  entries=%d" % len(entries)]
    for name in entries[:10]:
        full = os.path.join(path, name)
        try:
            if os.path.isdir(full):
                lines.append("     %s/" % name)
            else:
                lines.append("     %s  (%dB)" % (name, os.path.getsize(full)))
        except Exception:
            lines.append("     %s" % name)
    if len(entries) > 10:
        lines.append("     ... (+%d more)" % (len(entries) - 10))
    return {"_text": "\n".join(lines)}


_DISPATCH = {
    ".xlsx": ("xlsx", _probe_spreadsheet),
    ".xlsm": ("xlsx", _probe_spreadsheet),
    ".xls":  ("xlsx", _probe_spreadsheet),
    ".ods":  ("xlsx", _probe_spreadsheet),
    ".csv":  ("xlsx", _probe_spreadsheet),
    ".docx": ("text", _probe_doc),
    ".doc":  ("text", _probe_doc),
    ".odt":  ("text", _probe_doc),
    ".pptx": ("text", _probe_pptx),
    ".ppt":  ("text", _probe_pptx),
    ".odp":  ("text", _probe_pptx),
}

_result = {}
for _p in _PATHS:
    try:
        if not os.path.exists(_p):
            _result[_p] = {"_text": "(not found)", "_format": "none"}
            continue
        if os.path.isdir(_p):
            _entry = _probe_dir(_p); _entry["_format"] = "dir"
            _result[_p] = _entry
            continue
        _ext = os.path.splitext(_p)[1].lower()
        spec = _DISPATCH.get(_ext)
        if spec is None:
            _result[_p] = {"_text": "(no probe registered for %s)" % _ext,
                           "_format": "none"}
            continue
        _fmt, _fn = spec
        _entry = _fn(_p)
        _entry["_format"] = _fmt
        _result[_p] = _entry
    except Exception:
        _result[_p] = {"_text": "(probe crash: " +
                                 traceback.format_exc().strip().splitlines()[-1] + ")",
                       "_format": "error"}

sys.stdout.write(_SENTINEL + json.dumps(_result, default=str))
sys.stdout.write("\n")
'''


# --------------------------------------------------------------------------- #
# Host-side formatting — defer to the existing per-domain renderers
# --------------------------------------------------------------------------- #

def _render_one(path: str, entry: dict) -> str:
    """Format one VM-side entry into a text block.

    Spreadsheets carry a SheetPerceiver-shape dict → run through
    ``sheet_perceiver.render_block`` (identical to the per-turn block). docx /
    pptx / dirs went through soffice→txt (loses styles/layout), so we emit the
    ``_text`` summary directly — less rich but enough for init reasoning."""
    fmt = entry.get("_format")
    if entry.get("error"):
        return "── %s\n   (parse failed: %s)" % (path, entry["error"])
    if "_text" in entry:
        # dir / no-probe / not-found / crash / docx / pptx — already text
        return "── %s\n%s" % (path, entry["_text"])
    if fmt == "xlsx":
        try:
            from mm_agents.structagent.perception.sheet_perceiver import render_block
            text = render_block({k: v for k, v in entry.items()
                                 if not k.startswith("_")})
            return "── %s\n%s" % (path, text)
        except Exception as e:
            logger.warning("[FileProbe] sheet render failed for %s: %s",
                           path, e)
            return "── %s\n   (render failed: %s)" % (path, e)
    return "── %s\n   (no renderer for format=%s)" % (path, fmt)


def _format_block(results: dict) -> str:
    parts = [_render_one(p, e) for p, e in results.items()]
    return "\n\n".join(parts) if parts else ""


# --------------------------------------------------------------------------- #
# Public entry — host side
# --------------------------------------------------------------------------- #

def _probe_one(runner, path: str) -> Optional[dict]:
    """Run the VM-side script for ONE path. Parsed entry dict, or ``None`` on
    any failure (script error, no sentinel, bad JSON). Exceptions swallowed —
    skip this file, keep going."""
    script = _build_script(path)
    try:
        result = runner(script)
    except Exception as e:
        logger.warning("[FileProbe] runner raised on %s: %s", path, e)
        return None
    if not result or result.get("status") != "success":
        logger.info("[FileProbe] runner status != success for %s: %s",
                    path,
                    result.get("status") if result else None)
        return None
    out = (result.get("output") or "")
    for line in out.splitlines():
        line = line.strip()
        if line.startswith(_SENTINEL):
            try:
                d = json.loads(line[len(_SENTINEL):])
            except Exception as e:
                logger.warning(
                    "[FileProbe] JSON parse failed for %s: %s", path, e)
                return None
            if not isinstance(d, dict) or not d:
                return None
            # script keys by path; pull our entry out
            return d.get(path) or next(iter(d.values()), None)
    logger.info("[FileProbe] no sentinel for %s (%d chars out)",
                path, len(out))
    return None


def probe_useful_files(
    env: Any,
    paths: Iterable[str],
    *,
    per_file_budget: int = _PER_FILE_BUDGET_DEFAULT,
    wall_budget_s: float = 45.0,
) -> Optional[str]:
    """Run the disk-content probe on the VM; return a formatted text block
    (one ``── <path>`` section per file).

    One ``/run_python`` call per file (see ``_build_script``). A wall-clock
    budget caps total cost — once exceeded, remaining paths are skipped and we
    return what we've got. ``None`` only if setup is unusable (no env/runner)
    or nothing succeeded.
    """
    paths = [p for p in (paths or []) if p][:_MAX_FILES]
    if not paths:
        return None
    if not env or not getattr(env, "controller", None):
        return None
    runner = getattr(env.controller, "run_python_script", None)
    if runner is None:
        logger.info("[FileProbe] env.controller has no run_python_script")
        return None
    import time as _time
    deadline = _time.time() + wall_budget_s
    results: dict = {}
    for p in paths:
        if _time.time() > deadline:
            logger.info(
                "[FileProbe] wall budget %.1fs exceeded — skipping %d "
                "remaining path(s)", wall_budget_s,
                len(paths) - len(results))
            break
        entry = _probe_one(runner, p)
        if entry is not None:
            results[p] = entry
    if not results:
        return None
    return _format_block(results) or None

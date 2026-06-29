"""SheetPerceiver — calc-side analogue of DocPerceiver.

Probes the live LibreOffice Calc workbook via UNO and produces a compact
text block describing each sheet's structure (dimensions, header row,
column types, data sample head+tail, formula samples, merged ranges,
charts/pivots), injected into the planner prompt as the "WORKBOOK
STRUCTURE" block.

Triggered at task start + after every cli_run_uno mutation, mirroring
DocPerceiver. Read-only — never calls store(), so the probe is safe.

Token-budget strategy (calc data is much larger than writer prose):
  * Per-sheet sample: first N_HEAD + last N_TAIL data rows (default 10+5).
  * Wide sheets: cap column count to N_COLS (first 12 + last 3 + ellipsis).
  * Tall sheets: row count is reported but only the sample is rendered.
  * Multi-sheet: each sheet gets its own block; cap total at K_SHEETS,
    summarize the rest as a one-line count.

UNO API reference for the cells probed:
  * sheet.getCellByPosition(col, row)   — 0-indexed
  * cell.getType()                       — EMPTY=0 / VALUE=1 / STRING=2 / FORMULA=3
  * cell.getValue() / getString() / getFormula()
  * cell.getNumberFormat()              — int ID into doc.getNumberFormats()
  * sheet.createCursor().gotoEndOfUsedArea(True) — used-range bounds
  * sheet.Charts.getCount()             — chart count per sheet
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple


# Defaults — tune via run_inspect kwargs if a task needs more / less.
N_HEAD = 10
N_TAIL = 5
N_COLS_HEAD = 12
N_COLS_TAIL = 3
K_SHEETS = 6     # render up to K sheets fully; summarize the rest


# ---------------------------------------------------------------------- #
# UNO probe script (runs in the VM via run_python_script)
# ---------------------------------------------------------------------- #
INSPECT_SCRIPT = r'''
import uno
import json
import sys
import subprocess
import time

_BEGIN = "===SHEET_INSPECT_BEGIN==="
_END   = "===SHEET_INSPECT_END==="

def _connect():
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        'com.sun.star.bridge.UnoUrlResolver', local_ctx)
    return resolver.resolve(
        'uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')

try:
    try:
        ctx = _connect()
    except Exception:
        subprocess.Popen([
            'soffice',
            '--accept=socket,host=localhost,port=2002;urp;StarOffice.Service',
        ])
        time.sleep(2)
        ctx = _connect()
    desktop = ctx.ServiceManager.createInstanceWithContext(
        'com.sun.star.frame.Desktop', ctx)
    doc = desktop.getCurrentComponent()
    sheets = doc.Sheets
    controller = doc.getCurrentController()
except Exception as e:
    print(_BEGIN)
    print(json.dumps({"error": f"UNO connect failed: {type(e).__name__}: {e}"}))
    print(_END)
    sys.exit(0)

# Cell type enum: EMPTY=0, VALUE=1, STRING=2, FORMULA=3.
# On some Python UNO bridges cell.getType() returns an Enum object whose
# int() raises TypeError ("not 'Enum'"). _type_int handles both flavours:
# bare int (older bridges), Enum.value (newer), or str(Enum) name lookup
# as the last resort. We do this once-per-cell so the rest of the probe
# can compare against the canonical ints.
def _type_int_raw(cell):
    """Read cell.getType() and coerce to int via int(Enum) / Enum.value /
    name lookup — handles UNO bindings that don't allow direct int()."""
    try:
        t = cell.getType()
    except Exception:
        return -1
    try:
        return int(t)
    except (TypeError, ValueError):
        pass
    v = getattr(t, "value", None)
    if isinstance(v, int):
        return v
    name = str(getattr(t, "value", "") or t).upper()
    # "TEXT" is what pyuno's newer bindings return for string cells (older
    # bindings used "STRING"). Without TEXT here, every text-typed cell in
    # xlsx files falls through to -1 and gets skipped as EMPTY — column
    # headers and product-name columns then render as `·` and the planner
    # mis-infers schema (e.g. believing "Retail Price" sheet has only the
    # numeric price column).
    return {"EMPTY": 0, "VALUE": 1, "NUMERIC": 1, "STRING": 2, "TEXT": 2,
            "FORMULA": 3}.get(name, -1)

def _type_int(cell):
    """Cell type as int (0=EMPTY, 1=VALUE, 2=STRING, 3=FORMULA), with a
    cross-check for LO's lazy type metadata. Right after setString /
    setValue on a newly-touched cell, the cell SHOWS UP in the used
    range but getType() still reports EMPTY until something else
    forces type propagation. Without this cross-check the SP block
    misreports written cells as empty and the planner sees no
    progress (despite the doc actually having the data) — looping on
    the same write subgoal."""
    t = _type_int_raw(cell)
    if t == 0:
        try:
            s = cell.getString()
            if s:
                return 2     # has string content → STRING
        except Exception:
            pass
        try:
            v = cell.getValue()
            if v != 0:
                return 1     # has nonzero numeric content → VALUE
        except Exception:
            pass
    return t

def _type_label(cell):
    t = _type_int(cell)
    if t < 0: return None
    return {0: "EMPTY", 1: "NUMERIC", 2: "STRING", 3: "FORMULA"}.get(t)

def _cell_value(cell, tlabel):
    """Return a JSON-safe representation of the cell's value. For
    formula cells we return BOTH the formula and the cached numeric or
    string value, so the planner can see what the formula evaluates to."""
    try:
        if tlabel == "FORMULA":
            f = cell.getFormula()
            # The cached evaluation: getValue() if it's numeric, getString() otherwise
            try:
                v = cell.getValue()
                if isinstance(v, float) and v != v: v = None   # NaN
            except Exception:
                v = None
            return {"f": f, "v": v}
        if tlabel == "NUMERIC":
            return cell.getValue()
        if tlabel == "STRING":
            return cell.getString()
        return None
    except Exception:
        return None

def _used_range(sheet):
    """Return (n_cols, n_rows) of the used range, 0 if empty."""
    try:
        cur = sheet.createCursor()
        cur.gotoStartOfUsedArea(False)
        cur.gotoEndOfUsedArea(True)
        ra = cur.getRangeAddress()
        n_cols = ra.EndColumn + 1
        n_rows = ra.EndRow + 1
        # gotoStart of empty sheet → both ends at 0,0; check actual cell to confirm not empty
        if n_cols == 1 and n_rows == 1:
            c = sheet.getCellByPosition(0, 0)
            if _type_int(c) == 0:   # EMPTY
                return 0, 0
        return n_cols, n_rows
    except Exception:
        return 0, 0

def _scan_column_type(sheet, col, n_rows, max_scan=50):
    """Sample first `max_scan` cells in `col`, return dominant type label
    and a sample formula if any. Skips empties."""
    counts = {"NUMERIC": 0, "STRING": 0, "FORMULA": 0}
    sample_formula = None
    sample_format = None
    seen = 0
    for r in range(min(n_rows, max_scan)):
        c = sheet.getCellByPosition(col, r)
        tl = _type_label(c)
        if tl is None or tl == "EMPTY":
            continue
        counts[tl] = counts.get(tl, 0) + 1
        seen += 1
        if tl == "FORMULA" and sample_formula is None:
            sample_formula = c.getFormula()
        if sample_format is None:
            try:
                fmt_id = int(c.NumberFormat)
                if fmt_id > 0:
                    sample_format = fmt_id
            except Exception:
                pass
    if seen == 0:
        dominant = "EMPTY"
    else:
        dominant = max(counts.items(), key=lambda kv: kv[1])[0]
    return {"type": dominant, "sample_formula": sample_formula,
            "number_format_id": sample_format, "seen": seen}

def _merged_ranges(sheet, max_scan_rows=200, max_scan_cols=50):
    """Iterate cells over a bounded scan area and collect merged ranges
    as ``{"start_col", "start_row", "end_col", "end_row"}`` dicts.

    LO only reports ``IsMerged=True`` on the TOP-LEFT anchor of a
    merge (cells inside the merge report False). The merge bounds
    come from ``cursor.collapseToMergedArea()`` — the only public UNO
    accessor for "what range does this anchor span". Bounded by
    ``max_scan_*`` to keep the probe cheap on huge sheets.
    """
    seen = set()
    out = []
    n_cols, n_rows = _used_range(sheet)
    rmax = min(n_rows, max_scan_rows)
    cmax = min(n_cols, max_scan_cols)
    for r in range(rmax):
        for c in range(cmax):
            try:
                cell = sheet.getCellByPosition(c, r)
                if not hasattr(cell, 'getIsMerged') or not cell.getIsMerged():
                    continue
                addr = (c, r)
                if addr in seen:
                    continue
                seen.add(addr)
                # collapseToMergedArea expands the cursor to cover the
                # full merged region (the only known UNO API that
                # reports merge bounds — neither cell.MergeArea nor
                # range.IsMerged help: the former doesn't exist, the
                # latter returns True for any superset of the merge).
                end_c = c
                end_r = r
                try:
                    cur = sheet.createCursorByRange(cell)
                    cur.collapseToMergedArea()
                    cra = cur.getRangeAddress()
                    end_c = int(cra.EndColumn)
                    end_r = int(cra.EndRow)
                except Exception:
                    pass
                out.append({
                    "start_col": c, "start_row": r,
                    "end_col": end_c, "end_row": end_r,
                })
            except Exception:
                pass
    return out

def _sheet_record(sheet):
    n_cols, n_rows = _used_range(sheet)
    rec = {"name": sheet.getName(),
           "n_cols": int(n_cols),
           "n_rows": int(n_rows)}
    if n_cols == 0 or n_rows == 0:
        rec["empty"] = True
        return rec

    # Header-row detection: many calc docs leave row 0 blank for spacing
    # and put the actual column headers in row 1 or 2. Heuristic: scan
    # the first 5 rows and pick the one with the MOST STRING cells (tie-
    # broken by earliest row). Need >= 2 strings to qualify; fall back
    # to row 0 otherwise. Validated on 5 cached samples — 4/5 correct;
    # the 5th (cross-tab layout with row labels) has no clean header
    # row anyway so a wrong guess is harmless.
    header_row = 0
    best_strs = 0
    for r in range(min(5, n_rows)):
        strs = 0
        for c in range(n_cols):
            try:
                if _type_int(sheet.getCellByPosition(c, r)) == 2:
                    strs += 1
            except Exception:
                pass
        if strs > best_strs and strs >= 2:
            best_strs = strs
            header_row = r
    rec["header_row"] = header_row

    # Per-column type signature: scan a sample
    col_types = []
    for c in range(n_cols):
        ct = _scan_column_type(sheet, c, n_rows)
        # Read the header at the detected header_row.
        try:
            header_cell = sheet.getCellByPosition(c, header_row)
            if _type_int(header_cell) == 2:
                ct["header"] = header_cell.getString()
        except Exception:
            pass
        col_types.append(ct)
    rec["col_types"] = col_types

    # Row sample: head + tail. Each row is a list aligned with col indices.
    def _row(r):
        row = []
        for c in range(n_cols):
            cell = sheet.getCellByPosition(c, r)
            tl = _type_label(cell)
            row.append(_cell_value(cell, tl))
        return row

    head_n = min(N_HEAD_PLACEHOLDER, n_rows)
    head = [_row(r) for r in range(head_n)]
    tail = []
    tail_start = None
    if n_rows > head_n:
        # head + tail (middle omitted) — without this, n_rows in
        # [head_n+1 .. head_n+N_TAIL] would silently drop its last
        # rows (e.g. n_rows=11, head_n=10: row 11 was never rendered
        # and the planner wrote SUM ranges ending at row 10, off by
        # one). When the gap is wider than N_TAIL the render code
        # emits a "rows omitted" gap line between head and tail.
        tail_start = max(head_n, n_rows - N_TAIL_PLACEHOLDER)
        tail = [_row(r) for r in range(tail_start, n_rows)]
    rec["rows_head"] = head
    rec["rows_tail"] = tail
    rec["rows_head_first_idx"] = 0
    rec["rows_tail_first_idx"] = tail_start

    # Sub-table starts in the elided middle. A "sub-table start" is a
    # non-empty row whose immediate predecessor IS empty — i.e. the
    # first non-empty row after a blank gap. These rows usually carry
    # a section heading ("Personal Costs - 2020" in col A, rest empty)
    # OR the header row of a second adjacent table. Without surfacing
    # them, the head+tail window hides the existence of a second table
    # entirely and the planner treats one sheet with two stacked tables
    # as one continuous range — picking a chart data_range that spans
    # both, or naming a single Total row that overshoots into table 2.
    # We emit (row_idx, row_values, next_row_values_if_likely_header)
    # so the renderer can show two consecutive rows when the next row
    # looks like a header (mostly STRING cells, no formulas).
    middle = []
    if tail_start is not None and tail_start > head_n:
        def _row_empty(r):
            for c in range(n_cols):
                cell = sheet.getCellByPosition(c, r)
                if _type_int(cell) != 0:
                    return False
            return True
        # Scan the gap; surface up to 4 sub-table starts so the
        # middle elision doesn't balloon on a many-table sheet.
        prev_empty = (head_n == 0) or _row_empty(head_n - 1)
        for r in range(head_n, tail_start):
            empty = _row_empty(r)
            if prev_empty and not empty:
                entry = {"row": r, "values": _row(r)}
                # Pull the next row too if it looks like a header
                # (multiple STRING cells, no formulas) — that's the
                # column-name row of the second table.
                if r + 1 < tail_start:
                    nxt_str_count = 0
                    nxt_has_formula = False
                    for c in range(n_cols):
                        cell = sheet.getCellByPosition(c, r + 1)
                        ti = _type_int(cell)
                        if ti == 3:
                            nxt_has_formula = True
                            break
                        if ti == 2:
                            nxt_str_count += 1
                    if not nxt_has_formula and nxt_str_count >= 2:
                        entry["next_row"] = r + 1
                        entry["next_values"] = _row(r + 1)
                middle.append(entry)
                if len(middle) >= 4:
                    break
            prev_empty = empty
    rec["rows_middle_markers"] = middle

    # Sub-tables — explicit row-range partition of the sheet when blank
    # rows separate it into multiple non-empty sections. Computed across
    # the WHOLE sheet (not just the elided window) so the planner can
    # read off "Table 1: R1..R15, Table 2: R18..R32" without inferring
    # from middle markers. Single-section sheets emit an empty list.
    subtables = []
    run_start = None
    def _row_empty_full(r):
        for c in range(n_cols):
            cell = sheet.getCellByPosition(c, r)
            if _type_int(cell) != 0:
                return False
        return True
    for r in range(n_rows):
        empty = _row_empty_full(r)
        if not empty:
            if run_start is None:
                run_start = r
        else:
            if run_start is not None:
                subtables.append({"first_row": run_start, "last_row": r - 1})
                run_start = None
    if run_start is not None:
        subtables.append({"first_row": run_start, "last_row": n_rows - 1})
    # Only surface when there's more than one section — single-table
    # sheets don't need the line.
    rec["subtables"] = subtables if len(subtables) >= 2 else []

    # Formula sample — collect up to 5 unique formulas
    seen_f = set()
    fsamples = []
    for r in range(min(n_rows, 100)):
        for c in range(n_cols):
            cell = sheet.getCellByPosition(c, r)
            if _type_int(cell) == 3:
                f = cell.getFormula()
                if f and f not in seen_f:
                    seen_f.add(f)
                    fsamples.append({"row": r, "col": c, "f": f[:160]})
                    if len(fsamples) >= 5:
                        break
        if len(fsamples) >= 5:
            break
    rec["formulas_sample"] = fsamples

    # Charts on this sheet
    try:
        rec["n_charts"] = int(sheet.Charts.getCount())
    except Exception:
        rec["n_charts"] = 0

    # Merged regions — without this, a sheet whose only content lives
    # in merge anchors (e.g. ``A1='Investment Summary'`` spanning A1:C1)
    # renders as a 1×1 used range and the planner's vision LLM judges
    # from the screenshot that the cells are NOT merged. The merged
    # anchors themselves carry ``IsMerged=True`` and their bounds come
    # from ``cursor.collapseToMergedArea()``.
    try:
        rec["merged_regions"] = _merged_ranges(sheet)
    except Exception:
        rec["merged_regions"] = []

    # Sheet-wide anomaly scan — full-workbook pass to surface cells the
    # head+tail sample window omits. Without this, tasks that need to
    # enumerate ALL cells matching a pattern (e.g., "hide every row
    # with #N/A", "delete all rows whose Status is 'Error'") see only
    # the first 10 and last 5 data rows; matches in the middle gap
    # are invisible to the planner and silently dropped from the
    # action's row list. Cap scan to keep the probe under ~1s on
    # large sheets — 5000 rows × 30 cols is plenty for the typical
    # benchmark workbook (the largest seen is ~2000 rows).
    SCAN_MAX_ROWS = 5000
    SCAN_MAX_COLS = 30
    scan_rows = min(n_rows, SCAN_MAX_ROWS)
    scan_cols = min(n_cols, SCAN_MAX_COLS)
    na_cells = []          # =NA() formula OR rendered as "#N/A"
    error_cells = []       # other Calc errors (#VALUE!, #REF!, #DIV/0!, #NAME?)
    total_row_idxs = []    # row indices whose leftmost text cell is Total/Sum/Subtotal
    _TOTAL_LABELS = {
        "total", "totals", "sum", "subtotal", "subtotals",
        "grand total", "grand totals", "grand sum",
    }
    for _r in range(scan_rows):
        for _c in range(scan_cols):
            try:
                _cell = sheet.getCellByPosition(_c, _r)
                _t = _type_int(_cell)
                if _t == 3:  # FORMULA
                    _f = _cell.getFormula() or ""
                    _s_render = _cell.getString() or ""
                    if "NA()" in _f.upper() or _s_render == "#N/A":
                        na_cells.append((_c, _r))
                    elif (_s_render.startswith("#")
                          and _s_render.endswith(("!", "?"))
                          and _s_render != "#N/A"):
                        error_cells.append((_c, _r, _s_render))
                elif _t == 2:  # TEXT
                    _s_render = (_cell.getString() or "").strip().lower()
                    if _c == 0 and _s_render in _TOTAL_LABELS:
                        total_row_idxs.append(_r)
            except Exception:
                pass
    # Cap output sizes — full enumeration is the value-add, but a
    # pathological sheet with thousands of NA cells would balloon the
    # rendered block. 50 / 20 / 20 are generous for any realistic task.
    rec["anomalies"] = {
        "na_cells": [(_c, _r) for (_c, _r) in na_cells[:50]],
        "na_total": len(na_cells),
        "error_cells": [(_c, _r, _s) for (_c, _r, _s) in error_cells[:20]],
        "error_total": len(error_cells),
        "total_row_idxs": total_row_idxs[:20],
        "scan_truncated": (
            n_rows > SCAN_MAX_ROWS or n_cols > SCAN_MAX_COLS
        ),
    }

    return rec

# Active selection — render as "SheetName.A1" or "SheetName.A1:B10" if range
def _selection_str():
    try:
        sel = controller.getSelection()
        if hasattr(sel, "getRangeAddress"):
            ra = sel.getRangeAddress()
            sheet_name = sheets.getByIndex(ra.Sheet).getName()
            from string import ascii_uppercase
            def col_letter(n):
                s = ""
                while True:
                    s = ascii_uppercase[n % 26] + s
                    n = n // 26 - 1
                    if n < 0: break
                return s
            top = f"{col_letter(ra.StartColumn)}{ra.StartRow+1}"
            bot = f"{col_letter(ra.EndColumn)}{ra.EndRow+1}"
            if top == bot:
                return f"{sheet_name}.{top}"
            return f"{sheet_name}.{top}:{bot}"
    except Exception:
        pass
    return None

active_sheet_name = None
try:
    active_sheet_name = controller.getActiveSheet().getName()
except Exception:
    pass

# Render sheets (cap to K)
sheet_records = []
all_names = list(sheets.getElementNames())
for i, name in enumerate(all_names[:K_SHEETS_PLACEHOLDER]):
    try:
        sheet_records.append(_sheet_record(sheets.getByName(name)))
    except Exception as e:
        sheet_records.append({"name": name, "error": f"{type(e).__name__}: {e}"})
remaining = [
    {"name": n, "summary_only": True}
    for n in all_names[K_SHEETS_PLACEHOLDER:]
]

print(_BEGIN)
print(json.dumps({
    "globals": {
        "n_sheets": len(all_names),
        "active_sheet": active_sheet_name,
        "active_selection": _selection_str(),
    },
    "sheets": sheet_records,
    "sheets_truncated": remaining,
}, ensure_ascii=False, default=str))
print(_END)
'''

# String-substitute the tuning constants into INSPECT_SCRIPT so users
# can override them per call (run_inspect kwargs) without exposing the
# raw script body.
def _build_inspect_script(n_head: int, n_tail: int, k_sheets: int) -> str:
    return (INSPECT_SCRIPT
            .replace("N_HEAD_PLACEHOLDER", str(n_head))
            .replace("N_TAIL_PLACEHOLDER", str(n_tail))
            .replace("K_SHEETS_PLACEHOLDER", str(k_sheets)))


# ---------------------------------------------------------------------- #
# Agent-side renderer
# ---------------------------------------------------------------------- #

def _col_letter(n: int) -> str:
    s = ""
    while True:
        s = chr(ord("A") + (n % 26)) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


def _fmt_cell(v: Any) -> str:
    """Compact cell rendering for the markdown block. Trims noise:
    leading/trailing whitespace inside strings, long floats, and the
    verbose `datetime.datetime(...)` repr (which calc serial dates
    decode into when read by openpyxl proxies and sometimes UNO too)."""
    if v is None:
        return "·"   # middle-dot = empty
    if isinstance(v, dict) and "f" in v:
        # formula cell — show formula + cached value
        f = v.get("f", "")
        cached = v.get("v")
        if cached is not None:
            return f"{f!r}→{_fmt_cell(cached)}"
        return repr(f)
    if isinstance(v, str):
        # Visualize whitespace-only / heavy-leading-space cells
        s = v.strip()
        if not s and v:
            return f"<ws×{len(v)}>"
        # Use the stripped form for display (cell internal whitespace is
        # rarely meaningful for planner reasoning); cap length to 24.
        if len(s) > 24:
            return repr(s[:21] + "…")
        return repr(s)
    if isinstance(v, float):
        if v == int(v):
            return str(int(v))
        return f"{v:.4g}"
    # datetime / date / time — show ISO-ish compact form.
    # First by isinstance (when the proxy or UNO returns native objects),
    # then by str-prefix detection (when only their repr made it through
    # json.dumps + default=str on the VM side).
    try:
        import datetime as _dt
        if isinstance(v, _dt.datetime):
            return v.strftime("%Y-%m-%d") if (v.hour == 0 and v.minute == 0) else v.strftime("%Y-%m-%dT%H:%M")
        if isinstance(v, _dt.date):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, _dt.time):
            return v.strftime("%H:%M")
    except Exception:
        pass
    s = str(v)
    # `datetime.datetime(2023, 1, 1, 0, 0)` → '2023-01-01'
    if s.startswith("datetime.datetime("):
        try:
            inner = s[len("datetime.datetime("):-1]
            parts = [int(p.strip()) for p in inner.split(",")]
            if len(parts) >= 3 and all(p == 0 for p in parts[3:6] if len(parts) > 3):
                return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
            if len(parts) >= 5:
                return f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}T{parts[3]:02d}:{parts[4]:02d}"
        except Exception:
            pass
    if s.startswith("datetime.time("):
        try:
            inner = s[len("datetime.time("):-1]
            parts = [int(p.strip()) for p in inner.split(",")]
            if len(parts) >= 2:
                return f"{parts[0]:02d}:{parts[1]:02d}"
        except Exception:
            pass
    return repr(v)


def _render_row(row: List[Any], col_idxs: List[int]) -> str:
    """Render a row's cell values for the selected columns (handles wide-sheet truncation)."""
    parts = []
    for i in col_idxs:
        if i is None:
            parts.append("…")  # column-truncation marker
        elif i < len(row):
            parts.append(_fmt_cell(row[i]))
        else:
            parts.append("·")
    return "  ".join(parts)


def _pick_cols(n_cols: int) -> List[Optional[int]]:
    """Return a list of column indices to display, with None marking the
    truncation gap. Compact for wide sheets, full for narrow ones."""
    if n_cols <= N_COLS_HEAD + N_COLS_TAIL + 1:
        return list(range(n_cols))
    return list(range(N_COLS_HEAD)) + [None] + list(range(n_cols - N_COLS_TAIL, n_cols))


def _render_sheet(rec: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    name = rec.get("name", "?")
    if rec.get("error"):
        lines.append(f"── Sheet[{name}] ERROR: {rec['error']} ──")
        return lines
    n_cols = rec.get("n_cols", 0)
    n_rows = rec.get("n_rows", 0)
    n_charts = rec.get("n_charts", 0)
    if rec.get("empty"):
        lines.append(f"── Sheet[{name}] (empty) ──")
        return lines
    end_col = _col_letter(n_cols - 1) if n_cols else "A"
    header_row = rec.get("header_row", 0)
    hdr_marker = f"header=row{header_row + 1}" if header_row > 0 else "header=row1"
    merged = rec.get("merged_regions") or []
    merged_marker = ""
    if merged:
        merged_marker = f"  merges={len(merged)}"
    lines.append(
        f"── Sheet[{name}] used=A1:{end_col}{n_rows} ({n_rows}×{n_cols})  "
        f"{hdr_marker}  charts={n_charts}{merged_marker} ──"
    )
    if merged:
        # Render each merged region as A1-style range so the planner
        # can read them off the SP block instead of guessing from the
        # screenshot. Limit to first 20 to keep the SP compact on
        # heavily-merged dashboards; tail summary if more.
        items = []
        for m in merged[:20]:
            sc, sr = int(m["start_col"]), int(m["start_row"])
            ec, er = int(m["end_col"]), int(m["end_row"])
            items.append(
                f"{_col_letter(sc)}{sr+1}:{_col_letter(ec)}{er+1}")
        more = f" (+{len(merged) - 20} more)" if len(merged) > 20 else ""
        lines.append(f"merged: {', '.join(items)}{more}")
    subtables = rec.get("subtables") or []
    if subtables:
        # Explicit "this sheet is split into N tables" line — the
        # planner shouldn't have to infer table boundaries from the
        # middle markers. Use the first cell of each section's first
        # row as a tag so the boundaries are easy to skim.
        parts = []
        for i, st in enumerate(subtables, start=1):
            f, l = int(st["first_row"]), int(st["last_row"])
            parts.append(f"#{i} R{f+1}..R{l+1}")
        lines.append(
            f"sub-tables: {len(subtables)} sections separated by blank "
            f"rows — {', '.join(parts)}"
        )

    col_idxs = _pick_cols(n_cols)
    # Column-type signature header
    type_short = {"NUMERIC": "N", "STRING": "S", "FORMULA": "F",
                  "EMPTY": "·", "MIXED": "M"}
    col_types = rec.get("col_types") or []
    def _ct_line(label: str) -> str:
        parts = []
        for i in col_idxs:
            if i is None:
                parts.append("…")
            elif i < len(col_types):
                ct = col_types[i]
                if label == "letter":
                    parts.append(_col_letter(i))
                elif label == "header":
                    h = ct.get("header") or ""
                    parts.append(repr(h[:12]) if h else "·")
                elif label == "type":
                    parts.append(type_short.get(ct.get("type"), "?"))
                elif label == "fmt":
                    fid = ct.get("number_format_id")
                    parts.append(f"#{fid}" if fid else "·")
            else:
                parts.append("·")
        return "  ".join(parts)
    lines.append("col:    " + _ct_line("letter"))
    lines.append("hdr:    " + _ct_line("header"))
    lines.append("type:   " + _ct_line("type"))
    lines.append("fmt-id: " + _ct_line("fmt"))

    # Data rows: head + tail. Fold consecutive all-empty rows so a
    # sheet with 12 blank trailing rows shows as "[10-21] (×12 empty)"
    # instead of 12 distinct lines of dots.
    def _is_all_empty(row, idxs):
        return all((i is None) or (i >= len(row)) or row[i] is None for i in idxs)

    def _emit_rows(rows, start_idx_1based):
        i = 0
        while i < len(rows):
            if _is_all_empty(rows[i], col_idxs):
                j = i
                while j < len(rows) and _is_all_empty(rows[j], col_idxs):
                    j += 1
                k = j - i
                if k == 1:
                    lines.append(f"  [{start_idx_1based+i:3d}] (empty row)")
                else:
                    lines.append(
                        f"  [{start_idx_1based+i:3d}-{start_idx_1based+j-1:3d}] (×{k} empty rows)"
                    )
                i = j
                continue
            lines.append(f"  [{start_idx_1based+i:3d}] {_render_row(rows[i], col_idxs)}")
            i += 1

    head = rec.get("rows_head") or []
    tail = rec.get("rows_tail") or []
    if head:
        _emit_rows(head, 1)
    middle = rec.get("rows_middle_markers") or []
    if tail:
        tail_start = rec.get("rows_tail_first_idx", n_rows - len(tail))
        omitted = n_rows - len(head) - len(tail)
        if middle:
            # Bookend each sub-table marker with elision counts so the
            # planner sees exactly where in the gap the second table
            # starts (e.g. "rows 11-17 omitted" then row 18 marker).
            prev_idx = len(head)  # 0-based row idx of first omitted
            for mk in middle:
                r = mk["row"]
                gap = r - prev_idx
                if gap > 0:
                    lines.append(
                        f"  …  ({gap} row{'s' if gap != 1 else ''} omitted)"
                    )
                lines.append(
                    f"  [{r+1:3d}] {_render_row(mk['values'], col_idxs)}"
                )
                prev_idx = r + 1
                if "next_row" in mk:
                    lines.append(
                        f"  [{mk['next_row']+1:3d}] "
                        f"{_render_row(mk['next_values'], col_idxs)}"
                    )
                    prev_idx = mk['next_row'] + 1
            trailing = tail_start - prev_idx
            if trailing > 0:
                lines.append(
                    f"  …  ({trailing} row{'s' if trailing != 1 else ''} omitted)"
                )
        elif omitted > 0:
            lines.append(f"  …  ({omitted} rows omitted)")
        _emit_rows(tail, tail_start + 1)

    # Formulas
    fsamples = rec.get("formulas_sample") or []
    if fsamples:
        lines.append("formulas (sample, up to 5):")
        for fs in fsamples:
            addr = f"{_col_letter(fs['col'])}{fs['row']+1}"
            lines.append(f"  {addr} := {fs['f']!r}")

    # Sheet-wide anomalies — surfaces cells the head+tail rendering
    # window can't show (middle "rows omitted" gap). Lets the planner
    # enumerate ALL matching cells for tasks like "hide every #N/A
    # row" without having to guess from the visible sample.
    anomalies = rec.get("anomalies") or {}
    na_cells = anomalies.get("na_cells") or []
    na_total = anomalies.get("na_total", 0)
    error_cells = anomalies.get("error_cells") or []
    error_total = anomalies.get("error_total", 0)
    total_row_idxs = anomalies.get("total_row_idxs") or []
    if na_total or error_total or total_row_idxs:
        lines.append("sheet-wide anomalies (full-sheet scan):")
        if na_total:
            cell_refs = [f"{_col_letter(c)}{r+1}" for c, r in na_cells]
            row_idxs = sorted({r + 1 for c, r in na_cells})
            extra = (f" (+{na_total - len(na_cells)} more)"
                     if na_total > len(na_cells) else "")
            lines.append(
                f"  #N/A cells ({na_total}): {', '.join(cell_refs)}{extra}"
            )
            lines.append(
                f"  → rows containing #N/A: "
                f"{', '.join(str(r) for r in row_idxs)}"
            )
        if error_total:
            cell_strs = [f"{_col_letter(c)}{r+1}={s}"
                         for c, r, s in error_cells]
            extra = (f" (+{error_total - len(error_cells)} more)"
                     if error_total > len(error_cells) else "")
            lines.append(
                f"  formula errors ({error_total}): "
                f"{', '.join(cell_strs)}{extra}"
            )
        if total_row_idxs:
            lines.append(
                f"  Total/Sum/Subtotal row labels at rows: "
                f"{', '.join(str(r + 1) for r in total_row_idxs)}"
            )
        if anomalies.get("scan_truncated"):
            lines.append(
                "  (scan truncated at 5000 rows × 30 cols — sheet "
                "exceeds scan window; some anomalies may be missed)"
            )
    return lines


def render_block(parsed: Dict[str, Any]) -> str:
    if "error" in parsed:
        err = parsed["error"]
        # Detect "no Calc doc open" specifically — the natural symptom
        # is `desktop.getCurrentComponent()` returning the Start Center
        # component, then `doc.Sheets` raising `AttributeError: Sheets`.
        # Surface a strongly-worded warning so the planner doesn't
        # fall back to its own visual hallucination of an "open"
        # spreadsheet (Qwen3.5-VL has been observed describing a full
        # spreadsheet's contents when the screenshot actually shows
        # the Start Center).
        no_doc = ("AttributeError: Sheets" in err
                  or "no Calc document" in err.lower()
                  or "no spreadsheet" in err.lower())
        if no_doc:
            return (
                "═══════ WORKBOOK STRUCTURE (SheetPerceiver — UNO probe) ═══════\n"
                "⚠ NO LIBREOFFICE CALC DOCUMENT IS CURRENTLY OPEN ⚠\n"
                "\n"
                f"The UNO probe failed with: {err}\n"
                "\n"
                "This means `desktop.getCurrentComponent()` returned a non-\n"
                "Calc component — almost always the LibreOffice Start Center\n"
                "(the welcome screen with 'Open File', 'Recent Documents',\n"
                "'Create:' buttons). No spreadsheet file is loaded.\n"
                "\n"
                "CRITICAL: any <visible> description of an open spreadsheet\n"
                "(cell contents, sheet names, formula bar values) THIS TURN\n"
                "is HALLUCINATED. Trust this probe — the file is NOT loaded.\n"
                "\n"
                "FIX: the FIRST plan step must OPEN the target file via the\n"
                "GUI (File menu → Open, Ctrl+O, or click the file in the\n"
                "'Recent Documents' panel of the Start Center). cli_run_uno\n"
                "ops cannot work until a Calc document is the active component.\n"
                "═══════════════════════════════════════════════════════════════"
            )
        return f"WORKBOOK STRUCTURE — probe failed: {err}"
    g = parsed.get("globals") or {}
    sheets = parsed.get("sheets") or []
    truncated = parsed.get("sheets_truncated") or []

    lines: List[str] = []
    lines.append("═══════ WORKBOOK STRUCTURE (SheetPerceiver — UNO probe) ═══════")
    n = g.get("n_sheets", len(sheets))
    parts = [f"{n} sheet" + ("s" if n != 1 else "")]
    if g.get("active_sheet"):
        parts.append(f"active={g['active_sheet']}")
    if g.get("active_selection"):
        parts.append(f"selection={g['active_selection']}")
    lines.append(" | ".join(parts))
    lines.append("")
    for rec in sheets:
        lines.extend(_render_sheet(rec))
        lines.append("")
    if truncated:
        names = ", ".join(s.get("name", "?") for s in truncated)
        lines.append(f"(plus {len(truncated)} more sheet(s), summary-only: {names})")
    lines.append("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Dump artifacts + main entry point
# ---------------------------------------------------------------------- #

def _dump_artifacts(results_dir: Optional[str], step_idx: Optional[int],
                    raw_stdout: str, parsed: Optional[Dict[str, Any]],
                    block: Optional[str],
                    logger: Optional[logging.Logger]) -> None:
    if not results_dir:
        return
    try:
        out_dir = os.path.join(results_dir, "sheet_perceiver")
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"step_{(step_idx + 1):03d}" if step_idx is not None else "step_xxx"
        with open(os.path.join(out_dir, f"{prefix}_inspect_raw.txt"),
                  "w", encoding="utf-8") as f:
            f.write(raw_stdout or "")
        if parsed is not None:
            with open(os.path.join(out_dir, f"{prefix}_inspect_parsed.json"),
                      "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2, default=str)
        if block:
            with open(os.path.join(out_dir, f"{prefix}_inspect_rendered.txt"),
                      "w", encoding="utf-8") as f:
                f.write(block)
    except Exception as e:
        if logger:
            logger.info("[SheetPerceiver] dump failed: %s", e)


def _attempt_inspect(env, script: str,
                     logger: Optional[logging.Logger]
                     ) -> Tuple[str, Optional[Dict[str, Any]]]:
    try:
        resp = env.controller.run_python_script(script)
    except Exception as e:
        if logger: logger.info("[SheetPerceiver] run_python_script raised: %s", e)
        return (f"<exception: {e}>", None)
    if not isinstance(resp, dict):
        return (f"<unexpected resp type: {type(resp).__name__}>", None)
    if resp.get("status") != "success":
        return (str(resp.get("output") or resp.get("error"))[:2000], None)
    out = resp.get("output") or ""
    BEG, END = "===SHEET_INSPECT_BEGIN===", "===SHEET_INSPECT_END==="
    if BEG not in out or END not in out:
        return (out, None)
    payload = out.split(BEG, 1)[1].split(END, 1)[0].strip()
    try:
        parsed = json.loads(payload)
    except Exception:
        return (out, None)
    return (out, parsed)


def run_inspect(env, logger: Optional[logging.Logger] = None,
                results_dir: Optional[str] = None,
                step_idx: Optional[int] = None,
                max_retries: int = 5,
                retry_sleep_seconds: float = 2.0,
                n_head: int = N_HEAD,
                n_tail: int = N_TAIL,
                k_sheets: int = K_SHEETS) -> Optional[str]:
    """Execute the UNO probe in the VM, parse stdout, return a planner-ready
    text block (or None on persistent failure). Same retry + dump
    conventions as DocPerceiver."""
    if env is None or not hasattr(env, "controller"):
        if logger: logger.info("[SheetPerceiver] env.controller unavailable")
        return None
    script = _build_inspect_script(n_head, n_tail, k_sheets)
    last_raw = ""
    last_parsed: Optional[Dict[str, Any]] = None
    for attempt in range(max_retries):
        raw, parsed = _attempt_inspect(env, script, logger)
        last_raw, last_parsed = raw, parsed
        if parsed is not None and "error" not in parsed:
            block = render_block(parsed)
            if logger:
                ns = len(parsed.get("sheets") or [])
                logger.info(
                    "[SheetPerceiver] probe ok (attempt %d/%d): %d sheets, %d chars block",
                    attempt + 1, max_retries, ns, len(block))
            _dump_artifacts(results_dir, step_idx, raw, parsed, block, logger)
            return block
        err_msg = parsed["error"] if (parsed is not None and "error" in parsed) else None
        if logger:
            logger.info(
                "[SheetPerceiver] probe attempt %d/%d failed: %s — sleeping %.1fs",
                attempt + 1, max_retries,
                (err_msg or "no sentinel / parse fail")[:200],
                retry_sleep_seconds,
            )
        if attempt < max_retries - 1:
            try:
                import time
                time.sleep(retry_sleep_seconds)
            except Exception:
                pass
    if logger:
        logger.info("[SheetPerceiver] probe gave up after %d attempts", max_retries)
    # When the last attempt's error specifically indicates "no Calc doc
    # open" (the boilerplate's new fail-fast guard, or the natural
    # AttributeError: Sheets symptom), we DO want to surface a warning
    # to the planner — otherwise the planner sees no SheetPerceiver
    # block at all and may hallucinate spreadsheet contents from a
    # screenshot that actually shows the Start Center. render_block
    # handles this special case and returns a strongly-worded SystemNote.
    if last_parsed is not None and "error" in last_parsed:
        err = str(last_parsed.get("error", ""))
        if ("AttributeError: Sheets" in err
                or "No LibreOffice Calc document" in err
                or "no Calc document" in err.lower()
                or "no spreadsheet" in err.lower()):
            block = render_block(last_parsed)
            _dump_artifacts(results_dir, step_idx, last_raw, last_parsed,
                            block, logger)
            return block
    _dump_artifacts(results_dir, step_idx, last_raw, last_parsed, None, logger)
    return None


_DOMAINS_WITH_SHEET_PROBE = {"libreoffice_calc"}


def should_probe(domain: Optional[str]) -> bool:
    return (domain or "").lower() in _DOMAINS_WITH_SHEET_PROBE

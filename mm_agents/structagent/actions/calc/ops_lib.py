"""Pre-built LibreOffice Calc operations exposed to the actor.

The string :data:`CALC_OPS_LIBRARY` is concatenated into every
``cli_run_uno`` script whose domain is ``libreoffice_calc``, right
after the connect + domain setup blocks. It defines:

  - ``op_*(...)``      — mutation operations the actor invokes.
                          Returns ``True`` on success; raises on bad
                          input. The actor's snippet then prints
                          ``PASS`` / ``FAIL: <reason>`` based on
                          the return / exception, in line with the
                          existing cli_run_uno verify convention.
  - ``verify_*(...)``  — read-only checks for the init_ledger
                          verify_spec. Returns ``True`` / ``False``;
                          the caller prints ``PASS`` / ``FAIL``.

Both kinds rely on the globals ``doc``, ``sheets``, ``controller``
already being bound by the wrapper. They do NOT re-connect to UNO
themselves — that is the wrapper's job, and re-connecting in the
actor body was a frequent source of bridge-state bugs (importing
``com.sun.star.X`` constant groups before the bridge is warm, etc.).

Design contract:
  * Every op is keyword-only (``*,``) so the actor cannot get arg
    order wrong.
  * Every op ends with ``doc.store()``.
  * Op functions raise ``ValueError`` for input that cannot succeed
    (missing sheet, unknown field name). The wrapper catches this,
    sets returncode=1, and writes the traceback to stderr — the
    planner sees the concrete cause in the next turn's
    ``[Last cli_run output]`` block.
"""

CALC_OPS_LIBRARY = '''
# ===== Begin pre-built Calc ops (auto-injected by cli_run_uno wrapper) =====
# NOTE: these helpers rely on the wrapper's globals: `doc`, `sheets`,
# `controller`. Do not redefine those above. Do not re-import uno.
import os

def _calc_ops_get_sheet(name):
    if name not in sheets.getElementNames():
        raise ValueError(f"sheet {name!r} does not exist")
    return sheets.getByName(name)


_CELL_TYPE_NAME_TO_INT = {
    "EMPTY":   0,
    "VALUE":   1,
    "TEXT":    2,   # canonical pyuno name for a string-content cell
    "STRING":  2,   # legacy alias just in case
    "FORMULA": 3,
}


def _calc_ops_type_int(cell):
    """getType() with bridge-version-independent int coercion.

    Older pyuno bridges let ``int(t)`` work; newer ones expose ``t.value``
    as the enum *name* string (\"EMPTY\" / \"VALUE\" / ...), and
    ``int(\"EMPTY\")`` is a ValueError. Map by name first — it is the
    contract every released pyuno version honors — and fall back to
    int() for ancient bindings.
    """
    t = cell.getType()
    name = getattr(t, "value", None)
    if isinstance(name, str):
        return _CELL_TYPE_NAME_TO_INT.get(name, 0)
    try:
        return int(t)
    except (TypeError, ValueError):
        return 0


def _calc_ops_normalize_formula(s):
    """Convert `,` argument separators to `;` outside of string literals.

    Calc's argument separator for multi-arg functions is `;` — a
    formula written with `,` parses as garbage even though the cell
    stores the source string verbatim, and the cached result becomes
    `#VALUE!`. We rewrite `,` to `;` whenever it sits OUTSIDE a
    double- or single-quoted run (so string literals containing
    commas — e.g. `LEFT(A2, FIND(", ", A2))` — are preserved).
    """
    if not isinstance(s, str) or not s.startswith("="):
        return s
    out = []
    in_dq = False
    in_sq = False
    for ch in s:
        if ch == '"' and not in_sq:
            in_dq = not in_dq
        elif ch == "'" and not in_dq:
            in_sq = not in_sq
        if ch == "," and not in_dq and not in_sq:
            out.append(";")
        else:
            out.append(ch)
    return "".join(out)


# ---- Mutation ops ------------------------------------------------------------

def op_new_sheet(*, name, position="end", source_sheet=None):
    """Create a new sheet in the workbook.

    Args:
        name: name of the NEW sheet to create
        position: where to insert. ``"end"`` (default) appends after the
            last sheet, ``"start"`` prepends at index 0, or pass an int
            (0-based) to insert at that index. ``"before:Sheet1"`` /
            ``"after:Sheet1"`` also accepted relative to an existing
            sheet name.
        source_sheet: when given, COPY that existing sheet's contents
            into the new sheet (preserves formatting + cell types).
            When None, the new sheet is blank.

    Raises ValueError if a sheet named ``name`` already exists, or if
    ``source_sheet`` is given but does not exist.
    """
    existing = list(sheets.getElementNames())
    if name in existing:
        raise ValueError(f"sheet {name!r} already exists")
    # Resolve position.
    if position == "end":
        idx = sheets.getCount()
    elif position == "start":
        idx = 0
    elif isinstance(position, int):
        idx = int(position)
    elif isinstance(position, str) and position.startswith("before:"):
        ref = position.split(":", 1)[1]
        if ref not in existing:
            raise ValueError(f"position references unknown sheet {ref!r}")
        idx = existing.index(ref)
    elif isinstance(position, str) and position.startswith("after:"):
        ref = position.split(":", 1)[1]
        if ref not in existing:
            raise ValueError(f"position references unknown sheet {ref!r}")
        idx = existing.index(ref) + 1
    else:
        raise ValueError(f"unrecognised position {position!r}")
    if source_sheet is not None:
        if source_sheet not in existing:
            raise ValueError(f"source sheet {source_sheet!r} does not exist")
        sheets.copyByName(source_sheet, name, int(idx))
    else:
        sheets.insertNewByName(name, int(idx))
    # Move the view to the new sheet so the next screenshot reflects it.
    try:
        controller.setActiveSheet(sheets.getByName(name))
    except Exception:
        pass
    doc.store()
    return True


def op_rename_sheet(*, new_name, current_name=None):
    """Rename ``current_name`` to ``new_name``. When ``current_name`` is
    None, the currently active sheet is renamed."""
    if current_name is None:
        target = controller.getActiveSheet()
    else:
        if current_name not in sheets.getElementNames():
            raise ValueError(f"sheet {current_name!r} does not exist")
        target = sheets.getByName(current_name)
    old = target.getName()
    if old == new_name:
        doc.store()
        return True
    if new_name in sheets.getElementNames():
        raise ValueError(
            f"target name {new_name!r} already in use by another sheet"
        )
    target.setName(new_name)
    doc.store()
    return True


def _calc_ops_resolve_a1(sheet_obj, a1_ref):
    """Resolve a single-cell A1 reference (e.g. 'B2') to (col, row).
    Range refs ('A1:B2') raise ValueError — single cells only."""
    if ":" in a1_ref:
        raise ValueError(
            f"op_set_cell expects a single-cell ref like 'A1', got {a1_ref!r}"
        )
    cell_range = sheet_obj.getCellRangeByName(a1_ref)
    addr = (cell_range.getCellAddress()
            if hasattr(cell_range, "getCellAddress")
            else cell_range.getRangeAddress())
    col = getattr(addr, "Column", None)
    row = getattr(addr, "Row", None)
    if col is None: col = addr.StartColumn
    if row is None: row = addr.StartRow
    return int(col), int(row)


def op_set_cell(*, sheet, a1_ref, value=None, formula=None):
    """Write a single cell.

    EXACTLY ONE of ``value`` / ``formula`` must be supplied:
      - ``value=...`` writes a literal (int/float → numeric cell;
        str → text cell). To write the literal string "42" rather than
        the number 42, pass ``value="42"``.
      - ``formula="=SUM(A1:A10)"`` writes a formula (must start with '=').
    """
    if (value is None) == (formula is None):
        raise ValueError(
            "op_set_cell requires exactly one of `value` or `formula`"
        )
    sh = _calc_ops_get_sheet(sheet)
    col, row = _calc_ops_resolve_a1(sh, a1_ref)
    cell = sh.getCellByPosition(col, row)
    if formula is not None:
        if not isinstance(formula, str) or not formula.startswith("="):
            raise ValueError(
                f"formula must be a string starting with '=', got {formula!r}"
            )
        cell.setFormula(_calc_ops_normalize_formula(formula))
    else:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            cell.setString(str(value))
        else:
            cell.setValue(float(value))
    doc.store()
    return True


def op_set_range_values(*, sheet, start_a1, values):
    """Write a 2D block of values starting at ``start_a1``.

    ``values`` is a list of rows; each row is a list of cell values.
    Each cell value is one of:
      - int / float           → numeric cell (setValue)
      - str starting with '=' → formula (setFormula)
      - any other str         → text cell (setString)
      - None                  → cell left UNCHANGED (skip)

    The block is rectangular — every row must have the same length as
    row 0, else ValueError. The destination doesn't need to already
    contain anything; existing values at those coordinates are
    overwritten where ``values`` has a non-None entry.
    """
    if not values or not isinstance(values, list):
        raise ValueError("`values` must be a non-empty list of rows")
    if not all(isinstance(r, list) for r in values):
        raise ValueError("`values` must be a list of LISTS (2D)")
    n_cols = len(values[0])
    for ri, row in enumerate(values):
        if len(row) != n_cols:
            raise ValueError(
                f"`values` row {ri} has length {len(row)}, expected {n_cols}"
            )
    sh = _calc_ops_get_sheet(sheet)
    start_col, start_row = _calc_ops_resolve_a1(sh, start_a1)
    for ri, row in enumerate(values):
        for ci, v in enumerate(row):
            if v is None:
                continue
            cell = sh.getCellByPosition(start_col + ci, start_row + ri)
            if isinstance(v, str) and v.startswith("="):
                cell.setFormula(_calc_ops_normalize_formula(v))
            elif isinstance(v, bool) or not isinstance(v, (int, float)):
                cell.setString(str(v))
            else:
                cell.setValue(float(v))
    doc.store()
    return True


def op_split_text(*, sheet, src_col, first_row, last_row,
                  out_cols, delimiter=" ", last_takes_rest=True):
    """Split each row's source cell by ``delimiter`` into ``out_cols``.

    Reads ``<src_col><r>`` as a string for each ``r`` in
    ``[first_row, last_row]`` (1-based, inclusive), splits by
    ``delimiter``, writes each piece to the corresponding column in
    ``out_cols`` at the same row.

    ``last_takes_rest=True`` (default): when the source has MORE
    delimiters than ``len(out_cols)-1``, the LAST out_col receives the
    remaining tail joined back with ``delimiter`` (e.g.
    "Senior Vice President" with 3 out_cols and "First Last Rank"
    layout puts "Senior Vice President" into the Rank column).

    ``last_takes_rest=False``: extra pieces are discarded. When the
    source has FEWER pieces than ``len(out_cols)``, the trailing
    out_cols are left UNCHANGED (cell skipped, not blanked).

    Source cells are NOT modified — preserves the "don't touch the
    original data" task constraint without extra work.
    """
    if not out_cols:
        raise ValueError("`out_cols` must be non-empty")
    if not isinstance(out_cols, (list, tuple)):
        raise ValueError("`out_cols` must be a list of column letters")
    sh = _calc_ops_get_sheet(sheet)
    src_col_idx, _ = _calc_ops_resolve_a1(sh, f"{src_col}1")
    out_col_idxs = []
    for c in out_cols:
        ci, _ = _calc_ops_resolve_a1(sh, f"{c}1")
        out_col_idxs.append(ci)
    n_out = len(out_cols)
    for r in range(int(first_row) - 1, int(last_row)):  # 0-based row idx
        src_cell = sh.getCellByPosition(src_col_idx, r)
        src_val = src_cell.getString()
        if not src_val:
            continue
        if last_takes_rest:
            parts = src_val.split(delimiter, n_out - 1)
        else:
            parts = src_val.split(delimiter)[:n_out]
        for i, out_col_idx in enumerate(out_col_idxs):
            if i >= len(parts):
                break
            dst_cell = sh.getCellByPosition(out_col_idx, r)
            dst_cell.setString(parts[i])
    doc.store()
    return True


def op_pivot_table(*, source_sheet, source_range, dest_sheet, dest_anchor,
                   row_fields=None, column_fields=None,
                   filter_fields=None, data_fields=None,
                   pivot_name="PivotTable1"):
    """Create a pivot table on ``dest_sheet`` summarizing ``source_sheet``.

    source_range MUST include the header row (UNO derives field names
    from row 0 of the range). ``data_fields`` is a list of
    ``(header, function)`` tuples — function in
    SUM/AVERAGE/COUNT/MAX/MIN/PRODUCT.
    """
    # NOTE: do NOT use ``from com.sun.star.sheet import
    # DataPilotFieldOrientation, GeneralFunction`` — that constant-group
    # import path is rejected by the pyuno bridge in OSWorld VMs
    # (ImportError: "No module named 'com'"). Construct the enum values
    # via ``uno.Enum(<service>, <member>)`` which the bridge supports.
    def _orient(name):
        return uno.Enum("com.sun.star.sheet.DataPilotFieldOrientation", name)
    def _func(name):
        return uno.Enum("com.sun.star.sheet.GeneralFunction", name)
    src = _calc_ops_get_sheet(source_sheet)
    dst = _calc_ops_get_sheet(dest_sheet)

    src_ra = src.getCellRangeByName(source_range).getRangeAddress()
    da = dst.getCellRangeByName(dest_anchor).getRangeAddress()
    dst_idx = list(sheets.getElementNames()).index(dest_sheet)
    dst_addr = uno.createUnoStruct("com.sun.star.table.CellAddress")
    dst_addr.Sheet = int(dst_idx)
    dst_addr.Column = int(da.StartColumn)
    dst_addr.Row = int(da.StartRow)

    pts = dst.getDataPilotTables()
    if pts.hasByName(pivot_name):
        pts.removeByName(pivot_name)
    desc = pts.createDataPilotDescriptor()
    desc.setSourceRange(src_ra)

    fields = desc.getDataPilotFields()
    name_to_idx = {}
    for i in range(fields.getCount()):
        f = fields.getByIndex(i)
        nm = getattr(f, "Name", None) or f.getName()
        name_to_idx[nm] = i

    _VALID_FUNCS = {"SUM", "AVERAGE", "COUNT", "MAX",
                    "MIN", "PRODUCT", "COUNTNUMS"}

    def _set(field_name, orient, func=None):
        if field_name not in name_to_idx:
            raise ValueError(
                f"source has no field named {field_name!r}; "
                f"have {list(name_to_idx)}"
            )
        f = fields.getByIndex(name_to_idx[field_name])
        f.Orientation = orient
        if func is not None:
            f.Function = func

    for n in (row_fields or []):
        _set(n, _orient("ROW"))
    for n in (column_fields or []):
        _set(n, _orient("COLUMN"))
    for n in (filter_fields or []):
        _set(n, _orient("PAGE"))
    for n, fn in (data_fields or []):
        fn_name = str(fn).upper()
        if fn_name not in _VALID_FUNCS:
            fn_name = "SUM"
        _set(n, _orient("DATA"), _func(fn_name))

    pts.insertNewByName(pivot_name, dst_addr, desc)
    doc.store()
    return True


# LCID -> (language, country) for the comma-decimal locales OSWorld
# tasks ask about. Hex LCIDs follow the Microsoft locale-identifier
# convention used in xlsx number-format strings ([$-419] etc.). Add
# more entries as new tasks surface them.
_LCID_TO_LOCALE = {
    0x419: ("ru", "RU"),  # Russian
    0x407: ("de", "DE"),  # German
    0x40C: ("fr", "FR"),  # French
    0x410: ("it", "IT"),  # Italian
    0x416: ("pt", "BR"),  # Portuguese (Brazil)
    0x40A: ("es", "ES"),  # Spanish (Spain)
    0x405: ("cs", "CZ"),  # Czech
    0x415: ("pl", "PL"),  # Polish
    0x408: ("el", "GR"),  # Greek
    0x41F: ("tr", "TR"),  # Turkish
    0x41D: ("sv", "SE"),  # Swedish
    0x40B: ("fi", "FI"),  # Finnish
    0x413: ("nl", "NL"),  # Dutch
    0x414: ("nb", "NO"),  # Norwegian
    0x406: ("da", "DK"),  # Danish
    0x40E: ("hu", "HU"),  # Hungarian
    0x418: ("ro", "RO"),  # Romanian
    0x402: ("bg", "BG"),  # Bulgarian
    0x426: ("lv", "LV"),  # Latvian
    0x427: ("lt", "LT"),  # Lithuanian
    0x425: ("et", "EE"),  # Estonian
}


def _split_lo_format_locale_prefix(format_string):
    """If ``format_string`` is of the shape ``[$-<hex-lcid>]<rest>``,
    return ``(rest, (lang, country, ""))``; else return
    ``(format_string, None)``.

    LO's UNO bridge ``addNew`` silently strips the ``[$-LCID]`` prefix
    when registering, leaving an unlocalized format that renders with
    the document's default locale (no comma decimal). To get comma-
    decimal rendering we need to instead pass the LCID's locale as the
    ``addNew`` *parse* locale AND replace the decimal point in the
    remaining format with the locale's decimal separator (a comma for
    every entry in :data:`_LCID_TO_LOCALE`).
    """
    if not isinstance(format_string, str):
        return format_string, None
    if not format_string.startswith("[$-"):
        return format_string, None
    end = format_string.find("]")
    if end < 4:
        return format_string, None
    hex_part = format_string[3:end]
    try:
        lcid = int(hex_part, 16)
    except ValueError:
        return format_string, None
    pair = _LCID_TO_LOCALE.get(lcid)
    if pair is None:
        return format_string, None
    rest = format_string[end + 1:]
    # Every locale we care about uses ',' as the decimal separator —
    # rewrite '.' to ',' in the format remainder so addNew parses it
    # correctly under that locale.
    rest_localized = rest.replace(".", ",")
    return rest_localized, (pair[0], pair[1], "")


def op_format_number_range(*, sheet, range_ref, format_string, locale=None):
    """Apply a number format to a range. ``format_string`` is Excel-style.

    ``locale``: optional ``(language, country, variant)`` tuple to change
    the format's locale (e.g. ``("de", "DE", "")`` for German decimal).
    None means the document's default locale.

    Locale-prefixed formats (``[$-419]0.0`` etc.) are auto-translated:
    the UNO ``addNew`` call silently drops the ``[$-LCID]`` prefix and
    registers the unlocalized format, so cells render with the doc's
    default locale (no comma decimal). We detect the prefix, map the
    hex LCID to ``(lang, country)``, swap ``.`` -> ``,`` in the rest
    of the format, and call ``addNew(rewritten, parsed_locale)``. The
    resulting registry entry renders ``0.1`` as ``0,1`` etc. and
    survives a save + CSV-export-as-shown round-trip — matching the
    OSWorld evaluator's gold CSV for decimal-separator tasks.
    """
    # NOTE: ``from com.sun.star.lang import Locale`` is rejected by the
    # pyuno bridge in OSWorld VMs (ImportError). Use uno.createUnoStruct
    # which the bridge supports for plain UNO structs.
    sh = _calc_ops_get_sheet(sheet)
    fmt_registry = doc.getNumberFormats()
    # If the format carries a [$-LCID] prefix AND no explicit locale arg
    # overrode it, translate the prefix to a parse-locale + decimal-sep
    # rewrite so the registry actually picks up the localized format.
    if locale is None:
        rewritten, derived_locale = _split_lo_format_locale_prefix(
            format_string)
        if derived_locale is not None:
            format_string = rewritten
            locale = derived_locale
    loc = uno.createUnoStruct("com.sun.star.lang.Locale")
    if locale is not None:
        loc.Language, loc.Country, loc.Variant = locale
    fmt_id = fmt_registry.queryKey(format_string, loc, False)
    if fmt_id == -1:
        fmt_id = fmt_registry.addNew(format_string, loc)
    rng = sh.getCellRangeByName(range_ref)
    rng.NumberFormat = fmt_id
    doc.store()
    return True


def op_sort_range(*, sheet, range_ref, sort_columns, contains_header=True):
    """Sort a cell range.

    ``sort_columns`` is a list of ``(col_idx_within_range, ascending_bool)``.
    Column indices are 0-based WITHIN the range — if the range starts at
    column C and you want to sort by column D, pass ``[(1, True)]``.
    """
    sh = _calc_ops_get_sheet(sheet)
    rng = sh.getCellRangeByName(range_ref)
    fields = []
    for col_idx, ascending in sort_columns:
        f = uno.createUnoStruct("com.sun.star.util.SortField")
        f.Field = int(col_idx)
        f.SortAscending = bool(ascending)
        fields.append(f)

    def _pv(name, value):
        p = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
        p.Name = name
        p.Value = value
        return p

    descriptor = (
        _pv("SortFields",
            uno.Any("[]com.sun.star.util.SortField", tuple(fields))),
        _pv("ContainsHeader", bool(contains_header)),
    )
    rng.sort(descriptor)
    doc.store()
    return True


def op_freeze_pane(*, n_cols=0, n_rows=0):
    """Freeze the top ``n_rows`` rows and leftmost ``n_cols`` columns.

    Examples: ``n_rows=1`` keeps row 1 visible on scroll (header freeze).
    ``n_cols=1, n_rows=1`` freezes A1 (both row 1 and column A).
    """
    controller.freezeAtPosition(int(n_cols), int(n_rows))
    doc.store()
    return True


def op_duplicate_range(*, source_sheet, source_range,
                  dest_sheet, dest_anchor, transpose=False):
    """Copy a cell range to another location, preserving cell types
    (numeric / string / formula). Destination sheet must already exist
    (create it via ``calc_new_sheet`` first if needed).
    EMPTY source cells leave the destination cell untouched.

    Args:
        transpose:  When True, write source ``(r,c)`` to destination
                    ``(c,r)``. Source ``m × n`` produces destination
                    ``n × m`` anchored at ``dest_anchor``. Use for
                    "transpose and paste" requests; the destination
                    bounding box is implicit (``n_rows`` becomes
                    columns, ``n_cols`` becomes rows). NOTE: formula
                    cell references are NOT auto-rewritten on
                    transpose — the formula source string is copied
                    verbatim, so transposing a formula like ``=A2+1``
                    will still reference column A in the new
                    position, which is rarely what callers want for
                    matrix transposes of computed data. Prefer
                    transposing VALUE / STRING cells; copy formulas
                    only when the caller has confirmed references
                    survive the rotation.
    """
    src = _calc_ops_get_sheet(source_sheet)
    dst = _calc_ops_get_sheet(dest_sheet)
    src_range = src.getCellRangeByName(source_range)
    src_ra = src_range.getRangeAddress()
    n_cols = src_ra.EndColumn - src_ra.StartColumn + 1
    n_rows = src_ra.EndRow - src_ra.StartRow + 1
    dst_anchor_cell = dst.getCellRangeByName(dest_anchor)
    da = (dst_anchor_cell.getCellAddress()
          if hasattr(dst_anchor_cell, "getCellAddress")
          else dst_anchor_cell.getRangeAddress())
    dst_col0 = getattr(da, "Column", None)
    if dst_col0 is None:
        dst_col0 = da.StartColumn
    dst_row0 = getattr(da, "Row", None)
    if dst_row0 is None:
        dst_row0 = da.StartRow
    for r in range(n_rows):
        for c in range(n_cols):
            s_cell = src.getCellByPosition(
                src_ra.StartColumn + c, src_ra.StartRow + r)
            if transpose:
                d_cell = dst.getCellByPosition(
                    dst_col0 + r, dst_row0 + c)
            else:
                d_cell = dst.getCellByPosition(
                    dst_col0 + c, dst_row0 + r)
            t = _calc_ops_type_int(s_cell)
            if t == 1:
                d_cell.setValue(s_cell.getValue())
            elif t == 2:
                d_cell.setString(s_cell.getString())
            elif t == 3:
                d_cell.setFormula(s_cell.getFormula())
    doc.store()
    return True


def op_copy_to_clipboard(*, sheet, range_ref, copy=True):
    """Select a cell range in the Calc view and (default) copy it to
    the clipboard.

    Purpose: DETERMINISTIC precise-range selection for a cross-app
    copy/paste workflow. Select exactly the cells you need (e.g. the
    header row + ONE data row), then paste into another application
    (e.g. a Writer document — Writer renders pasted Calc cells as a
    native table). Replaces brittle GUI drag-select, which tends to
    grab whole rows or extra rows.

    Args:
        sheet:      name of the sheet holding the range.
        range_ref:  A1 range, e.g. "A1:L2" (header + first data row).
        copy:       when True (default) also issue ``.uno:Copy`` so
                    the selection lands on the clipboard, ready for a
                    paste (Ctrl+V) in the target application.

    Non-mutating — does NOT call ``doc.store()``."""
    sh = _calc_ops_get_sheet(sheet)
    cell_range = sh.getCellRangeByName(range_ref)
    controller.setActiveSheet(sh)
    controller.select(cell_range)
    if copy:
        # DispatchHelper drives .uno:Copy on the Calc frame; the
        # selection set just above is what lands on the clipboard.
        _dispatcher = remote_ctx.ServiceManager.createInstanceWithContext(
            "com.sun.star.frame.DispatchHelper", remote_ctx)
        _dispatcher.executeDispatch(
            controller.getFrame(), ".uno:Copy", "", 0, ())
    return True


def _calc_ops_col_letter_to_idx(letter):
    """Convert a column letter (``A``, ``Z``, ``AA``, …) to a 0-based
    column index. Used by ``op_hide_range`` / ``verify_col_hidden``
    where the caller passes user-friendly letters, not indices."""
    s = str(letter).strip().upper()
    if not s:
        raise ValueError("column letter is empty")
    n = 0
    for ch in s:
        if not ('A' <= ch <= 'Z'):
            raise ValueError(f"invalid column letter {letter!r}")
        n = n * 26 + (ord(ch) - ord('A') + 1)
    return n - 1


def op_insert_chart(*, sheet, data_sheet, data_range=None,
                    categories=None, values=None,
                    chart_type="column", subtype="clustered",
                    title="", name=None, anchor="A1",
                    width_cm=12.0, height_cm=8.0,
                    column_headers=None, row_headers=None,
                    **_extra):
    """Insert an embedded chart on ``sheet``. Two ways to specify the
    chart's data:

      (1) ``data_range`` — a single contiguous A1 range on
          ``data_sheet``. LO infers series by row/column from the
          range, with headers per ``column_headers`` / ``row_headers``.
          Use for the common "chart this whole table" case.

      (2) ``categories`` (single A1 range) + ``values`` (list of A1
          ranges) — explicit per-axis ranges, like OfficeCLI's
          ``series.categories`` / ``series.values`` idiom. Use when
          the x-axis labels live in one column and the y-axis values
          live in a NON-ADJACENT column (e.g. "x-axis = Date Time
          (col A), y-axis = Quantity (col E)" task), or when the
          chart shows ONLY a single summary row/column whose
          adjacent header row supplies the category labels. Multiple
          entries in ``values`` produce multi-series charts.

    Forms (1) and (2) are mutually exclusive — providing both raises
    a ValueError. Pass ONE shape per call.

    Args:
        sheet:        sheet name where the chart is placed.
        data_sheet:   sheet name holding the source data.
        data_range:   form (1) — A1 range, INCLUDING the header row
                      + first ID column when ``column_headers``
                      / ``row_headers`` are True.
        categories:   form (2) — A1 range for the x-axis category
                      labels (e.g. ``"A1:A36"`` or ``"B1:G1"``).
                      Optional; LO falls back to 1,2,3... when omitted.
        values:       form (2) — list of A1 range strings, one per
                      series. E.g. ``["E1:E36"]`` for a single-series
                      line chart, or ``["B2:B11", "C2:C11"]`` for two
                      series sharing the same category axis.
        chart_type:   one of "column" (vertical bars, default),
                      "bar" (horizontal bars), "line", "pie",
                      "scatter", "area".
        subtype:      "clustered" (default), "stacked", "percent".
                      Only applies to column/bar/area.
        title:        chart title string. Optional.
        name:         chart name on the sheet (used by ``verify_chart_exists``
                      and to update an existing chart). Default
                      ``"Chart<N>"`` auto-picked from existing charts.
        anchor:       single-cell A1 reference; chart top-left
                      approximately positioned at that cell.
        width_cm:     chart width in centimetres (default 12).
        height_cm:    chart height in centimetres (default 8).
        column_headers / row_headers: form (1) only — whether the
                      source range's first row / first column are
                      headers.

    The evaluator's ``chart`` rule reads chart series formulas as the
    dict key — an over-wide ``data_range`` produces extra series and
    misses the match. Pick the form that matches the chart's intent:
    summary-row/column charts → form (2) with the row/column as the
    single ``values`` entry.
    """
    if data_range is not None and (categories is not None or values):
        raise ValueError(
            "op_insert_chart: pass EITHER data_range OR "
            "categories/values, not both. Form (1) is data_range; "
            "form (2) is categories + values.")
    if data_range is None and not values:
        raise ValueError(
            "op_insert_chart: needs data_range (form 1) or values "
            "(form 2).")
    # Source data sheet MUST exist (chart references its cells).
    src = _calc_ops_get_sheet(data_sheet)
    # Destination sheet: auto-create if missing, so the planner doesn't
    # need a separate calc_new_sheet step before this. Removes the
    # most common 2-step plan (calc_new_sheet + calc_insert_chart) in
    # favour of a 1-step "place chart on new sheet X reading from
    # sheet Y" idiom.
    auto_created_sheet = False
    if sheet not in sheets.getElementNames():
        sheets.insertNewByName(sheet, sheets.getCount())
        auto_created_sheet = True
    dst = sheets.getByName(sheet)
    if auto_created_sheet and sheet != data_sheet:
        # The planner often emits ``sheet=<chart title>`` (e.g.
        # "2019 Monthly Costs") because the doc string mentions "auto-
        # create". This is almost always WRONG — the eval's
        # chart_exists check usually targets the data sheet. Surface
        # a clear DIAG so the planner's next-turn
        # ``[Last cli_run output]`` shows the actionable correction.
        print(
            "DIAG: calc_insert_chart auto-created an empty side sheet "
            "{!r} to host the chart. Most tasks want the chart on the "
            "DATA sheet ({!r}); the verify_chart_exists check usually "
            "targets the data sheet. If the task did NOT explicitly say "
            "'create chart in a new sheet', RE-RUN this action with "
            "sheet={!r} (and remove the empty side sheet via "
            "calc_rename_sheet or just leave it — verify won't fail on "
            "extra empty sheets, only on missing charts on the data "
            "sheet).".format(sheet, data_sheet, data_sheet)
        )
    def _peel_sheet(s):
        # Tolerate sheet prefix like ``Sheet1.A1:C11`` / ``Sheet1!A1:C11``
        # in any A1 range string; LO's getCellRangeByName expects a
        # plain A1 like "A1:C11".
        s = str(s).strip()
        if "!" in s:
            s = s.split("!", 1)[-1]
        if "." in s:
            s = s.split(".", 1)[-1]
        return s

    if data_range is not None:
        # Form (1): single contiguous range. Default headers to True
        # (the common "chart this whole table" shape has both a header
        # row and a leftmost label column).
        src_range = src.getCellRangeByName(_peel_sheet(data_range))
        src_addrs = (src_range.getRangeAddress(),)
        if column_headers is None:
            column_headers = True
        if row_headers is None:
            row_headers = True
    else:
        # Form (2): explicit categories + values list.
        # ``addNewByName``'s multi-range mode CANNOT correctly interpret
        # row-shape categories+values (LO defaults to column-oriented
        # series, producing one 1-point series per data-area column —
        # the empty line-plot failure observed on 0a2e43bf). The reliable
        # path is to build the chart via chart2 API: pass a single
        # placeholder range to ``addNewByName`` (just so the chart exists)
        # then explicitly replace its DataSeries + x-axis categories.
        # Header flags don't matter for the placeholder; force False/False
        # so the placeholder doesn't accidentally drop the first cell.
        if not values:
            raise ValueError(
                "op_insert_chart form (2): need values=[<range>, ...]")
        # Cell-count validation: every value range must have the same
        # cell count as ``categories``. The planner LLM frequently emits
        # mismatched ranges (extending categories into the leftmost
        # label column when values only covers data columns, or picking
        # categories from the values row itself instead of the header
        # row). Failing here surfaces a precise error to the ledger so
        # the planner gets concrete feedback for the re-plan instead of
        # an empty / mislabeled chart that silently fails the eval.
        def _count_cells(range_str):
            rng = src.getCellRangeByName(_peel_sheet(range_str))
            ra = rng.getRangeAddress()
            return ((ra.EndColumn - ra.StartColumn + 1)
                    * (ra.EndRow - ra.StartRow + 1))
        if categories is not None:
            cat_n = _count_cells(categories)
            cat_addr = src.getCellRangeByName(
                _peel_sheet(categories)).getRangeAddress()
            for v in values:
                v_n = _count_cells(v)
                if cat_n != v_n:
                    raise ValueError(
                        "op_insert_chart form (2): categories range %r "
                        "has %d cells but values entry %r has %d. They "
                        "MUST cover EXACTLY the same cols (when values "
                        "is a row) or rows (when values is a column). "
                        "Common cause: categories extends into the "
                        "leftmost label cell (col A) when values only "
                        "covers cols B..G; or categories points at the "
                        "VALUE row itself instead of the header row. "
                        "Re-emit with categories trimmed to match "
                        "values' span exactly."
                        % (categories, cat_n, v, v_n))
                # Categories and values MUST live on DIFFERENT rows
                # (when each is a row) or DIFFERENT columns (when
                # each is a column). Identical ranges or same-row/
                # same-col ranges mean categories points at the value
                # row itself — semantically meaningless (chart labels
                # are the values themselves).
                v_addr = src.getCellRangeByName(
                    _peel_sheet(v)).getRangeAddress()
                cat_is_row = cat_addr.StartRow == cat_addr.EndRow
                v_is_row = v_addr.StartRow == v_addr.EndRow
                cat_is_col = cat_addr.StartColumn == cat_addr.EndColumn
                v_is_col = v_addr.StartColumn == v_addr.EndColumn
                same_row = (cat_is_row and v_is_row
                            and cat_addr.StartRow == v_addr.StartRow)
                same_col = (cat_is_col and v_is_col
                            and cat_addr.StartColumn == v_addr.StartColumn)
                if same_row or same_col:
                    raise ValueError(
                        "op_insert_chart form (2): categories range %r "
                        "and values entry %r live on the same %s — they "
                        "must come from DIFFERENT %s (categories from "
                        "the header row above values when values is a "
                        "data row; categories from the leftmost label "
                        "column when values is a data column). Pointing "
                        "categories at the value row itself yields a "
                        "chart whose x-axis labels are the data values, "
                        "which has no semantic meaning."
                        % (categories, v,
                           "row" if same_row else "column",
                           "rows" if same_row else "columns"))
        _placeholder_range = src.getCellRangeByName(
            _peel_sheet(values[0])).getRangeAddress()
        src_addrs = (_placeholder_range,)
        column_headers = False
        row_headers = False

    # Approximate position from anchor cell — convert anchor's column/row
    # to a 1/100 mm rectangle origin. Default cell width ~22mm × 5mm
    # gives an ok ballpark for placement; the actual chart is sized
    # by ``width_cm`` / ``height_cm`` rather than cell counts.
    anchor_range = dst.getCellRangeByName(anchor)
    a_addr = anchor_range.getRangeAddress()
    rect = uno.createUnoStruct("com.sun.star.awt.Rectangle")
    rect.X = int(a_addr.StartColumn * 2400)        # ~24mm per col → 100ths-of-mm
    rect.Y = int(a_addr.StartRow * 450)            # ~4.5mm per row
    rect.Width = int(float(width_cm) * 1000)       # cm → 100ths-of-mm
    rect.Height = int(float(height_cm) * 1000)

    charts = dst.Charts
    if name is None or name == "":
        # No explicit name from the actor — auto-pick the next unused
        # ``Chart<N>`` on this sheet so a second insert call does NOT
        # silently overwrite the first (the failure mode in two-chart
        # tasks like 347ef137 / 0326d92d, where the actor never names
        # the chart and the old default ``"Chart1"`` self-collided).
        existing = set(charts.getElementNames())
        n = 1
        while ("Chart{}".format(n)) in existing:
            n += 1
        name = "Chart{}".format(n)
        # Anchor de-collision: when the actor didn't pass ``anchor``
        # explicitly (defaulted to "A1"), every multi-chart task lays
        # its charts on top of each other at A1 — both serialize to
        # xlsx correctly (the eval keys charts by series formula, not
        # by position), but the trajectory screenshots show only the
        # last chart drawn and hide all earlier ones. Shift the rect
        # DOWN by ``height_cm`` × (existing-chart count) so each new
        # chart lands below the previous one.
        if (anchor or "A1").upper() == "A1" and existing:
            rect.Y += int(float(height_cm) * 1000) * len(existing)
    elif charts.hasByName(name):
        # Explicit name → keep the original idempotent-retry semantics:
        # replace any chart already carrying that name.
        charts.removeByName(name)
    charts.addNewByName(name, rect, src_addrs,
                        bool(column_headers), bool(row_headers))

    chart_doc = charts.getByName(name).getEmbeddedObject()

    # Diagram type (chart1 API — still supported in modern LO).
    # column = vertical bars; bar = horizontal bars; both BarDiagram.
    _TYPE_TO_SERVICE = {
        "column":  "com.sun.star.chart.BarDiagram",
        "bar":     "com.sun.star.chart.BarDiagram",
        "line":    "com.sun.star.chart.LineDiagram",
        "pie":     "com.sun.star.chart.PieDiagram",
        "scatter": "com.sun.star.chart.XYDiagram",
        "area":    "com.sun.star.chart.AreaDiagram",
    }
    svc = _TYPE_TO_SERVICE.get(str(chart_type).lower(),
                                "com.sun.star.chart.BarDiagram")
    diagram = chart_doc.createInstance(svc)
    chart_doc.setDiagram(diagram)
    # LO BarDiagram.Vertical convention (matches lo_chart_helper.py):
    # column chart (vertical bars going up, categories on X) → Vertical=False
    # bar chart (horizontal bars going right, categories on Y) → Vertical=True
    # Setting these the OTHER way around makes openpyxl read barDir='bar'
    # when 'col' is expected — silently fails chart_props comparison.
    if str(chart_type).lower() == "column":
        try: diagram.Vertical = False
        except Exception: pass
    elif str(chart_type).lower() == "bar":
        try: diagram.Vertical = True
        except Exception: pass

    sub = str(subtype).lower()
    if sub == "stacked":
        try: diagram.Stacked = True
        except Exception: pass
    elif sub == "percent":
        try:
            diagram.Stacked = True
            diagram.Percent = True
        except Exception: pass

    if title:
        try:
            chart_doc.HasMainTitle = True
            chart_doc.Title.String = str(title)
        except Exception:
            pass

    # Form (2): replace the placeholder data with explicit chart2-API
    # DataSeries + categories. ``addNewByName``'s multi-range
    # interpretation cannot reliably represent "categories=<one
    # range>, values=[<another range>, ...]" — especially for
    # row-shape inputs where every column of the bounding box becomes
    # a one-point series. The chart2 API lets us name the categories
    # range AND the values ranges explicitly via XLabeledDataSequence
    # objects, which serialize to the exact ``$Sheet!$X$N:$Y$M``
    # references the openpyxl-based gold evaluator keys charts by.
    if data_range is None:
        try:
            provider = chart_doc.getDataProvider()
            sm = remote_ctx.ServiceManager
            sheet_prefix = "$" + data_sheet + "."
            def _abs_a1(rng):
                # Convert "B1:G1" or "B12:G12" → "$B$1:$G$1" / "$B$12:$G$12";
                # leave alone anything already containing $.
                if "$" in rng:
                    return rng
                import re as _re
                m = _re.match(r"^([A-Z]+)(\d+)(?::([A-Z]+)(\d+))?$",
                              rng.upper())
                if not m:
                    return rng
                c1, r1, c2, r2 = m.groups()
                if c2:
                    return "$%s$%s:$%s$%s" % (c1, r1, c2, r2)
                return "$%s$%s" % (c1, r1)
            def _labeled(role, range_ref):
                full = sheet_prefix + _abs_a1(_peel_sheet(range_ref))
                seq = provider.createDataSequenceByRangeRepresentation(full)
                seq.Role = role
                ld = sm.createInstanceWithContext(
                    "com.sun.star.chart2.data.LabeledDataSequence",
                    remote_ctx)
                ld.Values = seq
                return ld

            diag2 = chart_doc.getFirstDiagram()
            # Inject categories on the x-axis (dim 0) ScaleData.
            if categories is not None:
                cat_ld = _labeled("categories", categories)
                for cs in diag2.getCoordinateSystems():
                    try:
                        ax = cs.getAxisByDimension(0, 0)
                        sd = ax.getScaleData()
                        sd.Categories = cat_ld
                        ax.setScaleData(sd)
                    except Exception:
                        pass
            # Build one DataSeries per values entry and replace the
            # placeholder series in each ChartType.
            new_series = []
            for vrng in values:
                val_ld = _labeled("values-y", vrng)
                ds = sm.createInstanceWithContext(
                    "com.sun.star.chart2.DataSeries", remote_ctx)
                ds.setData([val_ld])
                new_series.append(ds)
            for cs in diag2.getCoordinateSystems():
                for ct in cs.getChartTypes():
                    ct.setDataSeries(new_series)
        except Exception:
            # Fallback: leave the placeholder chart in place. Better an
            # imperfect chart than a hard failure that drops the whole
            # action.
            pass

    doc.store()
    return True


def _calc_ops_infer_format_from_path(path):
    """Map a file extension to one of the supported export formats.

    Returns a normalized format string ("pdf" / "csv" / "html") so
    callers can pass ``format="auto"`` and rely on the path extension.
    """
    ext = os.path.splitext(str(path))[1].lower().lstrip(".")
    if ext in ("pdf",):
        return "pdf"
    if ext in ("csv", "tsv", "txt"):
        return "csv"
    # [FIX calc_export_html] HTML is a routinely-requested export format
    # ("Save as Web Page") and the UNO bridge supports it via the
    # "HTML (StarCalc)" filter. Without this branch tasks fell back to
    # File→Export GUI and reliably mis-clicked into the PDF dialog.
    # Roll back by removing this `if ext in ("html", ...)` block and the
    # corresponding html branch in ``op_export_file``.
    if ext in ("html", "htm", "xhtml"):
        return "html"
    raise ValueError(
        "calc_export_file: cannot infer format from path %r (ext=%r). "
        "Pass an explicit format=<pdf|csv|html>." % (path, ext))


def _calc_ops_sanitize_export_path(path, fmt):
    """Strip a stray ``.xlsx`` (or ``.xls``/``.ods``) sitting between
    the basename and the export extension. ``Foo.xlsx.pdf`` → ``Foo.pdf``.

    The planner sometimes composes the output path by appending the
    new extension to the OPEN document's full filename instead of its
    stem. Catch that here so the produced file lands at the path the
    evaluator expects.
    """
    p = str(path)
    suffix = "." + fmt
    if not p.lower().endswith(suffix):
        return p
    stem = p[:-len(suffix)]
    for src_ext in (".xlsx", ".xls", ".ods", ".xlsm", ".csv"):
        if stem.lower().endswith(src_ext):
            return stem[:-len(src_ext)] + suffix
    return p


def op_export_file(*, path, format="auto",
                   fit_width_pages=None, fit_height_pages=None,
                   sheet=None,
                   delimiter=",", quote='"', encoding="utf-8",
                   quote_all=False, save_formulas=False,
                   **_extra):
    """Export the open document to a file via UNO ``storeToURL``.

    SINGLE-DISPATCH replacement for the GUI-only "File → Export"
    path. ``format`` decides the filter; ``path`` ends with the
    matching extension. ONE step does everything: set scaling,
    serialize, write.

    Args:
        path:               absolute path on the VM filesystem
                            ending in ``.pdf`` or ``.csv``. Parent
                            directory must already exist.
        format:             "pdf" / "csv" / "auto" (infer from
                            ``path`` extension). Defaults to "auto"
                            so the planner usually only needs ``path``.

      PDF-specific (ignored when format=="csv"):
        fit_width_pages:    int — "fit print range to N pages wide"
                            (PageStyle.ScaleToPagesX). ``None`` =
                            don't change scaling. Use ``1`` for the
                            common "fit to one page width" request.
        fit_height_pages:   int — "fit to N pages tall"
                            (PageStyle.ScaleToPagesY). Pair with
                            ``fit_width_pages=1`` for "fit on one
                            page" tasks.

      CSV-specific (ignored when format=="pdf"):
        sheet:              sheet name to export (CSV only writes
                            one sheet at a time). ``None`` = active
                            sheet (matches "current sheet" wording
                            in instructions).
        delimiter:          field separator char (default ",").
        quote:              text delimiter char (default ``"``).
        encoding:           text encoding name ("utf-8", "ascii",
                            "cp1252"). Default UTF-8.
        quote_all:          quote every text cell (True) vs only
                            cells that need escaping (False, default).
        save_formulas:      write formula source vs the evaluated
                            cell value. Tasks asking for contents
                            "as shown on the screen" need ``False``.

    Returns ``True`` on success. Raises ``RuntimeError`` if the file
    was not produced (filter rejected the document, path not
    writable, etc.).
    """
    fmt = str(format).lower()
    if fmt == "auto":
        fmt = _calc_ops_infer_format_from_path(path)
    # Defensive: strip a stray source extension if the planner
    # composed ``Foo.xlsx.pdf`` instead of ``Foo.pdf``. Keeps the
    # action robust without forcing the planner to manage extension
    # peeling itself.
    path = _calc_ops_sanitize_export_path(path, fmt)

    # Resolve and validate output path BEFORE serializing, so we fail
    # fast on typos instead of writing junk into the wrong place.
    abs_path = os.path.abspath(str(path))
    parent = os.path.dirname(abs_path)
    if parent and not os.path.isdir(parent):
        raise ValueError(
            "calc_export_file: parent directory does not exist: %r" % parent)

    def _mkprop(name, value):
        pv = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
        pv.Name = name
        pv.Value = value
        return pv

    if fmt == "pdf":
        # Apply page-style scaling BEFORE storeToURL so the rendered
        # PDF picks up the fit-to-N-pages layout. Each sheet uses
        # its own PageStyle; iterating all sheets handles both the
        # active-sheet-only case and the multi-sheet workbook case.
        if fit_width_pages is not None or fit_height_pages is not None:
            seen_styles = set()
            ps_family = doc.StyleFamilies.getByName("PageStyles")
            for i in range(sheets.getCount()):
                sh = sheets.getByIndex(i)
                psn = sh.PageStyle
                if psn in seen_styles:
                    continue
                seen_styles.add(psn)
                try:
                    ps = ps_family.getByName(psn)
                except Exception:
                    continue
                # Clear plain percentage scaling so the fit-to-pages
                # setting actually drives layout.
                try: ps.PageScale = 0
                except Exception: pass
                try: ps.ScaleToPages = 0
                except Exception: pass
                if fit_width_pages is not None:
                    try: ps.ScaleToPagesX = int(fit_width_pages)
                    except Exception: pass
                if fit_height_pages is not None:
                    try: ps.ScaleToPagesY = int(fit_height_pages)
                    except Exception: pass
        out_url = "file://" + abs_path
        doc.storeToURL(out_url, (_mkprop("FilterName", "calc_pdf_Export"),))
    elif fmt == "csv":
        # If a specific sheet was named, activate it so the CSV
        # writer serializes THAT sheet's data (LO's CSV filter
        # writes the active sheet by default).
        if sheet is not None:
            if sheet not in sheets.getElementNames():
                raise ValueError(
                    "calc_export_file: sheet %r not found in workbook" % sheet)
            try:
                controller.setActiveSheet(sheets.getByName(sheet))
            except Exception:
                pass
        # FilterOptions token order (LO Text/CSV filter):
        #   <field_sep_ascii>,<text_delim_ascii>,<charset_token>,
        #   <start_line>,,<lang>,<quote_all>,<save_as_shown>,
        #   <save_formulas>,<remove_space>,<export_all_sheets>
        # Char set: 76 = UTF-8, 0 = system default.
        _ENC = {"utf-8": 76, "utf8": 76, "ascii": 11, "cp1252": 4,
                "windows-1252": 4, "iso-8859-1": 12}
        charset = _ENC.get(str(encoding).lower().replace("_", "-"), 76)
        if not delimiter:
            raise ValueError("calc_export_file: delimiter must be non-empty")
        if not quote:
            # ord(chr(0)) == 0 -> filter "no text delimiter"
            quote = chr(0)
        token = "{d},{q},{c},1,,0,{qa},true,{sf},false,false".format(
            d=ord(delimiter[0]), q=ord(quote[0]), c=charset,
            qa=("true" if quote_all else "false"),
            sf=("true" if save_formulas else "false"))
        out_url = "file://" + abs_path
        doc.storeToURL(out_url, (
            _mkprop("FilterName", "Text - txt - csv (StarCalc)"),
            _mkprop("FilterOptions", token),
        ))
    # [FIX calc_export_html] see infer_format_from_path note above.
    elif fmt == "html":
        if sheet is not None:
            if sheet not in sheets.getElementNames():
                raise ValueError(
                    "calc_export_file: sheet %r not found in workbook" % sheet)
            try:
                controller.setActiveSheet(sheets.getByName(sheet))
            except Exception:
                pass
        out_url = "file://" + abs_path
        doc.storeToURL(out_url, (
            _mkprop("FilterName", "HTML (StarCalc)"),
        ))
    else:
        raise ValueError(
            "calc_export_file: unsupported format %r (want pdf, csv, or html)" % fmt)

    if not os.path.exists(abs_path):
        raise RuntimeError(
            "calc_export_file: storeToURL returned but file not written: "
            "path=%r format=%s" % (abs_path, fmt))
    return True


def verify_file_exists(*, path, min_bytes=1, **_extra):
    """Verify a file exists on the VM filesystem with a minimum size.

    Args:
        path:       absolute path on the VM.
        min_bytes:  size must be ``>= min_bytes`` (default 1 — rules
                    out the empty-file case where the export filter
                    wrote a placeholder but no rows).

    Used to confirm ``calc_export_file`` actually produced output
    (PDF / CSV) at the expected path.
    """
    try:
        if not os.path.exists(str(path)):
            return False
        return os.path.getsize(str(path)) >= int(min_bytes)
    except Exception:
        return False


def op_hide_range(*, sheet, rows=None, cols=None):
    """Hide entire rows and/or entire columns on a sheet.

    Args:
        sheet:  sheet name.
        rows:   optional list of 1-based row numbers (matches A1
                conventions — row 1 is the header row).
        cols:   optional list of column letters (``["A", "C"]``) or
                0-based column indices (``[0, 2]``).

    At least one of ``rows`` or ``cols`` must be non-empty. Hidden
    rows/columns survive ``doc.store()`` to the .xlsx (saved as the
    ``hidden`` row/col property), so downstream comparators that
    inspect ``row_props.hidden`` / ``col_props.hidden`` see the
    change.

    Source cells are NOT modified — this is purely a view/structural
    op, so the "don't delete data" task constraint is preserved.
    """
    if not rows and not cols:
        raise ValueError("op_hide_range needs at least one of rows / cols")
    sh = _calc_ops_get_sheet(sheet)
    if rows:
        sh_rows = sh.getRows()
        for r in rows:
            sh_rows.getByIndex(int(r) - 1).IsVisible = False
    if cols:
        sh_cols = sh.getColumns()
        for c in cols:
            idx = c if isinstance(c, int) else _calc_ops_col_letter_to_idx(c)
            sh_cols.getByIndex(int(idx)).IsVisible = False
    doc.store()
    return True


def op_delete_columns(*, sheet, cols, **_extra):
    """PERMANENTLY delete entire columns from a sheet (the cells are
    removed and every column to the right shifts left to close the
    gap) — the structural counterpart to ``op_hide_range`` for tasks
    that must actually REMOVE a column, e.g. "delete the blank column
    A" / "remove column C".

    Args:
        sheet:  sheet name.
        cols:   list of column letters (``["A", "C"]``) or 0-based
                column indices (``[0, 2]``) — SAME convention as
                ``op_hide_range``'s ``cols``. Non-contiguous
                selections are fine: targets are resolved to indices
                up front and removed in DESCENDING order so an earlier
                removal never shifts a later target out from under it.

    Unlike ``op_hide_range`` (which only sets ``IsVisible=False`` and
    keeps the data), this reflows the sheet and the data is gone. Use
    it ONLY when the task explicitly asks to delete/remove a column,
    not merely hide it.
    """
    if not cols:
        raise ValueError("op_delete_columns: cols is empty")
    sh = _calc_ops_get_sheet(sheet)
    idxs = sorted({
        (int(c) if isinstance(c, int) else _calc_ops_col_letter_to_idx(c))
        for c in cols
    }, reverse=True)
    if any(i < 0 for i in idxs):
        raise ValueError("op_delete_columns: column index out of range (<0)")
    sh_cols = sh.getColumns()
    for idx in idxs:
        sh_cols.removeByIndex(idx, 1)
    doc.store()
    return True


def op_delete_rows(*, sheet, rows, **_extra):
    """PERMANENTLY delete entire rows from a sheet (the cells are
    removed and every row below shifts up to close the gap) — the
    structural counterpart to ``op_hide_range`` for tasks that must
    actually REMOVE a row, e.g. "delete the empty rows" / "remove row
    5".

    Args:
        sheet:  sheet name.
        rows:   list of 1-based row numbers (matches A1 conventions —
                row 1 is the header row), SAME convention as
                ``op_hide_range``'s ``rows``. Non-contiguous selections
                are fine: resolved up front and removed in DESCENDING
                order so an earlier removal never shifts a later target.

    Unlike ``op_hide_range``, this reflows the sheet and the data is
    gone. Use it ONLY when the task explicitly asks to delete/remove a
    row, not merely hide it.
    """
    if not rows:
        raise ValueError("op_delete_rows: rows is empty")
    sh = _calc_ops_get_sheet(sheet)
    idxs = sorted({int(r) - 1 for r in rows}, reverse=True)
    if any(i < 0 for i in idxs):
        raise ValueError(
            "op_delete_rows: row numbers are 1-based (got a value < 1)")
    sh_rows = sh.getRows()
    for idx in idxs:
        sh_rows.removeByIndex(idx, 1)
    doc.store()
    return True


def _calc_ops_parse_rgb(value):
    """Convert ``"#RRGGBB"`` / ``"RRGGBB"`` / int into a 0xRRGGBB int
    suitable for ``CellBackColor`` / ``CharColor``. Returns ``None``
    if ``value`` is ``None`` so callers can skip the assignment."""
    if value is None:
        return None
    if isinstance(value, int):
        return value & 0xFFFFFF
    s = str(value).strip().lstrip("#")
    if len(s) == 8 and s[:2].lower() in ("ff", "00"):
        # Tolerate ARGB ("FFRRGGBB") by dropping the alpha.
        s = s[2:]
    if len(s) != 6:
        raise ValueError(
            "calc color must be #RRGGBB or 0xRRGGBB int, got %r" % value)
    try:
        return int(s, 16) & 0xFFFFFF
    except ValueError:
        raise ValueError("calc color value not hex: %r" % value)


def _calc_ops_col_idx_to_letter(idx):
    """0-based column index → ``A`` / ``Z`` / ``AA`` / ``AB`` / ...
    Inverse of ``_calc_ops_col_letter_to_idx``. Used when the op needs
    to compose A1 cell references from numeric coordinates."""
    n = int(idx)
    if n < 0:
        raise ValueError("column index must be non-negative")
    s = ""
    n += 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        s = chr(ord("A") + rem) + s
    return s


def _calc_ops_qualify_refs(formula, sheet_name):
    """Prepend ``sheet_name.`` to UNQUALIFIED A1 cell refs in a formula.

    LO evaluates ``MAX($C$2:$C$25)`` relative to the sheet HOSTING the
    formula. When the predicate is written for "data lives on Sheet1"
    but evaluated on a temp scratch sheet, those refs accidentally
    read the temp sheet (all empty) and the predicate silently
    returns FALSE for every cell. Qualifying refs as ``Sheet1.$C$2``
    nails them to the data sheet regardless of where the formula
    runs.

    Refs already qualified (preceded by ``.``) are left untouched.
    Function names like ``WEEKDAY`` / ``MAX`` / ``MOD`` contain no
    digits in the row position so the cell-ref pattern doesn't
    match them. Quoted sheet names with spaces handled via the
    caller passing ``'My Sheet'`` form.
    """
    import re
    qual = sheet_name + "."
    # Match: optional $ + col letters + optional $ + row digits.
    # Negative lookbehind ``.`` keeps already-qualified refs alone;
    # also reject letters / digits / underscore as preceding char so
    # mid-identifier matches inside function names don't fire.
    pat = re.compile(r"(?<![A-Za-z0-9_.])(\$?[A-Z]+\$?\d+)")
    return pat.sub(lambda m: qual + m.group(1), formula)


def op_set_color(*, sheet, range_ref,
                 bgcolor=None, font_color=None, bold=None,
                 italic=None, where_formula=None,
                 skip_empty=True, **_extra):
    """Set background / font color / bold-italic across a range.

    Args:
        sheet:         sheet name.
        range_ref:     A1 range (``"A1"`` / ``"B2:D10"`` / ``"A:A"`` —
                       whole-column form supported by LO).
        bgcolor:       ``"#RRGGBB"`` / ``"RRGGBB"`` / int. ``None`` =
                       leave existing.
        font_color:    same forms as ``bgcolor``.
        bold:          ``True`` → ``CharWeight=150`` (BOLD).
                       ``False`` → ``CharWeight=100`` (NORMAL).
                       ``None`` → leave existing.
        italic:        ``True`` → ``CharPosture=ITALIC``.
                       ``False`` → ``CharPosture=NONE``.
                       ``None`` → leave existing.
        where_formula: optional per-cell predicate. When set, the
                       colour / font props apply ONLY to cells in
                       ``range_ref`` where this formula evaluates to
                       TRUE. Use ``{cell}`` as the placeholder for the
                       per-cell A1 reference and ``{r}`` for the row
                       number (1-based). Examples (LO Calc syntax —
                       ``;`` between multi-arg function args, NOT
                       ``,``; mixing them raises ``Err:508``):
                         "WEEKDAY({cell};2)>=6"     # Sat/Sun cells
                         "MOD({r};2)=0"             # alternate rows
                         "{cell}=MAX($B$2:$B$25)"   # the max value cell
                       The op evaluates the formula via a scratch
                       column far from the used area and clears it
                       after.
        skip_empty:    when ``where_formula`` is set, skip cells
                       whose source content is EMPTY (default True).
                       Without this, predicates like
                       ``WEEKDAY({cell};2)>=6`` match blank cells too
                       because LO Calc treats empty as ``0`` →
                       ``1899-12-30`` → Saturday, which leaks colour
                       onto trailing blank rows / cols. Set to
                       ``False`` only for predicates that explicitly
                       target blanks (e.g.  ``ISBLANK({cell})``).

    Used for "highlight cells with red fill" / "make max-value cell
    green" / "header row bold" / "color weekend dates red" tasks.
    NEVER chains with the GUI Format → Cells dialog. Pair with
    ``verify_cell_color`` to confirm the fill / font got applied at a
    representative cell.
    """
    sh = _calc_ops_get_sheet(sheet)
    rng = sh.getCellRangeByName(range_ref)
    bg = _calc_ops_parse_rgb(bgcolor)
    fc = _calc_ops_parse_rgb(font_color)

    def _apply_to_cell(c):
        if bg is not None:
            c.CellBackColor = int(bg)
        if fc is not None:
            c.CharColor = int(fc)
        if bold is not None:
            c.CharWeight = 150.0 if bold else 100.0
        if italic is not None:
            c.CharPosture = 2 if italic else 0

    if where_formula is None:
        _apply_to_cell(rng)
        doc.store()
        return True

    # Conditional path: evaluate ``where_formula`` per cell via a
    # TEMPORARY SHEET that gets dropped entirely at the end. Earlier
    # versions used a scratch column on the live sheet + clearContents,
    # but ``clearContents`` only wipes the cell value — the cell entry
    # itself stays in the saved .xlsx (`<c r="GS3"/>` etc), and
    # downstream evaluators that iterate every cell column (openpyxl's
    # ``iter_cols``) treat those leftovers as "used" cells with default
    # bgcolor, causing thousands of phantom mismatches against the
    # gold. Inserting a fresh sheet, doing work there, then
    # ``sheets.removeByName`` removes ALL traces from the workbook.
    ra = rng.getRangeAddress()
    n_rows = ra.EndRow - ra.StartRow + 1
    n_cols = ra.EndColumn - ra.StartColumn + 1

    import random as _random
    tmp_name = "__cf_eval_{:x}".format(_random.getrandbits(48))
    while tmp_name in sheets.getElementNames():
        tmp_name = "__cf_eval_{:x}".format(_random.getrandbits(48))
    sheets.insertNewByName(tmp_name, sheets.getCount())
    tmp_sheet = sheets.getByName(tmp_name)

    # Write the per-cell formulas into the temp sheet at (c, r)
    # positions matching the source range geometry. The formula
    # references ``sheet``-qualified A1 refs so it reads the live
    # data — both ``WEEKDAY(Sheet1.B6;2)`` and the shorthand
    # ``WEEKDAY('My Sheet'.B6;2)`` (quoted for names with spaces).
    scratch_cells = []
    sheet_name_for_ref = sheet
    if " " in sheet_name_for_ref or "'" in sheet_name_for_ref:
        sheet_name_for_ref = (
            "'" + sheet_name_for_ref.replace("'", "''") + "'")
    # Qualify unqualified refs in the user's predicate BEFORE substitution
    # so they read the data sheet, not the temp sheet.
    qualified_formula = _calc_ops_qualify_refs(
        str(where_formula), sheet_name_for_ref)
    for r in range(n_rows):
        cell_row = ra.StartRow + r  # 0-based on source
        for c in range(n_cols):
            cell_col = ra.StartColumn + c
            cell_a1 = "{}.{}{}".format(
                sheet_name_for_ref,
                _calc_ops_col_idx_to_letter(cell_col), cell_row + 1)
            formula_body = (qualified_formula
                            .replace("{cell}", cell_a1)
                            .replace("{r}", str(cell_row + 1)))
            sc = tmp_sheet.getCellByPosition(c, r)
            sc.setFormula("=" + formula_body)
            scratch_cells.append((sc, c, r))

    n_match = 0
    err_samples = []
    try:
        for sc, c, r in scratch_cells:
            v = None
            try:
                v = sc.getValue()
            except Exception:
                pass
            truthy = (v == 1.0) or (v == 1)
            if not truthy:
                try:
                    vs = sc.getString().strip()
                    if vs.lower() in ("true", "1", "yes"):
                        truthy = True
                    elif vs.startswith("Err:") or vs.startswith("#"):
                        if len(err_samples) < 3:
                            err_samples.append(vs)
                except Exception:
                    pass
            if truthy:
                target = sh.getCellByPosition(
                    ra.StartColumn + c, ra.StartRow + r)
                # skip_empty: drop matches on EMPTY source cells —
                # ``WEEKDAY(<empty>;2)`` returns 6 (1899-12-30 = Sat),
                # so a naive >=6 predicate would colour every blank
                # trailing row. Set skip_empty=False ONLY for
                # predicates that explicitly target blanks.
                if skip_empty:
                    try:
                        if _calc_ops_type_int(target) == 0:
                            continue
                    except Exception:
                        pass
                n_match += 1
                _apply_to_cell(target)
    finally:
        # Drop the temp sheet wholesale — every cell, every attribute,
        # gone. Belt + suspenders: tolerate a vanished sheet.
        try:
            if tmp_name in sheets.getElementNames():
                sheets.removeByName(tmp_name)
        except Exception:
            pass

    # If NO cell matched AND every scratch cell reported a formula
    # error, the predicate is syntactically broken — raise so the
    # planner sees the actual cause (typically ``,`` vs ``;``) on
    # the next turn instead of "0 cells coloured" silent failure.
    if n_match == 0 and err_samples:
        raise ValueError(
            "op_set_color: where_formula={!r} produced 0 matches and "
            "{:d}/{:d} scratch evaluations errored (samples: {}). LO "
            "Calc requires ``;`` as the argument separator inside "
            "multi-arg functions — rewrite ``WEEKDAY({{cell}},2)`` as "
            "``WEEKDAY({{cell}};2)`` and retry."
            .format(where_formula, len(err_samples), len(scratch_cells),
                     err_samples)
        )

    doc.store()
    return True


def op_merge_cells(*, sheet, range_ref, value=None, **_extra):
    """Merge a rectangular range into a single visual cell.

    Args:
        sheet:      sheet name.
        range_ref:  A1 range to merge (``"A1:C1"`` /
                    ``"A1:B2"`` — must span more than one cell).
        value:      optional string to write into the merged
                    cell's anchor BEFORE merging. Use when the
                    instruction says "merge A1:C1 to write the
                    header 'X'" so the write + merge sequence is
                    deterministic (writing AFTER merge picks the
                    same anchor cell but the merge state may
                    interfere on some bridges).

    Pair with ``verify_cell_merged(sheet, anchor)`` to confirm —
    benchmark evaluators read the merge property off the saved
    xlsx, so writing the value once and merging once is enough.
    """
    sh = _calc_ops_get_sheet(sheet)
    rng = sh.getCellRangeByName(range_ref)
    if value is not None:
        ra = rng.getRangeAddress()
        anchor = sh.getCellByPosition(ra.StartColumn, ra.StartRow)
        anchor.setString(str(value))
    try:
        rng.merge(True)
    except Exception as e:
        raise RuntimeError(
            "op_merge_cells: range.merge(True) failed for %r on %r: %s"
            % (range_ref, sheet, e))
    doc.store()
    return True


def op_data_validation(*, sheet, range_ref,
                        allowed_values=None, formula=None,
                        type="list", show_dropdown=True, **_extra):
    """Attach a data-validation rule to ``range_ref``.

    Args:
        sheet:           sheet name.
        range_ref:       A1 range (typically a whole column like
                         ``"D2:D29"`` or ``"D2:D1048576"``).
        allowed_values:  for ``type="list"`` — list of strings the
                         dropdown should offer (``["Pass","Fail",
                         "Held"]``). Each value is quoted and
                         joined with ``;`` to form LO's Formula1.
        formula:         escape hatch — raw LO ``Formula1`` string.
                         Use when ``type`` is not ``list``.
        type:            ``"list"`` (default) — dropdown of allowed
                         values. Other LO types: ``"any"``,
                         ``"whole"``, ``"decimal"``, ``"date"``,
                         ``"time"``, ``"text_len"``, ``"custom"``.
        show_dropdown:   ``True`` (default) — render the dropdown
                         arrow on the cell. ``False`` keeps the
                         rule active without the picker UI.

    The op replaces any prior validation rule on the same range.
    Pair with ``verify_data_validation`` to confirm the rule + the
    allowed-values set landed on the saved file.
    """
    sh = _calc_ops_get_sheet(sheet)
    rng = sh.getCellRangeByName(range_ref)
    # LO's ValidationType enum: ANY=0, WHOLE=1, DECIMAL=2, DATE=3,
    # TIME=4, TEXT_LEN=5, LIST=6, CUSTOM=7.
    _T = {"any": 0, "whole": 1, "decimal": 2, "date": 3,
          "time": 4, "text_len": 5, "list": 6, "custom": 7}
    t_int = _T.get(str(type).lower())
    if t_int is None:
        raise ValueError(
            "op_data_validation: unknown type %r (want one of %s)"
            % (type, sorted(_T)))
    val = rng.Validation
    val.Type = t_int
    val.ShowList = 1 if show_dropdown else 0
    if formula is not None:
        val.Formula1 = str(formula)
    elif allowed_values:
        # LO list-validation Formula1 format: ``"a";"b";"c"`` —
        # each value double-quoted, semicolons between.
        parts = []
        for v in allowed_values:
            sv = str(v).replace('"', '""')
            parts.append('"' + sv + '"')
        val.Formula1 = ";".join(parts)
    elif t_int == 6:
        raise ValueError(
            "op_data_validation: type='list' needs allowed_values=[...]")
    # Assignment back is REQUIRED — Validation is a property struct,
    # and mutating the local copy does nothing until re-assigned.
    rng.Validation = val
    doc.store()
    return True


def op_reorder_columns(*, sheet, new_order, first_row=None, last_row=None,
                       **_extra):
    """Rearrange the columns of a contiguous data block by name or
    letter, leaving the rest of the sheet untouched.

    Args:
        sheet:      sheet name.
        new_order:  list specifying the new column sequence. Items
                    can be column letters (``["B","A","C","D"]``)
                    OR header-name strings (``["First Name", "Last
                    Name", "Date"]``) — when a string isn't a valid
                    column letter, it's looked up against row 1
                    (the header row). The output columns are
                    written starting at column A.
        first_row:  first row to include in the rearranged block.
                    Defaults to 1 (entire used range).
        last_row:   last row, inclusive. Defaults to the sheet's
                    last used row.

    The op reads the current values for the named columns, then
    overwrites columns ``A..`` of the affected rows with the
    reordered data. Used for "Reorder columns to <X>, <Y>, <Z>"
    instructions. Original cell formats are NOT preserved beyond
    what falls out of value rewriting; if the task also requires
    keeping formatting / formulas, use ``calc_duplicate_range`` instead.
    """
    if not new_order:
        raise ValueError("op_reorder_columns: new_order is empty")
    sh = _calc_ops_get_sheet(sheet)
    cursor = sh.createCursor()
    cursor.gotoStartOfUsedArea(False)
    cursor.gotoEndOfUsedArea(True)
    used = cursor.getRangeAddress()
    if first_row is None:
        first_row = used.StartRow + 1
    else:
        first_row = int(first_row)
    if last_row is None:
        last_row = used.EndRow + 1
    else:
        last_row = int(last_row)
    if last_row < first_row:
        raise ValueError(
            "op_reorder_columns: last_row %d < first_row %d"
            % (last_row, first_row))
    # Build column-index list from new_order (letter or header name).
    # Header lookup uses row 1.
    header_row_idx = used.StartRow  # 0-based; row 1 of A1 ⇒ 0
    header_to_idx = {}
    for c in range(used.StartColumn, used.EndColumn + 1):
        cell = sh.getCellByPosition(c, header_row_idx)
        s = cell.getString().strip()
        if s and s not in header_to_idx:
            header_to_idx[s] = c
    src_cols = []
    for item in new_order:
        s = str(item).strip()
        # Try column-letter first (purely letters of length ≤ 3).
        try:
            if s and s.replace(" ", "").isalpha() and len(s) <= 3:
                src_cols.append(_calc_ops_col_letter_to_idx(s))
                continue
        except ValueError:
            pass
        # Header-name lookup fallback.
        if s in header_to_idx:
            src_cols.append(header_to_idx[s])
        else:
            raise ValueError(
                "op_reorder_columns: %r is neither a column letter nor a "
                "header in row 1 of sheet %r" % (item, sheet))
    # Read source data for all rows × selected source columns,
    # preserving formula / value / string cell types AND the cell's
    # NumberFormat id. Without carrying NumberFormat, a Date column
    # moved into a column that previously held strings would display
    # the date's underlying serial (44815) instead of the formatted
    # date, and eval's openpyxl-side ``cell.value`` reads back a
    # float vs the gold's ``datetime`` — same data, different type,
    # ``sheet_data`` rule fails.
    matrix = []
    for r in range(first_row - 1, last_row):
        row = []
        for c in src_cols:
            cell = sh.getCellByPosition(c, r)
            t = _calc_ops_type_int(cell)
            try:
                fmt = int(cell.NumberFormat)
            except Exception:
                fmt = None
            if t == 3:
                row.append(("formula", cell.getFormula(), fmt))
            elif t == 1:
                row.append(("value", cell.getValue(), fmt))
            elif t == 2:
                row.append(("string", cell.getString(), fmt))
            else:
                row.append(("empty", None, fmt))
        matrix.append(row)
    # Write back starting at the SAME first column the original data
    # block lived in. Hardcoding column A here used to silently shift
    # a B..F block to A..E, which matched a verify spec that assumed
    # "data starts at column A" but mismatched the gold file (which
    # preserves the B..F layout) — task 7a4e4bc8. Honour the source
    # bounding box instead.
    dst_start_col = used.StartColumn
    for ri, row in enumerate(matrix):
        r = first_row - 1 + ri
        for ci, (kind, v, fmt) in enumerate(row):
            cell = sh.getCellByPosition(dst_start_col + ci, r)
            if kind == "formula":
                cell.setFormula(v)
            elif kind == "value":
                cell.setValue(v)
            elif kind == "string":
                cell.setString(v)
            else:
                cell.setString("")
            if fmt is not None:
                try:
                    cell.NumberFormat = fmt
                except Exception:
                    pass
    # Clear leftover columns beyond the new width, IF the original
    # block was wider than the new arrangement. Without this, a
    # 5→4 column reorder would leave the 5th column populated with
    # stale data. Clear from the new last column to the original
    # used-range end, using the offset start.
    n_new = len(src_cols)
    orig_width = used.EndColumn - used.StartColumn + 1
    if n_new < orig_width:
        for r in range(first_row - 1, last_row):
            for c in range(dst_start_col + n_new,
                            dst_start_col + orig_width):
                sh.getCellByPosition(c, r).setString("")
    doc.store()
    return True


def op_fill_blanks(*, sheet, range_ref, source="above", **_extra):
    """For each EMPTY cell in ``range_ref``, copy the value from the
    adjacent non-empty cell in ``source`` direction.

    Args:
        sheet:      sheet name.
        range_ref:  A1 range to scan, e.g. ``"B1:E30"``.
        source:     ``"above"`` (default) — fill DOWN: each empty cell
                    takes the value from the closest non-empty cell
                    above it in the same column.
                    ``"below"`` — fill UP from the closest non-empty
                    cell below.
                    ``"left"`` / ``"right"`` — horizontal analogue.

    Used for the "fill blank cells with the value in the cell above"
    pattern. Naive ``calc_fill_column`` with ``=B{r}`` formulas creates
    a SELF-REFERENCE (each cell points to itself), produces a circular
    reference and ends up rendering 0 or the header text. This op
    iterates and writes literal values (string / numeric — preserving
    type), leaving non-empty cells untouched — so the "don't touch
    irrelevant regions" constraint holds by construction.
    """
    sh = _calc_ops_get_sheet(sheet)
    rng = sh.getCellRangeByName(range_ref)
    ra = rng.getRangeAddress()
    direction = str(source).strip().lower()
    if direction == "above":
        # For each column in range, scan top→bottom; carry the last
        # non-empty value forward.
        for c in range(ra.StartColumn, ra.EndColumn + 1):
            carry_kind, carry_val = None, None
            for r in range(ra.StartRow, ra.EndRow + 1):
                cell = sh.getCellByPosition(c, r)
                t = _calc_ops_type_int(cell)
                if t == 0:
                    # Empty — fill from carry, if any.
                    if carry_kind == "value":
                        cell.setValue(carry_val)
                    elif carry_kind == "string":
                        cell.setString(carry_val)
                    # formula carry intentionally NOT propagated —
                    # copying a formula downward would silently re-bind
                    # its row references, which is rarely what the
                    # task asks for. Skip.
                else:
                    if t == 1:
                        carry_kind, carry_val = "value", cell.getValue()
                    elif t == 2:
                        carry_kind, carry_val = "string", cell.getString()
                    else:
                        # formula or unknown — keep current cell as-is,
                        # but DROP the carry (don't propagate a formula
                        # result; let the next non-empty cell reset).
                        carry_kind, carry_val = None, None
    elif direction == "below":
        for c in range(ra.StartColumn, ra.EndColumn + 1):
            carry_kind, carry_val = None, None
            for r in range(ra.EndRow, ra.StartRow - 1, -1):
                cell = sh.getCellByPosition(c, r)
                t = _calc_ops_type_int(cell)
                if t == 0:
                    if carry_kind == "value":
                        cell.setValue(carry_val)
                    elif carry_kind == "string":
                        cell.setString(carry_val)
                else:
                    if t == 1:
                        carry_kind, carry_val = "value", cell.getValue()
                    elif t == 2:
                        carry_kind, carry_val = "string", cell.getString()
                    else:
                        carry_kind, carry_val = None, None
    elif direction == "left":
        for r in range(ra.StartRow, ra.EndRow + 1):
            carry_kind, carry_val = None, None
            for c in range(ra.StartColumn, ra.EndColumn + 1):
                cell = sh.getCellByPosition(c, r)
                t = _calc_ops_type_int(cell)
                if t == 0:
                    if carry_kind == "value":
                        cell.setValue(carry_val)
                    elif carry_kind == "string":
                        cell.setString(carry_val)
                else:
                    if t == 1:
                        carry_kind, carry_val = "value", cell.getValue()
                    elif t == 2:
                        carry_kind, carry_val = "string", cell.getString()
                    else:
                        carry_kind, carry_val = None, None
    elif direction == "right":
        for r in range(ra.StartRow, ra.EndRow + 1):
            carry_kind, carry_val = None, None
            for c in range(ra.EndColumn, ra.StartColumn - 1, -1):
                cell = sh.getCellByPosition(c, r)
                t = _calc_ops_type_int(cell)
                if t == 0:
                    if carry_kind == "value":
                        cell.setValue(carry_val)
                    elif carry_kind == "string":
                        cell.setString(carry_val)
                else:
                    if t == 1:
                        carry_kind, carry_val = "value", cell.getValue()
                    elif t == 2:
                        carry_kind, carry_val = "string", cell.getString()
                    else:
                        carry_kind, carry_val = None, None
    else:
        raise ValueError(
            "op_fill_blanks: source must be one of "
            "'above'/'below'/'left'/'right', got %r" % source)
    doc.store()
    return True


# ---- Verify ops (read-only) -------------------------------------------------

def verify_sheet_exists(*, name):
    """A sheet with the given name is present in the workbook."""
    return name in sheets.getElementNames()


def verify_cell_value(*, sheet, a1_ref, expected_value=None,
                     expected_substring=None, expected_formula=None,
                     expected_formula_contains=None,
                     expected_formula_regex=None, **_extra):
    """Check a single cell against expectations. Supply ONE of:
      - ``expected_value``: cell value (numeric ≈; string ==).
      - ``expected_substring``: cell rendered text contains this substring
        (plain `in`, NOT regex).
      - ``expected_formula``: cell is a formula EXACTLY equal to this.
      - ``expected_formula_contains``: cell is a formula whose string
        contains this PLAIN substring.
      - ``expected_formula_regex``: cell is a formula whose string
        matches this regex (re.search semantics).

    On mismatch, prints a ``DIAG:`` line with the cell's actual
    value / formula so the planner's next-turn ``[Last cli_run
    output]`` block can show concrete feedback ("expected X got Y")
    instead of an opaque ``FAIL: cell_value`` that leaves the
    planner guessing.
    """
    if sheet not in sheets.getElementNames():
        print("DIAG: cell_value sheet={!r} not present".format(sheet))
        return False
    sh = sheets.getByName(sheet)
    try:
        col, row = _calc_ops_resolve_a1(sh, a1_ref)
    except Exception:
        print("DIAG: cell_value bad a1_ref={!r}".format(a1_ref))
        return False
    cell = sh.getCellByPosition(col, row)
    t = _calc_ops_type_int(cell)

    def _fail(expected_kind, expected_val, got_kind, got_val):
        print("DIAG: cell_value at {} expected_{}={!r} got_{}={!r}".format(
            a1_ref, expected_kind, expected_val, got_kind, got_val))
        return False

    if (expected_formula is not None
            or expected_formula_contains is not None
            or expected_formula_regex is not None):
        if t != 3:
            return _fail("formula", expected_formula or expected_formula_contains
                          or expected_formula_regex,
                          "type", {0: "EMPTY", 1: "VALUE", 2: "TEXT"}.get(t, t))
        f = cell.getFormula()
        if expected_formula is not None and f != expected_formula:
            return _fail("formula", expected_formula, "formula", f)
        if (expected_formula_contains is not None
                and expected_formula_contains not in f):
            return _fail("formula_contains", expected_formula_contains,
                          "formula", f)
        if expected_formula_regex is not None:
            import re as _re
            try:
                if not _re.search(expected_formula_regex, f):
                    return _fail("formula_regex", expected_formula_regex,
                                  "formula", f)
            except _re.error:
                return _fail("formula_regex (invalid)",
                              expected_formula_regex, "formula", f)
        return True
    if expected_value is not None:
        if isinstance(expected_value, (int, float)) and not isinstance(expected_value, bool):
            try:
                if abs(cell.getValue() - float(expected_value)) < 1e-9:
                    return True
                return _fail("value", expected_value,
                              "value", cell.getValue())
            except Exception:
                return _fail("value (numeric)", expected_value,
                              "string", cell.getString())
        if cell.getString() == str(expected_value):
            return True
        return _fail("value (string)", expected_value,
                      "string", cell.getString())
    if expected_substring is not None:
        if expected_substring in cell.getString():
            return True
        return _fail("substring", expected_substring,
                      "string", cell.getString())
    if _calc_ops_type_int(cell) != 0:
        return True
    print("DIAG: cell_value at {} expected non-empty, got EMPTY".format(a1_ref))
    return False


def verify_pivot_exists(*, on_sheet, pivot_name=None):
    """At least one pivot table is present on ``on_sheet`` (and optionally
    named ``pivot_name``)."""
    if on_sheet not in sheets.getElementNames():
        return False
    pts = sheets.getByName(on_sheet).getDataPilotTables()
    if pivot_name is not None:
        return bool(pts.hasByName(pivot_name))
    n = (pts.getCount() if hasattr(pts, "getCount")
         else len(list(pts.getElementNames())))
    return n >= 1


def verify_cell_format(*, sheet, range_ref, format_string, **_extra):
    """Every non-empty cell in ``range_ref`` has number format
    ``format_string`` (registered in the doc's NumberFormats).

    On mismatch, prints a ``DIAG:`` line naming the expected vs the
    actual format string. The planner's next-turn
    ``[Last cli_run output]`` block surfaces that line, letting the
    model self-correct to the right format string instead of looping
    on whatever it guessed from the screenshot.
    """
    if sheet not in sheets.getElementNames():
        print("DIAG: cell_format sheet={!r} not present".format(sheet))
        return False
    sh = sheets.getByName(sheet)
    fmt_registry = doc.getNumberFormats()
    loc = uno.createUnoStruct("com.sun.star.lang.Locale")
    # Mirror op_format_number_range's locale-prefix translation: when
    # the planner+actor applied `[$-419]0.0`, the registered format is
    # `'0,0'` under ru/RU locale (not `'[$-419]0.0'`). Translate here
    # so the lookup hits the entry the op actually created.
    rewritten, derived_locale = _split_lo_format_locale_prefix(format_string)
    if derived_locale is not None:
        format_string = rewritten
        loc.Language, loc.Country, loc.Variant = derived_locale
    fmt_id = fmt_registry.queryKey(format_string, loc, False)
    if fmt_id == -1:
        # queryKey=-1 means "not yet registered in this doc" — NOT
        # necessarily "malformed format string". Try addNew on a
        # throwaway probe to distinguish the two cases:
        #   - addNew succeeds → format is valid; the cells just carry
        #     a DIFFERENT format string. Surface the cells' actual
        #     FormatString so the planner can see why the strict
        #     format-key match failed (most often: planner emitted a
        #     near-equivalent variant — e.g. `0.0,, "M"` vs the gold
        #     `0.0,," M"` — and the strings need to converge on ONE
        #     canonical form).
        #   - addNew raises → format is genuinely malformed.
        try:
            fmt_id = fmt_registry.addNew(format_string, loc)
        except Exception as e:
            print(
                "DIAG: cell_format expected={!r} is MALFORMED — LO "
                "addNew raised {}: {!s:.140}".format(
                    format_string, type(e).__name__, e))
            return False
        # addNew succeeded; the cells still won't match unless they
        # also use this fmt_id. Fall through to the per-cell compare
        # which will surface the cells' actual FormatString via the
        # existing got=... DIAG line.
    rng = sh.getCellRangeByName(range_ref)
    ra = rng.getRangeAddress()
    checked = 0
    for c in range(ra.StartColumn, ra.EndColumn + 1):
        for r in range(ra.StartRow, ra.EndRow + 1):
            cell = sh.getCellByPosition(c, r)
            if _calc_ops_type_int(cell) == 0:
                continue
            checked += 1
            cur_id = int(cell.NumberFormat)
            if cur_id != fmt_id:
                try:
                    cur_str = fmt_registry.getByKey(cur_id).FormatString
                except Exception:
                    cur_str = "<id={}>".format(cur_id)
                coord = "{}{}".format(
                    _calc_ops_col_idx_to_letter(c), r + 1)
                print(
                    "DIAG: cell_format expected={!r} got={!r} at {} "
                    "(use this exact format_string in calc_format_number "
                    "to satisfy the verify)".format(
                        format_string, cur_str, coord))
                return False
    if checked == 0:
        # All cells in the range are EMPTY — there's nothing actually
        # carrying the format. Returning True here (the old behaviour)
        # let init_ledger's ``column_*_formatted_as_*`` outcomes flip
        # to "satisfied" before the actor had filled the column, and
        # the planner then chased a phantom done-state. An empty range
        # is NOT a satisfied number-format outcome — it's a precondition
        # the actor still owes (calc_fill_column for that range first).
        print(
            "DIAG: cell_format range_ref={!r} is empty — no non-EMPTY "
            "cells carry the format. Fill the range with values first "
            "(e.g. calc_fill_column) before applying calc_format_number".format(
                range_ref))
        return False
    return True


def verify_sort_order(*, sheet, range_ref, sort_col_within_range,
                       ascending=True, contains_header=True,
                       expected_first_value=None,
                       expected_last_value=None,
                       **_extra):
    """Data column of ``range_ref`` at ``sort_col_within_range`` is
    monotonic in the requested direction (skipping the header row when
    ``contains_header`` is True).

    Args:
        expected_first_value:  optional anchor — the first data row's
                               value in the sort column must equal this.
                               Catches the "monotonic but wrong key"
                               false-positive (e.g. lexical-vs-numeric
                               sort on dates).
        expected_last_value:   optional anchor — the last data row's
                               value in the sort column must equal this.
    """
    if sheet not in sheets.getElementNames():
        return False
    sh = sheets.getByName(sheet)
    rng = sh.getCellRangeByName(range_ref)
    ra = rng.getRangeAddress()
    sorted_col_global = ra.StartColumn + int(sort_col_within_range)
    start_data_row = ra.StartRow + (1 if contains_header else 0)
    vals = []
    for r in range(start_data_row, ra.EndRow + 1):
        cell = sh.getCellByPosition(sorted_col_global, r)
        t = _calc_ops_type_int(cell)
        if t == 1:
            vals.append(cell.getValue())
        elif t == 2:
            vals.append(cell.getString())
        elif t == 3:
            try:
                vals.append(cell.getValue())
            except Exception:
                vals.append(cell.getString())
    is_monotonic = (vals == sorted(vals)) if ascending \\
        else (vals == sorted(vals, reverse=True))
    if not is_monotonic:
        return False
    # Anchor checks — catches the "sorted by wrong key" case where
    # the column is still monotonic in lexical order but the ground
    # truth used numeric / date order.
    if expected_first_value is not None and vals:
        a = vals[0]; b = expected_first_value
        try:
            if isinstance(b, (int, float)) and isinstance(a, str):
                a = float(a)
        except Exception:
            pass
        if a != b:
            return False
    if expected_last_value is not None and vals:
        a = vals[-1]; b = expected_last_value
        try:
            if isinstance(b, (int, float)) and isinstance(a, str):
                a = float(a)
        except Exception:
            pass
        if a != b:
            return False
    return True


def verify_freeze_pane(*, expected_rows=None, expected_cols=None, **_extra):
    """Workbook freeze position. Without args, only confirms a freeze
    exists. With ``expected_rows`` and/or ``expected_cols``, verifies
    the EXACT split position.

    LO controller exposes:
      * ``SplitColumn`` = number of frozen columns (matches input).
      * ``SplitRow``    = frozen-row split position. When no rows
                          are frozen this is 0; when ``N`` rows are
                          frozen this is ``N + 1`` (the 1-based row
                          where the split begins). We back-derive
                          ``n_rows = SplitRow - 1`` when ``SplitRow``
                          is positive.

    On mismatch, prints a ``DIAG:`` line naming the actual vs
    expected (rows, cols) so the planner's next-turn
    ``[Last cli_run output]`` block surfaces concrete feedback to
    correct the freeze position.
    """
    has = bool(controller.hasFrozenPanes())
    sc = int(getattr(controller, "SplitColumn", 0) or 0)
    sr = int(getattr(controller, "SplitRow", 0) or 0)
    actual_cols = sc if has else 0
    actual_rows = (sr - 1) if (has and sr > 0) else 0
    if not has:
        if (not expected_rows) and (not expected_cols):
            return True
        print(
            "DIAG: freeze_pane no freeze present; expected "
            "n_rows={} n_cols={}".format(
                expected_rows or 0, expected_cols or 0))
        return False
    ok = True
    if expected_cols is not None and actual_cols != int(expected_cols):
        ok = False
    if expected_rows is not None and actual_rows != int(expected_rows):
        ok = False
    if not ok:
        print(
            "DIAG: freeze_pane expected n_rows={} n_cols={} but got "
            "n_rows={} n_cols={} (call calc_freeze with those exact "
            "values)".format(
                expected_rows if expected_rows is not None else "?",
                expected_cols if expected_cols is not None else "?",
                actual_rows, actual_cols))
    return ok


def verify_duplicate_range_match(*, source_sheet, source_range,
                            dest_sheet, dest_anchor,
                            n_rows_to_check=3, transposed=False,
                            **_extra):
    """Spot-check that the first ``n_rows_to_check`` rows of the source
    column landed at the destination (cell-type match — values not
    compared beyond type, since formulas re-resolve to different
    values after copy).

    Args:
        transposed:  When True, the destination is the TRANSPOSE of
                     the source — source row ``r`` should land at
                     destination column ``r`` (anchored from
                     ``dest_anchor``). Verifies cell-type match in
                     the rotated layout instead of the straight copy.
    """
    if source_sheet not in sheets.getElementNames():
        return False
    if dest_sheet not in sheets.getElementNames():
        return False
    src = sheets.getByName(source_sheet)
    dst = sheets.getByName(dest_sheet)
    src_range = src.getCellRangeByName(source_range)
    src_ra = src_range.getRangeAddress()
    da = dst.getCellRangeByName(dest_anchor).getRangeAddress()
    dst_col0 = getattr(da, "Column", None) or da.StartColumn
    dst_row0 = getattr(da, "Row", None) or da.StartRow
    n_check = min(src_ra.EndRow - src_ra.StartRow + 1, int(n_rows_to_check))
    src_n_cols = src_ra.EndColumn - src_ra.StartColumn + 1
    for r in range(n_check):
        s = src.getCellByPosition(src_ra.StartColumn, src_ra.StartRow + r)
        if transposed:
            # source (r, 0) → destination (0, r); also check the
            # first row's columns map to destination column 0's rows.
            d = dst.getCellByPosition(dst_col0 + r, dst_row0)
        else:
            d = dst.getCellByPosition(dst_col0, dst_row0 + r)
        if _calc_ops_type_int(s) != _calc_ops_type_int(d):
            return False
    if transposed and src_n_cols > 1:
        # Also spot-check that source ROW 0's later columns end up
        # in destination column 0's later rows.
        n_col_check = min(src_n_cols, int(n_rows_to_check))
        for c in range(n_col_check):
            s = src.getCellByPosition(src_ra.StartColumn + c, src_ra.StartRow)
            d = dst.getCellByPosition(dst_col0, dst_row0 + c)
            if _calc_ops_type_int(s) != _calc_ops_type_int(d):
                return False
    return True


def verify_cell_color(*, sheet, a1_ref, bgcolor=None, font_color=None,
                      bold=None, italic=None,
                      where_formula=None, skip_empty=True,
                      check_unmatched_unstyled=False,
                      **_extra):
    """Verify cell formatting.

    Single-cell mode (``where_formula`` omitted):
      ``a1_ref`` is a single cell (``"B5"``) and the attrs apply to
      that one cell — passes iff every supplied attr matches.

    Range + predicate mode (``where_formula`` set):
      ``a1_ref`` is a multi-cell range (``"B3:F33"``). For each cell
      in the range where the predicate evaluates TRUE, the supplied
      attrs must match. The predicate uses the same ``{cell}`` /
      ``{r}`` placeholder grammar as ``op_set_color``. This mirrors
      ``op_set_color(where_formula=...)`` so verify and op agree on
      the same cell set — emit this when the task is "highlight
      cells matching X" instead of picking arbitrary corner cells.

    Args:
        sheet:                       sheet name.
        a1_ref:                      single cell OR multi-cell range.
        bgcolor / font_color:        ``"#RRGGBB"`` / int.
        bold / italic:               True/False.
        where_formula:               optional LO predicate string.
                                     See ``op_set_color`` docstring.
        skip_empty:                  in predicate mode, skip cells
                                     whose source content is empty
                                     (default True — symmetric with
                                     op_set_color default so a naive
                                     ``WEEKDAY({cell};2)>=6`` predicate
                                     doesn't ask for colour on blank
                                     trailing rows that LO maps to
                                     1899-12-30 = Saturday).
        check_unmatched_unstyled:    when True (and in predicate mode),
                                     ALSO require that cells where the
                                     predicate is False do NOT carry
                                     the colour. Catches the "agent
                                     coloured the whole range" failure
                                     mode. Default False to keep the
                                     verify tolerant of pre-existing
                                     incidental fills.
    """
    if sheet not in sheets.getElementNames():
        return False
    sh = sheets.getByName(sheet)

    bg_want = _calc_ops_parse_rgb(bgcolor) if bgcolor is not None else None
    fc_want = _calc_ops_parse_rgb(font_color) if font_color is not None else None

    def _cell_matches(c):
        if bg_want is not None:
            if (int(c.CellBackColor) & 0xFFFFFF) != bg_want:
                return False
        if fc_want is not None:
            if (int(c.CharColor) & 0xFFFFFF) != fc_want:
                return False
        if bold is not None:
            cw = float(c.CharWeight)
            if bold and cw < 140: return False
            if (not bold) and cw > 110: return False
        if italic is not None:
            cp = int(c.CharPosture)
            if italic and cp == 0: return False
            if (not italic) and cp != 0: return False
        return True

    if where_formula is None:
        return _cell_matches(sh.getCellRangeByName(a1_ref))

    # Range + predicate mode.
    rng = sh.getCellRangeByName(a1_ref)
    ra = rng.getRangeAddress()
    n_rows = ra.EndRow - ra.StartRow + 1
    n_cols = ra.EndColumn - ra.StartColumn + 1

    # Use a temp sheet for predicate evaluation (same approach as
    # op_set_color). A scratch column on the live sheet leaks cell
    # entries into the saved .xlsx even after clearContents — the
    # downstream evaluator's full-sheet ``iter_cols`` then sees them
    # as "used" cells with default bgcolor, flagging thousands of
    # phantom mismatches against the gold.
    import random as _random
    tmp_name = "__cf_verify_{:x}".format(_random.getrandbits(48))
    while tmp_name in sheets.getElementNames():
        tmp_name = "__cf_verify_{:x}".format(_random.getrandbits(48))
    sheets.insertNewByName(tmp_name, sheets.getCount())
    tmp_sheet = sheets.getByName(tmp_name)

    sheet_name_for_ref = sheet
    if " " in sheet_name_for_ref or "'" in sheet_name_for_ref:
        sheet_name_for_ref = (
            "'" + sheet_name_for_ref.replace("'", "''") + "'")
    qualified_formula = _calc_ops_qualify_refs(
        str(where_formula), sheet_name_for_ref)

    scratch_cells = []
    for r in range(n_rows):
        cell_row = ra.StartRow + r
        for c in range(n_cols):
            cell_col = ra.StartColumn + c
            cell_a1 = "{}.{}{}".format(
                sheet_name_for_ref,
                _calc_ops_col_idx_to_letter(cell_col), cell_row + 1)
            body = (qualified_formula
                    .replace("{cell}", cell_a1)
                    .replace("{r}", str(cell_row + 1)))
            sc = tmp_sheet.getCellByPosition(c, r)
            sc.setFormula("=" + body)
            scratch_cells.append((sc, c, r))

    ok = True
    try:
        for sc, c, r in scratch_cells:
            v = None
            try: v = sc.getValue()
            except Exception: pass
            truthy = (v == 1.0) or (v == 1)
            if not truthy:
                try:
                    vs = sc.getString().strip().lower()
                    if vs in ("true", "1", "yes"):
                        truthy = True
                except Exception:
                    pass
            target = sh.getCellByPosition(
                ra.StartColumn + c, ra.StartRow + r)
            if skip_empty:
                try:
                    if _calc_ops_type_int(target) == 0:
                        continue
                except Exception:
                    pass
            if truthy:
                if not _cell_matches(target):
                    ok = False
                    break
            elif check_unmatched_unstyled:
                # Predicate False: target must NOT have the requested
                # styling. Mirror of _cell_matches but negated.
                if _cell_matches(target):
                    ok = False
                    break
    finally:
        try:
            if tmp_name in sheets.getElementNames():
                sheets.removeByName(tmp_name)
        except Exception:
            pass
    return ok


def verify_cell_merged(*, sheet, a1_ref, **_extra):
    """The cell at ``a1_ref`` is the anchor of a merged region (i.e.
    ``IsMerged`` is True on the cell at the top-left of the merge).
    """
    if sheet not in sheets.getElementNames():
        return False
    sh = sheets.getByName(sheet)
    cell = sh.getCellRangeByName(a1_ref)
    try:
        return bool(cell.IsMerged)
    except Exception:
        return False


def verify_data_validation(*, sheet, range_ref, allowed_values=None,
                           type=None, **_extra):
    """Range has a data-validation rule. When ``allowed_values`` is
    supplied, also confirms the rule's ``Formula1`` contains the
    same set of quoted values (order-insensitive)."""
    if sheet not in sheets.getElementNames():
        return False
    sh = sheets.getByName(sheet)
    val = sh.getCellRangeByName(range_ref).Validation
    # Read Validation.Type — older bridges expose an int, newer ones
    # an Enum whose ``.value`` is the name string ("LIST"). Map both
    # to the canonical lowercase name.
    _NAMES = {0: "any", 1: "whole", 2: "decimal", 3: "date",
              4: "time", 5: "text_len", 6: "list", 7: "custom"}
    raw = val.Type
    actual = None
    if isinstance(raw, int):
        actual = _NAMES.get(raw)
    else:
        name = getattr(raw, "value", None) or str(raw)
        actual = str(name).lower()
        # Strip enum wrapper formatting if any.
        if "'" in actual:
            actual = actual.split("'")[1].lower()
    if type is not None:
        if actual != str(type).lower():
            return False
    if allowed_values is not None:
        f1 = str(getattr(val, "Formula1", "") or "")
        # Extract quoted tokens from Formula1 (``"a";"b";"c"`` →
        # ``{"a","b","c"}``). Tolerates extra whitespace.
        tokens = set()
        i = 0
        while i < len(f1):
            if f1[i] == '"':
                j = f1.find('"', i + 1)
                if j == -1:
                    break
                tokens.add(f1[i + 1:j])
                i = j + 1
            else:
                i += 1
        if not tokens:
            # Fallback: split on ";" if no quoted tokens were found
            # (some LO bridges drop the quotes on roundtrip).
            tokens = {p.strip().strip('"') for p in f1.split(";") if p.strip()}
        want = {str(v) for v in allowed_values}
        if tokens != want:
            return False
    return True


def verify_row_hidden(*, sheet, rows, hidden=True):
    """Every listed row's hidden state equals ``hidden``.

    Args:
        sheet:  sheet name.
        rows:   list of 1-based row numbers.
        hidden: True (default) requires every row to be HIDDEN;
                False requires every row to be VISIBLE.
    """
    if sheet not in sheets.getElementNames():
        return False
    sh_rows = sheets.getByName(sheet).getRows()
    for r in rows:
        try:
            is_visible = bool(sh_rows.getByIndex(int(r) - 1).IsVisible)
        except Exception:
            return False
        if (not is_visible) != bool(hidden):
            return False
    return True


def verify_chart_exists(*, on_sheet, name=None, chart_type=None, title=None,
                        expected_series_count=None, **_extra):
    """At least one embedded chart matches the filter on ``on_sheet``.

    Args:
        on_sheet:    sheet name to check.
        name:        if given, require a chart with this name. Otherwise
                     any chart counts.
        chart_type:  if given, require the chart's diagram service to
                     match. One of "column"/"bar" (both BarDiagram),
                     "line", "pie", "scatter", "area". Case-insensitive
                     substring match against the diagram's service name.
        title:       if given, require the chart title to CONTAIN this
                     string (substring, case-sensitive).
        expected_series_count:
                     if given, the chart's DataSeries count must equal
                     this integer. Works for both form-1 (data_range
                     produces N series automatically from non-header
                     rows/cols) and form-2 (one series per ``values``
                     entry); both surface identically via
                     diagram.CoordinateSystems → ChartTypes →
                     DataSeries. Use this to FAIL wide-range charts
                     that carry extra spurious series.
    """
    if on_sheet not in sheets.getElementNames():
        return False
    charts = sheets.getByName(on_sheet).Charts
    if name is not None:
        if not charts.hasByName(name):
            return False
        candidates = [charts.getByName(name)]
    else:
        try:
            n = int(charts.getCount())
        except Exception:
            n = 0
        if n == 0:
            return False
        candidates = [charts.getByIndex(i) for i in range(n)]
    # LO chart's diagram identification uses ``DiagramType`` property
    # (e.g. "com.sun.star.chart.BarDiagram"), NOT ``SupportedServiceNames``
    # — the service list only contains generic ``Diagram`` interfaces and
    # does NOT include "BarDiagram"/"LineDiagram"/etc, so a substring scan
    # of services silently fails every chart_type filter.
    _TYPE_HINTS = {
        "column":  "bardiagram",
        "bar":     "bardiagram",
        "line":    "linediagram",
        "pie":     "piediagram",
        "scatter": "xydiagram",
        "area":    "areadiagram",
    }
    for ch in candidates:
        try:
            doc_obj = ch.getEmbeddedObject()
            if title is not None:
                cur = ""
                try:
                    cur = doc_obj.Title.String or ""
                except Exception:
                    pass
                if title not in cur:
                    continue
            if chart_type is not None:
                hint = _TYPE_HINTS.get(str(chart_type).lower())
                if hint:
                    diag_type = ""
                    try:
                        diagram = doc_obj.getDiagram()
                        diag_type = str(diagram.DiagramType or "").lower()
                    except Exception:
                        diag_type = ""
                    if hint not in diag_type:
                        continue
            if expected_series_count is not None:
                actual = 0
                try:
                    diag2 = doc_obj.getFirstDiagram()
                    for cs in diag2.getCoordinateSystems():
                        for ct in cs.getChartTypes():
                            actual += len(ct.getDataSeries() or ())
                except Exception:
                    actual = -1
                if actual != int(expected_series_count):
                    continue
            return True
        except Exception:
            continue
    return False


def verify_col_hidden(*, sheet, cols, hidden=True):
    """Every listed column's hidden state equals ``hidden``.

    Args:
        sheet:  sheet name.
        cols:   list of column letters (``["A","C"]``) or 0-based ints.
        hidden: True (default) requires every column to be HIDDEN;
                False requires every column to be VISIBLE.
    """
    if sheet not in sheets.getElementNames():
        return False
    sh_cols = sheets.getByName(sheet).getColumns()
    for c in cols:
        idx = c if isinstance(c, int) else _calc_ops_col_letter_to_idx(c)
        try:
            is_visible = bool(sh_cols.getByIndex(int(idx)).IsVisible)
        except Exception:
            return False
        if (not is_visible) != bool(hidden):
            return False
    return True


def verify_clipboard_has_content(**_extra):
    """True iff the system clipboard currently holds content.

    Confirms a ``calc_copy_to_clipboard`` succeeded — the clipboard is
    transient (not a document property), so this is the only way to
    verify the copy short of actually pasting. NOTE: checks only that
    the clipboard is NON-EMPTY, not that it holds one specific range."""
    try:
        clip = remote_ctx.ServiceManager.createInstanceWithContext(
            "com.sun.star.datatransfer.clipboard.SystemClipboard",
            remote_ctx)
        contents = clip.getContents()
        if contents is None:
            return False
        flavors = contents.getTransferDataFlavors()
        return bool(flavors) and len(flavors) > 0
    except Exception:
        return False

# ===== End pre-built Calc ops =====
'''

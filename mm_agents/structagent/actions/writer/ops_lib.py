"""Pre-built LibreOffice Writer operations exposed to the actor.

The string :data:`WRITER_OPS_LIBRARY` is concatenated into every
``cli_run_uno`` script whose domain is ``libreoffice_writer``, right
after the connect + domain setup blocks (see
``cli_run_uno_helpers._ops_library_for``). It defines ``op_*(...)``
mutation operations the actor invokes through ``writer_*`` structured
JSON actions — the actor never writes UNO Python itself.

This module is the first piece of the Writer structured-action
subsystem (the Calc / Impress equivalents already exist as
``calc_ops_lib`` / ``impress_ops_lib``). It currently exposes one op:
``op_paste_at`` — paste the clipboard at a precise document location.

Design contract (same as calc_ops_lib):
  * Every op is keyword-only (``*,``) so the actor cannot get arg
    order wrong.
  * Ops rely on the wrapper-bound globals — for Writer:
    ``doc``, ``text``, ``view_cursor``, ``remote_ctx`` — and do NOT
    re-connect to UNO.
  * Mutating ops end with ``doc.store()``.
  * Ops raise ``ValueError`` for input that cannot succeed; the
    wrapper catches it and surfaces the traceback to the planner.
"""

WRITER_OPS_LIBRARY = '''
# ===== Begin pre-built Writer ops (auto-injected by cli_run_uno wrapper) =====
# NOTE: these helpers rely on the wrapper's globals: `doc`, `text`,
# `view_cursor`, `remote_ctx`. Do not redefine those above. Do not
# re-import uno.


def _writer_ops_paragraphs():
    """Live list of body paragraphs (same order the doc-perceiver
    reports, so a caller's paragraph_index lines up with what the
    planner saw)."""
    return [p for p in text.createEnumeration()
            if p.supportsService("com.sun.star.text.Paragraph")]


def _writer_ops_set_para_text(t_obj, para, new_s):
    """Replace one paragraph's text via cursor selection + insertString —
    the Text-service replace path. Unlike ``Paragraph.setString`` it keeps
    cursors/views consistent: raw setString on a GUI-open document can leave
    the view repainting a disposed range and crash soffice (seen live on
    case-conversion over a whole document)."""
    c = t_obj.createTextCursorByRange(para)
    t_obj.insertString(c, new_s, True)


def _writer_ops_transform_text(t_obj, fn):
    """Apply ``fn(str) -> str`` to every paragraph under ``t_obj`` (an XText:
    the document body or a table cell), RECURSING into text tables — the body
    enumeration alone SKIPS tables entirely, so a whole-document text op that
    only walks body paragraphs silently leaves table-cell text untouched
    (and the evaluator then fails on the table content). Returns #changed."""
    n = 0
    for el in t_obj.createEnumeration():
        if el.supportsService("com.sun.star.text.Paragraph"):
            s = el.getString()
            if s.strip():
                new_s = fn(s)
                if new_s != s:
                    _writer_ops_set_para_text(t_obj, el, new_s)
                    n += 1
        elif el.supportsService("com.sun.star.text.TextTable"):
            for _cn in el.getCellNames():
                n += _writer_ops_transform_text(el.getCellByName(_cn), fn)
    return n


def op_paste_at(*, paragraph_index=None, after_heading=None):
    """Paste the current clipboard content at a PRECISE location in the
    document, then save.

    The location is resolved deterministically via UNO and the view
    cursor is moved there programmatically — no GUI clicking — so the
    paste lands exactly where intended. This is the Writer half of a
    cross-app copy/paste (e.g. ``calc_copy_to_clipboard`` copies a sheet
    range, then ``op_paste_at`` drops it into the report as a native
    table).

    Exactly ONE locator must be given:
      paragraph_index:  0-based body-paragraph index (the index the
                        doc-perceiver structure block reports).
      after_heading:    substring of a heading paragraph's text; the
                        paste goes into the paragraph immediately
                        AFTER the matched heading.
    """
    paras = _writer_ops_paragraphs()
    if not paras:
        raise ValueError("document has no body paragraphs")

    target = None
    if paragraph_index is not None:
        idx = int(paragraph_index)
        if idx < 0 or idx >= len(paras):
            raise ValueError(
                "paragraph_index %d out of range (0..%d)"
                % (idx, len(paras) - 1))
        target = paras[idx]
    elif after_heading:
        needle = str(after_heading).strip().lower()
        if not needle:
            raise ValueError("after_heading is empty")
        for i, p in enumerate(paras):
            if needle in p.getString().strip().lower():
                target = paras[i + 1] if (i + 1) < len(paras) else p
                break
        if target is None:
            raise ValueError(
                "after_heading %r matched no paragraph" % (after_heading,))
    else:
        raise ValueError(
            "op_paste_at requires paragraph_index or after_heading")

    controller = doc.getCurrentController()
    vc = controller.getViewCursor()
    # Move the VISIBLE cursor to the start of the target paragraph —
    # .uno:Paste pastes at the view cursor.
    vc.gotoRange(target.getStart(), False)

    dispatcher = remote_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.frame.DispatchHelper", remote_ctx)
    dispatcher.executeDispatch(
        controller.getFrame(), ".uno:Paste", "", 0, ())
    doc.store()
    return True


# ---- Paragraph / character property ops -----------------------------------

def _writer_ops_idxs(target_indices):
    """Resolve a ``target_indices`` kwarg to a concrete RAW-index list
    over the current paragraph list.

    SEMANTIC (unified across ALL writer paragraph ops): indices are
    0-based RAW paragraph positions — the SAME index the doc-perceiver
    structure block reports. Blank paragraphs are counted in their
    raw position; pass the raw index even if it lands on a blank.

    None → list of ALL raw indices. Negative indices count from the
    end of the raw paragraph list.
    """
    paras = _writer_ops_paragraphs()
    if target_indices is None:
        return paras, list(range(len(paras)))
    res = [i if i >= 0 else len(paras) + i for i in target_indices]
    return paras, res


def op_line_spacing(*, multiplier, target_indices=None):
    """Set LINE SPACING of paragraphs. multiplier: 1.0=single, 1.5,
    2.0=double, ...

    ``target_indices``: 0-based RAW paragraph indices — the SAME index
    the doc-perceiver's structure block reports (blank paragraphs are
    counted as themselves, NOT skipped from the count). Pass
    ``target_indices=[2, 4]`` to target raw paragraphs 2 and 4.
    ``None`` → apply to every non-blank paragraph (we still skip blanks
    on the apply-to-all path because changing the line spacing of a
    blank paragraph affects the vertical white-space the reader sees
    and is almost never what the caller asks for).

    Negative indices count from the end of the raw paragraph list.
    """
    import uno
    ls = uno.createUnoStruct("com.sun.star.style.LineSpacing")
    ls.Mode = 0
    ls.Height = int(float(multiplier) * 100)
    paras = _writer_ops_paragraphs()
    if target_indices is None:
        for p in paras:
            if p.getString().strip():
                p.ParaLineSpacing = ls
    else:
        n = len(paras)
        for i in target_indices:
            ci = int(i) if int(i) >= 0 else n + int(i)
            if 0 <= ci < n:
                paras[ci].ParaLineSpacing = ls
    doc.store()
    return True


def op_font_size(*, size, pattern=None, target_indices=None):
    """Set FONT SIZE (pt). pattern: optional regex to scope within a
    paragraph; None = whole paragraph. target_indices: None = all."""
    import re
    paras, idxs = _writer_ops_idxs(target_indices)
    for i in idxs:
        if pattern is None:
            c = text.createTextCursorByRange(paras[i])
            c.CharHeight = float(size)
        else:
            rx = re.compile(pattern)
            for m in rx.finditer(paras[i].getString()):
                c = text.createTextCursorByRange(paras[i].getStart())
                c.goRight(m.start(), False)
                c.goRight(m.end() - m.start(), True)
                c.CharHeight = float(size)
    doc.store()
    return True


def op_set_font(*, font_name, target_indices=None):
    """Restyle EXISTING text with a font name. target_indices: None = all."""
    paras, idxs = _writer_ops_idxs(target_indices)
    for i in idxs:
        c = text.createTextCursorByRange(paras[i])
        c.CharFontName = font_name
        c.CharFontNameAsian = font_name
        c.CharFontNameComplex = font_name
    doc.store()
    return True


def op_default_font(*, font_name):
    """Set the application's DEFAULT font (registrymodifications.xcu —
    what new typing uses). NOT a document edit; no doc.store()."""
    import uno
    cp = remote_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.configuration.ConfigurationProvider", remote_ctx)
    _pv = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    _pv.Name = "nodepath"
    _pv.Value = "/org.openoffice.Office.Writer/DefaultFont"
    ua = cp.createInstanceWithArguments(
        "com.sun.star.configuration.ConfigurationUpdateAccess", (_pv,))
    ua.setPropertyValue("Standard", font_name)
    ua.commitChanges()
    return True


def op_alignment(*, alignment, target_indices=None):
    """Set paragraph ALIGNMENT. alignment: left/center/right/justify."""
    from com.sun.star.style.ParagraphAdjust import (
        LEFT, CENTER, RIGHT, BLOCK)
    _m = {"left": LEFT, "center": CENTER, "centre": CENTER,
          "right": RIGHT, "justify": BLOCK, "justified": BLOCK,
          "block": BLOCK}
    adj = _m.get(str(alignment).strip().lower())
    if adj is None:
        raise ValueError("alignment must be left/center/right/justify, "
                          "got %r" % (alignment,))
    paras, idxs = _writer_ops_idxs(target_indices)
    for i in idxs:
        paras[i].ParaAdjust = adj
    doc.store()
    return True


def op_case_conversion(*, mode, target_indices=None):
    """Convert text to 'lower' or 'upper' case.

    ``target_indices=None`` converts the WHOLE document — body paragraphs AND
    table-cell text (tables are not in the body enumeration; skipping them
    left e.g. a grading-rubric table uppercase while the body converted, and
    the full-document evaluator then failed). Explicit indices keep the usual
    body-paragraph RAW-index semantics.
    GUI safety: mutations run under lockControllers and replace text through
    a cursor (not Paragraph.setString) — bulk setString on a GUI-open doc
    raced the repaint and could crash soffice into Document Recovery."""
    m = str(mode).strip().lower()
    if m not in ("lower", "upper"):
        raise ValueError("mode must be 'lower' or 'upper', got %r" % (mode,))
    fn = (lambda s: s.lower()) if m == "lower" else (lambda s: s.upper())
    doc.lockControllers()
    try:
        if target_indices is None:
            _writer_ops_transform_text(text, fn)
        else:
            paras, idxs = _writer_ops_idxs(target_indices)
            for i in idxs:
                s = paras[i].getString()
                if not s.strip():
                    continue
                _writer_ops_set_para_text(text, paras[i], fn(s))
    finally:
        doc.unlockControllers()
    doc.store()
    return True


def op_strikethrough(*, target_indices=None, pattern=None):
    """Apply SINGLE strikethrough. pattern: optional regex to scope
    within a paragraph; None = whole paragraph."""
    import re
    from com.sun.star.awt.FontStrikeout import SINGLE
    paras, idxs = _writer_ops_idxs(target_indices)
    for i in idxs:
        if pattern is None:
            c = text.createTextCursorByRange(paras[i])
            c.CharStrikeout = SINGLE
        else:
            rx = re.compile(pattern)
            for m in rx.finditer(paras[i].getString()):
                c = text.createTextCursorByRange(paras[i].getStart())
                c.goRight(m.start(), False)
                c.goRight(m.end() - m.start(), True)
                c.CharStrikeout = SINGLE
    doc.store()
    return True


def op_insert_page_break(*, after_paragraph_index=0):
    """Insert a hard page break AFTER the given RAW paragraph index
    (matches what the doc-perceiver structure block reports).

    Out-of-range handling:
      * ``after_paragraph_index >= len(paragraphs)`` is clamped to the
        last raw paragraph (insert at end of document).
      * Negative indices count from the end (``-1`` = last paragraph).
      * Raises ``ValueError`` only when the document has zero paragraphs.
    """
    from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
    from com.sun.star.style.BreakType import PAGE_BEFORE
    paras = _writer_ops_paragraphs()
    n = len(paras)
    if n == 0:
        raise ValueError("op_insert_page_break: document has no paragraphs")
    idx = int(after_paragraph_index)
    if idx < 0:
        idx = n + idx
    if idx >= n:
        idx = n - 1
    if idx < 0:
        idx = 0
    c = text.createTextCursorByRange(paras[idx].getEnd())
    text.insertControlCharacter(c, PARAGRAPH_BREAK, False)
    paras = _writer_ops_paragraphs()
    # After insertion the newly-created paragraph sits at idx+1. If the
    # original idx was the last paragraph (clamped above), idx+1 is
    # within range after the insertion.
    if idx + 1 < len(paras):
        paras[idx + 1].BreakType = PAGE_BEFORE
    doc.store()
    return True


def op_page_numbers(*, position="bottom_left"):
    """Add live page-number fields to the page header/footer.
    position: bottom_left|bottom_center|bottom_right|top_*."""
    from com.sun.star.style.ParagraphAdjust import LEFT, CENTER, RIGHT
    page_styles = doc.StyleFamilies.getByName("PageStyles")
    if page_styles.hasByName("Default Page Style"):
        style = page_styles.getByName("Default Page Style")
    else:
        style = page_styles.getByName("Standard")
    pos = str(position).strip().lower()
    if pos.startswith("top"):
        style.HeaderIsOn = True
        target = style.HeaderText
    else:
        style.FooterIsOn = True
        target = style.FooterText
    c = target.createTextCursor()
    c.gotoStart(False)
    c.gotoEnd(True)
    c.setString("")
    c.gotoStart(False)
    if pos.endswith("_left"):
        c.ParaAdjust = LEFT
    elif pos.endswith("_center"):
        c.ParaAdjust = CENTER
    elif pos.endswith("_right"):
        c.ParaAdjust = RIGHT
    page_num = doc.createInstance("com.sun.star.text.TextField.PageNumber")
    page_num.NumberingType = 4
    target.insertTextContent(c, page_num, False)
    doc.store()
    return True


def op_capitalize_words(*, target_indices=None):
    """Title-case paragraph text — capitalize the first letter of every
    word (apostrophes keep a word together; hyphens break it)."""
    import re
    _word_re = re.compile(r"[A-Za-z][A-Za-z'’]*")

    def _cap(m):
        w = m.group(0)
        return (w[0].upper() if w[0].isalpha() else w[0]) + w[1:]

    paras, idxs = _writer_ops_idxs(target_indices)
    doc.lockControllers()
    try:
        for i in idxs:
            s = paras[i].getString()
            if not s.strip():
                continue
            new_s = _word_re.sub(_cap, s)
            if new_s != s:
                _writer_ops_set_para_text(text, paras[i], new_s)
    finally:
        doc.unlockControllers()
    doc.store()
    return True


def op_remove_highlighting(*, target_indices=None):
    """Remove background highlighting (CharBackColor) from paragraphs —
    clears it on the paragraph cursor AND each text portion."""
    paras, idxs = _writer_ops_idxs(target_indices)
    for i in idxs:
        para = paras[i]
        c = text.createTextCursorByRange(para)
        c.CharBackColor = -1
        for portion in para.createEnumeration():
            if hasattr(portion, "CharBackColor"):
                pc = text.createTextCursorByRange(portion)
                pc.CharBackColor = -1
    doc.store()
    return True


def op_subscript_text(*, pattern, target_indices=None):
    """Convert text matching ``pattern`` (a Python regex) to subscript."""
    import re
    rx = re.compile(pattern)
    paras, idxs = _writer_ops_idxs(target_indices)
    for i in idxs:
        for m in rx.finditer(paras[i].getString()):
            c = text.createTextCursorByRange(paras[i].getStart())
            c.goRight(m.start(), False)
            c.goRight(m.end() - m.start(), True)
            c.CharEscapement = -14000
            c.CharEscapementHeight = 58
    doc.store()
    return True


def op_insert_table(*, rows, cols):
    """Insert a NEW empty table (rows x cols) at the view cursor."""
    table = doc.createInstance("com.sun.star.text.TextTable")
    table.initialize(int(rows), int(cols))
    insert_cur = text.createTextCursorByRange(view_cursor.getStart())
    text.insertTextContent(insert_cur, table, False)
    doc.store()
    return True


def op_delete_table(*, table_index=None, after_heading=None):
    """Delete Writer table(s) — the rollback for a wrong / duplicate
    insert (e.g. verify reports "found 3 tables, expected 1").

    Pass exactly one selector:
      * table_index: 1-based index among the document's tables in
        document order — deletes that single table.
      * after_heading: text of a heading — deletes EVERY table in that
        heading's section EXCEPT the first one (de-duplicates a
        section back to a single table).

    Raises ValueError on a missing / out-of-range selector so the
    planner gets an actionable message.
    """
    if table_index is not None:
        tables = doc.getTextTables()
        n = tables.getCount()
        idx = int(table_index) - 1
        if idx < 0 or idx >= n:
            raise ValueError(
                "table_index %s out of range (1..%d)" % (table_index, n))
        text.removeTextContent(tables.getByIndex(idx))
        doc.store()
        return True
    if after_heading and str(after_heading).strip():
        needle = str(after_heading).strip().lower()
        els = _writer_ops_elements()
        start = -1
        for i, (kind, el) in enumerate(els):
            if kind == "para" and needle in el.getString().strip().lower():
                start = i
                break
        if start < 0:
            raise ValueError(
                "heading %r not found in the document" % (after_heading,))
        section_tables = []
        for kind, el in els[start + 1:]:
            if kind == "para" and _writer_ops_is_heading(el):
                break
            if kind == "table":
                section_tables.append(el)
        # Keep the first, delete the rest.
        for tbl in section_tables[1:]:
            text.removeTextContent(tbl)
        doc.store()
        return True
    raise ValueError(
        "op_delete_table needs either table_index (1-based) or "
        "after_heading")


def op_export_pdf(*, output_path=None, output_filename=None):
    """Export the document to PDF via the writer_pdf_Export filter."""
    import uno
    import os as _os
    doc_url = doc.getURL()
    if doc_url:
        doc_dir = uno.fileUrlToSystemPath(_os.path.dirname(doc_url))
        doc_name = _os.path.splitext(_os.path.basename(doc_url))[0]
    else:
        doc_dir = "/home/user/Desktop"
        doc_name = "export"
    final_dir = output_path or doc_dir
    final_name = output_filename or (doc_name + ".pdf")
    if not final_name.lower().endswith(".pdf"):
        final_name += ".pdf"
    url_out = uno.systemPathToFileUrl(_os.path.join(final_dir, final_name))
    _f = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    _f.Name = "FilterName"
    _f.Value = "writer_pdf_Export"
    _o = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    _o.Name = "Overwrite"
    _o.Value = True
    doc.storeToURL(url_out, (_f, _o))
    return True


def op_tabstops_split_align(*, n_words_before_tab=3, right_tab_pos=16510,
                            target_indices=None):
    """Split each non-blank paragraph at the Nth word — first N words
    stay left, the rest jump to a RIGHT tab stop.

    ``target_indices``: 0-based RAW paragraph indices (matches the
    doc-perceiver structure block). Blank paragraphs at the listed
    indices are silently skipped (cannot split nothing).
    ``None`` → every non-blank paragraph.
    """
    import uno
    ts = uno.createUnoStruct("com.sun.star.style.TabStop")
    ts.Position = int(right_tab_pos)
    ts.Alignment = 2       # RIGHT
    ts.FillChar = 32       # space
    n = int(n_words_before_tab)
    paras = _writer_ops_paragraphs()

    def _apply(p):
        s = p.getString()
        if not s.strip():
            return
        words = s.split()
        if len(words) >= n:
            _writer_ops_set_para_text(
                text, p, " ".join(words[:n]) + "\t" + " ".join(words[n:]))
            p.ParaTabStops = (ts,)

    doc.lockControllers()
    try:
        if target_indices is None:
            for p in paras:
                _apply(p)
        else:
            total = len(paras)
            for i in target_indices:
                ci = int(i) if int(i) >= 0 else total + int(i)
                if 0 <= ci < total:
                    _apply(paras[ci])
    finally:
        doc.unlockControllers()
    doc.store()
    return True


def _writer_norm_color(v):
    """Normalize a color kwarg to a 0xRRGGBB int. Accepts an int or a
    '#RRGGBB' / 'RRGGBB' hex string."""
    if isinstance(v, int):
        return v
    s = str(v).strip().lstrip("#")
    return int(s, 16)


def op_set_color(*, color, pattern=None, walk_tables=True,
                 target_indices=None):
    """Set TEXT COLOR (CharColor). pattern: optional regex for
    per-match coloring; None = whole paragraph. walk_tables: also
    color matching text inside table cells."""
    import re
    cv = _writer_norm_color(color)
    paras = _writer_ops_paragraphs()

    def _color_in(para, owner):
        if pattern is None:
            c = owner.createTextCursorByRange(para)
            c.CharColor = cv
            return
        rx = re.compile(pattern)
        for m in rx.finditer(para.getString()):
            c = owner.createTextCursorByRange(para.getStart())
            c.goRight(m.start(), False)
            c.goRight(m.end() - m.start(), True)
            c.CharColor = cv

    idxs = range(len(paras)) if target_indices is None else target_indices
    for i in idxs:
        _color_in(paras[i], text)
    if walk_tables:
        n = doc.TextTables.getCount() if hasattr(doc, "TextTables") else 0
        for ti in range(n):
            tbl = doc.TextTables.getByIndex(ti)
            for cn in tbl.CellNames:
                cell = tbl.getCellByName(cn)
                ct = cell.Text
                for el in cell.createEnumeration():
                    if el.supportsService("com.sun.star.text.Paragraph"):
                        _color_in(el, ct)
    doc.store()
    return True


def op_find_replace(*, mode, pattern=None, replacement=None,
                    dedup_pattern=None, target_indices=None):
    """mode='regex': regex find+replace across paragraphs.
    mode='dedup': delete duplicate paragraphs (key = whole line, or
    capture-group 1 of dedup_pattern)."""
    import re
    paras = _writer_ops_paragraphs()
    if mode == "regex":
        if pattern is None or replacement is None:
            raise ValueError("regex mode requires pattern + replacement")
        rx = re.compile(pattern)
        idxs = range(len(paras)) if target_indices is None else target_indices
        for i in idxs:
            new_s, n = rx.subn(replacement, paras[i].getString())
            if n > 0:
                paras[i].setString(new_s)
    elif mode == "dedup":
        def _key(s):
            if dedup_pattern is None:
                return s
            m = re.search(dedup_pattern, s)
            return m.group(1) if (m and m.groups()) else s
        seen = set()
        to_delete = []
        for p in paras:
            s = p.getString()
            if not s.strip():
                continue
            k = _key(s)
            if k in seen:
                to_delete.append(p)
            else:
                seen.add(k)
        for p in reversed(to_delete):
            pc = text.createTextCursorByRange(p.getStart())
            pc.gotoEndOfParagraph(True)
            pc.goRight(1, True)
            pc.setString("")
    else:
        raise ValueError("mode must be 'regex' or 'dedup', got %r" % (mode,))
    doc.store()
    return True


def op_insert_image(*, image_path, width_cm=None, height_cm=None):
    """Insert an image file at the view cursor (anchored AS_CHARACTER).
    width_cm / height_cm optional (float cm); omit for native size."""
    import os as _os
    import uno
    from com.sun.star.text.TextContentAnchorType import AS_CHARACTER
    ip = _os.path.abspath(_os.path.expanduser(str(image_path)))
    if not _os.path.exists(ip):
        raise ValueError("source image not found at %s" % ip)
    gp = remote_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.graphic.GraphicProvider", remote_ctx)
    _u = uno.createUnoStruct("com.sun.star.beans.PropertyValue")
    _u.Name = "URL"
    _u.Value = "file://" + ip
    g = gp.queryGraphic((_u,))
    if g is None:
        raise ValueError("GraphicProvider could not load %s" % ip)
    tg = doc.createInstance("com.sun.star.text.TextGraphicObject")
    tg.Graphic = g
    tg.AnchorType = AS_CHARACTER
    if width_cm is not None:
        tg.Width = int(float(width_cm) * 1000)
    if height_cm is not None:
        tg.Height = int(float(height_cm) * 1000)
    vc = view_cursor.getStart()
    if vc.Text == text:
        ic = text.createTextCursorByRange(vc)
    else:
        ic = text.createTextCursor()
        ic.gotoEnd(False)
    text.insertTextContent(ic, tg, False)
    doc.store()
    return True


def op_insert_paragraph_breaks(*, target_idx=0):
    """Split the paragraph at RAW index ``target_idx`` on sentence
    boundaries — first sentence stays, the rest become new paragraphs
    each preceded by an empty paragraph.

    ``target_idx`` is the SAME 0-based raw paragraph index the
    doc-perceiver structure block reports. If the indexed paragraph
    is blank or has fewer than 2 sentences, raises ``ValueError``.
    Negative indices count from the end.
    """
    import re
    from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
    paras = _writer_ops_paragraphs()
    n = len(paras)
    idx = int(target_idx)
    if idx < 0:
        idx = n + idx
    if not (0 <= idx < n):
        raise ValueError("target_idx %s out of range (0..%d)"
                         % (target_idx, n - 1))
    target_para = paras[idx]
    if not target_para.getString().strip():
        raise ValueError("paragraph at raw index %d is blank — "
                         "nothing to split" % idx)
    pt = target_para.getString().strip()
    parts = [s for s in re.findall(r"[^.!?]+[.!?]\s*", pt) if s.strip()]
    if len(parts) < 2:
        raise ValueError("paragraph has only %d sentence(s) — nothing "
                         "to split" % len(parts))
    target_para.setString(parts[0])
    ins = text.createTextCursorByRange(target_para.getEnd())
    for s in parts[1:]:
        text.insertControlCharacter(ins, PARAGRAPH_BREAK, False)
        text.insertControlCharacter(ins, PARAGRAPH_BREAK, False)
        text.insertString(ins, s, False)
    doc.store()
    return True


def op_text_to_table(*, start_index, end_index, separator=","):
    """Convert paragraphs in the inclusive RAW-index range
    [start_index, end_index] — each a separator-delimited row — into a
    Writer TextTable, replacing those paragraphs.

    ``start_index`` / ``end_index`` are 0-based RAW paragraph indices,
    matching what the doc-perceiver structure block reports. Blank
    paragraphs within the range are silently skipped (they would
    become empty rows in the table otherwise — almost never the
    caller's intent).
    """
    paras = _writer_ops_paragraphs()
    s_ri, e_ri = int(start_index), int(end_index)
    total = len(paras)
    # Normalize negative indices (count from end of raw paragraph list).
    if s_ri < 0:
        s_ri = total + s_ri
    if e_ri < 0:
        e_ri = total + e_ri
    if s_ri > e_ri or s_ri < 0 or e_ri >= total:
        raise ValueError("invalid raw paragraph range (%d, %d) for a "
                         "document with %d paragraphs"
                         % (s_ri, e_ri, total))
    target = [p for i, p in enumerate(paras)
              if s_ri <= i <= e_ri and p.getString().strip()]
    if not target:
        raise ValueError(
            "no non-blank paragraphs in raw range (%d, %d)"
            % (s_ri, e_ri))
    rows = [[c.strip() for c in p.getString().split(separator)]
            for p in target]
    ncols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < ncols:
            r.append("")
    table = doc.createInstance("com.sun.star.text.TextTable")
    table.initialize(len(rows), ncols)
    anchor = text.createTextCursorByRange(target[0].getStart())
    text.insertTextContent(anchor, table, False)

    def _cn(c, r):
        s = ""
        n = c
        while True:
            s = chr(ord("A") + (n % 26)) + s
            n = n // 26 - 1
            if n < 0:
                break
        return s + str(r + 1)

    for r in range(len(rows)):
        for c in range(ncols):
            table.getCellByName(_cn(c, r)).setString(rows[r][c])
    dc = text.createTextCursorByRange(target[0].getStart())
    dc.gotoRange(target[-1].getEnd(), True)
    dc.goRight(1, True)
    dc.setString("")
    doc.store()
    return True


def op_cross_reference(*, citation_text, marker):
    """Append ``citation_text`` to the reference list and replace
    ``marker`` in the body with a numbered citation marker [N]."""
    from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK
    paras = _writer_ops_paragraphs()
    kws = ("references", "reference list", "bibliography", "works cited")
    refs_idx = None
    for i, p in enumerate(paras):
        if p.getString().strip().lower() in kws:
            refs_idx = i
            break
    if refs_idx is None:
        raise ValueError("no References / Bibliography heading found")
    existing = [p for p in paras[refs_idx + 1:] if p.getString().strip()]
    num = len(existing) + 1
    replaced = False
    for p in paras:
        s = p.getString()
        if marker in s:
            p.setString(s.replace(marker, "[%d]" % num))
            replaced = True
            break
    if not replaced:
        raise ValueError("marker %r not found in any paragraph" % (marker,))
    anchor = existing[-1] if existing else paras[refs_idx]
    ec = text.createTextCursorByRange(anchor.getEnd())
    text.insertControlCharacter(ec, PARAGRAPH_BREAK, False)
    text.insertString(ec, citation_text, False)
    doc.store()
    return True


# ---- Read-only verify ops (for kind="writer_verify" verify specs) ----------

def _writer_ops_elements():
    """Document body elements in order: list of ("para", paragraph) /
    ("table", texttable). Tables appear inline at their anchor position,
    so a caller can tell which heading a table follows."""
    out = []
    e = text.createEnumeration()
    while e.hasMoreElements():
        el = e.nextElement()
        if el.supportsService("com.sun.star.text.Paragraph"):
            out.append(("para", el))
        elif el.supportsService("com.sun.star.text.TextTable"):
            out.append(("table", el))
    return out


def _writer_ops_is_heading(para):
    """True iff a paragraph is heading-styled (ParaStyleName starts
    with 'Heading' — LibreOffice's built-in heading styles)."""
    try:
        return str(para.ParaStyleName or "").lower().startswith("heading")
    except Exception:
        return False


def _writer_ops_cell_name(row, col):
    """(0-based row, 0-based col) -> Writer cell name, e.g. (1,0)->'A2'."""
    letters = ""
    c = int(col)
    while True:
        letters = chr(ord("A") + c % 26) + letters
        c = c // 26 - 1
        if c < 0:
            break
    return letters + str(int(row) + 1)


def verify_table_count(*, expected=None, min=None, max=None, **_extra):
    """Verify the TOTAL number of tables in the document.

    Returns ``(ok, reason)`` — reason is "" on success, a specific
    message on failure (so the verifier can surface WHY to the planner).

    expected: exact count. min/max: bounds. Use ``expected`` to catch a
    DUPLICATE paste (e.g. expected=1)."""
    n = doc.getTextTables().getCount()
    if expected is not None and n != int(expected):
        return (False, "document has %d table(s), expected exactly %d"
                % (n, int(expected)))
    if min is not None and n < int(min):
        return (False, "document has %d table(s), need at least %d"
                % (n, int(min)))
    if max is not None and n > int(max):
        return (False, "document has %d table(s), at most %d allowed"
                % (n, int(max)))
    return (True, "")


def verify_table_after_heading(*, heading, single=True,
                               exact_rows=None, min_rows=None,
                               n_cols=None, cell_contains=None, **_extra):
    """Verify a table sits in the section that starts at ``heading``.

    Walks the document in order: finds the (heading-styled) paragraph
    whose text contains ``heading``, then collects the tables between
    it and the NEXT heading-styled paragraph.

      single:        require EXACTLY ONE table in that section
                     (default True — catches a duplicate paste).
      exact_rows / min_rows / n_cols: shape checks on the FIRST such
                     table.
      cell_contains: list of [row, col, substring] (0-based row/col) —
                     each named cell's text must contain the substring.

    Returns ``(ok, reason)`` — reason names the FIRST failing condition
    so the planner gets actionable feedback (e.g. "table has 1 row,
    need at least 2" → the planner knows to include the header row)."""
    needle = str(heading).strip().lower()
    if not needle:
        return (False, "heading argument is empty")
    els = _writer_ops_elements()
    start = -1
    for i, (kind, el) in enumerate(els):
        if kind == "para" and needle in el.getString().strip().lower():
            start = i
            break
    if start < 0:
        return (False, "heading %r not found in the document" % (heading,))
    section_tables = []
    for kind, el in els[start + 1:]:
        if kind == "para" and _writer_ops_is_heading(el):
            break
        if kind == "table":
            section_tables.append(el)
    if not section_tables:
        return (False, "no table found in the '%s' section" % (heading,))
    if single and len(section_tables) != 1:
        return (False, "expected exactly 1 table in the '%s' section, "
                "found %d (duplicate paste?)"
                % (heading, len(section_tables)))
    tbl = section_tables[0]
    rows = tbl.getRows().getCount()
    cols = tbl.getColumns().getCount()
    if exact_rows is not None and rows != int(exact_rows):
        return (False, "table has %d row(s), expected exactly %d"
                % (rows, int(exact_rows)))
    if min_rows is not None and rows < int(min_rows):
        return (False, "table has %d row(s), need at least %d — a "
                "header row plus the data row(s) is missing?"
                % (rows, int(min_rows)))
    if n_cols is not None and cols != int(n_cols):
        return (False, "table has %d column(s), expected %d"
                % (cols, int(n_cols)))
    for spec in (cell_contains or []):
        try:
            r, c, sub = spec[0], spec[1], str(spec[2])
        except Exception:
            return (False, "malformed cell_contains entry %r" % (spec,))
        try:
            cell_text = tbl.getCellByName(
                _writer_ops_cell_name(r, c)).getString()
        except Exception:
            return (False, "cell (row %s, col %s) does not exist in the "
                    "table" % (r, c))
        if sub not in cell_text:
            return (False, "cell (row %s, col %s) = %r does not contain "
                    "%r" % (r, c, cell_text, sub))
    return (True, "")
'''

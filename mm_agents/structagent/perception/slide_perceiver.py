"""SlidePerceiver — impress-side analogue of SheetPerceiver.

Probes the live Impress deck via UNO and emits a compact "PRESENTATION
STRUCTURE" text block per slide: layout, background, notes preview, and
per-shape attributes (type, name, pos, size, text, font).

Read-only (never calls ``doc.store()``). Caps rendered slides at
K_SLIDES; the rest get a one-line count.
"""
from __future__ import annotations
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional


K_SLIDES = 20
TEXT_PREVIEW_CHARS = 60


# ---------------------------------------------------------------------- #
# UNO probe script (runs in the VM via run_python_script)
# ---------------------------------------------------------------------- #
INSPECT_SCRIPT = r'''
import uno
import json
import sys
import subprocess
import time

_BEGIN = "===SLIDE_INSPECT_BEGIN==="
_END   = "===SLIDE_INSPECT_END==="


def _connect():
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        'com.sun.star.bridge.UnoUrlResolver', local_ctx)
    return resolver.resolve(
        'uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext')


def _shape_kind(svcs):
    """Classify a shape into a short type tag."""
    s = " ".join(svcs)
    if "PageShape" in s:
        return "page"
    if "GraphicObjectShape" in s:
        return "pic"
    if "OLE2Shape" in s:
        return "chart"   # OLE2 in Impress is almost always a chart
    if "TableShape" in s:
        return "table"
    if "ConnectorShape" in s:
        return "conn"
    if "LineShape" in s:
        return "line"
    if "GroupShape" in s:
        return "group"
    if "TitleTextShape" in s:
        return "title"
    if "SubTitleTextShape" in s:
        return "subtitle"
    if "OutlineTextShape" in s:
        return "outline"
    if "TextShape" in s:
        return "txt"
    return "shape"


def _len_cm(uno_units):
    try:
        return round(float(uno_units) / 1000.0, 2)
    except Exception:
        return None


def _color_hex(c):
    try:
        return "#%06X" % (int(c) & 0xFFFFFF)
    except Exception:
        return None


def _first_para_text(shape, max_chars):
    try:
        txt = shape.getString() or ""
    except Exception:
        return None
    if not txt:
        return None
    line = txt.split("\n", 1)[0]
    if len(line) > max_chars:
        line = line[:max_chars - 1] + "…"
    return line


def _all_paragraphs(shape, max_chars=50, max_paras=12):
    """Return a list of {i, text, indent, bullet} for paragraphs in a
    text shape. Lets the planner see multi-paragraph structure
    (heading vs body bullets vs sub-bullets) so it can target the
    right paragraph_index for "first line / second line" tasks.

    Empty paragraphs are kept (so 1-based indices line up with the
    actual UNO paragraph order). Returns None for non-text shapes.
    """
    try:
        txt = shape.getText()
    except Exception:
        return None
    out = []
    try:
        for i, para in enumerate(txt.createEnumeration()):
            if i >= max_paras:
                out.append({"i": -1, "text": "...", "indent": None,
                            "bullet": None})
                break
            try:
                p_text = para.getString() or ""
            except Exception:
                p_text = ""
            if len(p_text) > max_chars:
                p_text = p_text[:max_chars - 1] + "…"
            try:
                indent = int(para.NumberingLevel)
            except Exception:
                indent = None
            try:
                bullet = bool(para.NumberingIsNumber)
            except Exception:
                bullet = None
            out.append({"i": i + 1, "text": p_text, "indent": indent,
                        "bullet": bullet})
    except Exception:
        return None
    return out if len(out) > 1 else None


def _shape_font_summary(shape):
    """Read font name/size/bold/italic/underline/strike/color from the
    FIRST run. Tasks frequently ask for bold/underline/strike toggles
    so the planner must be able to verify the post-state from the SP
    block — CharUnderline / CharStrikeout were previously omitted and
    left the planner blind to those properties."""
    try:
        txt = shape.getText()
    except Exception:
        return None
    try:
        for para in txt.createEnumeration():
            try:
                for run in para.createEnumeration():
                    out = {}
                    try: out["name"] = str(run.CharFontName)
                    except Exception: pass
                    try: out["size"] = float(run.CharHeight)
                    except Exception: pass
                    try: out["bold"] = float(run.CharWeight) >= 140.0
                    except Exception: pass
                    try:
                        from com.sun.star.awt.FontSlant import ITALIC as _IT
                        out["italic"] = (run.CharPosture == _IT)
                    except Exception:
                        pass
                    # FontUnderline: NONE=0, SINGLE=1, DOUBLE=2, ...
                    # any non-zero value means "underlined" for our purposes.
                    try: out["underline"] = bool(int(run.CharUnderline) != 0)
                    except Exception: pass
                    # FontStrikeout: NONE=0, SINGLE=1, DOUBLE=2, ...
                    try: out["strike"] = bool(int(run.CharStrikeout) != 0)
                    except Exception: pass
                    try: out["color"] = _color_hex(run.CharColor)
                    except Exception: pass
                    try: out["align"] = int(para.ParaAdjust)
                    except Exception: pass
                    return out
            except Exception:
                continue
    except Exception:
        return None
    return None


_LAYOUT_NAME = {
    0: "title", 1: "title_content", 2: "title_2content", 3: "two_content",
    19: "title_only", 20: "blank", 32: "centered_text",
}


def _notes_text(slide_obj):
    try:
        np = slide_obj.NotesPage
    except Exception:
        return None
    n = np.getCount()
    # Notes body is the last non-PageShape TextShape on the NotesPage.
    for i in range(n - 1, -1, -1):
        sh = np.getByIndex(i)
        svcs = list(getattr(sh, "SupportedServiceNames", ()) or ())
        if any("PageShape" in s for s in svcs):
            continue
        if any("TextShape" in s for s in svcs):
            try:
                txt = sh.getString() or ""
            except Exception:
                txt = ""
            return txt[:120]
    return None


def _bg_color(slide_obj):
    try:
        bg = slide_obj.Background
        if bg is None:
            return None
        return _color_hex(bg.FillColor)
    except Exception:
        return None


def _try_get_table_model(sh):
    """Robust table detection — pptx tables imported into LO Impress
    don't reliably carry 'TableShape' in service names; some appear as
    a generic graphicFrame whose .Model has Rows/Columns. Return the
    cell-range model on success, else None."""
    model = None
    try:
        model = getattr(sh, "Model", None)
    except Exception:
        model = None
    for candidate in (model, sh):
        if candidate is None:
            continue
        try:
            if hasattr(candidate, "Rows") and hasattr(candidate, "Columns"):
                _ = candidate.Rows.getCount(); _ = candidate.Columns.getCount()
                return candidate
        except Exception:
            continue
    return None


def _shape_record(sh, i):
    """Build a record for one shape. Recurses into group children so
    text-bearing shapes nested inside ``GroupShape`` show up in the
    SP block (otherwise the planner sees only the group's outer hull
    and assumes the group has no editable text)."""
    svcs = list(getattr(sh, "SupportedServiceNames", ()) or ())
    kind = _shape_kind(svcs)
    name = getattr(sh, "Name", "") or ""
    sz = None; pos = None
    try:
        sz = sh.Size
        pos = sh.Position
    except Exception:
        pass
    rec = {
        "i": i,
        "kind": kind,
        "name": name,
        "x_cm": _len_cm(pos.X) if pos else None,
        "y_cm": _len_cm(pos.Y) if pos else None,
        "w_cm": _len_cm(sz.Width) if sz else None,
        "h_cm": _len_cm(sz.Height) if sz else None,
    }
    # Probe for table — pptx tables often arrive as kind="shape" because
    # the service-name check missed them. The Model.Rows probe is
    # authoritative.
    tbl = _try_get_table_model(sh)
    if tbl is not None:
        rec["kind"] = "table"
        try:
            nr = tbl.Rows.getCount(); nc = tbl.Columns.getCount()
            rec["rows"] = int(nr); rec["cols"] = int(nc)
            # Sample first cell's first run for font + preview text
            try:
                first_cell = tbl.getCellByPosition(0, 0)
                first_txt = first_cell.getText().getString().strip()
                if first_txt:
                    rec["preview"] = first_txt[:60]
                # font of first run in first cell
                for para in first_cell.getText().createEnumeration():
                    for run in para.createEnumeration():
                        out = {}
                        try: out["name"] = str(run.CharFontName)
                        except Exception: pass
                        try: out["size"] = float(run.CharHeight)
                        except Exception: pass
                        try: out["bold"] = float(run.CharWeight) >= 140.0
                        except Exception: pass
                        try: out["color"] = _color_hex(run.CharColor)
                        except Exception: pass
                        try:
                            out["underline"] = int(run.CharUnderline) != 0
                        except Exception: pass
                        rec["font"] = out
                        break
                    break
            except Exception:
                pass
        except Exception:
            pass
        return rec
    if kind == "group":
        children = []
        try:
            n = sh.getCount()
        except Exception:
            n = 0
        for j in range(n):
            try:
                child = sh.getByIndex(j)
                children.append(_shape_record(child, j + 1))
            except Exception:
                continue
        # Sort group children by VISUAL position (top→bottom, left→right)
        # so the planner's "first textbox in this group" reasoning
        # matches a human reading the slide top-down. The child's
        # original z-order index stays available via ``i`` for any
        # shape_index= selector that needs it.
        def _child_sort_key(c):
            y = c.get("y_cm"); x = c.get("x_cm")
            return (float(y) if y is not None else 1e9,
                    float(x) if x is not None else 1e9,
                    int(c.get("i") or 0))
        children.sort(key=_child_sort_key)
        rec["children"] = children
    elif kind in ("title", "subtitle", "outline", "txt", "shape"):
        rec["text"] = _first_para_text(sh, 60)
        rec["font"] = _shape_font_summary(sh)
        # Master-driven placeholder detection. Impress renders the
        # SlideNumber/Footer/Header/DateTime TextField placeholders as
        # their angle-bracketed marker (e.g. ``<number>``) — UNLESS
        # the slide is currently active in the controller, in which
        # case LO substitutes the live value (the actual page number)
        # and the marker disappears. So text-content matching is not
        # reliable on its own.
        #
        # Primary signal: iterate the shape's text runs and check
        # ``run.TextField.SupportedServiceNames`` for ``PageNumber`` /
        # ``Footer`` / ``Header`` / ``DateTime`` — this is invariant
        # to whether the slide is active. Fall back to text-content
        # matching only when the field probe fails.
        _ROLE_BY_SVC = (
            ("PageNumber", "slide_num"),
            ("Footer", "footer"),
            ("Header", "header"),
            ("DateTime", "date"),
            ("Date", "date"),
        )
        master_role = None
        try:
            txt = sh.getText()
            for para in txt.createEnumeration():
                for run in para.createEnumeration():
                    try:
                        tf = getattr(run, "TextField", None)
                    except Exception:
                        tf = None
                    if tf is None:
                        continue
                    tf_svcs = " ".join(
                        list(getattr(tf, "SupportedServiceNames", ()) or ()))
                    for needle, role in _ROLE_BY_SVC:
                        if needle in tf_svcs:
                            master_role = role
                            break
                    if master_role:
                        break
                if master_role:
                    break
        except Exception:
            pass
        if master_role is None:
            t = (rec.get("text") or "").strip().lower()
            _MASTER_FIELD = {
                "<number>": "slide_num",
                "<page>": "slide_num",
                "<footer>": "footer",
                "<header>": "header",
                "<date>": "date",
                "<time>": "date",
                "<date/time>": "date",
            }
            master_role = _MASTER_FIELD.get(t)
        if master_role:
            rec["kind"] = master_role
            rec["master_driven"] = True
        paras = _all_paragraphs(sh)
        if paras is not None:
            rec["paragraphs"] = paras
        # Overclassify 1: pptx <p:pic> imports often arrive as
        # ``drawing.CustomShape`` with a ``FillStyle.BITMAP`` —
        # functionally a picture. Tag it as "pic" so the planner can
        # match instruction phrases like "the picture / image" via
        # the SP block instead of guessing CustomShape names.
        # CHECK THIS FIRST: an empty-text CustomShape with BITMAP fill
        # is a picture, NOT a text shape — even though
        # ``_shape_font_summary`` returns a default-font record. If we
        # ran the text overclassify first, this shape would land as
        # ``txt`` with empty text and get collapsed as decorative.
        if rec["kind"] == "shape":
            try:
                fs = sh.FillStyle
                if "BITMAP" in repr(fs):
                    rec["kind"] = "pic"
            except Exception:
                pass
        # Overclassify 2: imported decks (Google Slides → pptx)
        # frequently land as ``drawing.CustomShape`` whose service
        # name does not contain "TextShape", but the shape carries
        # real text content. Treat text-bearing shapes as ``txt`` so
        # the planner doesn't confuse them with decorative shapes.
        # Require actual TEXT content (not just a font summary —
        # ``_shape_font_summary`` returns defaults even for shapes
        # with no real text, which would otherwise misclassify
        # decorative shapes as txt).
        if rec["kind"] == "shape" and (rec.get("text") or "").strip():
            rec["kind"] = "txt"
    return rec


def _slide_record(slide_obj):
    rec = {}
    rec["name"] = getattr(slide_obj, "Name", None)
    rec["layout"] = _LAYOUT_NAME.get(int(getattr(slide_obj, "Layout", -1)),
                                     str(getattr(slide_obj, "Layout", -1)))
    try:
        rec["width_cm"]  = _len_cm(slide_obj.Width)
        rec["height_cm"] = _len_cm(slide_obj.Height)
    except Exception:
        rec["width_cm"] = None
        rec["height_cm"] = None
    rec["background"] = _bg_color(slide_obj)
    rec["notes"] = _notes_text(slide_obj)
    shapes = []
    for i in range(slide_obj.getCount()):
        try:
            sh = slide_obj.getByIndex(i)
            shapes.append(_shape_record(sh, i + 1))
        except Exception:
            continue
    rec["shapes"] = shapes
    return rec


try:
    remote_ctx = _connect()
except Exception:
    subprocess.Popen([
        'soffice',
        '--accept=socket,host=localhost,port=2002;urp;StarOffice.Service',
    ])
    time.sleep(2)
    remote_ctx = _connect()

desktop = remote_ctx.ServiceManager.createInstanceWithContext(
    'com.sun.star.frame.Desktop', remote_ctx)
doc = desktop.getCurrentComponent()

try:
    slides = doc.DrawPages
    n_slides = slides.getCount()
except Exception:
    print(_BEGIN)
    print(json.dumps({"error": "no presentation open"}))
    print(_END)
    sys.exit(0)

try:
    controller = doc.getCurrentController()
    active = controller.getCurrentPage()
    active_idx = None
    # ``is`` comparison is unreliable for UNO proxies — two
    # getXxx() calls can return distinct proxies pointing at the same
    # remote object. Compare by .Name (which is stable per slide) or
    # by the slide's Number property as a fallback.
    try:
        active_name = getattr(active, "Name", None)
    except Exception:
        active_name = None
    try:
        active_number = int(getattr(active, "Number", -1))
    except Exception:
        active_number = None
    for i in range(n_slides):
        s = slides.getByIndex(i)
        # Try the cheap proxy-identity check first (sometimes works),
        # then fall back to name/number comparison.
        try:
            if s is active:
                active_idx = i + 1
                break
        except Exception:
            pass
        try:
            if active_name is not None and getattr(s, "Name", None) == active_name:
                active_idx = i + 1
                break
        except Exception:
            pass
        try:
            if (active_number is not None and active_number > 0
                    and int(getattr(s, "Number", -2)) == active_number):
                active_idx = i + 1
                break
        except Exception:
            pass
    # Last resort: use the controller's CurrentPage.Number as the
    # 1-based slide index (LO sets this to the current slide number).
    if active_idx is None and active_number is not None and 1 <= active_number <= n_slides:
        active_idx = active_number
except Exception:
    active_idx = None

records = []
K = 20
for i in range(min(n_slides, K)):
    try:
        records.append({"index": i + 1,
                        **_slide_record(slides.getByIndex(i))})
    except Exception:
        continue

out = {
    "n_slides": n_slides,
    "active": active_idx,
    "slides": records,
    "elided": max(0, n_slides - K),
}
print(_BEGIN)
print(json.dumps(out))
print(_END)
'''


# ---------------------------------------------------------------------- #
# Rendering — parsed JSON → text block
# ---------------------------------------------------------------------- #

_ALIGN_INT_TO_NAME = {0: "left", 1: "right", 2: "justify", 3: "center",
                      4: "stretch"}


def _render_shape(sh: Dict[str, Any], depth: int = 0) -> str:
    """Render one shape record as a text block.

    ``depth`` = indent level (0=slide child, 1=in group, 2+=nested).
    Group children render inline so the planner sees their text —
    imported decks (Google Slides → pptx) wrap real title/body text in
    a GroupShape that a non-recursive render would hide.
    """
    indent = "  " * (depth + 1)
    parts = [
        f"{indent}[{sh.get('i', '?'):>2}] {sh.get('kind', '?'):<6}",
    ]
    name = sh.get("name") or ""
    if name:
        parts.append(f"'{name}'")
    if sh.get("x_cm") is not None and sh.get("y_cm") is not None:
        parts.append(f"pos=({sh['x_cm']}cm,{sh['y_cm']}cm)")
    if sh.get("w_cm") is not None and sh.get("h_cm") is not None:
        parts.append(f"size=({sh['w_cm']}cm,{sh['h_cm']}cm)")
    font = sh.get("font")
    if font:
        font_parts = []
        if font.get("name"):
            font_parts.append(str(font["name"]))
        if font.get("size") is not None:
            font_parts.append(f"{font['size']:.0f}pt")
        if font.get("bold"):
            font_parts.append("bold")
        if font.get("italic"):
            font_parts.append("italic")
        if font.get("underline"):
            font_parts.append("underline")
        if font.get("strike"):
            font_parts.append("strike")
        if font.get("color"):
            font_parts.append(str(font["color"]))
        if font.get("align") is not None:
            align_name = _ALIGN_INT_TO_NAME.get(font["align"])
            if align_name and align_name != "left":
                font_parts.append(align_name)
        if font_parts:
            parts.append(" ".join(font_parts))
    # Tables: rows × cols + first-cell preview.
    if sh.get("kind") == "table":
        r = sh.get("rows"); c = sh.get("cols")
        if r is not None and c is not None:
            parts.append(f"{r}×{c}")
    line = " ".join(parts)
    preview = sh.get("preview")
    if preview:
        text_indent = "  " * (depth + 2)
        line += f"\n{text_indent}cell[0,0] \"{preview}\""
    text = sh.get("text")
    paras = sh.get("paragraphs")
    text_indent = "  " * (depth + 2)
    if paras:
        # List each paragraph with its 1-based index for paragraph_index=.
        # A non-bullet paragraph immediately followed by bullets is a
        # section header → tag ``(title)``; bullets get ``•``. Lets the
        # planner count list items and skip headers when "line N" / "item
        # N" refers to the list itself.
        real_paras = [p for p in paras if p.get("i") != -1]
        n_real = len(real_paras)
        for idx, p in enumerate(paras):
            if p.get("i") == -1:
                line += f"\n{text_indent}{p.get('text', '...')}"
                continue
            i_ = p.get("i")
            t_ = p.get("text") or ""
            indent_ = p.get("indent") or 0
            bullet = p.get("bullet")
            if bullet is True:
                bullet_tag = " •"
            elif bullet is False:
                # Is the next real paragraph a bullet?
                next_is_bullet = False
                for j in range(idx + 1, len(paras)):
                    nxt = paras[j]
                    if nxt.get("i") == -1:
                        continue
                    next_is_bullet = (nxt.get("bullet") is True)
                    break
                bullet_tag = " (title)" if next_is_bullet else ""
            else:
                bullet_tag = ""
            indent_pad = "  " * int(indent_)
            line += (f"\n{text_indent}p{i_}{bullet_tag} "
                     f"{indent_pad}\"{t_}\"")
    elif text:
        line += f"\n{text_indent}\"{text}\""
    # Master-driven placeholders (slide-number, footer, date, …): hint
    # the planner to route through impress_set_master_color, not
    # impress_set_text_color.
    if sh.get("master_driven"):
        kind = sh.get("kind") or "?"
        # perceiver kind → impress_set_master_color role
        role_map = {"slide_num": "slide_number"}
        role = role_map.get(kind)
        if role:
            line += (f"\n{text_indent}→ master-driven; to recolor use "
                     f"impress_set_master_color(role='{role}')")
        else:
            line += (f"\n{text_indent}→ master-driven placeholder "
                     f"({kind}); color lives on the slide MASTER")
    # Render group children inline at depth+1; drop purely decorative
    # ones (no text, no font) to keep the block small.
    children = sh.get("children") or []
    if children:
        kept = []
        for c in children:
            c_kind = c.get("kind") or ""
            has_text = bool((c.get("text") or "").strip())
            has_font = bool(c.get("font"))
            # Keep text/font children; show all if the group has ≤4.
            if has_text or has_font or len(children) <= 4 \
                    or c_kind in ("group", "table", "title", "subtitle", "outline"):
                kept.append(c)
        if kept:
            child_lines = [_render_shape(c, depth + 1) for c in kept]
            line += "\n" + "\n".join(child_lines)
            skipped = len(children) - len(kept)
            if skipped > 0:
                line += (f"\n{'  ' * (depth + 2)}⋯ {skipped} decorative "
                         f"child shape{'s' if skipped != 1 else ''} collapsed")
    return line


def _slide_role_label(is_focus: bool, is_active: bool) -> str:
    """English role label for a slide header. Words not symbols so a 9B
    model never needs a legend."""
    tags = []
    if is_focus:
        tags.append("task target")
    if is_active:
        tags.append("currently viewed")
    if not tags:
        return ""
    return "  [" + " | ".join(tags) + "]"


_TEXT_KINDS = {"title", "subtitle", "outline", "txt"}
_DECOR_RE = re.compile(r"^[\d\W_]+$")


def _shape_is_off_canvas(sh: Dict[str, Any],
                          slide_w: Optional[float],
                          slide_h: Optional[float]) -> bool:
    """Off-canvas iff the top-left anchor is outside the slide rect.
    Anchor-only (not bbox-intersection) is intentional: template
    artifacts often anchor above the canvas (Y<0) with an overlapping
    bbox but are never user-visible task targets."""
    x = sh.get("x_cm")
    y = sh.get("y_cm")
    if x is None or y is None:
        return False
    if x < 0 or y < 0:
        return True
    if slide_w is not None and x >= slide_w:
        return True
    if slide_h is not None and y >= slide_h:
        return True
    return False


def _shape_is_empty_text(sh: Dict[str, Any]) -> bool:
    """Text-kind or generic ``shape`` record with no text — template
    residue / decorative CustomShape outlines, useless to the planner.

    Includes plain ``"shape"`` (untagged CustomShape) so empty ones that
    miss the BITMAP overclassify don't leak in as noise. ``"pic"`` is
    excluded even when empty — pictures stay first-class resize/move
    targets addressed by Name / shape_index.
    """
    if sh.get("kind") not in _TEXT_KINDS and sh.get("kind") != "shape":
        return False
    t = (sh.get("text") or "").strip()
    return not t


def _pick_title_candidate(shapes: List[Dict[str, Any]],
                           slide_h: Optional[float]
                           ) -> Optional[Dict[str, Any]]:
    """Host-side mirror of impress_ops_lib._impress_pick_text_shape_by_role
    (role='title') over the perceiver's shape records (no UNO calls).
    Picks the largest-font, top-positioned, non-decorative text shape.
    """
    if not shapes or not slide_h or slide_h <= 0:
        return None
    cands = []
    for sh in shapes:
        if sh.get("kind") not in _TEXT_KINDS:
            continue
        text = (sh.get("text") or "").strip()
        if not text or len(text) <= 2 or _DECOR_RE.match(text):
            continue
        font = sh.get("font") or {}
        try:
            size = float(font.get("size") or 0)
        except Exception:
            size = 0.0
        if size <= 0:
            continue
        y = sh.get("y_cm")
        cands.append({"rec": sh, "size": size,
                      "top": float(y) if y is not None else 0.0,
                      "text": text})
    if not cands:
        return None
    half = slide_h / 2.0
    top_zone = [c for c in cands if c["top"] < half] or cands
    top_zone.sort(key=lambda c: (-c["size"], c["top"]))
    best = top_zone[0]
    if best["size"] < 18.0:
        return None
    if len(top_zone) >= 2:
        runner = top_zone[1]
        if (abs(best["size"] - runner["size"]) < 3.0
                and runner["top"] < best["top"]):
            return runner["rec"]
    return best["rec"]


def _render_slide(rec: Dict[str, Any], *, is_focus: bool = False,
                  is_active: bool = False) -> List[str]:
    idx = rec.get("index")
    layout = rec.get("layout")
    bg = rec.get("background") or "-"
    shapes_all = rec.get("shapes", []) or []
    slide_w = rec.get("width_cm")
    slide_h = rec.get("height_cm")
    n_shapes = len(shapes_all)
    role = _slide_role_label(is_focus, is_active)
    head = (f"── Slide[{idx}]{role}  layout={layout}  bg={bg}  "
            f"shapes={n_shapes} ──")
    lines = [head]

    # Collapse off-canvas + empty-text shapes, but still count them in
    # the per-slide total (so "shapes=25" matches the UNO count);
    # collapsed ones are summarized at the tail.
    visible: List[Dict[str, Any]] = []
    off_idx: List[int] = []
    empty_idx: List[int] = []
    for sh in shapes_all:
        if _shape_is_off_canvas(sh, slide_w, slide_h):
            off_idx.append(sh.get("i") or 0)
            continue
        if _shape_is_empty_text(sh):
            empty_idx.append(sh.get("i") or 0)
            continue
        visible.append(sh)
    n_off = len(off_idx); n_empty = len(empty_idx)

    # Title-candidate hint (★): same heuristic as the op layer's
    # placeholder='title' fallback, so the planner can pick a Name
    # directly instead of guessing across the SP body.
    title_cand = _pick_title_candidate(shapes_all, slide_h)
    if title_cand is not None:
        font = title_cand.get("font") or {}
        size = font.get("size")
        size_str = f"{size:.0f}pt" if isinstance(size, (int, float)) else ""
        text = title_cand.get("text") or ""
        lines.append(
            f"  ★ title candidate: '{title_cand.get('name') or ''}'"
            + (f"  {size_str}" if size_str else "")
            + (f'  "{text}"' if text else "")
        )

    # Sort by visual position (top→bottom, then left→right) so "first
    # textbox / second textbox" matches a top-down read. Each shape
    # still prints its own ``[N]`` (UNO z-order), so shape_index=N is
    # unaffected — only render order changes.
    def _sort_key(sh):
        y = sh.get("y_cm"); x = sh.get("x_cm")
        return (float(y) if y is not None else 1e9,
                float(x) if x is not None else 1e9,
                int(sh.get("i") or 0))
    visible_sorted = sorted(visible, key=_sort_key)

    for sh in visible_sorted:
        lines.append(_render_shape(sh))

    # List collapsed shape indices so the planner can still target them
    # via shape_index= ("delete the icons" often hits empty-text
    # decorative shapes).
    if n_off or n_empty:
        bits = []
        if n_off:
            bits.append(f"{n_off} off-canvas (indices {off_idx})")
        if n_empty:
            bits.append(f"{n_empty} empty-text (indices {empty_idx})")
        lines.append(f"  ⋯ {' + '.join(bits)} collapsed")

    notes = rec.get("notes")
    if notes:
        lines.append(f"  notes: {notes!r}")
    return lines


def _render_slide_summary(rec: Dict[str, Any], *,
                          is_active: bool = False) -> str:
    """One-line summary for slides outside the focus set: layout, bg, and
    shape-kind census (so "slide 7 has a chart" is still visible without
    rendering every shape).
    """
    idx = rec.get("index")
    layout = rec.get("layout")
    bg = rec.get("background") or "-"
    shapes = rec.get("shapes", [])
    n = len(shapes)
    census = {}
    for sh in shapes:
        k = sh.get("kind") or "shape"
        census[k] = census.get(k, 0) + 1
    if census:
        kinds = sorted(census.items(), key=lambda x: -x[1])
        census_str = ", ".join(f"{c}×{k}" for k, c in kinds)
    else:
        census_str = "empty"
    role = _slide_role_label(False, is_active)
    return (f"── Slide[{idx}]{role}  layout={layout}  bg={bg}  "
            f"shapes={n} ({census_str}) ──")


def render_block(parsed: Dict[str, Any],
                 focus_slides: Optional[List[int]] = None) -> str:
    """Build the PRESENTATION STRUCTURE block.

    Args:
        parsed: JSON output of the inspect script.
        focus_slides: 1-based indices to render in full; others get a
            one-line summary. ``None`` → render every slide full.
    """
    if "error" in parsed:
        return (f"═══ PRESENTATION STRUCTURE (SlidePerceiver) ═══\n"
                f"(probe error: {parsed['error']})\n"
                f"═══════════════════════════════")
    n = parsed.get("n_slides", 0)
    active = parsed.get("active")
    # Deck dims off the first slide (all slides share page size). Needed
    # for "stretch to fill" / "full-bleed background" width/height kwargs.
    deck_dims = ""
    try:
        first = (parsed.get("slides") or [{}])[0]
        w = first.get("width_cm"); h = first.get("height_cm")
        if w is not None and h is not None:
            deck_dims = f" | slide_size={w}cm×{h}cm"
    except Exception:
        pass
    header_lines = [
        "═══════ PRESENTATION STRUCTURE (SlidePerceiver — UNO probe) ═══════",
        (f"{n} slide{'s' if n != 1 else ''} (1-based indexing)"
         + (f" | active=Slide {active}" if active else "")
         + (f" | task targets={sorted(set(focus_slides))}"
            if focus_slides else "")
         + deck_dims),
    ]
    lines = list(header_lines)
    focus_set = set(focus_slides or [])
    for rec in parsed.get("slides", []):
        idx = rec.get("index")
        is_active = (active is not None and idx == active)
        if focus_slides and idx not in focus_set:
            lines.append(_render_slide_summary(rec, is_active=is_active))
        else:
            is_focus = bool(focus_slides) and idx in focus_set
            lines.extend(_render_slide(rec, is_focus=is_focus,
                                       is_active=is_active))
    elided = parsed.get("elided", 0)
    if elided:
        lines.append(f"(+ {elided} more slide{'s' if elided != 1 else ''} not shown)")
    lines.append("══════════════════════════")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Focus extraction: pick which slides the task actually touches
# ---------------------------------------------------------------------- #

_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
    "eleventh": 11, "twelfth": 12,
}


def _regex_extract_focus(instruction: str,
                         n_slides: int,
                         active_idx: Optional[int]) -> Optional[List[int]]:
    """Regex / ordinal / pronoun parse. Returns:
      ``None`` — universal scope ("all slides" / "every page");
      ``[]``   — no explicit reference;
      list     — explicit indices.
    """
    if not instruction or n_slides <= 0:
        return []
    text = instruction.lower()
    # Strip quoted spans before the digit scan so e.g. `Add "Page 1" ...
    # on Slide 2` matches only Slide 2, not the quoted "Page 1". Keep
    # the original text for "all" / pronoun checks.
    digit_text = re.sub(
        r"\"[^\"]*\"|'[^']*'|“[^”]*”|‘[^’]*’",
        " __Q__ ", text)
    if re.search(r"\b(?:all|every|each)\s+(?:slide|page)s?\b", text):
        return None
    indices: set = set()
    # Numeric refs "slide N" / "page N" + ranges "slide 2 to 5".
    for m in re.finditer(
            r"(?:slide|page)s?\s+(\d+)(?:\s*(?:-|to|–|—|through)\s*(\d+))?",
            digit_text):
        a = int(m.group(1))
        b = int(m.group(2)) if m.group(2) else a
        lo, hi = min(a, b), max(a, b)
        for i in range(lo, hi + 1):
            if 1 <= i <= n_slides:
                indices.add(i)
    # Comma/"and" lists: "slides 2, 3, 5", "slides 2 and 4".
    for m in re.finditer(
            r"(?:slide|page)s?\s+(\d+(?:\s*(?:,|\band\b)\s*\d+)+)",
            digit_text):
        for nstr in re.split(r"\s*(?:,|\band\b)\s*", m.group(1)):
            try:
                i = int(nstr)
                if 1 <= i <= n_slides:
                    indices.add(i)
            except ValueError:
                pass
    # ("on slide 1 and on slide 2" already covered by the first loop.)
    # Ordinal words ("first slide" / "third page")
    for word, num in _ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\s+(?:slide|page)\b", text):
            if 1 <= num <= n_slides:
                indices.add(num)
    # "last slide" / "last N slides"
    if re.search(r"\blast\s+(?:slide|page)\b", text):
        indices.add(n_slides)
    m = re.search(
        r"\blast\s+(\d+|two|three|four|five)\s+(?:slide|page)s\b", text)
    if m:
        nstr = m.group(1)
        k = {"two": 2, "three": 3, "four": 4, "five": 5}.get(
            nstr, int(nstr) if nstr.isdigit() else 0)
        for off in range(k):
            i = n_slides - off
            if 1 <= i <= n_slides:
                indices.add(i)
    # "this/current/the slide" → active
    if (active_idx is not None
            and re.search(r"\b(?:this|current|the)\s+(?:slide|page)\b", text)):
        indices.add(active_idx)
    return sorted(indices)


def _slide_brief_for_llm(rec: Dict[str, Any]) -> str:
    """One-line per-slide summary fed to the LLM focus picker."""
    idx = rec.get("index")
    layout = rec.get("layout", "?")
    bg = rec.get("background") or "-"
    shapes = rec.get("shapes", []) or []
    census: Dict[str, int] = {}
    for sh in shapes:
        k = sh.get("kind") or "shape"
        census[k] = census.get(k, 0) + 1
    census_str = ", ".join(f"{c}×{k}" for k, c in sorted(
        census.items(), key=lambda x: -x[1])) or "empty"
    # first text preview, for semantic queries
    first_text = ""
    for sh in shapes:
        if sh.get("text"):
            first_text = sh["text"]
            break
    line = f"Slide {idx}: layout={layout} bg={bg} {census_str}"
    if first_text:
        line += f" first_text={first_text!r}"
    return line


def _llm_extract_focus(instruction: str,
                       parsed: Dict[str, Any],
                       call_llm,
                       model: Optional[str],
                       logger: Optional[logging.Logger] = None) -> Optional[List[int]]:
    """LLM fallback for semantic slide refs the regex misses ("the title
    slide" / "the slide before the conclusion").

    Same return shape as ``_regex_extract_focus``: ``None`` = all,
    ``[]`` = undecided, ``[...]`` = explicit indices.
    """
    n = parsed.get("n_slides", 0)
    if n <= 0:
        return []
    briefs = "\n".join(
        _slide_brief_for_llm(rec) for rec in parsed.get("slides", []))
    active = parsed.get("active")
    sys_msg = (
        "You pick which SLIDE indices a presentation task targets.\n"
        "Output JSON only, no prose. Exactly ONE of these THREE forms:\n"
        '  "all"         — task applies to EVERY slide. Output the\n'
        "                  literal 4-character string ``\"all\"`` — do\n"
        "                  NOT enumerate the indices.\n"
        '  [N, M, ...]   — a SUBSET of 1-based SLIDE indices that the\n'
        "                  instruction explicitly names.\n"
        '  []            — no slide reference; task targets the active slide.\n'
        "\n"
        'ALWAYS use "all" (not an enumerated list) when the instruction\n'
        'mentions "all slides" / "every slide" / "each slide" / "my\n'
        'slides" / "all my slides" / "the entire presentation" / "the\n'
        'whole deck" / "every page" — regardless of how many slides the\n'
        'deck has. Enumerating wastes tokens and risks truncation.\n'
        "\n"
        "STRICT scoping rules:\n"
        "  * Only return slide indices when the instruction names a SLIDE\n"
        "    or PAGE: 'slide 3', 'page 4', 'the title slide', 'the\n"
        "    Features slide', 'the last slide'.\n"
        "  * Ordinals followed by 'line' / 'paragraph' / 'bullet' / 'row'\n"
        "    / 'column' / 'item' / 'cell' / 'textbox' / 'text box' /\n"
        "    'shape' / 'image' / 'character' / 'word' / 'point' /\n"
        "    'entry' are SUB-SLIDE references — they live WITHIN ONE\n"
        "    shape, NOT across slides. Return [] in that case (the\n"
        "    active slide is the target).\n"
        '    "first and second line" → [] (paragraphs within a textbox)\n'
        '    "the third row" → [] (row inside a table)\n'
        '    "the second textbox" → [] (shape on the active slide)\n'
        '    "the first slide" → [1]\n'
        '    "second page" → [2]\n'
        '    "all slides" / "all my slides" / "every slide" → "all"\n'
        "  * Verb phrases like 'I am checking ...' / 'I want to ...'\n"
        "    that describe context but never bind to a slide → [].\n"
        "  * Prefer the smallest set the task literally needs."
    )
    user_msg = (
        f"Task instruction:\n{instruction}\n\n"
        f"Presentation has {n} slide(s); active slide is "
        f"{active if active else 'unknown'}.\n\n"
        f"Per-slide brief (kind census + first text preview):\n{briefs}\n\n"
        "Output the JSON value now."
    )
    try:
        resp = call_llm(
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            model=model,
            temperature=0.0,
            max_tokens=64,
        )
    except Exception as e:
        if logger:
            logger.info("[SlidePerceiver] focus-LLM call failed: %s", e)
        return []
    if not isinstance(resp, str):
        return []
    txt = resp.strip()
    # Strip code fences
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", txt, re.DOTALL)
    if m:
        txt = m.group(1).strip()
    if txt.strip().strip('"').lower() == "all":
        return None
    try:
        val = json.loads(txt)
    except Exception:
        if logger:
            logger.info("[SlidePerceiver] focus-LLM unparseable: %r",
                        txt[:120])
        return []
    if isinstance(val, str) and val.lower() == "all":
        return None
    if not isinstance(val, list):
        return []
    out = []
    for v in val:
        try:
            i = int(v)
            if 1 <= i <= n:
                out.append(i)
        except (ValueError, TypeError):
            continue
    uniq = sorted(set(out))
    # If the LLM enumerated every slide, treat as "all" — guards against
    # a truncated list being read as a narrow subset.
    if n > 0 and len(uniq) >= n:
        return None
    return uniq


def extract_focus_slides(instruction: Optional[str],
                         parsed: Dict[str, Any],
                         *,
                         call_llm=None,
                         model: Optional[str] = None,
                         logger: Optional[logging.Logger] = None) -> Optional[List[int]]:
    """Decide which slide indices to render in full.

    1. Regex/ordinal/pronoun parse (fast). 2. On empty, fall back to one
    LLM call for semantic refs (may also say "all"). 3. The active slide
    is always merged in (implicit "this slide" tasks target it).

    Returns ``None`` → no filter (all slides full); ``[]`` → no info,
    caller decides; list → sorted focus set bounded to [1..n_slides].
    """
    n = parsed.get("n_slides", 0)
    active = parsed.get("active")
    if n <= 0:
        return []
    regex_result = _regex_extract_focus(instruction or "", n, active)
    if regex_result is None:
        return None  # explicit "all"
    if regex_result:
        focus = set(regex_result)
        if active is not None:
            focus.add(active)
        return sorted(i for i in focus if 1 <= i <= n)
    # Regex empty → LLM
    if call_llm is not None and instruction:
        llm_result = _llm_extract_focus(
            instruction, parsed, call_llm, model, logger=logger)
        if llm_result is None:
            return None  # LLM says all
        if llm_result:
            focus = set(llm_result)
            if active is not None:
                focus.add(active)
            return sorted(i for i in focus if 1 <= i <= n)
    # Last resort: the active slide.
    if active is not None:
        return [active]
    return []


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #

def probe(env, logger: Optional[logging.Logger] = None) -> Optional[Dict[str, Any]]:
    """Run the inspector script on the VM, return parsed JSON (or ``None``
    after retries). Side-effect-free; caller renders/caches.
    """
    if env is None or not hasattr(env, "controller"):
        if logger:
            logger.warning("[SlidePerceiver] env has no controller")
        return None
    for attempt in range(1, 6):
        try:
            resp = env.controller.run_python_script(INSPECT_SCRIPT)
        except Exception as e:  # noqa: BLE001
            if logger:
                logger.info("[SlidePerceiver] probe attempt %d failed: %s",
                            attempt, e)
            continue
        if not resp or resp.get("status") != "success":
            if logger:
                logger.info("[SlidePerceiver] probe attempt %d non-success: %r",
                            attempt, (resp or {}).get("status"))
            continue
        out = resp.get("output") or ""
        begin = out.find("===SLIDE_INSPECT_BEGIN===")
        end = out.find("===SLIDE_INSPECT_END===")
        if begin < 0 or end <= begin:
            continue
        try:
            payload = out[begin + len("===SLIDE_INSPECT_BEGIN==="):end].strip()
            return json.loads(payload)
        except Exception as e:
            if logger:
                logger.info(
                    "[SlidePerceiver] probe parse failed (attempt %d): %s",
                    attempt, e)
            continue
    if logger:
        logger.warning("[SlidePerceiver] probe failed after 5 attempts")
    return None


def run_inspect(env, logger: Optional[logging.Logger] = None,
                results_dir: Optional[str] = None,
                step_idx: Optional[int] = None,
                focus_slides: Optional[List[int]] = None) -> Optional[str]:
    """probe + render in one call, for callers that don't cache the
    parsed JSON / focus. Prefer explicit ``probe()`` + ``render_block()``
    when you want to extract and cache focus_slides once.

    ``focus_slides``: 1-based indices to render in full (others get a
    one-line summary); ``None`` → all full.
    """
    parsed = probe(env, logger=logger)
    if parsed is None:
        return None
    block = render_block(parsed, focus_slides=focus_slides)
    if results_dir and step_idx is not None:
        try:
            d = os.path.join(results_dir, "slide_perceiver")
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, f"step_{step_idx:03d}_block.txt"),
                      "w") as f:
                f.write(block)
        except Exception:
            pass
    return block


def should_probe(domain: Optional[str]) -> bool:
    """Whether SlidePerceiver should run for this domain."""
    return (domain or "").lower() == "libreoffice_impress"

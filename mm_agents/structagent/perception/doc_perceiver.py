"""DocPerceiver — structural probe for libreoffice_writer tasks on the UNO route.

Document-model counterpart to the GUI Perceiver: tells the planner "what's in
the doc" (paragraphs with content-index/style/font/color/alignment/spacing/
comma_count/snippet, plus tables and view-cursor position).

Fires at task start and after every completed cli_run_uno mutation. Runs via
``env.controller.run_python_script`` against the OSWorld VM; read-only (no
``doc.store()``). Output is a compact text block for the planner prompt.
"""
from __future__ import annotations
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------- #
# UNO probe script (runs in the VM via run_python_script)
# ---------------------------------------------------------------------- #
# Reads the live LibreOffice doc via the UNO socket and prints a JSON blob
# between ``===DOC_INSPECT_BEGIN===`` / ``===DOC_INSPECT_END===`` sentinels.
# Read-only: walks paragraphs (body + text frames) and tables.
#
# Output schema (JSON):
#   {
#     "globals":    {"page_width": int, "page_height": int,
#                    "left_margin": int, "right_margin": int,    # all 1/100 mm
#                    "default_font": str|None,
#                    "default_size": float|None},
#     "paragraphs": [{raw_idx, ci, style, text, length, comma_count,
#                     para_adjust, line_height, line_mode,
#                     font_name, font_size, color, bold, italic,
#                     frame_idx?}, ...],
#     "tables":     [{raw_idx, rows, cols, first_row: [str, ...]}, ...],
#     "view_cursor":{"in_body": bool},
#   }
INSPECT_SCRIPT = r"""
import uno
import json
import sys
import subprocess
import time

_BEGIN = "===DOC_INSPECT_BEGIN==="
_END   = "===DOC_INSPECT_END==="

# Mirrors mm_agents/cli_run_uno_helpers._CONNECT_BLOCK: OSWorld VMs
# don't start LO with --accept by default, so the first resolve() will
# fail with "Connection refused". The fallback Popens soffice with
# --accept= which BINDS the listener to the already-running LO
# instance (idempotent — second soffice exits on the existing
# instance's lockfile after enabling the socket).
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
    text = doc.Text
except Exception as e:
    print(_BEGIN)
    print(json.dumps({"error": f"UNO connect failed: {type(e).__name__}: {e}"}))
    print(_END)
    sys.exit(0)

# --- Globals: page size + margins (default page style) + default font ---
globals_rec = {"page_width": None, "page_height": None,
               "left_margin": None, "right_margin": None,
               "default_font": None, "default_size": None}
try:
    page_styles = doc.StyleFamilies.getByName('PageStyles')
    default_page = None
    for n in page_styles.getElementNames():
        if 'Default' in n or n == 'Standard':
            default_page = page_styles.getByName(n); break
    if default_page is None and page_styles.getCount() > 0:
        default_page = page_styles.getByIndex(0)
    if default_page is not None:
        globals_rec["page_width"]    = int(default_page.Width)
        globals_rec["page_height"]   = int(default_page.Height)
        globals_rec["left_margin"]   = int(default_page.LeftMargin)
        globals_rec["right_margin"]  = int(default_page.RightMargin)
except Exception:
    pass
try:
    para_styles = doc.StyleFamilies.getByName('ParagraphStyles')
    df = para_styles.getByName('Default Paragraph Style')
    globals_rec["default_font"] = df.CharFontName
    globals_rec["default_size"] = float(df.CharHeight)
except Exception:
    pass

# --- Paragraph walker (body + frames) ---
def _para_record(p, raw_idx, ci, frame_idx=None):
    s = p.getString()
    is_blank = not s.strip()
    rec = {"raw_idx": raw_idx,
           "ci": None if is_blank else ci,
           "style": p.ParaStyleName,
           "text": s,
           "length": len(s),
           "comma_count": s.count(',')}
    if frame_idx is not None:
        rec["frame_idx"] = frame_idx
    try: rec["para_adjust"] = int(p.ParaAdjust)
    except Exception: rec["para_adjust"] = None
    try:
        ls = p.ParaLineSpacing
        rec["line_height"] = int(ls.Height) if ls else None
        rec["line_mode"]   = int(ls.Mode)   if ls else None
    except Exception:
        rec["line_height"] = None; rec["line_mode"] = None
    # First portion's char attrs (proxy for paragraph default)
    rec["font_name"] = None; rec["font_size"] = None
    rec["color"] = None; rec["bold"] = False; rec["italic"] = False
    try:
        portions = list(p.createEnumeration())
        fp = portions[0] if portions else None
        if fp is not None:
            try: rec["font_name"] = fp.CharFontName
            except Exception: pass
            try: rec["font_size"] = float(fp.CharHeight)
            except Exception: pass
            try: rec["color"] = int(fp.CharColor)
            except Exception: pass
            try: rec["bold"] = float(fp.CharWeight) >= 150.0
            except Exception: pass
            try: rec["italic"] = int(fp.CharPosture) > 0
            except Exception: pass
    except Exception:
        pass
    return rec

paragraphs = []
tables = []
raw_idx = 0
ci = 0

def _cell_name(c, r):
    s = ''
    n = c
    while True:
        s = chr(ord('A') + (n % 26)) + s
        n = n // 26 - 1
        if n < 0: break
    return s + str(r + 1)

for el in text.createEnumeration():
    if el.supportsService('com.sun.star.text.Paragraph'):
        rec = _para_record(el, raw_idx, ci)
        if rec["ci"] is not None:
            ci += 1
        paragraphs.append(rec)
        raw_idx += 1
    elif el.supportsService('com.sun.star.text.TextTable'):
        rows = el.Rows.getCount(); cols = el.Columns.getCount()
        first_row = []
        try:
            for c in range(cols):
                try:
                    first_row.append(el.getCellByName(_cell_name(c, 0)).getString()[:25])
                except Exception:
                    first_row.append('?')
        except Exception:
            pass
        tables.append({"raw_idx": raw_idx, "rows": rows, "cols": cols,
                       "first_row": first_row})
        raw_idx += 1

# Walk paragraphs inside text frames (floating-content docs like 0a0faba3)
try:
    n_frames = doc.TextFrames.getCount()
    for fi in range(n_frames):
        tf = doc.TextFrames.getByIndex(fi)
        f_ci = 0
        for el in tf.createEnumeration():
            if el.supportsService('com.sun.star.text.Paragraph'):
                rec = _para_record(el, -1, f_ci, frame_idx=fi)
                if rec["ci"] is not None:
                    f_ci += 1
                paragraphs.append(rec)
except Exception:
    pass

# View cursor
view_cursor = {"in_body": False, "raw_para_idx": None}
try:
    vc = doc.getCurrentController().getViewCursor()
    view_cursor["in_body"] = (vc.getStart().Text == text)
except Exception:
    pass

print(_BEGIN)
print(json.dumps({"globals": globals_rec,
                  "paragraphs": paragraphs,
                  "tables": tables,
                  "view_cursor": view_cursor},
                 ensure_ascii=False))
print(_END)
"""


# ---------------------------------------------------------------------- #
# Agent-side parser + collapser
# ---------------------------------------------------------------------- #

# Per-paragraph fingerprint for collapse-eligibility. Length/text excluded
# (they vary even within a coherent block).
def _fingerprint(p: Dict[str, Any]) -> Tuple:
    return (
        p.get("style"),
        p.get("font_name"),
        # bucket font size to 0.1pt so float noise doesn't break runs
        round(p["font_size"], 1) if p.get("font_size") else None,
        p.get("color"),
        p.get("para_adjust"),
        # bucket line height to 5/100mm so 100/100 align but 100/150 differ
        (p["line_height"] // 5 * 5) if p.get("line_height") else None,
        p.get("line_mode"),
        bool(p.get("bold")),
        bool(p.get("italic")),
        p.get("comma_count"),
    )


_ADJUST_NAMES = {0: "L", 1: "R", 2: "BLK", 3: "C", 4: "STR"}  # ParagraphAdjust enum


def _fmt_attrs(p: Dict[str, Any]) -> str:
    style = (p.get("style") or "?")[:18]
    font = (p.get("font_name") or "-")[:14]
    sz = p.get("font_size")
    sz_s = f"{sz:.0f}pt" if sz else "-"
    col = p.get("color")
    col_s = f"#{col:06X}" if col is not None and col >= 0 else "-"
    adj = p.get("para_adjust")
    adj_s = _ADJUST_NAMES.get(adj, "-") if adj is not None else "-"
    lh = p.get("line_height")
    lh_s = f"{lh}" if lh else "-"
    bi = ("B" if p.get("bold") else "-") + ("I" if p.get("italic") else "-")
    cc = p.get("comma_count", 0)
    return f"{style} {font} {sz_s} col={col_s} {adj_s} lh={lh_s} {bi} cc={cc}"


def _snippet(p: Dict[str, Any], n: int = 60) -> str:
    s = p.get("text") or ""
    return s[:n].replace("\n", " ").replace("\t", "→")


def _render_paragraphs(paragraphs: List[Dict[str, Any]]) -> List[str]:
    """Collapse blank runs to ``(×K blank)`` and consecutive same-fingerprint
    paragraphs to a range header + indented text list."""
    lines: List[str] = []
    i = 0
    body = [p for p in paragraphs if "frame_idx" not in p]
    while i < len(body):
        p = body[i]
        if p.get("ci") is None:    # blank
            j = i
            while j < len(body) and body[j].get("ci") is None and "frame_idx" not in body[j]:
                j += 1
            k = j - i
            if k == 1:
                lines.append(f"  [{p['raw_idx']:3d}]  -   (blank)")
            else:
                lines.append(
                    f"  [{p['raw_idx']:3d}-{body[j-1]['raw_idx']}] (×{k} blank)"
                )
            i = j
            continue
        fp = _fingerprint(p)
        j = i + 1
        while (j < len(body)
               and body[j].get("ci") is not None
               and _fingerprint(body[j]) == fp):
            j += 1
        run = body[i:j]
        if len(run) == 1:
            lines.append(
                f"  [{p['raw_idx']:3d}] ci={p['ci']:2d} {_fmt_attrs(p)} "
                f"l={p['length']:4d} | {_snippet(p)!r}"
            )
        else:
            ci_lo, ci_hi = run[0]['ci'], run[-1]['ci']
            lines.append(
                f"  [{run[0]['raw_idx']:3d}-{run[-1]['raw_idx']}] "
                f"ci={ci_lo}-{ci_hi} {_fmt_attrs(p)} (×{len(run)} same-fp):"
            )
            for q in run:
                lines.append(f"      [{q['raw_idx']:3d}] l={q['length']:4d} | {_snippet(q)!r}")
        i = j

    # Frame paragraphs — listed separately (rare; floating text)
    frame_paras = [p for p in paragraphs if "frame_idx" in p]
    if frame_paras:
        lines.append("")
        lines.append("  ── Inside TextFrames (floating content) ──")
        for p in frame_paras:
            lines.append(
                f"    [frame {p['frame_idx']}] {_fmt_attrs(p)} "
                f"l={p['length']:4d} | {_snippet(p)!r}"
            )
    return lines


def _emu_to_inch(emu: Optional[int]) -> float:
    """UNO Width/Height are 1/100 mm, not EMU (the name is legacy): 1 inch =
    2540, so divide by 2540, not 914400."""
    if emu is None: return 0.0
    return emu / 2540.0


def render_block(parsed: Dict[str, Any]) -> str:
    """Format the parsed JSON into a planner-ready text block."""
    if "error" in parsed:
        return f"DOC STRUCTURE — probe failed: {parsed['error']}"
    g = parsed.get("globals") or {}
    pp = parsed.get("paragraphs") or []
    tt = parsed.get("tables") or []
    vc = parsed.get("view_cursor") or {}

    pw = _emu_to_inch(g.get("page_width"))
    ph = _emu_to_inch(g.get("page_height"))
    lm = _emu_to_inch(g.get("left_margin"))
    rm = _emu_to_inch(g.get("right_margin"))

    n_content = sum(1 for p in pp if "frame_idx" not in p and p.get("ci") is not None)
    n_blank = sum(1 for p in pp if "frame_idx" not in p and p.get("ci") is None)
    n_frame = sum(1 for p in pp if "frame_idx" in p)

    lines: List[str] = []
    lines.append("═══════ DOCUMENT STRUCTURE (DocPerceiver — UNO probe) ═══════")
    lines.append(
        f"Page: {pw:.2f}\" x {ph:.2f}\", margins L={lm:.2f}\" R={rm:.2f}\"  "
        f"(1/100mm units: pw={g.get('page_width')}, lm={g.get('left_margin')}, "
        f"text_width={(g.get('page_width') or 0) - (g.get('left_margin') or 0) - (g.get('right_margin') or 0)})"
    )
    df = g.get("default_font") or "(inherits)"
    ds = g.get("default_size")
    ds_s = f"{ds:.1f}pt" if ds else "-"
    lines.append(f"Default font: {df} {ds_s}")
    lines.append(
        f"Paragraphs: {len(pp) - n_frame} body (content: {n_content}, blank: {n_blank}); "
        f"Tables: {len(tt)}; TextFrame-paragraphs: {n_frame}"
    )
    lines.append("")
    lines.append("── Paragraphs (raw_idx|ci|style font sz color adjust lh BI cc|len|text) ──")
    lines.extend(_render_paragraphs(pp))
    if tt:
        lines.append("")
        lines.append("── Tables ──")
        for ti, t in enumerate(tt):
            row0 = t.get("first_row") or []
            row0_disp = [c[:20] for c in row0]
            lines.append(
                f"  [{ti}] raw_idx={t['raw_idx']} {t['rows']}x{t['cols']}: row0={row0_disp}"
            )
    if vc.get("in_body") is False:
        lines.append("")
        lines.append("── ViewCursor ── outside body (in header/footer/frame/cell)")
    lines.append("══════════════════════════════════════════════════════════════")
    return "\n".join(lines)


# ---------------------------------------------------------------------- #
# Public entry points
# ---------------------------------------------------------------------- #

def _dump_artifacts(results_dir: Optional[str], step_idx: Optional[int],
                    raw_stdout: str, parsed: Optional[Dict[str, Any]],
                    block: Optional[str],
                    logger: Optional[logging.Logger]) -> None:
    """Persist per-step probe I/O under
    ``<results_dir>/doc_perceiver/step_NNN_inspect_{raw,parsed,rendered}.*``
    (raw stdout / extracted JSON / rendered planner block) for offline
    debugging. Any error logs and returns; never raises."""
    if not results_dir:
        return
    try:
        out_dir = os.path.join(results_dir, "doc_perceiver")
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"step_{(step_idx + 1):03d}" if step_idx is not None else "step_xxx"
        # always dump raw stdout (diagnoses probe-script crashes)
        with open(os.path.join(out_dir, f"{prefix}_inspect_raw.txt"),
                  "w", encoding="utf-8") as f:
            f.write(raw_stdout or "")
        if parsed is not None:
            with open(os.path.join(out_dir, f"{prefix}_inspect_parsed.json"),
                      "w", encoding="utf-8") as f:
                json.dump(parsed, f, ensure_ascii=False, indent=2)
        if block:
            with open(os.path.join(out_dir, f"{prefix}_inspect_rendered.txt"),
                      "w", encoding="utf-8") as f:
                f.write(block)
    except Exception as e:
        if logger:
            logger.info("[DocPerceiver] dump failed: %s", e)


def _attempt_inspect(env, logger: Optional[logging.Logger]
                     ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """One probe attempt: run the script in the VM, return (raw_stdout, parsed).

    parsed is None when the response can't be parsed (no sentinel / bad JSON).
    A parsed dict with an ``"error"`` key means the script ran but its UNO
    connect failed — typical at task start before LO has bound port 2002.
    """
    try:
        resp = env.controller.run_python_script(INSPECT_SCRIPT)
    except Exception as e:
        if logger: logger.info("[DocPerceiver] run_python_script raised: %s", e)
        return (f"<exception: {e}>", None)
    if not isinstance(resp, dict):
        return (f"<unexpected resp type: {type(resp).__name__}>", None)
    if resp.get("status") != "success":
        return (str(resp.get("output") or resp.get("error"))[:2000], None)
    out = resp.get("output") or ""
    BEG, END = "===DOC_INSPECT_BEGIN===", "===DOC_INSPECT_END==="
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
                retry_sleep_seconds: float = 2.0) -> Optional[str]:
    """Run the UNO probe in the VM and return a planner-ready text block, or
    None on failure.

    Retries transient UNO connect failures: at task start LO is still binding
    port 2002, so the first probe usually gets "Connection refused". On
    persistent failure returns None (no DOCUMENT STRUCTURE block — less
    misleading to the planner than a "probe failed" block).

    With ``results_dir`` set, persists per-step artifacts (see _dump_artifacts).
    """
    if env is None or not hasattr(env, "controller"):
        if logger: logger.info("[DocPerceiver] env.controller unavailable")
        return None

    last_raw = ""
    last_parsed: Optional[Dict[str, Any]] = None
    for attempt in range(max_retries):
        raw, parsed = _attempt_inspect(env, logger)
        last_raw, last_parsed = raw, parsed
        # Success: parsed JSON with no "error" key.
        if parsed is not None and "error" not in parsed:
            block = render_block(parsed)
            npara = len(parsed.get("paragraphs") or [])
            ntab = len(parsed.get("tables") or [])
            if logger:
                logger.info(
                    "[DocPerceiver] probe ok (attempt %d/%d): "
                    "%d paragraphs, %d tables, %d chars block",
                    attempt + 1, max_retries, npara, ntab, len(block),
                )
            _dump_artifacts(results_dir, step_idx, raw, parsed, block, logger)
            return block
        # Retry-eligible: connect refused, missing sentinel, or JSON parse fail.
        err_msg = None
        if parsed is not None and "error" in parsed:
            err_msg = parsed["error"]
        if logger:
            logger.info(
                "[DocPerceiver] probe attempt %d/%d failed: %s — sleeping %.1fs",
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

    # Retries exhausted — dump the last attempt and return None ("no doc
    # structure" rather than "probe failed").
    if logger:
        logger.info("[DocPerceiver] probe gave up after %d attempts", max_retries)
    _dump_artifacts(results_dir, step_idx, last_raw, last_parsed, None, logger)
    return None


# Writer only; calc/impress have their own data models needing a different probe.
_DOMAINS_WITH_DOC_PROBE = {"libreoffice_writer"}


def should_probe(domain: Optional[str]) -> bool:
    return (domain or "").lower() in _DOMAINS_WITH_DOC_PROBE

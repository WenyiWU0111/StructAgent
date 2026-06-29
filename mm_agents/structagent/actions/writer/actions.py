"""LibreOffice Writer structured actions — the Writer counterpart of
``calc_actions`` / ``impress_actions``.

The actor emits a flat-JSON ``writer_*`` action; this module
deterministically builds the ``op_*`` runtime script (no UNO Python
written by the model). It is the first piece of the Writer
structured-action subsystem; today it exposes one action:

  * ``writer_paste_at`` — paste the clipboard at a precise document
    location (by paragraph index, or after a heading).

Public surface (mirrors ``calc_actions``):
  * :data:`WRITER_ACTION_NAMES` — membership gate.
  * :func:`step_to_wire` — decomposer JSON dict → AST-parseable wire
    string.
  * :func:`action_to_runtime_script` — action + inputs → VM script.
  * :func:`action_schema_block` — prompt block describing the actions,
    injected into the actor's system prompt for the
    ``libreoffice_writer`` domain.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# Spec-driven actions: action_name -> (op_func, required, optional).
# A generic wire/build pair handles all of them — the actor emits flat
# JSON kwargs, the framework renders the op_* call deterministically.
_WRITER_OP_SPECS: Dict[str, tuple] = {
    "writer_line_spacing":      ("op_line_spacing",
                                 ("multiplier",), ("target_indices",)),
    "writer_font_size":         ("op_font_size",
                                 ("size",), ("pattern", "target_indices")),
    "writer_set_font":          ("op_set_font",
                                 ("font_name",), ("target_indices",)),
    "writer_default_font":      ("op_default_font",
                                 ("font_name",), ()),
    "writer_alignment":         ("op_alignment",
                                 ("alignment",), ("target_indices",)),
    "writer_case_conversion":   ("op_case_conversion",
                                 ("mode",), ("target_indices",)),
    "writer_strikethrough":     ("op_strikethrough",
                                 (), ("target_indices", "pattern")),
    "writer_insert_page_break": ("op_insert_page_break",
                                 ("after_paragraph_index",), ()),
    "writer_page_numbers":      ("op_page_numbers",
                                 (), ("position",)),
    "writer_capitalize_words":  ("op_capitalize_words",
                                 (), ("target_indices",)),
    "writer_remove_highlighting": ("op_remove_highlighting",
                                 (), ("target_indices",)),
    "writer_subscript_text":    ("op_subscript_text",
                                 ("pattern",), ("target_indices",)),
    "writer_insert_table":      ("op_insert_table",
                                 ("rows", "cols"), ()),
    "writer_delete_table":      ("op_delete_table",
                                 (), ("table_index", "after_heading")),
    "writer_export_pdf":        ("op_export_pdf",
                                 (), ("output_path", "output_filename")),
    "writer_tabstops_split_align": ("op_tabstops_split_align",
                                 (), ("n_words_before_tab",
                                      "right_tab_pos", "target_indices")),
    "writer_set_color":         ("op_set_color",
                                 ("color",),
                                 ("pattern", "walk_tables",
                                  "target_indices")),
    "writer_find_replace":      ("op_find_replace",
                                 ("mode",),
                                 ("pattern", "replacement",
                                  "dedup_pattern", "target_indices")),
    "writer_insert_image":      ("op_insert_image",
                                 ("image_path",),
                                 ("width_cm", "height_cm")),
    "writer_insert_paragraph_breaks": ("op_insert_paragraph_breaks",
                                 (), ("target_idx",)),
    "writer_text_to_table":     ("op_text_to_table",
                                 ("start_index", "end_index"),
                                 ("separator",)),
    "writer_cross_reference":   ("op_cross_reference",
                                 ("citation_text", "marker"), ()),
}

WRITER_ACTION_NAMES: frozenset = frozenset(
    {"writer_paste_at"} | set(_WRITER_OP_SPECS))


# --------------------------------------------------------------------
# wire rendering  (decomposer JSON  →  action_name(k=v, ...) string)
# --------------------------------------------------------------------

def _kw_wire(action_name: str, kwargs: Dict[str, Any]) -> str:
    """Render ``action_name(k=v, ...)``. Skips None; ``repr`` for
    JSON-safe round-trip through the AST parser."""
    parts = [f"{k}={v!r}" for k, v in kwargs.items() if v is not None]
    return f"{action_name}({', '.join(parts)})"


def _paste_at_kwargs(step: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate + normalize writer_paste_at inputs. Exactly one of
    ``paragraph_index`` / ``after_heading`` is required."""
    pidx = step.get("paragraph_index")
    heading = step.get("after_heading") or step.get("heading")
    if pidx is not None:
        try:
            pidx = int(pidx)
        except (TypeError, ValueError):
            return None
        return {"paragraph_index": pidx}
    if heading and str(heading).strip():
        return {"after_heading": str(heading).strip()}
    return None


def _wire_paste_at(step: Dict[str, Any]) -> Optional[str]:
    kwargs = _paste_at_kwargs(step)
    if kwargs is None:
        return None
    return _kw_wire("writer_paste_at", kwargs)


def _spec_kwargs(action: str, src: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Collect a spec-driven action's kwargs from ``src`` (decomposer
    step or build inputs). Returns None when a required kwarg is
    missing/empty."""
    _, required, optional = _WRITER_OP_SPECS[action]
    kw: Dict[str, Any] = {}
    for k in required:
        v = src.get(k)
        if v is None or v == "":
            return None
        kw[k] = v
    for k in optional:
        if src.get(k) is not None:
            kw[k] = src[k]
    return kw


def _wire_spec(step: Dict[str, Any]) -> Optional[str]:
    """Generic wire renderer for every spec-driven writer action."""
    action = step.get("action")
    kwargs = _spec_kwargs(action, step)
    if kwargs is None:
        return None
    return _kw_wire(action, kwargs)


_WIRE_HANDLERS = {
    "writer_paste_at": _wire_paste_at,
}
_WIRE_HANDLERS.update({name: _wire_spec for name in _WRITER_OP_SPECS})


def step_to_wire(step: Dict[str, Any]) -> Optional[str]:
    """Convert a decomposer JSON action dict to an AST-parseable wire
    string. Returns None on validation failure — caller logs."""
    handler = _WIRE_HANDLERS.get(step.get("action"))
    return handler(step) if handler else None


# --------------------------------------------------------------------
# runtime-script building  (action_type + inputs  →  VM python script)
# --------------------------------------------------------------------

def _kw_call(op_name: str, kwargs: Dict[str, Any]) -> str:
    """Render ``op_name(k=v, ...)`` — a valid Python call."""
    parts = [f"{k}={v!r}" for k, v in kwargs.items() if v is not None]
    return f"{op_name}({', '.join(parts)})"


def _emit_script(call_lines: List[str], domain: Optional[str],
                 logger: Optional[Any]) -> Optional[str]:
    """Append ``print('PASS')`` and run through the cli_run_uno wrapper
    (which prepends the connect + Writer setup + WRITER_OPS_LIBRARY)."""
    # canonical module lives in the sibling ``uno`` package (see calc/impress)
    from ..uno.cli_run_uno_helpers import build_runtime_script
    code = "\n".join(list(call_lines) + ["print('PASS')"])
    try:
        # Ops library/setup must match the ACTION's domain (writer), not the
        # active-window __current_domain (cli_run_uno binds ``doc`` by service
        # type; focus is irrelevant). See impress/actions for the bug context.
        return build_runtime_script(code, domain="libreoffice_writer")
    except ValueError as e:
        if logger:
            logger.warning("[writer action] build failed: %s", e)
        return None


def _build_paste_at(inputs: Dict[str, Any], logger) -> Optional[str]:
    kwargs = _paste_at_kwargs(inputs)
    domain = inputs.get("__current_domain")
    if kwargs is None:
        if logger:
            logger.warning("[writer_paste_at] needs paragraph_index "
                            "or after_heading")
        return None
    return _emit_script([_kw_call("op_paste_at", kwargs)], domain, logger)


def _make_spec_build(action: str):
    """Build a runtime-script builder for a spec-driven writer action."""
    op_name = _WRITER_OP_SPECS[action][0]

    def _build(inputs: Dict[str, Any], logger) -> Optional[str]:
        kwargs = _spec_kwargs(action, inputs)
        if kwargs is None:
            if logger:
                logger.warning("[%s] missing a required kwarg", action)
            return None
        return _emit_script(
            [_kw_call(op_name, kwargs)],
            inputs.get("__current_domain"), logger,
        )
    return _build


_BUILD_HANDLERS = {
    "writer_paste_at": _build_paste_at,
}
_BUILD_HANDLERS.update(
    {name: _make_spec_build(name) for name in _WRITER_OP_SPECS})


def action_to_runtime_script(
    action_type: str,
    action_inputs: Dict[str, Any],
    *,
    logger: Optional[Any] = None,
) -> Optional[str]:
    """Build the wrapped Python runtime script for a parsed ``writer_*``
    action. Returns None on validation failure / unknown action."""
    builder = _BUILD_HANDLERS.get(action_type)
    if builder is None:
        return None
    return builder(action_inputs, logger)


# --------------------------------------------------------------------
# actor prompt block
# --------------------------------------------------------------------

_SCHEMA_BLOCK = """\
Writer structured actions — the framework builds the underlying UNO
Python deterministically from your flat JSON kwargs, so you never
write Python source.

# Actions

* writer_paste_at — paste the clipboard content at a PRECISE location
  in the document (then saves). Use this — NOT a GUI click + Ctrl+V —
  to drop copied content (e.g. a range copied from Calc via
  calc_copy_to_clipboard) into the document; the framework positions the
  cursor by UNO so the paste lands exactly where intended. Give ONE
  locator:
  {"action":"writer_paste_at", "paragraph_index":<0-based index>}
    — paste into the paragraph at this index. Use the indices from
      the "Live document structure" block (e.g. the blank paragraph
      right after a heading).
  {"action":"writer_paste_at", "after_heading":"<heading text>"}
    — paste into the paragraph immediately after the heading whose
      text contains this substring.

For the paragraph-property actions below, ``target_indices`` is an
optional list of 0-based RAW paragraph indices (omit ⇒ apply to ALL
paragraphs). The index is the SAME number the doc-perceiver
structure block reports — blank paragraphs are counted in their raw
position; pass the raw index even if the perceiver shows it as
blank. Negative indices count from the end of the raw paragraph
list. The semantics are uniform across every writer paragraph
action below — there is NO content-paragraph vs raw-paragraph
distinction per op.

* writer_line_spacing — set line spacing.
  {"action":"writer_line_spacing", "multiplier":<1.0|1.5|2.0|...>,
   "target_indices":[<int>, ...]}

* writer_font_size — set font size in points.
  {"action":"writer_font_size", "size":<pt>,
   "pattern":"<optional regex to scope within a paragraph>",
   "target_indices":[<int>, ...]}

* writer_set_font — restyle existing text with a font name.
  {"action":"writer_set_font", "font_name":"<font>",
   "target_indices":[<int>, ...]}

* writer_default_font — set the APPLICATION default font (what new
  typing uses); not a document edit.
  {"action":"writer_default_font", "font_name":"<font>"}

* writer_alignment — set paragraph alignment.
  {"action":"writer_alignment",
   "alignment":"left|center|right|justify",
   "target_indices":[<int>, ...]}

* writer_case_conversion — convert paragraph text case.
  {"action":"writer_case_conversion", "mode":"lower|upper",
   "target_indices":[<int>, ...]}

* writer_strikethrough — apply strikethrough.
  {"action":"writer_strikethrough", "target_indices":[<int>, ...],
   "pattern":"<optional regex to scope within a paragraph>"}

* writer_insert_page_break — insert a hard page break after a
  paragraph.
  {"action":"writer_insert_page_break",
   "after_paragraph_index":<0-based index>}

* writer_page_numbers — add live page numbers to the header/footer.
  {"action":"writer_page_numbers",
   "position":"bottom_left|bottom_center|bottom_right|top_left|top_center|top_right"}

* writer_capitalize_words — Title-Case paragraph text.
  {"action":"writer_capitalize_words", "target_indices":[<int>, ...]}

* writer_remove_highlighting — clear background highlight colour.
  {"action":"writer_remove_highlighting", "target_indices":[<int>, ...]}

* writer_subscript_text — subscript the text matching a regex.
  {"action":"writer_subscript_text",
   "pattern":"<Python regex matching ONLY the char(s) to subscript>",
   "target_indices":[<int>, ...]}

* writer_insert_table — insert a NEW empty table at the cursor.
  {"action":"writer_insert_table", "rows":<int>, "cols":<int>}

* writer_delete_table — delete table(s); the rollback for a wrong or
  duplicate insert. Pass ONE selector:
  {"action":"writer_delete_table", "table_index":<1-based int>}
  {"action":"writer_delete_table", "after_heading":"<heading text>"}
  ``table_index`` deletes that single table (document order).
  ``after_heading`` de-duplicates that section — deletes every table
  in it EXCEPT the first. Use when verify reports "found N tables,
  expected 1 (duplicate paste?)".

* writer_export_pdf — export the document to PDF.
  {"action":"writer_export_pdf",
   "output_path":"<optional dir; default = doc's dir>",
   "output_filename":"<optional name; default = docbasename.pdf>"}

* writer_tabstops_split_align — split each non-blank paragraph at the
  Nth word, sending the remainder to a right tab stop.
  {"action":"writer_tabstops_split_align",
   "n_words_before_tab":<int, default 3>,
   "target_indices":[<int>, ...]}

* writer_set_color — set text colour.
  {"action":"writer_set_color", "color":"#RRGGBB",
   "pattern":"<optional regex for per-match colouring>",
   "walk_tables":<bool, default true>,
   "target_indices":[<int>, ...]}

* writer_find_replace — regex find+replace, or delete duplicate
  paragraphs.
  {"action":"writer_find_replace", "mode":"regex",
   "pattern":"<regex>", "replacement":"<text>",
   "target_indices":[<int>, ...]}
  {"action":"writer_find_replace", "mode":"dedup",
   "dedup_pattern":"<optional regex w/ 1 capture group = the dedup key>"}

* writer_insert_image — insert an image file at the cursor.
  {"action":"writer_insert_image",
   "image_path":"<absolute / ~ path>",
   "width_cm":<optional float>, "height_cm":<optional float>}

* writer_insert_paragraph_breaks — split one paragraph into multiple
  on sentence boundaries.
  {"action":"writer_insert_paragraph_breaks",
   "target_idx":<0-based RAW paragraph index>}

* writer_text_to_table — convert a block of separator-delimited paragraphs
  into a table.
  {"action":"writer_text_to_table",
   "start_index":<first RAW paragraph index, inclusive>,
   "end_index":<last RAW paragraph index, inclusive>,
   "separator":"<delimiter, default ','>"}
  Blank paragraphs inside the raw range are skipped (would otherwise
  become empty rows).

* writer_cross_reference — append a citation to the reference list and
  replace a marker in the body with a numbered [N].
  {"action":"writer_cross_reference",
   "citation_text":"<the exact citation string, verbatim>",
   "marker":"<the placeholder string to replace>"}
"""


def action_schema_block() -> str:
    """Prompt block describing every ``writer_*`` action — injected
    into the actor's system prompt for the ``libreoffice_writer``
    domain only."""
    return _SCHEMA_BLOCK


# ====================================================================
# writer_verify — structured JSON verify checks (kind="writer_verify")
# ====================================================================
#
# Counterpart of calc_verify. A ``writer_checks`` item is
# ``{"op": <op>, ...kwargs}``; kwargs pass 1:1 to the matching
# ``verify_*`` helper in writer_ops_lib.py. The framework builds the
# python3 -c body server-side so init_ledger never writes UNO Python.

WRITER_VERIFY_OPS: Dict[str, str] = {
    "table_count":          "verify_table_count",
    "table_after_heading":  "verify_table_after_heading",
}

_WRITER_VERIFY_REQUIRED_KWARGS: Dict[str, tuple] = {
    "table_count":          (),
    "table_after_heading":  ("heading",),
}


def validate_writer_verify_check(chk: Dict[str, Any]) -> bool:
    """True iff ``chk`` is a well-formed writer_verify check item.
    Structural-only — value correctness is checked at run time inside
    the verify_* helper."""
    op = chk.get("op")
    if op not in WRITER_VERIFY_OPS:
        return False
    for k in _WRITER_VERIFY_REQUIRED_KWARGS[op]:
        if k not in chk or chk[k] in (None, ""):
            return False
    return True


def build_writer_verify_python_body(checks: List[Dict[str, Any]]) -> str:
    """Return the python3 -c body for a list of writer_verify checks.

    Each ``verify_*`` op returns ``(ok, reason)``. ``all`` semantics —
    prints ``PASS`` iff every check's ``ok`` is truthy, else
    ``FAIL: <op> -- <reason>`` naming the first failing check AND why
    (so the planner gets actionable feedback, not an opaque FAIL).
    The UNO connect + WRITER_OPS_LIBRARY boilerplate is prepended by
    the verifier wrapper."""
    if not checks:
        return "print('FAIL: no checks')"
    lines: List[str] = ["_results = []"]
    for chk in checks:
        op = chk["op"]
        fn = WRITER_VERIFY_OPS[op]
        kwargs = {k: v for k, v in chk.items() if k != "op"}
        lines.append(f"_results.append({_kw_call(fn, kwargs)})")
    lines.append("_ok = all(_r[0] for _r in _results)")
    lines.append(
        "if _ok:\n    print('PASS')\nelse:\n"
        "    _i = next(_j for _j, _r in enumerate(_results) if not _r[0])\n"
        "    print('FAIL: ' + str(_OPS[_i]) + ' -- ' "
        "+ str(_results[_i][1]))"
    )
    ops_repr = [chk["op"] for chk in checks]
    return "_OPS = " + repr(ops_repr) + "\n" + "\n".join(lines)


_WRITER_VERIFY_SCHEMA_BLOCK = """\
WRITER_VERIFY — structured verify spec for libreoffice_writer

For a libreoffice_writer outcome, emit ``kind="writer_verify"`` — NOT
``kind="shell_command"`` with a hand-written ``python3 -c`` UNO
one-liner (that route hallucinates UNO API and breaks). The framework
builds the verifier deterministically from flat JSON.

Shape:

  "verify": {
    "kind": "writer_verify",
    "writer_checks": [
      {"op": "<op-name>", ...kwargs},
      ...
    ]
  }

All checks AND together — the outcome verifies iff EVERY check is
truthy.

# Ops

* ``table_after_heading`` — a table sits in the section that starts at
  a heading. Catches "no table" / "wrong location" / a DUPLICATE
  paste (``single`` defaults true → requires EXACTLY ONE table in the
  section).
  {"op":"table_after_heading", "heading":<heading substring>,
   "single":true,
   "exact_rows":<int, optional>, "min_rows":<int, optional>,
   "n_cols":<int, optional>,
   "cell_contains":[[<row>,<col>,<substring>], ...]  // 0-based row/col
  }
  Emit ``cell_contains`` / row / col anchors ONLY when their values
  are read verbatim from the task instruction or the live document
  structure — a hallucinated anchor permanently FAILs verify.

* ``table_count`` — total number of tables in the document.
  {"op":"table_count", "expected":<int>}        // exact
  {"op":"table_count", "min":<int>, "max":<int>}
  Use ``expected`` to assert the task added exactly the intended
  number of tables (the document's pre-existing tables count too —
  only assert an exact total when you know the starting count).
"""


def writer_verify_schema_block() -> str:
    """Prompt block describing the structured ``writer_verify`` kind —
    injected into the init_ledger prompt for libreoffice_writer tasks."""
    return _WRITER_VERIFY_SCHEMA_BLOCK

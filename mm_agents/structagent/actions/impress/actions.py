"""Structured Impress action handlers — flat JSON kwargs, no nested
Python.

Mirrors :mod:`mm_agents.actions.calc.actions`. The actor emits a JSON dict like
``{"action":"impress_set_text", "slide":1, "placeholder":"title",
"text":"..."}``; this module deterministically builds the equivalent
``op_impress_*`` call. The Python source is generated server-side —
guaranteed syntactically correct.

Three slices:

  * :data:`IMPRESS_ACTION_NAMES` — membership gate for dispatch.
  * ``_wire_impress_*`` validators — convert a decomposer's JSON
    action dict to an AST-parseable wire string.
  * ``_build_impress_*`` codegen — given parsed kwargs, build the
    wrapped runtime script (boilerplate + ops library + actor body)
    ready to ``exec`` on the VM.

Adding a new ``impress_*`` action: add the name to
:data:`IMPRESS_ACTION_NAMES`, add a ``_wire_impress_*`` validator
returning a wire string, add a ``_build_impress_*`` Python-code
generator. The dispatch tables at the bottom of the file pick up the
new handlers automatically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


# ====================================================================
# Membership set + verify-op registry
# ====================================================================

IMPRESS_ACTION_NAMES: frozenset = frozenset({
    "impress_set_text",
    "impress_set_font",
    "impress_set_paragraph",
    "impress_set_text_color",
    "impress_delete_shape",
    "impress_move_shape",
    "impress_set_shape_size",
    "impress_new_slide",
    "impress_delete_slide",
    "impress_duplicate_slide",
    "impress_move_slide",
    "impress_set_slide_orientation",
    "impress_set_slide_size",
    "impress_set_notes",
    "impress_set_transition",
    "impress_set_background",
    "impress_insert_shape",
    "impress_insert_image",
    "impress_insert_media",
    "impress_insert_chart",
    "impress_insert_table",
    "impress_set_master_color",
    "impress_set_app_setting",
    "impress_export_file",
})


# Companion verify-op registry — feeds the init_ledger's calc_verify-
# style structured spec for the impress domain.

IMPRESS_VERIFY_OPS: Dict[str, str] = {
    "slide_count":              "verify_impress_slide_count",
    "slide_exists_at_index":    "verify_impress_slide_exists_at_index",
    "shape_text":               "verify_impress_shape_text",
    "font_props":               "verify_impress_font_props",
    "paragraph_props":          "verify_impress_paragraph_props",
    "shape_exists":             "verify_impress_shape_exists",
    "slide_background":         "verify_impress_slide_background",
    "slide_notes":              "verify_impress_slide_notes",
    "slide_orientation":        "verify_impress_slide_orientation",
    "slide_transition":         "verify_impress_slide_transition",
    "chart_exists":             "verify_impress_chart_exists",
    "table_exists":             "verify_impress_table_exists",
    "image_dims":               "verify_impress_image_dims",
    "master_color":             "verify_impress_master_color",
    "app_setting":              "verify_impress_app_setting",
    "file_exists":              "verify_impress_file_exists",
}

_IMPRESS_VERIFY_REQUIRED_KWARGS: Dict[str, Tuple[str, ...]] = {
    "slide_count":              ("expected",),
    "slide_exists_at_index":    ("slide",),
    "shape_text":               ("slide",),
    "font_props":               ("slide",),
    "paragraph_props":          ("slide",),
    "shape_exists":             ("slide",),
    "slide_background":         ("slide", "color"),
    "slide_notes":              ("slide",),
    "slide_orientation":        ("expected",),
    "slide_transition":         ("slide",),
    "chart_exists":             ("slide",),
    "table_exists":             ("slide",),
    "image_dims":               ("slide",),
    "master_color":             ("role", "color"),
    "app_setting":              ("key", "expected"),
    "file_exists":              ("path",),
}


def validate_impress_verify_check(chk: Dict[str, Any]) -> bool:
    """True iff ``chk`` is a well-formed impress_verify check item."""
    op = chk.get("op")
    if op not in IMPRESS_VERIFY_OPS:
        return False
    for k in _IMPRESS_VERIFY_REQUIRED_KWARGS[op]:
        if k not in chk or chk[k] in (None, ""):
            return False
    return True


def build_impress_verify_python_body(checks: List[Dict[str, Any]]) -> str:
    """Return the python3 -c body string for a list of impress_verify
    checks. ``all`` semantics — prints PASS iff every check is truthy;
    otherwise prints ``FAIL: check[<i>] '<op>'`` naming the first
    failing check."""
    if not checks:
        return "print('FAIL: no checks')"
    lines: List[str] = ["_results = []"]
    for chk in checks:
        op = chk["op"]
        fn = IMPRESS_VERIFY_OPS[op]
        kwargs = {k: v for k, v in chk.items() if k != "op"}
        call = _kw_call(fn, kwargs)
        lines.append(f"_results.append(bool({call}))")
    lines.append("_ok = all(_results)")
    lines.append(
        "if _ok:\n    print('PASS')\nelse:\n"
        "    _i = _results.index(False)\n"
        "    print('FAIL: check[' + str(_i) + '] ' + repr(_OPS[_i]))"
    )
    ops_repr = [chk["op"] for chk in checks]
    return "_OPS = " + repr(ops_repr) + "\n" + "\n".join(lines)


def impress_verify_schema_block() -> str:
    """Prompt block describing the structured ``impress_verify`` spec
    for init_ledger. Mirrors calc_verify_schema_block.
    """
    return _IMPRESS_VERIFY_SCHEMA_BLOCK


_IMPRESS_VERIFY_SCHEMA_BLOCK = """\
IMPRESS_VERIFY — structured verify spec for libreoffice_impress

Impress tasks MUST use ``kind="impress_verify"`` (NOT
``kind="shell_command"``). The framework builds the verifier subprocess
from flat JSON; you never write Python, never write argv, never worry
about quote escaping.

Shape:

  "verify": {
    "kind": "impress_verify",
    "impress_checks": [
      {"op": "<op-name>", ...kwargs},
      ...
    ]
  }

Multiple checks AND together — outcome verifies iff EVERY check returns
truthy. Failure message names the first failing check + its ``op``.

# CRITICAL — do NOT hallucinate precise anchors

Every check has REQUIRED structural kwargs and OPTIONAL expected-value
anchors (``expected_value`` / ``expected_substring`` / ``color`` /
``size`` / ``expected``). An anchor permanently FAILs verify if its
value doesn't match the actual post-action state.

Rule: emit an anchor ONLY when you can read its exact value either
(a) VERBATIM from the task instruction (e.g. "set font color to
yellow" ⇒ ``color="#FFFF00"``; "make it 60pt" ⇒ ``size=60``), or
(b) VERBATIM from the SP (presentation structure) block as plain
text/number.

Otherwise OMIT the anchor — the structural / shape-exists-only check
alone passes when the agent does the right thing. Hallucinated anchors
(e.g. ``color="#1E2761"`` invented for a task that names no color)
GUARANTEE task failure.

# Ops

* ``slide_count`` — the presentation has exactly N slides
  {"op":"slide_count", "expected":<int>}

* ``slide_exists_at_index`` — a slide exists at the given 1-based
  index (used to confirm new-slide / duplicate insertion).
  {"op":"slide_exists_at_index", "slide":<1-based int>}

* ``shape_text`` — text content of a shape on a slide. Selector is
  one of ``shape`` (Name) / ``shape_index`` (1-based z-order) /
  ``placeholder`` ("title"/"subtitle"/"body"/"content"). Supply
  EXACTLY ONE expectation kwarg:
  {"op":"shape_text", "slide":<int>, "placeholder":"title",
   "expected_value":<text>}
  {"op":"shape_text", "slide":<int>, "shape":<name>,
   "expected_substring":<text>}
  {"op":"shape_text", "slide":<int>, "shape_index":<int>,
   "expected_regex":<pattern>}

* ``font_props`` — font properties on every run of the shape's text.
  All non-None kwargs must hold:
  {"op":"font_props", "slide":<int>, "placeholder":"title",
   "size":<int>, "bold":<bool>, "color":<"#RRGGBB">,
   "italic":<bool>, "underline":<bool>, "name":<font-name>}

* ``paragraph_props`` — alignment / indent on shape paragraphs.
  ``paragraph_index`` (1-based) optional; default checks ALL paragraphs.
  {"op":"paragraph_props", "slide":<int>, "shape":<name>,
   "alignment":"center"}

* ``shape_exists`` — slide K has at least one shape of the given type.
  type: "text"/"picture"/"image"/"chart"/"table"/"connector"/"any".
  {"op":"shape_exists", "slide":<int>, "shape_type":"picture"}

* ``slide_background`` — slide background fill matches ``color``.
  {"op":"slide_background", "slide":<int>, "color":<"#RRGGBB">}

* ``slide_notes`` — slide K speaker notes contain substring or
  equal exact value.
  {"op":"slide_notes", "slide":<int>, "expected_substring":<text>}

* ``slide_orientation`` — first slide is "portrait" or "landscape".
  {"op":"slide_orientation", "expected":"portrait"}

* ``slide_transition`` — slide K has any transition; optionally
  ``transition_type``.
  {"op":"slide_transition", "slide":<int>, "transition_type":"fade"}

* ``chart_exists`` — slide K has at least one chart. OMIT
  ``chart_type`` / ``title`` unless task literally specifies them.
  {"op":"chart_exists", "slide":<int>, "chart_type":"column"}

* ``table_exists`` — slide K has a table; optional ``rows`` / ``cols``.
  {"op":"table_exists", "slide":<int>, "rows":3, "cols":2}

* ``image_dims`` — slide K's picture shape has expected width/height.
  ``shape`` or ``shape_index`` selector required.
  {"op":"image_dims", "slide":<int>, "shape":"Picture1",
   "width":"1cm", "height":"1cm"}

* ``master_color`` — master placeholder of ``role`` has font color.
  role: "slide_number" / "title" / "body".
  {"op":"master_color", "role":"slide_number", "color":"#FF0000"}

* ``app_setting`` — LibreOffice registry value matches.
  key: "presenter_console_disable" (bool), "auto_save_interval" (int).
  {"op":"app_setting", "key":"auto_save_interval", "expected":3}

* ``file_exists`` — a file at ``path`` exists with size >= min_bytes.
  {"op":"file_exists", "path":<absolute path>, "min_bytes":<int>}

# Patterns — match the task to one shape

* text edit: ``shape_text`` (substring or exact) + optionally
  ``font_props`` when task names font / size / color.
* slide structural: ``slide_count`` for add/delete/duplicate; or
  ``slide_exists_at_index`` when only the K-th slide matters.
* image insertion: ``shape_exists`` (shape_type="picture") +
  optionally ``image_dims`` when task names dimensions.
* chart insertion: ``chart_exists`` with optional chart_type /
  title only when literally named.
* table insertion: ``table_exists`` with optional rows/cols.
* background change: ``slide_background``.
* notes: ``slide_notes``.
* orientation: ``slide_orientation``.
* transition: ``slide_transition``.
* export: ``file_exists`` on the output path.
* app setting: ``app_setting`` with the relevant key.

Do NOT emit ``kind="shell_command"`` for libreoffice_impress —
the impress_verify kind covers every Impress evaluator shape.
"""


# ====================================================================
# Schema block injected into the actor decomposer prompt
# ====================================================================

def action_schema_block() -> str:
    """Action catalogue describing every impress_* action's flat JSON
    shape. Injected into the decomposer system prompt when domain is
    libreoffice_impress. Mirrors calc's action_schema_block.
    """
    return _ACTION_SCHEMA_BLOCK


_ACTION_SCHEMA_BLOCK = """\
Impress structured actions — the framework builds the underlying Python
deterministically from your flat JSON kwargs, so you NEVER write Python
source. Brackets, escapes, newlines: all handled for you.

# Common conventions

* `slide`        : 1-based slide index or slide name (string).
* shape selector : EXACTLY ONE of `shape` (Name), `shape_index`
                   (1-based z-order), or `placeholder` ("title"/
                   "subtitle"/"body"/"content").
* lengths        : "2cm" / "0.5in" / "12pt" / "300" (raw 1/100mm).
* colors         : "#RRGGBB" hex.

NEVER hand-write raw UNO Python for libreoffice_impress — every
Impress mutation has a structured impress_* action below, and there
is no raw-UNO escape hatch.

# Action catalogue

* impress_set_text — set text content of a shape on a slide.
  {"action":"impress_set_text",
   "slide":<int>, "placeholder":"title"|"body"|"subtitle"|"content",
   "text":<replacement text>}
  Alternative selectors: ``"shape":"<Name>"`` or
  ``"shape_index":<1-based int>``.
  TABLE CELL — to write a specific cell, pass ``row`` + ``col`` (both
  1-based) alongside a ``shape="<Table Name>"`` selector. Rewriting
  one whole row uses one action per cell:
    {"action":"impress_set_text", "slide":4, "shape":"Table 3",
     "row":1, "col":1, "text":"T1"}
    (and one each for col=2,3,4)

* impress_set_font — set font properties on text runs. Each property
  kwarg optional; non-None values are applied. Also iterates table
  cells when the target shape is a table.
  {"action":"impress_set_font", "slide":<int>, "placeholder":"title",
   "bold":<bool>, "italic":<bool>, "underline":<bool>,
   "strike":<bool>, "name":"Arial", "size":<int>,
   "color":<"#RRGGBB">}
  SCOPE EXPANSION — for "all text" / "all text boxes" / whole-deck
  font changes, set ``scope`` instead of enumerating shapes:
    {"action":"impress_set_font", "scope":"slide", "slide":<int>, ...}
      → every text-bearing shape (incl. tables) on that slide
    {"action":"impress_set_font", "scope":"all", ...}
      → every text-bearing shape across ALL slides
  ``paragraph_index`` (1-based int or list) restricts to specific
  paragraphs within each text shape. The VALUE is the literal
  ``pN`` number printed on the SP block row — NOT a bullet number.
  HARD RULE for ordinal mapping when the SP block contains
  ``(title)`` rows: when the instruction says "first / second / Nth
  line / item / row / bullet", count only ``•`` rows from the SP
  block, then emit the LITERAL ``pN`` index of the chosen row.
  Example mapping (schematic):
    SP rows:  ``p1 (title) "..."  p2 • "..."  p3 • "..."  p4 • "..."``
    Instruction "first line" → first ``•`` row is p2 →
                                  ``paragraph_index=2``
    Instruction "second line" → second ``•`` row is p3 →
                                  ``paragraph_index=3``
  For a shape with no ``(title)`` rows, the literal ``pN`` already
  matches the natural ordinal — "line N" maps to
  ``paragraph_index=N``.

* impress_set_paragraph — alignment / indent / bullets on shape
  paragraphs. ``paragraph_index`` (1-based int or list) optional —
  default applies to ALL paragraphs. Supports the same ``scope``
  modes as ``impress_set_font`` ("shape" / "slide" / "all").
  {"action":"impress_set_paragraph", "slide":<int>, "shape":<Name>,
   "alignment":"center", "indent_level":<int>, "bullets":<bool>}

  CRITICAL — ``shape=`` MUST be a LEAF text shape name, never a
  ``group`` name. In the SP block, text shapes are tagged ``txt``
  (or ``title``/``subtitle``/``outline``) and groups are tagged
  ``group``. Passing a group's Name silently recurses into EVERY
  text child of the group, which over-applies the alignment and
  fails any task that wants only one textbox changed (e.g. "align
  the first textbox" when the slide has a group containing two
  stacked textboxes — only one of them is the target).
  When the SP block shows a group with text-bearing children
  indented under it, pick the specific child's name. The first
  child line under the group (top-most by position, since group
  children are sorted top→bottom) is the "first textbox" for
  reasoning like "the first/topmost/title textbox on slide K".

* impress_set_text_color — convenience shortcut for color-only edits.
  Supports the same ``scope`` modes.
  {"action":"impress_set_text_color", "slide":<int>, "placeholder":"title",
   "color":<"#RRGGBB">}

* impress_delete_shape — delete ONE shape from a slide. Use this when
  the task asks to remove specific content (icons, a textbox, a
  picture) while keeping the slide itself; do NOT confuse with
  ``impress_delete_slide`` which removes the whole slide.
  {"action":"impress_delete_shape", "slide":<int>, "shape":<Name>}
  Alternative selectors: ``"shape_index":<int>`` or
  ``"placeholder":"title"|...``.

* impress_move_shape — move a shape to a new position on its slide.
  {"action":"impress_move_shape", "slide":<int>, "shape":<Name>,
   "x":"<length>", "y":"<length>"}
  Length specs accept "Ncm" / "Nin" / "Nmm" / "Npt" (e.g. "10.5cm").
  ``anchor`` (optional) controls which point of the shape's bounding
  box lands at (x, y) — "top-left" (default), "center",
  "bottom-right", etc.

  ``edge`` (optional) — "top" / "bottom" / "left" / "right". Snaps
  the corresponding edge of the shape to the same edge of the slide.
  STRONGLY PREFER ``edge`` over computing y/x yourself when the
  instruction says "move to the {top,bottom,left,right} of the
  slide". Reasons:
   * ``y=<length>`` puts the shape's TOP-LEFT corner at that y.
     The shape's BOTTOM ends up at y + shape_height — which usually
     OVERFLOWS the slide when you wanted the bottom snug against the
     slide's bottom edge. ``edge="bottom"`` does the math for you
     (places top-left at slide_height − shape_height).
   * You don't need to read the shape's height from the SP block.
   * The slide's dimensions are taken from UNO at runtime, not from
     a hardcoded guess.

  Schematic — replace <K>/<NAME>:
    "move title to the bottom of the slide":
      {"slide":<K>, "placeholder":"title", "edge":"bottom"}
    "move the table to the bottom":
      {"slide":<K>, "shape":"<Table Name>", "edge":"bottom"}
    "move the image to the right side":
      {"slide":<K>, "shape":"<NAME>", "edge":"right"}
  Use explicit ``y=`` / ``x=`` ONLY when the instruction names a
  specific coordinate ("y=5cm", "at the middle"), not for "to the
  bottom/top/sides".

* impress_set_shape_size — resize a shape on its slide.
  {"action":"impress_set_shape_size", "slide":<int>, "shape":<Name>,
   "width":"<length>", "height":"<length>"}
  Pass only one of width/height to resize a single axis.
  ``keep_position`` (default true): the top-left corner stays put.
  Set false to keep the shape centered on its current centroid.

  "Stretch to fill the (entire) page / slide" — REQUIRES TWO steps,
  because set_shape_size only resizes; the shape's top-left stays
  put unless you also move it. Pattern:
    Step 1: impress_move_shape  → snap origin to (0, 0)
        {"slide":<K>, "shape":"<NAME>", "x":"0cm", "y":"0cm"}
    Step 2: impress_set_shape_size  → fill with the SLIDE's own dims
        {"slide":<K>, "shape":"<NAME>",
         "width":"<slide_w_from_SP_block>",
         "height":"<slide_h_from_SP_block>"}
  Read slide_w / slide_h from the SP block's slide header (the
  ``width=Wcm height=Hcm`` line on the target slide); never hardcode
  defaults — slide dimensions vary across decks.

* impress_new_slide — insert one or more new slides.
  {"action":"impress_new_slide", "position":"end"|<int>|"after:<N>"|
   "before:<N>"|"start", "layout":"blank"|"title"|"content"|
   "two_content"|"title_only"|<int>, "name":<optional Name>,
   "count":<int default 1>}
  Use ``count`` to batch-create N slides at the same position — e.g.
  "create six blank slides at the end":
    {"action":"impress_new_slide", "position":"end", "layout":"blank",
     "count":6}
  Do NOT emit six separate impress_new_slide steps; ``count`` does it
  in one action.

* impress_delete_slide — delete the K-th slide (1-based).
  {"action":"impress_delete_slide", "slide":<int>}

* impress_duplicate_slide — duplicate slide K; copy inserted at K+1.
  {"action":"impress_duplicate_slide", "slide":<int>}

* impress_move_slide — reorder a slide to a new 1-based position.
  {"action":"impress_move_slide", "slide":<int>, "to":<int>}

* impress_set_slide_orientation — page orientation for every slide.
  {"action":"impress_set_slide_orientation",
   "orientation":"portrait"|"landscape"}

* impress_set_slide_size — page dimensions.
  {"action":"impress_set_slide_size", "width":"25.4cm",
   "height":"19.05cm"}

* impress_set_notes — speaker notes on a slide.
  {"action":"impress_set_notes", "slide":<int>, "text":<notes text>}

* impress_set_transition — slide transition.
  {"action":"impress_set_transition", "slide":<int>,
   "transition_type":"dissolve"|"fade"|"wipe_left"|"wipe_right"|
   "wipe_up"|"wipe_down"|"push_up"|"push_down"|"cover"|"split"|
   "diamond"|"circle"|"random"|"none",
   "speed":"slow"|"medium"|"fast"}

* impress_set_background — slide background fill.
  {"action":"impress_set_background", "slide":<int>,
   "color":<"#RRGGBB">}
  ``slide`` omitted ⇒ apply to ALL slides.
  ``"image_path":<absolute path>`` for bitmap fill (mutually
  exclusive with ``color``).

* impress_insert_shape — insert a free shape on a slide.
  {"action":"impress_insert_shape", "slide":<int>,
   "preset":"textbox"|"rect"|"rectangle"|"ellipse"|"circle"|"line",
   "x":"2cm", "y":"2cm", "width":"10cm", "height":"3cm",
   "text":<optional>, "name":<optional Name>,
   "fill_color":<"#RRGGBB">,
   "font_size":<int>, "font_color":<"#RRGGBB">,
   "bold":<bool>, "italic":<bool>, "alignment":<alignment>}

* impress_insert_media — embed an audio / video media file (mp3,
  wav, mp4, etc.) on a slide as a playable shape.
  {"action":"impress_insert_media", "slide":<int>,
   "media_path":"/home/user/Desktop/clip.mp3",
   "x":"<length>", "y":"<length>",
   "width":"<length>", "height":"<length>", "name":<optional Name>}
  The media file must already exist at ``media_path`` on the VM; the
  saved .pptx embeds the bytes so the file travels with the doc.

* impress_insert_image — insert an image from a path on the VM.
  {"action":"impress_insert_image", "slide":<int>,
   "image_path":"/home/user/photo.png",
   "x":"1cm", "y":"1cm", "width":"5cm", "height":"5cm",
   "name":<optional Name>}

* impress_insert_chart — chart on a slide (best-effort: LO Impress
  embedded charts are fiddly; verify via shape_exists).
  {"action":"impress_insert_chart", "slide":<int>,
   "categories":["Q1","Q2","Q3"],
   "values":[[10,20,30],[5,15,25]],
   "chart_type":"column"|"bar"|"line"|"pie"|"area"|"scatter",
   "title":<optional>,
   "x":"2cm", "y":"2cm", "width":"18cm", "height":"12cm",
   "name":<optional Name>}

* impress_insert_table — table on a slide.
  {"action":"impress_insert_table", "slide":<int>,
   "rows":<int>, "cols":<int>,
   "data":[["A","B"],["1","2"]],
   "x":"1cm", "y":"1cm", "width":"20cm", "height":"5cm",
   "name":<optional Name>}

  AUTO-REPLACE MODE — when the instruction is just "insert a table
  on slide K" / "in the <X> slide, insert a table" (no explicit
  position phrasing like "at the top", "at 5cm,3cm", "next to the
  picture"), OMIT all of ``x`` / ``y`` / ``width`` / ``height``.
  The op then finds the slide's empty body / content placeholder
  and REPLACES it with the new table at the same pos+size. This
  matches the standard PowerPoint/Impress "Insert Table" UX where
  the table fills the slide's content area; the eval gold for
  these tasks expects the original placeholder to be gone, NOT
  left behind underneath an extra new table. Passing any explicit
  x/y disables auto-replace (use that only when the task specifies
  a coordinate).

* impress_set_master_color — master placeholder color (slide number /
  master title / master body). USE THIS when the instruction talks
  about "the slide number" / "the page number" / "slide numbering" /
  "the footer". The slide-number / page-number color lives on the
  MASTER slide, not on any individual slide — using impress_set_font
  on a "SlideNumber" shape on one slide does NOT change what the
  evaluator reads. Always go through impress_set_master_color for
  these targets.
  {"action":"impress_set_master_color",
   "role":"slide_number"|"title"|"body",
   "color":<"#RRGGBB">}
  ROLES — ``slide_number`` / ``title`` / ``body`` are the ONLY valid
  values. There is NO ``"background"`` role. To change a slide's
  background fill, use ``impress_set_background`` (works for one
  slide via ``slide=K`` or for ALL slides by OMITTING the ``slide``
  kwarg — much shorter than emitting one action per slide).

* impress_set_app_setting — LO config keys.
  {"action":"impress_set_app_setting",
   "key":"presenter_console_disable"|"auto_save_interval",
   "value":<bool>|<int minutes>}

* impress_export_file — export to PDF / PPTX / PNG / ODP.
  {"action":"impress_export_file", "path":<absolute path>,
   "format":"auto"|"pdf"|"pptx"|"png"|"odp",
   "slide":<int>  // PNG only: per-slide export
  }
"""


# ====================================================================
# data_layout init / planner blocks — Q1-Q4 framework
# ====================================================================

def data_layout_init_block() -> str:
    """Prompt block describing the data_layout JSON shape init_ledger
    must emit for libreoffice_impress tasks. Mirrors calc's init block.
    """
    return _DATA_LAYOUT_INIT_BLOCK


def data_layout_planner_block() -> str:
    """Prompt block describing the <data_layout>+<step> XML shape the
    planner must emit for libreoffice_impress tasks.
    """
    return _DATA_LAYOUT_PLANNER_BLOCK


_DATA_LAYOUT_INIT_BLOCK = """\
DATA_LAYOUT — REQUIRED FOR libreoffice_impress TASKS

init_ledger MUST emit a ``data_layout`` object as the FIRST top-level
field after ``initial_context``. The fields are the Q1-Q4 STOP chain;
each answer must be grounded in the PRESENTATION STRUCTURE
(SlidePerceiver) block, not training memory.

JSON shape:

{
  "initial_context": { ... },
  "data_layout": {
    "q0_subtasks": ["<short imperative phrase>", ...],
    "subtasks_reasoning": [
      {
        "q1_slide_semantics": "<one sentence: what does the relevant slide hold?>",
        "q2_task_pattern":   "<letter + impress_* action name>",
        "q3_output_target":  "<slide index + shape selector + property>",
        "q4_action_spec":    "<pattern-specific kwargs>"
      },
      ...                         // one per q0_subtasks entry
    ]
  },
  "required_outcomes": [ ... ]    // one per q0_subtasks entry
}

  q0_subtasks: list of sub-tasks the instruction asks for.
       **RULE: one impress_* action ⇒ one q0 entry.** Split at every
       conjunction ("Next" / "Then" / "And then" / "Also" / ";").
       Two cell-write directives against DIFFERENT shapes / slides are
       always separate sub-tasks. Use ``depends_on`` to chain when a
       later sub-task reads from an earlier one.

  Then answer Q1-Q4 SEPARATELY for EACH sub-task. Within each
  sub-task's answers:

  q1_slide_semantics: one sentence. What does the RELEVANT slide hold?
       Read off the SlidePerceiver block (`Slide[K] layout=... shapes=N`).
       Be concrete (e.g., "Slide 3 has a title + 3 bullet points;
       title is the 'Conclusion' header").

  q2_task_pattern: pick ONE of (a-j) and name the impress_* action it
       maps to.
        (a) text/font edit on existing shape → impress_set_text /
            impress_set_font / impress_set_paragraph /
            impress_set_text_color  (use ``scope="slide"`` / ``"all"``
            for whole-slide / whole-deck text changes; ``paragraph_index``
            to restrict to specific paragraphs/lines.)
        (a') delete a specific shape (icon / textbox / picture) from a
            slide while keeping the slide → impress_delete_shape
        (b) slide structural → impress_new_slide / impress_delete_slide /
            impress_duplicate_slide / impress_move_slide /
            impress_set_slide_orientation / impress_set_slide_size
        (c) shape insert (free shape / text box / preset geometric) →
            impress_insert_shape
        (c') move / resize an EXISTING shape (image, table, textbox)
            on a slide → impress_move_shape (re-position) /
            impress_set_shape_size (resize)
        (d) image insert → impress_insert_image
        (e) chart insert → impress_insert_chart
        (f) table insert → impress_insert_table
        (g) background / master → impress_set_background /
            impress_set_master_color
        (h) notes / transition → impress_set_notes /
            impress_set_transition
        (i) export → impress_export_file
        (j) app setting → impress_set_app_setting

  q3_output_target: slide index + shape selector + property name.
       Example: "Slide 1 title shape, text content + font size";
       "Slide 3 background color"; "All slides orientation".

  q4_action_spec: the flat JSON kwargs the chosen impress_* action
       requires. EVERY required slot MUST be filled. Spec slots per
       pattern letter — see the Impress structured actions block for
       the full catalogue.

The ``data_layout`` object goes IMMEDIATELY after ``initial_context``
and BEFORE ``required_outcomes``. ``required_outcomes`` MUST contain
ONE outcome per ``q0_subtasks`` entry (1:1, same order); each
outcome's verify spec uses the slide index / shape name from that
sub-task's ``q3_output_target`` verbatim. Use ``depends_on`` to chain
when a later sub-task reads from an earlier one.
"""


_DATA_LAYOUT_PLANNER_BLOCK = """\
DATA_LAYOUT — REQUIRED FOR libreoffice_impress TASKS

For libreoffice_impress tasks, the planner MUST emit a ``<data_layout>``
block as the FIRST child of ``<plan>``, before ``<strategy>``.

**Q1 — directive analysis (in <data_layout>)** is the CRITICAL
decomposition step. It has TWO parts that BOTH must appear:

  <q1_directives>
    PART 1 — Slide inventory (1 line per slide the instruction touches).
    Format:  "Slide K: <layout>. Shapes you may target: title=…,
              body=…, table=…  (Name=… from the SP block)".
    Read straight off the SP body — never invent Names.

    PART 2 — Decomposition. Enumerate EVERY atomic directive in the
    instruction as a numbered list. One directive per row:
      D1: verb=<imperative>; property=<what changes>;
          target_phrase=<verbatim noun phrase from the instruction>
      D2: ...

    Rules for splitting directives:
      * A conjunction ("and", "also", ";", "then", "next") between two
        verbs / property changes ⇒ separate directives.
      * Two property changes that share the same verb but apply to
        DIFFERENT shapes / scopes ⇒ separate directives.
      * One verb that lists multiple values (e.g. "set color red,
        green, blue respectively" across 3 shapes) ⇒ N directives.
      * Two property changes that share a verb AND a target_phrase
        (e.g. "underline and bold the title") ⇒ ONE directive with a
        compound property.

    ONE directive ⇒ exactly ONE <step>.

**Q2 / Q3 / Q4 are PER-STEP** — each <step> carries its own block:

  <q2_task_pattern>  — letter (a-j) + the impress_* action that
                       implements THIS directive.
  <q3_target_binding> — translate THIS directive's target_phrase from
                       Q1 PART 2 into a concrete addressing tuple.
                       Answer THREE sub-questions in order:

                       (a) SHAPE selector — pick ONE form:
                             placeholder="title"|"subtitle"|"body"|"notes"
                             shape="<Name copied from SP block>"
                             shape_index=<int>
                             slide=<K> + scope="slide"   (whole slide)
                             scope="all"                 (whole deck)
                           Conventions for translating target_phrase:
                             "the title"               → placeholder="title"
                             "the subtitle"            → placeholder="subtitle"
                             "the content" / "the body" /
                             "the body text" / "the main text"
                                                       → placeholder="body"
                             "the notes" / "speaker notes"
                                                       → placeholder="notes"
                             "the table"
                                  → shape="<Table Name from SP block>"
                             "the picture" / "the image"
                                  → shape="<Picture Name from SP block>"
                             "this slide" / "the slide" (when the
                             property spans multiple shapes) → scope="slide"
                             "all text" / "everything on this slide" /
                             "every textbox" (one slide)    → scope="slide"
                             "every slide" / "all my slides" /
                             "the whole presentation"       → scope="all"
                           NEVER pick scope="slide" / scope="all" when
                           target_phrase names a specific placeholder /
                           shape — that over-applies and breaks the eval.

                       (b) PARAGRAPH filter — set ``paragraph_index=N``
                           (1-based int, or a list like [1,2]) IF AND
                           ONLY IF the target_phrase points at a
                           sub-shape paragraph range:
                             * Ordinal + "line" / "paragraph" / "bullet"
                               / "row" / "item" (e.g. "the first line",
                               "the second paragraph", "the third
                               bullet") — paragraph_index = that ordinal.
                             * Two such ordinals joined by "and" / "to"
                               (e.g. "the first and second line") —
                               paragraph_index = [N, M] (or a range).
                             * target_phrase is verbatim text that
                               matches exactly ONE paragraph in the
                               shape chosen in (a) — paragraph_index =
                               that paragraph's 1-based index (read it
                               off the SP body of the target shape).
                           OMIT paragraph_index when target_phrase
                           names a whole shape ("the title", "the
                           body", "the table") or a whole slide /
                           deck — the property then applies to every
                           run in the shape(s).

                       (c) PROPERTY — name the kwarg the chosen action
                           will set (underline / bold / color / size /
                           indent_level / bullets / text / etc.) plus
                           its value from Q1 PART 2's directive.
  <q4_action_spec>   — flat kwargs for the impress_* action, copying
                       the selector from Q3 and the property values
                       from Q1 PART 2's directive. EVERY required
                       slot filled.

Bundling Q2 / Q3 ("(a) set_font + (a) set_paragraph") in one <step>
collapses both directives into one reasoning block; the second action's
kwargs end up improvised inside <text> and skip the protocol. One
directive ⇒ one step ⇒ one Q2 ⇒ one Q3 ⇒ one Q4.

Pattern-letter PRECEDENCE for <q2_task_pattern>: image keyword → (d);
chart keyword → (e); table keyword → (f); transition keyword →
(h, transition). Text edits on existing shapes default to (a).

After ``<data_layout>`` (which carries only Q1), emit ``<strategy>``
then ONE ``<step>`` per directive from Q1 PART 2. EACH ``<step>``
carries its own ``<q2_task_pattern>`` + ``<q3_target_binding>`` +
``<q4_action_spec>`` + ``<text>`` + ``<expected_post_state>``. The
``<text>`` MUST carry EVERY kwarg from this step's
``<q4_action_spec>`` verbatim — the actor decomposer only reads
``<text>``; missing kwargs lose information and the actor re-guesses
(usually wrong).

NEVER COMBINE TWO Q1-DIRECTIVES INTO ONE STEP
  Each ``impress_set_font`` (or set_paragraph / set_text_color) call
  applies EVERY non-None property kwarg to EVERY run it iterates. If
  two directives in Q1 PART 2 have DIFFERENT target_phrases — even by
  one word — they MUST be two ``<step>``s with two ``<q3>`` selectors.
  Bundling them collapses the narrower target_phrase into the broader
  one and over-applies the property.

  Schematic only — placeholders ⟨A⟩ / ⟨B⟩, NEVER task wording:
    Q1 directive D1: target_phrase=⟨narrow noun, e.g. "the body"⟩
                     property=⟨P1⟩
    Q1 directive D2: target_phrase=⟨broad noun, e.g. "this slide"⟩
                     property=⟨P2⟩

    Wrong → ONE step combining both:
      impress_set_font scope="slide" slide=K ⟨P1⟩=v1 ⟨P2⟩=v2
        (P1 leaks onto every shape, not just ⟨A⟩)

    Right → TWO steps, one per directive:
      Step 1: impress_set_font slide=K placeholder=⟨A's role⟩ ⟨P1⟩=v1
      Step 2: impress_set_font scope="slide" slide=K ⟨P2⟩=v2

  Heuristic: count the distinct noun phrases the instruction binds
  properties to. Two distinct target nouns ⇒ two steps. One target
  noun ⇒ one step.

NEVER CONFUSE delete_shape WITH delete_slide
  "remove the personal info / icons / a textbox on slide K" →
    impress_delete_shape (slide stays, shape goes).
  "remove slide K entirely" → impress_delete_slide.

Output skeleton — Q1 in <data_layout>, Q2/Q3/Q4 inside each <step>:

<plan>
  <data_layout>
    <q1_directives>
      PART 1 — Slide inventory (one line per relevant slide).
      PART 2 — Directives (D1 ... DN, one per row).
    </q1_directives>
  </data_layout>
  <strategy>...</strategy>
  <step>                       <!-- one per directive in Q1 PART 2 -->
    <q2_task_pattern>(letter) impress_<action></q2_task_pattern>
    <q3_target_binding>selector for THIS directive's target_phrase +
                       property name</q3_target_binding>
    <q4_action_spec>kwargs for THIS step</q4_action_spec>
    <text>Use impress_<action> with <full kwargs>.</text>
    <expected_post_state>...</expected_post_state>
  </step>
</plan>
"""


# ====================================================================
# Wire helpers
# ====================================================================

def _kw_wire(action_name: str, kwargs: Dict[str, Any]) -> str:
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        parts.append(f"{k}={v!r}")
    return f"{action_name}({', '.join(parts)})"


def _kw_call(op_name: str, kwargs: Dict[str, Any]) -> str:
    parts = []
    for k, v in kwargs.items():
        if v is None:
            continue
        parts.append(f"{k}={v!r}")
    return f"{op_name}({', '.join(parts)})"


def _emit_script(call_lines: List[str], domain: Optional[str],
                 logger: Optional[Any]) -> Optional[str]:
    # canonical module lives in the sibling ``uno`` package (see calc/writer)
    from ..uno.cli_run_uno_helpers import build_runtime_script
    code = "\n".join(list(call_lines) + ["print('PASS')"])
    try:
        # The UNO ops library + setup MUST match the ACTION's domain (impress),
        # NOT the active-window/__current_domain — cli_run_uno re-binds ``doc``
        # by document service type, so window focus is irrelevant. Routing by
        # __current_domain loaded the wrong ops lib when another app was focused
        # (6f4073b8/eb303e01: impress_set_notes under a Writer window → the
        # Writer ops lib has no op_impress_set_notes → every call failed).
        return build_runtime_script(code, domain="libreoffice_impress")
    except ValueError as e:
        if logger:
            logger.warning("[impress action] build failed: %s", e)
        return None


# ====================================================================
# Wire functions (JSON → wire string)
# ====================================================================

def _wire_set_text(step):
    slide = step.get("slide")
    text = step.get("text")
    if slide is None or text is None:
        return None
    return _kw_wire("impress_set_text", {
        "slide": slide,
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
        "row": step.get("row"),
        "col": step.get("col"),
        "text": text,
    })


def _wire_set_font(step):
    # ``scope`` defaults to "shape" → ``slide`` is then required.
    # For scope="slide" only ``slide`` is required; for scope="all"
    # nothing geometric is required.
    scope = (step.get("scope") or "shape").lower()
    slide = step.get("slide")
    if scope == "shape" and slide is None:
        return None
    if scope == "slide" and slide is None:
        return None
    if scope not in ("shape", "slide", "all"):
        return None
    return _kw_wire("impress_set_font", {
        "slide": slide,
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
        "scope": scope if scope != "shape" else None,
        "paragraph_index": step.get("paragraph_index"),
        "bold": step.get("bold"),
        "italic": step.get("italic"),
        "underline": step.get("underline"),
        "strike": step.get("strike"),
        "name": step.get("name"),
        "size": step.get("size"),
        "color": step.get("color"),
    })


def _wire_set_paragraph(step):
    scope = (step.get("scope") or "shape").lower()
    slide = step.get("slide")
    if scope == "shape" and slide is None:
        return None
    if scope == "slide" and slide is None:
        return None
    if scope not in ("shape", "slide", "all"):
        return None
    return _kw_wire("impress_set_paragraph", {
        "slide": slide,
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
        "scope": scope if scope != "shape" else None,
        "paragraph_index": step.get("paragraph_index"),
        "alignment": step.get("alignment"),
        "indent_level": step.get("indent_level"),
        "bullets": step.get("bullets"),
    })


def _wire_set_text_color(step):
    color = step.get("color")
    if color is None:
        return None
    scope = (step.get("scope") or "shape").lower()
    slide = step.get("slide")
    if scope in ("shape", "slide") and slide is None:
        return None
    if scope not in ("shape", "slide", "all"):
        return None
    return _kw_wire("impress_set_text_color", {
        "slide": slide,
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
        "scope": scope if scope != "shape" else None,
        "color": color,
    })


def _wire_delete_shape(step):
    if step.get("slide") is None:
        return None
    return _kw_wire("impress_delete_shape", {
        "slide": step.get("slide"),
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
    })


def _wire_move_shape(step):
    slide = step.get("slide")
    if slide is None:
        return None
    # Need at least one of: x, y, edge
    if step.get("x") is None and step.get("y") is None and step.get("edge") is None:
        return None
    return _kw_wire("impress_move_shape", {
        "slide": slide,
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
        "x": step.get("x"),
        "y": step.get("y"),
        "anchor": step.get("anchor"),
        "edge": step.get("edge"),
    })


def _wire_set_shape_size(step):
    slide = step.get("slide")
    if slide is None:
        return None
    if step.get("width") is None and step.get("height") is None:
        return None
    return _kw_wire("impress_set_shape_size", {
        "slide": slide,
        "shape": step.get("shape"),
        "shape_index": step.get("shape_index"),
        "placeholder": step.get("placeholder"),
        "width": step.get("width"),
        "height": step.get("height"),
        "keep_position": step.get("keep_position"),
    })


def _wire_new_slide(step):
    return _kw_wire("impress_new_slide", {
        "position": step.get("position", "end"),
        "layout":   step.get("layout", "blank"),
        "name":     step.get("name"),
        "count":    step.get("count"),
    })


def _wire_delete_slide(step):
    if step.get("slide") is None:
        return None
    return _kw_wire("impress_delete_slide", {"slide": step.get("slide")})


def _wire_duplicate_slide(step):
    if step.get("slide") is None:
        return None
    return _kw_wire("impress_duplicate_slide", {"slide": step.get("slide")})


def _wire_move_slide(step):
    if step.get("slide") is None or step.get("to") is None:
        return None
    return _kw_wire("impress_move_slide", {
        "slide": step.get("slide"), "to": step.get("to")})


def _wire_set_slide_orientation(step):
    orient = step.get("orientation")
    if not orient:
        return None
    return _kw_wire("impress_set_slide_orientation",
                    {"orientation": orient})


def _wire_set_slide_size(step):
    w = step.get("width"); h = step.get("height")
    if w is None or h is None:
        return None
    return _kw_wire("impress_set_slide_size",
                    {"width": w, "height": h})


def _wire_set_notes(step):
    slide = step.get("slide"); text = step.get("text")
    if slide is None or text is None:
        return None
    return _kw_wire("impress_set_notes",
                    {"slide": slide, "text": text})


def _wire_set_transition(step):
    slide = step.get("slide"); tt = step.get("transition_type")
    if slide is None or not tt:
        return None
    return _kw_wire("impress_set_transition", {
        "slide": slide,
        "transition_type": tt,
        "speed": step.get("speed"),
    })


def _wire_set_background(step):
    color = step.get("color"); image_path = step.get("image_path")
    if (color is None) == (image_path is None):
        return None
    return _kw_wire("impress_set_background", {
        "slide": step.get("slide"),
        "color": color, "image_path": image_path,
    })


def _wire_insert_shape(step):
    slide = step.get("slide")
    if slide is None:
        return None
    return _kw_wire("impress_insert_shape", {
        "slide": slide,
        "preset":     step.get("preset", "textbox"),
        "x":          step.get("x", "1cm"),
        "y":          step.get("y", "1cm"),
        "width":      step.get("width", "10cm"),
        "height":     step.get("height", "3cm"),
        "text":       step.get("text"),
        "name":       step.get("name"),
        "fill_color": step.get("fill_color"),
        "font_size":  step.get("font_size"),
        "font_color": step.get("font_color"),
        "bold":       step.get("bold"),
        "italic":     step.get("italic"),
        "alignment":  step.get("alignment"),
    })


def _wire_insert_image(step):
    slide = step.get("slide"); image_path = step.get("image_path")
    if slide is None or not image_path:
        return None
    return _kw_wire("impress_insert_image", {
        "slide": slide,
        "image_path": image_path,
        "x": step.get("x", "1cm"),
        "y": step.get("y", "1cm"),
        "width":  step.get("width"),
        "height": step.get("height"),
        "name":   step.get("name"),
    })


def _wire_insert_media(step):
    slide = step.get("slide"); media_path = step.get("media_path")
    if slide is None or not media_path:
        return None
    return _kw_wire("impress_insert_media", {
        "slide": slide,
        "media_path": media_path,
        "x": step.get("x", "1cm"),
        "y": step.get("y", "1cm"),
        "width":  step.get("width"),
        "height": step.get("height"),
        "name":   step.get("name"),
    })


def _wire_insert_chart(step):
    slide = step.get("slide")
    if slide is None:
        return None
    categories = step.get("categories")
    values = step.get("values")
    if not values:
        return None
    # Form-2-style emission protocol borrowed from calc: count check.
    cat_len = (len(categories) if isinstance(categories, list)
               else (len(categories.split("|"))
                     if isinstance(categories, str) else 0))
    for v in values:
        if isinstance(v, list):
            v_len = len(v)
        elif isinstance(v, str):
            v_len = len(v.split("|"))
        else:
            v_len = 0
        if cat_len and v_len and cat_len != v_len:
            return None
    return _kw_wire("impress_insert_chart", {
        "slide": slide,
        "categories": categories,
        "values": values,
        "chart_type": step.get("chart_type", "column"),
        "title": step.get("title"),
        "x": step.get("x", "2cm"),
        "y": step.get("y", "2cm"),
        "width":  step.get("width",  "18cm"),
        "height": step.get("height", "12cm"),
        "name":   step.get("name"),
    })


def _wire_insert_table(step):
    slide = step.get("slide")
    rows = step.get("rows"); cols = step.get("cols")
    if slide is None or rows is None or cols is None:
        return None
    # Pass position/size through as-is (no hard-coded defaults). When all
    # four are omitted, the op falls into "replace empty content
    # placeholder" mode — see op_impress_insert_table.
    return _kw_wire("impress_insert_table", {
        "slide": slide,
        "rows": int(rows), "cols": int(cols),
        "x": step.get("x"),
        "y": step.get("y"),
        "width":  step.get("width"),
        "height": step.get("height"),
        "data":   step.get("data"),
        "name":   step.get("name"),
    })


def _wire_set_master_color(step):
    role = step.get("role"); color = step.get("color")
    if not role or not color:
        return None
    return _kw_wire("impress_set_master_color",
                    {"role": role, "color": color})


def _wire_set_app_setting(step):
    key = step.get("key"); value = step.get("value")
    if not key or value is None:
        return None
    return _kw_wire("impress_set_app_setting",
                    {"key": key, "value": value})


def _wire_export_file(step):
    path = step.get("path")
    if not path:
        return None
    return _kw_wire("impress_export_file", {
        "path": path,
        "format": step.get("format", "auto"),
        "slide": step.get("slide"),
    })


_WIRE_HANDLERS = {
    "impress_set_text":               _wire_set_text,
    "impress_set_font":               _wire_set_font,
    "impress_set_paragraph":          _wire_set_paragraph,
    "impress_set_text_color":         _wire_set_text_color,
    "impress_delete_shape":           _wire_delete_shape,
    "impress_move_shape":             _wire_move_shape,
    "impress_set_shape_size":         _wire_set_shape_size,
    "impress_new_slide":              _wire_new_slide,
    "impress_delete_slide":           _wire_delete_slide,
    "impress_duplicate_slide":        _wire_duplicate_slide,
    "impress_move_slide":             _wire_move_slide,
    "impress_set_slide_orientation":  _wire_set_slide_orientation,
    "impress_set_slide_size":         _wire_set_slide_size,
    "impress_set_notes":              _wire_set_notes,
    "impress_set_transition":         _wire_set_transition,
    "impress_set_background":         _wire_set_background,
    "impress_insert_shape":           _wire_insert_shape,
    "impress_insert_image":           _wire_insert_image,
    "impress_insert_media":           _wire_insert_media,
    "impress_insert_chart":           _wire_insert_chart,
    "impress_insert_table":           _wire_insert_table,
    "impress_set_master_color":       _wire_set_master_color,
    "impress_set_app_setting":        _wire_set_app_setting,
    "impress_export_file":            _wire_export_file,
}


def step_to_wire(step: Dict[str, Any]) -> Optional[str]:
    """Convert a decomposer JSON action dict to an AST-parseable
    ``impress_xxx(arg=val, ...)`` wire string. Returns None if the
    JSON is malformed."""
    handler = _WIRE_HANDLERS.get(step.get("action") or "")
    if not handler:
        return None
    return handler(step)


# ====================================================================
# Build functions (JSON → full runtime script for cli_run_uno)
# ====================================================================

def _build_generic(op_name: str, inputs: Dict[str, Any], logger) -> Optional[str]:
    """Default builder — takes inputs minus ``__current_domain`` /
    ``action`` and forwards them to op_impress_<name>.

    Used for actions whose kwargs map 1:1 to the op signature.

    Also captures the action's target slide (when present) into a module
    global ``_pa_target_slide`` BEFORE the op call. The cli_run_uno
    post-block reads this and switches the view to that slide so the
    next screenshot reflects the edit instead of whichever slide
    happened to be visible. Mirrors the calc_* sheet-tracking design.
    """
    domain = inputs.get("__current_domain")
    kw = {k: v for k, v in inputs.items()
          if k not in ("__current_domain", "action")}
    pre_lines: list = []
    slide_val = kw.get("slide")
    if slide_val is not None:
        # Stash the slide selector verbatim — the post-block re-resolves
        # via _impress_get_slide so it handles either int or name.
        pre_lines.append(f"_pa_target_slide = {slide_val!r}")
    return _emit_script(pre_lines + [_kw_call(op_name, kw)], domain, logger)


def _build_set_text(inputs, logger):
    return _build_generic("op_impress_set_text", inputs, logger)


def _build_set_font(inputs, logger):
    return _build_generic("op_impress_set_font", inputs, logger)


def _build_set_paragraph(inputs, logger):
    return _build_generic("op_impress_set_paragraph", inputs, logger)


def _build_set_text_color(inputs, logger):
    return _build_generic("op_impress_set_text_color", inputs, logger)


def _build_delete_shape(inputs, logger):
    return _build_generic("op_impress_delete_shape", inputs, logger)


def _build_move_shape(inputs, logger):
    return _build_generic("op_impress_move_shape", inputs, logger)


def _build_set_shape_size(inputs, logger):
    return _build_generic("op_impress_set_shape_size", inputs, logger)


def _build_new_slide(inputs, logger):
    return _build_generic("op_impress_new_slide", inputs, logger)


def _build_delete_slide(inputs, logger):
    return _build_generic("op_impress_delete_slide", inputs, logger)


def _build_duplicate_slide(inputs, logger):
    return _build_generic("op_impress_duplicate_slide", inputs, logger)


def _build_move_slide(inputs, logger):
    return _build_generic("op_impress_move_slide", inputs, logger)


def _build_set_slide_orientation(inputs, logger):
    return _build_generic("op_impress_set_slide_orientation", inputs, logger)


def _build_set_slide_size(inputs, logger):
    return _build_generic("op_impress_set_slide_size", inputs, logger)


def _build_set_notes(inputs, logger):
    return _build_generic("op_impress_set_notes", inputs, logger)


def _build_set_transition(inputs, logger):
    return _build_generic("op_impress_set_transition", inputs, logger)


def _build_set_background(inputs, logger):
    return _build_generic("op_impress_set_background", inputs, logger)


def _build_insert_shape(inputs, logger):
    return _build_generic("op_impress_insert_shape", inputs, logger)


def _build_insert_image(inputs, logger):
    return _build_generic("op_impress_insert_image", inputs, logger)


def _build_insert_media(inputs, logger):
    return _build_generic("op_impress_insert_media", inputs, logger)


def _build_insert_chart(inputs, logger):
    return _build_generic("op_impress_insert_chart", inputs, logger)


def _build_insert_table(inputs, logger):
    return _build_generic("op_impress_insert_table", inputs, logger)


def _build_set_master_color(inputs, logger):
    return _build_generic("op_impress_set_master_color", inputs, logger)


def _build_set_app_setting(inputs, logger):
    return _build_generic("op_impress_set_app_setting", inputs, logger)


def _build_export_file(inputs, logger):
    return _build_generic("op_impress_export_file", inputs, logger)


_BUILD_HANDLERS = {
    "impress_set_text":               _build_set_text,
    "impress_set_font":               _build_set_font,
    "impress_set_paragraph":          _build_set_paragraph,
    "impress_set_text_color":         _build_set_text_color,
    "impress_delete_shape":           _build_delete_shape,
    "impress_move_shape":             _build_move_shape,
    "impress_set_shape_size":         _build_set_shape_size,
    "impress_new_slide":              _build_new_slide,
    "impress_delete_slide":           _build_delete_slide,
    "impress_duplicate_slide":        _build_duplicate_slide,
    "impress_move_slide":             _build_move_slide,
    "impress_set_slide_orientation":  _build_set_slide_orientation,
    "impress_set_slide_size":         _build_set_slide_size,
    "impress_set_notes":              _build_set_notes,
    "impress_set_transition":         _build_set_transition,
    "impress_set_background":         _build_set_background,
    "impress_insert_shape":           _build_insert_shape,
    "impress_insert_image":           _build_insert_image,
    "impress_insert_media":           _build_insert_media,
    "impress_insert_chart":           _build_insert_chart,
    "impress_insert_table":           _build_insert_table,
    "impress_set_master_color":       _build_set_master_color,
    "impress_set_app_setting":        _build_set_app_setting,
    "impress_export_file":            _build_export_file,
}


def action_to_runtime_script(action_type: str,
                              inputs: Dict[str, Any],
                              logger=None) -> Optional[str]:
    """Given an ``impress_*`` action_type + parsed kwargs, build the
    full runtime script ready for cli_run_uno on the VM.

    Returns None when the kwargs are malformed; the planner's next
    turn will see [Last cli_run output] empty and re-plan.
    """
    builder = _BUILD_HANDLERS.get(action_type)
    if builder is None:
        return None
    return builder(inputs, logger)

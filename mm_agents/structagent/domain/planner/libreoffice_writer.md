# LibreOffice Writer domain knowledge

This block is injected into the planner and init_ledger prompts ONLY
when the task's domain prefix is `[libreoffice_writer]`. It does NOT
apply to other domains.

## Path: structured `writer_*` JSON actions only

Writer operations go through structured **`writer_*` actions** (flat
JSON kwargs). The framework deterministically builds the underlying
UNO Python from the kwargs — actors never write Python / UNO source,
planners never plan it. `cli_run_uno` is OFF for libreoffice_writer
(stripped from the actor's action space), exactly as it is for
libreoffice_calc / libreoffice_impress.

Available `writer_*` actions — the full schema (per-action kwargs) is
in the "Writer structured actions" block injected into the actor's
prompt:

* **Paragraph / character format**: `writer_line_spacing`,
  `writer_font_size`, `writer_set_font`, `writer_alignment`,
  `writer_case_conversion`, `writer_capitalize_words`,
  `writer_strikethrough`, `writer_subscript_text`, `writer_set_color`,
  `writer_remove_highlighting`.
* **Structure**: `writer_insert_page_break`, `writer_insert_table`,
  `writer_text_to_table`, `writer_insert_paragraph_breaks`,
  `writer_tabstops_split_align`, `writer_page_numbers`.
* **Content**: `writer_paste_at` (paste the clipboard at a precise
  paragraph — pair with `calc_copy_to_clipboard` for cross-app data
  moves), `writer_insert_image`, `writer_find_replace` (regex
  replace, or dedup paragraphs), `writer_cross_reference`.
* **App / export**: `writer_default_font`, `writer_export_pdf`.

Each `<step>`'s `<text>` SHOULD name the action + kwargs (e.g.
"Use writer_alignment with alignment='center', target_indices=[0]");
the actor builds the JSON deterministically.

Why structured (not `cli_run` + python-docx): LibreOffice has the
document open in memory, and a later save from that open window
overwrites any external on-disk edit. The `writer_*` actions edit the
in-memory document via the UNO bridge, so the change lives in the open
instance and survives that save. `cli_run` +
python-docx edits the file on disk while LO's stale in-memory copy
then overwrites it on `ctrl+s` — that route is BANNED for Writer.

## Bringing tabular data from another app into Writer

When the task is "insert a table with data from <a spreadsheet / another
source>", the table is created BY THE PASTE — do NOT pre-insert an
empty table.

Correct, two steps total:
  1. (source app) `calc_copy_to_clipboard` the range you need.
  2. (writer) `writer_paste_at` with `after_heading=` or
     `paragraph_index=` at the target location.
LibreOffice renders pasted spreadsheet cells as a NATIVE Writer table
automatically — step 2 alone produces the finished table.

Do NOT plan `writer_insert_table` followed by a paste. That inserts a
SEPARATE empty table; the paste then drops its own table next to it,
leaving you with a duplicate (an empty table + the data table) — the
exact failure `table_after_heading` verify reports as
"found N tables, expected 1 (duplicate paste?)".

`writer_insert_table` is ONLY for creating a blank table you then fill
cell-by-cell with typed values — never as a container for a paste.

If you have already created duplicate tables, use `writer_delete_table`
(`after_heading=` de-duplicates a section to one table) to roll back.

## When to fall back to GUI

GUI (click / type / hotkey, Patterns 1-12) is reserved for what no
`writer_*` action covers:

* **Transient UI state** — success criterion is "a dialog stays
  open" / "wizard mid-step visible", not a persisted document
  property.
* **Real-time collaboration / document sharing** — UNO has no API;
  the share flow is purely GUI.


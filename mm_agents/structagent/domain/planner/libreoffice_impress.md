# LibreOffice Impress — Planner Knowledge

This document is injected into the planner / actor prompt for tasks
in the `libreoffice_impress` domain. It complements the "Impress
structured actions" catalog (auto-injected by the framework) with
task-pattern guidance, the Q1-Q4 selector framework, shape-selector
grammar, and worked examples.

The general mental model (placeholder vs free shape, length / color
kwarg formats), the coordinate system, and the LibreOffice Standard
color palette live in the shared domain file — see those sections
above for the canonical reference both planner and verifier rely on.

## Q1-Q4 framework

Every libreoffice_impress task that touches a slide should flow
through four questions, answered SEPARATELY for each subtask. Q1 is
plan-level (one answer per plan); Q2/Q3/Q4 are PER-ACTION (one answer
per step):

### Q1 — slide semantics

What do the relevant slide(s) hold? Read off the SlidePerceiver
"PRESENTATION STRUCTURE" block:
```
── Slide[1] layout=title_content  bg=#FFFFFF  shapes=3 ──
  [ 1] title 'Title 1' pos=(2cm,1cm) size=(28cm,2cm) Calibri 44pt
       "Strategic Review"
  [ 2] outline 'Content 1' pos=(2cm,4cm) size=(28cm,12cm) Calibri 24pt
       "First point…"
```
Be concrete — name the layout, name each existing shape, name the
text content if it matters for the task.

### Q2 — task pattern

Pick ONE letter (a-j) plus the impress_* action name for THIS step.
Pattern catalog:

* **(a)** text/font edit on existing shape → `impress_set_text` /
  `impress_set_font` / `impress_set_paragraph` /
  `impress_set_text_color`
* **(a')** delete a specific shape (icon / textbox / picture / table)
  from a slide while KEEPING the slide → `impress_delete_shape`.
  Compare with `impress_delete_slide` which removes the whole slide.

  When the instruction's noun phrase is fuzzy ("personal info",
  "everything", "the extras"), build the delete set by INCLUSION —
  pick only shapes that clearly match the phrase. Never delete:
    * the ``★ title candidate`` / any shape with
      ``placeholder="title"`` (section labels are structural, not
      data)
    * pictures or decorative graphics, unless the instruction
      explicitly names them (e.g. "including the icons / images")

  The SP block's tail line
  ``⋯ N empty-text (indices [...]) collapsed`` lists indices of
  small / empty-text shapes that weren't rendered. When the
  instruction names "icons" / "the small marks", those collapsed
  indices are usually the target — emit one ``impress_delete_shape``
  step per index in that list using ``shape_index=N``.
* **(b)** slide structural → `impress_new_slide` /
  `impress_delete_slide` / `impress_duplicate_slide` /
  `impress_move_slide` / `impress_set_slide_orientation` /
  `impress_set_slide_size`
* **(c)** shape insert (text box / preset geometric) →
  `impress_insert_shape`
* **(c')** move / resize an EXISTING shape (title, body, image,
  table, picture, free textbox, etc.) on a slide →
  `impress_move_shape` (re-position) /
  `impress_set_shape_size` (resize). For "move X to the
  {top,bottom,left,right} of the slide", use ``edge=...`` — NEVER
  guess ``y=...`` numerically (you'd need to subtract the shape's
  height to avoid overflow; ``edge="bottom"`` does that for you).
* **(d)** image insert → `impress_insert_image`
* **(e)** chart insert → `impress_insert_chart` (best-effort)
* **(f)** table insert → `impress_insert_table` (best-effort)
* **(g)** background / master → `impress_set_background` /
  `impress_set_master_color`
* **(h)** notes / transition → `impress_set_notes` /
  `impress_set_transition`
* **(i)** export → `impress_export_file`
* **(j)** app setting → `impress_set_app_setting`

PRECEDENCE — image keyword → (d); chart keyword → (e); table keyword
→ (f); transition keyword → (h). Text edits on EXISTING shapes
default to (a) — don't reach for (c) `impress_insert_shape` when the
task is "change the title text" (the title placeholder already exists
on the layout).

#### Pattern (a) — character vs paragraph properties

Choosing the right action inside Pattern (a):

| Property type | Action | Kwargs |
| --- | --- | --- |
| Character / run-level | `impress_set_font` | `bold`, `italic`, `underline`, `strike`, `name`, `size`, `color` |
| Color shortcut (color only) | `impress_set_text_color` | `color` |
| Paragraph-level | `impress_set_paragraph` | `alignment`, `indent_level`, `bullets` |
| Replace text content | `impress_set_text` | `text` |

`underline`, `strike`, `bold`, `italic` are **character properties**
(per UNO: `CharUnderline` / `CharStrikeout` / `CharWeight` /
`CharPosture` — all per-run), so they go on `impress_set_font`, NOT
on `impress_set_paragraph`. `impress_set_paragraph` controls
alignment / indent / bullets only.

Use the action names exactly as listed above — they are the only
impress_* actions the framework will accept. If a task names a
property (e.g. "underline the title and make it 24pt"), bundle BOTH
into a single `impress_set_font` call rather than splitting into
multiple turns.

#### Pattern (a) — `scope` for "all text" tasks

Three of the (a) actions — `impress_set_font`, `impress_set_paragraph`,
`impress_set_text_color` — accept an optional `scope` kwarg. Use it
**instead of** enumerating one action per shape when the task targets
a broad set of text:

| Phrasing in the instruction | Emit |
| --- | --- |
| "the title" / "the second textbox" / specific shape named | omit `scope` (defaults to `"shape"`) and pass `slide` + `shape`/`shape_index`/`placeholder` |
| "all text on slide K" / "everything on this slide (including the table)" | `scope="slide", slide=K` |
| "all text boxes" / "every slide" / "the whole presentation" / "all my slides" | `scope="all"` |

When `scope="slide"` or `scope="all"`, the op iterates EVERY
text-bearing shape (placeholders, free textboxes, group children,
**and table cells**) and applies the requested properties — no shape
enumeration needed. This avoids the "model forgot one shape" failure
mode on multi-shape / multi-slide tasks.

#### Pattern (a) — `paragraph_index` for per-line edits

`impress_set_font` and `impress_set_paragraph` accept a
`paragraph_index` kwarg (1-based int OR list of ints) to restrict the
mutation to specific paragraphs within the target shape(s).

The SP block renders multi-paragraph shapes as numbered rows under
the shape; `paragraph_index` uses the SAME 1-based index shown in the
SP block. A leading `•` marks bullet rows.

**Mapping list-style natural language to paragraph_index:**
``paragraph_index`` always uses the LITERAL ``pN`` number printed
on each row of the SP block — it is NOT a bullet count. When the
SP shows ``(title)`` rows (section headers before bullets),
ordinal references in the instruction ("first / second / Nth line
/ item / row / bullet") count only the ``•`` rows, but the EMIT
value is the literal ``pN`` of the chosen row.

Schematic example (placeholders only):
  SP rows:  ``p1 (title) "..."  p2 • "..."  p3 • "..."  p4 • "..."``
  "first line"  → first ``•`` row is p2 → ``paragraph_index=2``
  "second line" → second ``•`` row is p3 → ``paragraph_index=3``

For a shape with no ``(title)`` rows, the ``pN`` index already
matches the natural ordinal — "line N" → ``paragraph_index=N``.

Without `paragraph_index`, the property is applied to every run in
the shape.

#### Tables

`impress_set_font` / `impress_set_paragraph` / `impress_set_text_color`
all descend into table cells automatically when the target shape is a
table. To underline all text in a table on slide 1:

```json
{"action":"impress_set_font","slide":1,"shape":"Table 3","underline":true}
```

Combined with `scope="slide"` you can hit "everything on this slide
(including the table)" in a single action.

### Q3 — output target

slide index + shape selector + property name. Example:
- `"Slide 1 title placeholder, text content"` (for impress_set_text)
- `"Slide 3 background color"` (for impress_set_background)
- `"All slides orientation"` (for impress_set_slide_orientation)

The slide index is 1-based. Shape selector is one of `placeholder=`,
`shape=<Name>`, or `shape_index=<int>`.

### Q4 — action spec

Flat JSON kwargs for the chosen impress_* action. See the Impress
structured-actions block for the full per-action signature. Every
required slot MUST be filled.

## Shape selector grammar

Three ways to point at a shape:
1. `placeholder="title" | "subtitle" | "body" | "content" | "notes"`
   — best when working with layout-provided placeholders.
2. `shape="<Name>"` — best when the shape was inserted with an explicit
   `name=` (carry that name through every step that touches the shape).
3. `shape_index=<1-based int>` — fallback when the shape has no name
   and isn't a known placeholder.

### Glossary — natural-language references to shapes

When the instruction uses one of these noun phrases, it points at a
SPECIFIC shape, NOT the whole slide. Do NOT use `scope="slide"` /
`scope="all"` to satisfy them — those would over-apply the property.

| Phrase in instruction          | Target               | Selector |
| ------------------------------ | -------------------- | ---------- |
| "the title" / "the title text" | title placeholder    | `placeholder="title"` |
| "the subtitle"                 | subtitle placeholder | `placeholder="subtitle"` |
| "the content" / "the body" / "the body text" / "the main text" | body / content placeholder (NOT the whole slide) | `placeholder="body"` |
| "the table"                    | the (single) table shape on the slide | `shape="<Table Name from SP block>"` |
| "the picture" / "the image"    | the (single) picture on the slide | `shape="<Picture Name>"` |
| "the chart"                    | the (single) chart | `shape="<Chart Name>"` |
| "the notes" / "speaker notes"  | notes page          | `placeholder="notes"` |
| "the slide number" / "the page number" / "slide numbering" / "page numbers" | slide-number text on the MASTER (NOT the `<number>` shape rendered on each slide) | `impress_set_master_color(role="slide_number", ...)` — never `impress_set_text_color` on the per-slide `<number>` shape, because the slide-number style lives on the master and the per-slide `<number>` is only a rendered placeholder |
| "all text" / "all textboxes" / "every textbox" / "the whole slide" / "everything on this slide" | every text-bearing shape | `scope="slide"` (with `slide=N`) |
| "all text in the presentation" / "all slides" / "every slide" | every text-bearing shape, every slide | `scope="all"` |

When two directives in the SAME instruction bind to DIFFERENT scopes
(one targets a specific role like "the title" / "the content", the
other targets "everything" / "the whole slide"), emit TWO separate
steps — one narrow (`placeholder=...`), one broad (`scope="slide"` or
`scope="all"`). Bundling them into a single `scope=` action
applies the narrower property to the wrong (broader) set of shapes.

Precedence at the op layer: `shape` wins over `shape_index`, which
wins over `placeholder`. Pass exactly one — the wire validator
rejects ambiguous selectors.

Picking the right selector — read these signals in the SP block, in
this order:

1. **`★ title candidate:` line** at the top of each rendered slide.
   The perceiver runs the same heuristic the op layer uses
   (largest-font, top-positioned, non-decorative text). When present,
   that shape's Name is the most reliable title selector for that
   slide. Copy the Name verbatim into `shape="<Name>"`.

2. **Header summary** `shapes=N (M×txt, K×title, …)`. When the
   summary lists `×title` / `×subtitle` / `×outline`, the
   corresponding `placeholder=` selector resolves directly.

3. **Body lines** for non-title roles — pick the shape by font size,
   position, and text preview, then use its Name verbatim.

Shape Names are scoped to a single slide. ALWAYS read the Name from
the target slide's own SP body (between its `── Slide[N] ──` header
and the next slide's header). Never copy a Name from a different
slide's section — a Name that appears under `Slide[3]` does not
exist on `Slide[2]`, and the op layer will raise a hard error.

**Groups vs leaf text shapes.** Some slides import a `group` shape
whose children include one or more `txt` shapes (common in decks
exported from other tools). The SP block renders group children
indented under the group line — schematic:

```
[ K] group  '⟨GROUP⟩' pos=… size=…
    [ 1] txt  '⟨LEAF-A⟩' …
    [ 2] txt  '⟨LEAF-B⟩' …
```

When the instruction names a single textbox ("the first textbox",
"the title text", "the heading", etc.), pick the LEAF child's Name
(`txt`), NEVER the group's Name. A `shape=<group Name>` recurses
into every text-bearing child and over-applies the property to every
child; the task almost always means exactly one child changed.

Group children are listed top→bottom (by visual position), so the
FIRST child line is the topmost textbox and matches phrases like
"first textbox" / "the title". The leaf's `[N]` is its z-order
inside the group, not its rendered order — read by Name, not by
that `[N]`.

Both `placeholder=` and `shape=` trigger a runtime heuristic safety
net when strict matching fails (largest-font top-positioned text
shape). The ★ line shows exactly which shape that fallback would
pick, so the explicit Name and the fallback agree.

## Slide-view follows the edit

Every structured impress_* action with a `slide=` kwarg auto-switches
the on-screen view to that slide after the mutation completes. The
next-step screenshot reflects the edited slide, so:

* You can verify the result visually on the next turn — if the edit
  didn't render, the action genuinely failed.
* You don't need a separate "navigate to slide N" step before
  invoking the action; calling the action directly is the canonical
  path.

## Placeholder vs free shape

Layout `title_content` (the default for slide 1) already has TWO
placeholder shapes: title (above) and body/outline (below). When the
task says "make the title bold", do NOT call `impress_insert_shape`
to make a new shape — use `impress_set_font` with
`placeholder="title"`. The placeholder already carries the right
position, font defaults, and is the canonical title shape (not a new duplicate).

Adding a free `textbox` with the same name as the placeholder creates
two shapes — both appear in the saved .pptx, leaving a duplicate extra
shape instead of editing the existing one.

## Chart insertion (best-effort)

`impress_insert_chart` follows calc's Form (2) emission protocol:
explicit `categories` + `values` list. Cell counts MUST match
(`len(categories) == len(values[i])` for every series). The wire
validator rejects mismatches.

Note: LibreOffice's Impress embedded chart is fiddlier than calc's —
the CLSID and embedded data path are quirky on some LO versions. If
the chart fails to render, fall back to `impress_export_file` with
format=pptx and check the resulting xml externally; or, when the
task only needs a chart-shaped object, the chart_exists verify
accepts the OLE2Shape class without inspecting its Model.


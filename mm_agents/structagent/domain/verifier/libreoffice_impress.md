# LibreOffice Impress — Domain Knowledge

## Verify recipes — OMIT anchors by default

The same "no hallucinated anchors" rule from calc applies:

* `shape_text` — use `expected_substring` when the task names exact
  wording; use no expectation at all when the task is "add a title
  with appropriate text" (no specific value).
* `font_props` — only emit `color` / `size` / `bold` when the task
  literally names them (e.g., "make it 60pt", "color it yellow ⇒
  #FFFF00"). Don't guess hex codes.
* `slide_background` — same rule: only emit `color` when the task
  states the color literally.
* `chart_exists` / `table_exists` — emit `chart_type` / `rows` /
  `cols` only when literally named.

A hallucinated `color="#1E2761"` for a task that names no color
guarantees verify FAIL forever even when the agent's work is
correct.

## Worked example — change a title's font color

Task: "Change the title on slide 1 to red."

Planner emits:
```xml
<plan>
  <data_layout>
    <q1_slide_semantics>Slide 1 has a title placeholder containing the
    deck heading; layout is title_content.</q1_slide_semantics>
  </data_layout>
  <strategy>Recolor the title placeholder text on slide 1 via
  impress_set_font.</strategy>
  <step>
    <q2_task_pattern>(a) impress_set_font for title text color</q2_task_pattern>
    <q3_output_target>Slide 1 title placeholder, font color</q3_output_target>
    <q4_action_spec>slide=1; placeholder="title"; color="#FF0000"</q4_action_spec>
    <text>Use impress_set_font with slide=1, placeholder="title",
    color="#FF0000".</text>
    <expected_post_state>Slide 1's title text is rendered in red.</expected_post_state>
  </step>
</plan>
```

init_ledger emits:
```json
{"verify": {
  "kind": "impress_verify",
  "impress_checks": [
    {"op": "font_props", "slide": 1, "placeholder": "title",
     "color": "#FF0000"}
  ]
}}
```

## Worked example — add a new slide

Task: "Add a blank slide at the end of the deck."

```xml
<step>
  <q2_task_pattern>(b) impress_new_slide append blank</q2_task_pattern>
  <q3_output_target>Last slide position, blank layout</q3_output_target>
  <q4_action_spec>position="end"; layout="blank"</q4_action_spec>
  <text>Use impress_new_slide with position="end", layout="blank".</text>
  <expected_post_state>Slide count increased by 1, new slide is blank.</expected_post_state>
</step>
```

Verify:
```json
{"op": "slide_count", "expected": <N+1>}
```

where N is the original slide count (from the SP block).

## Anti-patterns

* **Don't** chain a GUI Pattern (click / type) for spreadsheet-like
  mutations. Use the structured impress_* actions exclusively for
  document-property edits. GUI is for transient UI state (dialogs,
  menus, navigation).
* **Don't** call `impress_insert_shape` and then immediately call
  `impress_set_text` on the new shape — `impress_insert_shape` already
  takes `text=` directly. Bundling saves one step + one verify cycle.
* **Don't** emit `placeholder="title"` AND `shape="Title 1"` AND
  `shape_index=1` together — exactly ONE selector. The wire layer
  rejects ambiguous selectors.
* **Don't** invent a slide title in the verify when the task doesn't
  specify one. The `shape_text` verify must use the exact substring
  the task names; arbitrary titles fail the task.

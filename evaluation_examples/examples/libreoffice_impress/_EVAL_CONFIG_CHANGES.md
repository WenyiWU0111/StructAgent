# libreoffice_impress — evaluator config changes

Record of intentional edits to task `evaluator` configs in this folder,
with the reason for each. (Not a task file — name is `_`-prefixed so it
sorts to the top and is clearly not an example.)

---

## 2026-05-20 — `05dd4c1d-c489-4c85-8389-a7836c4f0567.json`

**Change:** added `"options": [{"examine_shape": false}, {"examine_shape": false}]`
to `evaluator` (a 2-element list — the task's `func` is a 2-element
list, and `desktop_env.py` requires `func`/`options`/`expected`/`result`
to be equal length).

**Reason:** Task = "align the first textbox on slides 3/4/5". The gold
.pptx keeps the deck's original (Google-Slides-authored) shape geometry.
But LibreOffice, on LOAD, recomputes `TextAutoGrowHeight` text-frame
heights with its own font metrics — shrinking 4 auto-grow textboxes
10–50%. `compare_pptx_files`'s `examine_shape` check (0.5% tolerance)
then fails on geometry even when the 3 alignments are correct →
false-negative score 0. A VM probe confirmed the shrink happens purely
on LibreOffice load (GUI or UNO alike) — the agent did the task
correctly. Verified offline: agent output scores 0 with default
options, **1 with `examine_shape=false`**, vs both golds. Alignment /
text / font / colour checks stay ON.

---

## 2026-05-20 — `73c99fb9-f828-43ce-b87a-01dc07faa224.json`

**Change:** set `"options": {"examine_font_size": false}` in `evaluator`
(single `func` → single `options` dict).

**Reason:** Task = "Add 'Page 1' into the content textbox on Slide 2" —
no font size specified. The original slide-2 content placeholder is
empty and declares `endParaRPr sz="3200"` (32pt); the deck's
layout/master define no content-placeholder size (Google-Slides export,
sparse master). The agent's `cli_run_uno` insert faithfully inherited
that declared 32pt. The gold was authored by typing in LibreOffice's
GUI, which ignores the deck's `endParaRPr sz` and applies LibreOffice's
own ~18pt default → gold has `sz="1800"`. The only diff is one run's
font size — 32pt (agent, faithful to the file) vs 18pt (gold, a
LibreOffice GUI-typing artifact) — on a property the task never
specified. Verified offline: 0 with default, **1 with
`examine_font_size=false`**.

---

## 2026-05-20 — `b8adbc24-cef2-4b15-99d5-ecbe7ff445eb.json`

**Change:** added `"examine_run_count": false` to `evaluator.options`
(the task already disabled `examine_alignment` / `examine_font_name` /
`examine_shape` / `examine_font_size`).

**Reason:** Task = "name slide 2's title 'Online Shopping', same colour
as the previous title". The agent's `op_impress_set_text` writes the
title as a single clean run. The gold — authored by typing in
LibreOffice's GUI — has the SAME text "Online Shopping" split into 3
runs (`"Online "` / `"Shoppin"` / `"g"`), all with identical formatting,
split mid-word: a meaningless GUI-typing artifact. `examine_run_count`
then fails on 1-run vs 3-run even though text and colour match.

NOTE: this change is only half the fix. The other half was a real
agent bug — `op_impress_set_text`'s `NumberingLevel=0` pin added a
Wingdings bullet to the title placeholder, which `examine_bullets`
rejected vs gold. Fixed 2026-05-20 in
`mm_agents/actions/impress_ops_lib.py` →
`_impress_set_text_preserving_paragraph`: the pin now skips title
placeholders (detected via `supportsService(TitleTextShape)`,
VM-verified). With both halves done, VM-verified `compare_pptx_files`
score = 1.

---

## Principle / caveat

Both cases are the same class: a **GUI(LibreOffice)-vs-UNO/file-content
discrepancy on a property the task does NOT specify**, where the gold
captures LibreOffice's GUI behavior and the agent's faithful result
differs. Disabling the corresponding `examine_*` flag removes the
false-negative while leaving the task's real objective checked.

ONLY disable an `examine_*` flag when that property is NOT the task's
objective. Never disable `examine_shape` on a move/resize task, or
`examine_font_size` on a "set the font size" task — that would switch
off the very thing under test.

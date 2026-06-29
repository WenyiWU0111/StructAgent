# LibreOffice Impress — shared mental model + colors

This block is read by BOTH the planner (so the action it emits sets
the right hex / picks the right placeholder selector) AND the
init_ledger verifier-author (so the verify spec it writes matches
the exact hex / shape selector the saved .pptx will contain). The
two audiences MUST agree on these values; that's why this section is
factored out instead of duplicated across planner/ and verifier/.

## Mental model

A presentation is a sequence of slides (`doc.DrawPages`). Each slide
has a layout (`title` / `title_content` / `blank` / …), an optional
background fill, a notes page, and a set of shapes. Shapes come in
two flavors:

* **Placeholder shapes** — pre-positioned, pre-styled by the layout
  (title, subtitle, body/content/outline). Their service names contain
  `TitleTextShape` / `SubTitleTextShape` / `OutlineTextShape`. Setting
  text on a placeholder by `placeholder="title"` is the safe path
  for "edit the title" tasks.
* **Free shapes** — added explicitly via `impress_insert_shape` /
  `impress_insert_image`. Identify them by `name=<Name>` or
  `shape_index=<N>` (1-based z-order).

Every length kwarg accepts `"2cm"` / `"0.5in"` / `"12pt"` / a bare int
(treated as raw 1/100 mm, the UNO unit). Every color kwarg accepts
`"#RRGGBB"` hex.

## Coordinate system

LibreOffice UNO uses **1/100 mm** internally. The structured actions
all parse human-readable length strings — you should never write raw
1/100 mm values unless the task literally specifies them in those
units. Standard A4-landscape slide ≈ 25.4cm × 19.05cm.

## Named colors — LibreOffice Standard palette

When the task names a color in plain words (e.g. "make it green",
"set the title to blue"), use the LibreOffice **Standard palette**
hex below — NOT the CSS / HTML hex. LibreOffice writes its own
palette's `#RRGGBB` into the saved file, and several common color names
diverge between the two palettes (notably green, blue, magenta).

|    Name | Hex       | Note vs. CSS                          |
| ------- | --------- | ------------------------------------- |
| Black   | `#000000` | same                                  |
| White   | `#FFFFFF` | same                                  |
| Yellow  | `#FFFF00` | same                                  |
| Gold    | `#FFBF00` | LO-only                               |
| Orange  | `#FF8000` | LO uses 8000, CSS uses A500           |
| Brick   | `#FF4000` | LO-only                               |
| Red     | `#FF0000` | same                                  |
| Magenta | `#BF0041` | CSS is `#FF00FF` (LO calls that "Light Magenta 2") |
| Purple  | `#800080` | same                                  |
| Indigo  | `#55308D` | CSS is `#4B0082`                      |
| Blue    | `#0000FF` | PRIMARY blue (same as CSS). LO color-picker "Blue" swatch is `#2A6099` (darker) — use that ONLY when the task explicitly says "LO Blue swatch" / picks a darker variant |
| Teal    | `#158466` | CSS is `#008080`                      |
| Green   | `#00A933` | **CSS is `#008000`; pure `#00FF00` is LO "Light Green 2" / Lime** |
| Lime    | `#81D41A` | LO-only                               |

Rule: bare color names ("red", "green", "blue", …) → the row above.

When the task names a **shade qualifier** ("dark red 2", "light green
1", "dark blue 2") it is referring to LibreOffice's Standard-palette
shade variants (the rows below the base color in LO's color dropdown).
Use LibreOffice's own Standard-palette shade hexes below:

|         Variant | Hex       |
| --------------- | --------- |
| Dark Red 1      | `#CC0000` |
| Dark Red 2      | `#C9211E` |
| Light Red 1     | `#F0CDCD` |
| Light Red 2     | `#FFA6A6` |
| Dark Orange 1   | `#E8A33D` |
| Dark Orange 2   | `#BF6E04` |
| Light Orange 1  | `#FFD8B1` |
| Dark Yellow 1   | `#E6B412` |
| Dark Yellow 2   | `#A28200` |
| Dark Green 1    | `#006B2D` |
| Dark Green 2    | `#003D1A` |
| Light Green 1   | `#CCFFCC` |
| Light Green 2   | `#00FF00` |
| Dark Blue 1     | `#2A6099` |
| Dark Blue 2     | `#1C4587` |
| Light Blue 1    | `#B4C7E7` |
| Dark Purple 1   | `#6B238E` |
| Dark Magenta 1  | `#8E1C4A` |
| Dark Magenta 2  | `#5C0F30` |

If a qualifier doesn't appear here (e.g. "navy blue", "forest green"),
fall back to a sensible CSS-style hex — but PREFER the LO variant when
the wording uses "dark/light X 1/2" exactly.

The phrasing "use exactly these colors—no variations (e.g., no dark
red, light green, etc.)" in a task means the planner MUST use the
BARE-NAME row from the upper table, never a qualified variant.

Instructions that emphasize strict-color language ("use exactly
these colors—no variations", "no dark red, no light green") are
reinforcing this rule — they want the Standard-palette entry, not a
CSS primary.

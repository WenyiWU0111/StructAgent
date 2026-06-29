# GIMP domain knowledge

This block is injected into the planner and init_ledger prompts ONLY when
the task's domain prefix is `[gimp]`. It does NOT apply to other domains.

## GUI is the primary path

GIMP is a raster image editor whose operations live behind menus:
`Image → Mode / Transform / ...`, `Colors → Brightness-Contrast /
Hue-Saturation / ...`, `Filters → ...`, `Layer → Transparency / ...`.

For nearly every `[gimp]` task the actor should drive the **GUI**:

* Image-editing tasks (brightness / saturation / contrast / mirror /
  flip / transparent background / color-mode conversion / resize / etc.)
  → GIMP menus on the open image, then `File → Export As` to write
  the result.
* GIMP preferences toggles → `Edit → Preferences` (writes to
  `~/.config/GIMP/<ver>/gimprc` on dialog OK / app exit).

CLI alternatives are **mostly not viable**:

* `gimp -i -b '(...)'` (Script-Fu console batch) requires correct
  Script-Fu syntax for every operation and tends to fail silently when
  the script has a typo. Don't write `cli_run` actions invoking
  Script-Fu unless the instruction itself supplies the script.
* Image manipulation via `convert` / `mogrify` (ImageMagick) is a
  different tool — its output differs from GIMP's (metadata, palette
  handling, color profiles), so for a `[gimp]` task produce the result
  through GIMP. Stay in GIMP unless the task explicitly names
  ImageMagick.

CLI is appropriate only for: checking that the output file exists
(`test -s <path>`), grepping `gimprc` content, or listing `~/Desktop`
to confirm input files.

## Save vs Export — a recurring trap

GIMP's `File → Save` writes the native `.xcf` format (GIMP's project
file). To produce `.png` / `.jpg` / etc., use `File → Export As`
(keyboard `shift+ctrl+e`). The dialog asks for a filename and
extension; the extension determines the format.

When a task says "save as `foo.png`" / "save it on the Desktop as X",
the actor must use **Export As**, not Save. Verifying that a
non-`.xcf` file appeared on disk is the deterministic check.

## Config / preferences

GIMP user state lives under `~/.config/GIMP/<major.minor>/` (e.g.
`~/.config/GIMP/2.10/`). Common files:

| State | File |
|---|---|
| Application preferences (theme, default tool sizes, dialog positions) | `gimprc` |
| Toolbox / dockable layout | `sessionrc` |
| Custom keyboard shortcuts | `menurc` |
| Color management | `colorrc` |

`gimprc` is plain text: each entry is an S-expression like
`(default-image (...))` or `(quick-mask-color (color-rgba ...))`.
file_grep works against it.

## Color modes supported

GIMP's native `Image → Mode` offers **RGB**, **Indexed (Palette)**,
and **Grayscale** only. **CMYK is NOT supported** in stock GIMP — only
via third-party plugins (Separate+ etc.). A task asking to convert an
image to CMYK in stock GIMP is infeasible — flag it as such rather
than producing a fake result.

## Scope — what GIMP does NOT do

GIMP is a **single-image, raster** editor. It does NOT do:

* Video editing / trimming (use a video tool instead)
* Audio (anything)
* Batch processing across many files from the GUI (one file at a time)
* Vector-only formats (SVG opens as raster import, edits are raster)
* Downloading images from the web (no built-in browser)

Tasks asking for any of the above in stock GIMP are infeasible.


# VLC domain knowledge

## file_grep verify for vlcrc edits

vlcrc is plain text — file_grep is the right `kind`. The pattern
MUST match the full line form including the value, e.g.
`^<key>\s*=\s*<expected_value>\b`. A pattern that only matches
`^<key>\s*=` (no value constraint) passes whenever the key exists
with any value — including the wrong one.

## Verify kind for preference toggles — `a11y_match` is the DEFAULT

For tasks that toggle or set a VLC preference (anything reachable
through `Tools → Preferences` — UI options, hotkeys, volume, splash
behavior, appearance, network, …), the DEFAULT verify kind is
`a11y_match` on the GUI control's post-commit state, NOT `file_grep`
on `vlcrc`.

Use `file_grep` on `vlcrc` ONLY when the exact target key NAME appears
verbatim in the probe's `[VLC config]` "active key=value lines" block.
If the key is not visible in the probe, do not write file_grep with
a guessed key — use `a11y_match`.

Map preference controls to `a11y_match` shapes:

* checkbox setting → `tag=checkbox`, `text=[<control label>]`,
                     `state=[checked]` (or `unchecked`)
* dropdown / mode  → `tag=combobox`, `text=[<chosen value>]`
* color-picker     → `tag=button`,   `text=[<hex or rgb value>]`
* text / spin      → `tag=entry`,    `text=[<typed value>]`

VLC writes the real key to `vlcrc` on Preferences → Save, so the
on-disk config reflects the change once saved — your verify can target
the persisted result without naming the exact key.

## Tasks that produce media files (mp3 / mp4 / png / ...)

Several VLC tasks output a media file (convert / rotate / snapshot /
save) whose correctness is a perceptual property of the content
(audio / video / image similarity) that a local probe cannot compute.

* file_grep on the BINARY content cannot assert anything useful —
  the bytes are an audio / video / image stream, not text. A pattern
  matching the file name string by accident inside the binary header
  is meaningless.
* The strongest local verify is "file exists and is non-empty":
    shell_command `["test", "-s", "<absolute-path>"]`
    `expected_exit_code: 0`
  (argv list, no shell metacharacters — the executor does not spawn
  a shell.)
* Whether the content is perceptually correct cannot be checked by a
  local verify spec — verify the file exists and has the right
  format / size, not its pixel or audio content. Do NOT add
  expected_substring patterns that pretend to.

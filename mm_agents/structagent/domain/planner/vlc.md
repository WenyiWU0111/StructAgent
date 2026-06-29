# VLC domain knowledge

This block is injected into the planner and init_ledger prompts ONLY when
the task's domain prefix is `[vlc]`. It does NOT apply to other domains.

The OSWorld VM baseline (Linux user, sudo password, `ctrl+alt+t`
terminal shortcut) and the actual environment snapshot (existing
config files, running processes) are provided elsewhere — do NOT
duplicate them here. This file documents only the VLC conventions a
reader can find in the public VLC docs.

## Where the success state lives — triage BEFORE choosing verify kind

Not every `[vlc]` task is a vlcrc edit. The task instruction tells you
WHAT to change; before writing verify decide WHERE the post-action
ground truth actually lives:

  (a) A `vlcrc` INI key changed (VLC's own preferences / appearance /
      hotkeys / interface behavior)
       → file_grep on `~/.config/vlc/vlcrc`

  (b) A media file produced on disk (mp3 / mp4 / png / wav / ...
      conversion, rotation, snapshot, recording)
       → shell_command shape (4) `["test", "-s", "<absolute-path>"]`

  (c) A GNOME / Ubuntu system setting changed because VLC is being
      used to produce it (desktop wallpaper / global media key /
      screensaver image / ...)
       → shell_command shape (5) `["gsettings", "get",
         "<schema>", "<key>"]` — schema and key COPIED from the
         perceiver's `[Relevant GSettings schemas]` block

  <!-- [FIX default_app_xdg_mime] Setting the "default app for files
       of type X" on modern GNOME is xdg-mime + ~/.config/mimeapps.list,
       NOT gsettings. Without this carve-out, init_ledger kept
       hallucinating ``gsettings get org.gnome.desktop.default-
       applications.<type>`` which has not existed for years; verify
       reported ``No such key`` every run, outcome stayed pending,
       DONE always rejected, planner died in REPLAN. General fix —
       applies to "default video player", "default image viewer",
       "default browser", etc. Roll back by deleting this (d) branch. -->
  (d) The DEFAULT APPLICATION for a category of files (default video
      player / default image viewer / default web browser / etc.)
       → mechanism: `xdg-mime default <app>.desktop <mime/type>`
         (writes / merges `~/.config/mimeapps.list`).
       → verify: shell_command shape `["xdg-mime", "query", "default",
         "<mime/type>"]` with ``expected_substring: ["<app>.desktop"]``.
       → IMPORTANT: a category like "default video player" spans MANY
         MIME subtypes (``video/mp4``, ``video/x-msvideo``, ``video/webm``,
         ``video/x-matroska``, … dozens). Setting the default for one
         subtype leaves the others unchanged — set it for the whole
         category (all its subtypes), not a single one.
         The actor needs to loop over all MIME subtypes for the
         category — list them from
         ``/usr/share/mime/packages/freedesktop.org.xml`` or
         hardcode the common ones — and run `xdg-mime default
         <app>.desktop <mime/type>` for EACH. Verify on at least 3
         representative MIME subtypes; if any returns the wrong
         desktop file the outcome is not satisfied.
       → NEVER use `gsettings get org.gnome.desktop.default-
         applications.<type>` for this — that schema/key is from old
         GNOME and does not exist on Ubuntu 22.04+. Verify will
         silently fail with ``No such key`` if you write it.

When the instruction names a system-wide concept (wallpaper /
background picture, global shortcut, login screen, screensaver) the
answer is (c). When it names "default <app-category>" the answer is
(d). VLC is the source/tool in both cases; the success state lives
in GNOME settings / xdg-mime, not vlcrc.

For (c) `picture-uri` / `*-uri` image keys: the value must point to a
still image (`.png` / `.jpg` / ...), NEVER a video (`.mp4` / ...) —
GNOME cannot render video as wallpaper. If the source is a video,
the actor must snapshot a frame first; verify on an image extension.

## Taking a video snapshot

VLC menu `Video → Take Snapshot`, or hotkey `shift+s`. Saves to
`~/Pictures/vlcsnap-<timestamp>.png` by default (NOT next to the
video, NOT to Desktop). If the task names a specific output path or
filename, take the snapshot to the default location first, then
`mv ~/Pictures/vlcsnap-*.png <target-path>` — the snapshot filename
is not configurable from the menu.

## Where VLC stores its settings, and INI structure

See the shared domain file — vlcrc path + INI format conventions
are factored out so the planner and the verify-spec author write
the exact same key names / sections.

## Editing vlcrc

Prefer a single `cli_run` over walking the GUI Preferences pages:

* update an existing key:
    `pkill -9 vlc 2>/dev/null; sleep 1; sed -i 's/^<key>=.*/<key>=<new_val>/' ~/.config/vlc/vlcrc`
* append a missing key (creating parent dir / file if needed):
    `pkill -9 vlc 2>/dev/null; sleep 1; mkdir -p ~/.config/vlc && printf '%s\n' '<key>=<new_val>' >> ~/.config/vlc/vlcrc`

The `pkill -9 vlc` prefix is required: VLC overwrites vlcrc with its
in-memory state on clean exit, so any sed/echo edit made while VLC is
running gets clobbered. SIGKILL (-9) prevents the write-back.

The Preferences GUI is paginated and search-driven; many keys live
2-3 panes deep behind a "Show settings: All" toggle, and the field
naming in the GUI does not always match the INI key.


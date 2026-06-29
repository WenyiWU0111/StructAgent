# VLC — shared vlcrc location + INI conventions

Both the planner (when emitting a vlcrc edit) and the init_ledger
verifier-author (when writing a file_grep against vlcrc) need the
same path + INI shape. Factored here to avoid drift.

## Where VLC stores its settings

VLC's user preferences live in a plain INI file at:

  `~/.config/vlc/vlcrc`

The file is **NOT** stored in GSettings or dconf. NEVER write a verify
spec or actor action that uses `gsettings get/set` on a schema like
`org.vlc.*`, `org.gnome.*.vlc`, or
`org.gnome.desktop.default-applications.vlc` — those schemas do not
exist. NEVER read `dconf` paths under `/org/vlc/...` — VLC does not
use the dconf tree at all.

## INI structure

`vlcrc` is a plain INI file: optional `[section]` headers followed by
`<key>=<value>` lines. Keys are LOWERCASE with `-` as the word
separator (NOT dots, NOT camelCase). Generic examples:

```
volume-save=1
qt-system-tray=1
preferred-resolution=-1
```

Some keys store multiple values as a single semicolon-separated
string. The form `<key>=<v1>;<v2>;<v3>;...` is ONE INI value, NOT
several lines.

`key=value` lines may appear under a section header OR at file top
level. When verifying via file_grep, match `^<key>\s*=` anywhere in
the file — do not require a specific section.

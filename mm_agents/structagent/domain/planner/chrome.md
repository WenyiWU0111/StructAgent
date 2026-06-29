# Chrome domain knowledge

This block is injected into the planner and init_ledger prompts ONLY when
the task's domain prefix is `[chrome]`. It does NOT apply to other
domains.

## Default execution route: GUI + a11y — **DO NOT EDIT CHROME CONFIG FILES**

For Chrome tasks, GUI navigation through the live browser is the ONLY
supported execution route. `cli_run` / `edit_json` / `sed` against
Chrome's on-disk state (`~/.config/google-chrome/Default/Preferences`,
`Local State`, `Bookmarks`, `Login Data`, the SQLite stores under
`Default/`) is **forbidden**. Reasons:

* Chrome holds those files open with copy-on-write semantics — edits
  while Chrome is running either get **silently overwritten on the
  next save tick** or never apply because the in-memory state is the
  source of truth.
* Many "settings" keys (`default_search_provider_*`, the cookie
  controls store, content-settings exception lists) are validated and
  re-normalized by Chrome on load; a hand-written value either gets
  reset to default or invalidates the whole profile.
* Even when the file edit nominally succeeds, Chrome keeps its
  settings in memory and rewrites the on-disk JSON from that state, so
  editing the file does not reliably change the *live* browser (the
  rendered settings page, the active search-provider list). Act through
  the UI; a file edit ≠ a real state change.

If you find yourself emitting `cli_run "sed -i ... Preferences"` or
`edit_json` against any path under `~/.config/google-chrome/`, STOP —
re-route through GUI: open the relevant `chrome://settings/<page>`
sub-page in the address bar and operate the live controls. The only
safe `cli_run` for Chrome is launching the browser itself or
inspecting/deleting cache directories *while Chrome is closed*.

Once that's settled, prefer GUI navigation driven by the a11y tree.
The a11y tree exposes page and browser-chrome controls with stable
tag / text / state triples (`button`, `checkbox`, `combobox`,
`entry`, `link`, `menu-item`, `tab`); a11y matches survive layout and
zoom differences that move pixel coordinates.

The a11y tree is **viewport-centric** — it lists nodes around the
currently rendered region, not the whole page DOM. A target that is
not in the tree may simply be below the fold; scroll the relevant
region before concluding it is absent.

## Settings entry points

Most browser-level settings live under `chrome://settings/` sub-pages,
reachable from the ⋮ (three-dot) menu → Settings, or by navigating the
address bar directly to the sub-page:

* `chrome://settings/` — appearance, search engine, on-startup, downloads.
* `chrome://settings/defaultBrowser`, `.../search` — default search
  engine selection.
* `chrome://settings/privacy`, `.../cookies`, `.../content` — privacy
  and per-site permissions.
* `chrome://bookmarks/`, `chrome://history/`, `chrome://extensions/`,
  `chrome://downloads/` — open as full tabs, with the literal
  `chrome://<name>` string as the tab/URL.

Navigating the address bar straight to the target `chrome://` sub-page
is usually shorter and more reliable than clicking through nested
menus.

## Committing input

A value typed into the address bar (omnibox) or into a typeahead /
autocomplete field is **not committed** while a suggestion dropdown is
still showing beneath it. Commit it explicitly — press Enter, or click
the intended suggestion row — before treating the value as set. An
open suggestion list next to typed text is the "not committed yet"
signal.

A setting changed inside a dialog or a `chrome://settings` toggle
generally commits immediately on click (there is no separate Save
button); confirm via the control's post-change a11y state.

## Coupled controls

Some pages couple form controls — toggling one checkbox / radio can
auto-flip another (sites with linked "flexible dates" / "auto-renew" /
payment options). After an action that toggles a checkbox / radio /
switch, check the state of the OTHER toggles in the same form or
dialog, not only the one targeted.


## Navigation discipline

Do NOT navigate to external URLs or new websites unless the task
explicitly requires it. Work within the currently open page / tab.
Exception: `chrome://` settings URLs are allowed (and often the
fastest way to reach a settings sub-page — see "Settings entry
points" above).

## JSON config edits — DO NOT GUI-type into a JSON literal

This applies whenever a task touches a JSON-shaped file on disk
(Chrome `Preferences` / `Bookmarks` / `Local State`, browser-side
config, an extension's stored settings, etc.).

Do NOT plan a GUI keystroke sequence — `Ctrl+A`, type the whole
literal, `Ctrl+S` — to apply the content change. That pattern
corrupts JSON via partial-content overwrite. Instead, the LAST step
of the plan (the one that actually applies the change) must be
written as a HIGH-LEVEL FILE-EDIT INTENT, and the actor's decomposer
will emit a deterministic JSON edit (jq / Python json edit / etc.).

  GOOD `<text>`: "Modify ~/.config/<app>/<file>.json to set <key> to <value>." (intent only — actor picks a deterministic edit)
  BAD  `<text>`: "Press Ctrl+A, type the JSON entry, then Ctrl+S." (ctrl+a+type pattern destroys the buffer)

The earlier GUI steps (open the file in the editor for visual context) stay as-is — only the actual content-change step uses the file-edit intent style.

(That said: per the section above, for Chrome's own state files —
`Preferences`, `Local State`, `Bookmarks`, the SQLite stores —
direct file edits are forbidden because Chrome owns those files
live. Use the in-browser `chrome://settings/` pages instead. The
JSON-edit rule above applies to OTHER JSON config files the task
might ask you to modify.)

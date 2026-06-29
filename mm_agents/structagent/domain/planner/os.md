# OS / Ubuntu domain knowledge

This block is injected into the planner and init_ledger prompts when
the task's domain (or any related app) is `os`. Generic shell /
xdg-spec / gsettings idioms that don't belong to a specific GUI app
go here. Keep it surgical — bloated OS knowledge bleeds into every
multi-app task.

<!-- [FIX os_default_app_knowledge] Created so default-application
     tasks ("set default video player", "make Firefox the default
     browser", "open .md files with VS Code by default") stop
     hallucinating non-existent ``gsettings`` keys. The mechanism is
     ``xdg-mime`` / ``~/.config/mimeapps.list`` — general across
     ANY default-app task (no leak of any one task's answer).
     Roll back by deleting this entire file. -->

## Default application for a file category — use `xdg-mime`, NOT gsettings

When the instruction reads "set `<App>` as the default `<category>`
{player, viewer, browser, editor}" or "open `.<ext>` files with `<App>`
by default":

* **Mechanism**: `xdg-mime default <App>.desktop <mime/type>`. This
  writes / merges into `~/.config/mimeapps.list`.
* **NO `sudo`**: `xdg-mime` targets the user's own `~/.config/
  mimeapps.list`. `sudo xdg-mime …` is wrong (it'd write
  `/root/.config/…` and the user's defaults don't change), and
  `echo password | sudo -S …` is also wrong AND a security
  anti-pattern. Run the command as the user, directly.
* A category like "video" / "image" / "text" maps to MANY MIME
  subtypes (`video/mp4`, `video/x-msvideo`, `video/webm`,
  `video/x-matroska`, …). A robust "set the default for the
  category" change applies `xdg-mime default` to **all** of them,
  not just one — applying to a single subtype leaves the rest
  pointing at the previous default and the system-wide behaviour
  for the category is unchanged. Enumerate the full set with
  `grep -oE 'type="<category>/[a-z0-9.+-]+"'
  /usr/share/mime/packages/freedesktop.org.xml | cut -d'"' -f2 |
  sort -u` and loop:
  ```
  for m in $(grep -oE 'type="<category>/[a-z0-9.+-]+"' \
              /usr/share/mime/packages/freedesktop.org.xml \
              | cut -d'"' -f2 | sort -u); do
      xdg-mime default <App>.desktop "$m"
  done
  ```

* **DO NOT** use `gsettings get/set org.gnome.desktop.default-
  applications.*`. That schema has been deprecated / removed on
  Ubuntu 22.04+ — the verify returns `No such key '<X>'` and
  silently fails the outcome forever. Even if the env perceiver's
  `[Relevant GSettings schemas]` block lists `org.gnome.desktop.
  default-applications`, the KEY for `<category>` does not exist
  there in current GNOME.

### Verify spec for "default app" tasks

`kind="shell_command"` with `command=["xdg-mime","query","default",
"<mime/type>"]` and `expected_substring=["<App>.desktop"]`. Author
several such outcomes — one per representative MIME subtype of the
category — so a verify that passed on a single subtype while the
rest were untouched cannot false-positive the task.


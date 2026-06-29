"""Task-agnostic environment probe.

One batched bash script run on the OSWorld VM at task start, capturing user
identity, available binaries, app config-file locations, the GSettings schema
pool, and running processes — to ground the ledger initializer (and planner) in
the VM's actual state rather than pre-baked domain knowledge.

It does NOT consume the user's instruction; per-task filtering (which schemas /
binaries matter for *this* task) happens later in
:func:`mm_agents.perceiver.perceiver.perceive_environment`.
"""
from __future__ import annotations
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Inline shell. Each command swallows errors so a missing tool never blocks the
# rest; ``head`` caps the noisier commands to keep the payload ~4-10KB.
PROBE_SCRIPT = r"""
echo "### identity"
echo "user=$(whoami)  home=$HOME"
grep -E '^(NAME|VERSION)=' /etc/os-release 2>/dev/null | head -2

echo "### binaries"
for b in code chrome google-chrome firefox thunderbird soffice gimp vlc \
         python3 git gsettings snap dpkg pgrep useradd passwd ssh \
         systemctl gnome-terminal nautilus dconf \
         cp mv rm mkdir ls find sed awk grep cat echo tr sort uniq cut wc tee tar gzip chmod chown; do
  printf "  %-22s %s\n" "$b" "$(command -v "$b" 2>/dev/null || echo NOT_FOUND)"
done

echo "### user dirs"
# Absolute paths (not bare names), two levels deep, so the environment
# perceiver can quote them verbatim — the downstream agent then opens a
# task file straight from its path via open_app instead of hunting in
# the GUI file picker (and risking a blank document by mistake).
for d in ~/Desktop ~/Documents ~/Downloads; do
  find "$d" -maxdepth 2 -mindepth 1 ! -name '.*' 2>/dev/null | sort | head -60
done
echo "### .config entries"
ls -1a ~/.config 2>/dev/null | head -25

echo "### gsettings schemas"
gsettings list-schemas 2>/dev/null | sort | head -400

echo "### gsettings keys (common GNOME namespaces)"
# For each schema in the well-known GNOME / Ubuntu-fork namespace
# prefixes, list its keys. Lets a downstream consumer pair a key
# (e.g. ``lock-enabled``) with its actual schema (e.g.
# ``org.gnome.desktop.screensaver``) instead of the LLM guessing.
# Capped by `head -50` to keep total probe output manageable; the
# common namespaces fit comfortably under that cap.
for sch in $(gsettings list-schemas 2>/dev/null \
        | grep -E '^(org\.gnome\.(desktop|settings-daemon|shell|mutter|SessionManager)|com\.ubuntu)' \
        | sort | head -150); do
  echo "  schema=$sch"
  gsettings list-keys "$sch" 2>/dev/null | sort | sed 's/^/    /'
done

echo "### running processes"
# [FIX probe_filter_helper_procs] Filter out helper subprocesses so the
# perceiver sees the main browser/editor lines, not the noise from
# crashpad handlers + chrome's per-tab renderer / gpu / utility procs
# (each one matches "chrome" and floods the head -10 window). Drop:
#   crashpad_handler   — Chrome's crash reporter
#   --type=<X>         — every Chrome / VS Code subprocess flag
#   --monitor-self     — crashpad-internal flag
#   --initial-client-fd — IPC fd for sub-processes; main never has it
# Roll back by reverting to the bare ``pgrep -af '(...)' | head -10``.
pgrep -af '(code|chrome|firefox|thunderbird|soffice|gimp|vlc)' 2>/dev/null \
  | grep -vE 'crashpad_handler|--type=|--monitor-self|--initial-client-fd' \
  | head -10
"""


# Domain-conditional tails, appended to PROBE_SCRIPT when the domain matches.
# Keep each bash-only and fully read-only — the probe never mutates VM state.

_VSCODE_PROBE_TAIL = r"""
echo "### vscode config"
ls ~/.config/Code/User 2>/dev/null
find ~ -maxdepth 3 -name '*.code-workspace' -type f 2>/dev/null | head -10
"""

_CHROME_PROBE_TAIL = r"""
echo "### chrome config"
ls ~/.config/google-chrome/Default 2>/dev/null | head -15
"""

_THUNDERBIRD_PROBE_TAIL = r"""
echo "### thunderbird profile"
ls -d ~/.thunderbird/*.default* 2>/dev/null
# profiles.ini names which profile dir is the ACTIVE default — multiple
# profile dirs can coexist (e.g. *.default plus *.default-release) and
# only the one under [Install*]'s Default= is the live one Thunderbird
# reads from. Dump it so downstream doesn't have to guess.
if [ -f ~/.thunderbird/profiles.ini ]; then
  echo "  --- profiles.ini ---"
  sed 's/^/    /' ~/.thunderbird/profiles.ini
fi
"""

_LIBREOFFICE_PROBE_TAIL = r"""
echo "### libreoffice"
ls ~/.config/libreoffice 2>/dev/null | sed 's/^/  /'
echo "  --- soffice version (headless conversions, PDF export, etc.) ---"
soffice --version 2>&1 | head -2 | sed 's/^/    /'
# Python office libraries — the OSWorld evaluator imports python-docx /
# python-pptx / openpyxl to read documents. If they're available, the
# planner can author shell_command verify specs using the python3 -c
# shape (whitelisted) and the actor can patch documents via cli_run
# with python-docx scripts (preferred over GUI menus for routine
# property edits).
echo "  --- python office libs available on VM ---"
python3 -c "
for name in ['docx', 'openpyxl', 'pptx']:
    try:
        m = __import__(name); v = getattr(m, '__version__', '?')
        print(f'    {name}: {v}')
    except Exception as e:
        print(f'    {name}: not available ({type(e).__name__})')
" 2>/dev/null
"""

_OS_PROBE_TAIL = r"""
echo "### gnome-terminal profile (relocatable schema introspection)"
# GNOME Terminal profile settings (size, theme, font) live at a dconf
# path instanced with the profile UUID. The schema
# ``org.gnome.Terminal.Legacy.Profile`` is RELOCATABLE — its keys are
# defined statically but the dconf path is bound at use-time. So:
#   • ``gsettings list-keys org.gnome.Terminal.Legacy.Profile:<path>``
#     returns the key NAMES from the schema definition, NO dconf data
#     required (works before gnome-terminal has ever launched).
#   • ``dconf dump <path>`` returns the current VALUES, may be empty
#     before first launch.
# Together the LLM downstream sees both the UUID and the exact key
# names (e.g. ``default-size-columns`` / ``default-size-rows``),
# without us launching gnome-terminal or otherwise mutating VM state.
echo "  default-profile-uuid: $(gsettings get org.gnome.Terminal.ProfilesList default 2>/dev/null)"
echo "  profile-list:         $(gsettings get org.gnome.Terminal.ProfilesList list 2>/dev/null)"
TERM_UUID=$(gsettings get org.gnome.Terminal.ProfilesList default 2>/dev/null | tr -d "'\"")
if [ -n "$TERM_UUID" ]; then
  TERM_PATH="/org/gnome/terminal/legacy/profiles:/:$TERM_UUID/"
  echo "  --- gsettings list-keys org.gnome.Terminal.Legacy.Profile:$TERM_PATH ---"
  gsettings list-keys "org.gnome.Terminal.Legacy.Profile:$TERM_PATH" 2>/dev/null \
      | sort | sed 's/^/    /'
  echo "  --- dconf dump $TERM_PATH (current values, may be empty before first launch) ---"
  dconf dump "$TERM_PATH" 2>/dev/null | head -50 | sed 's/^/    /'
fi
"""

_GIMP_PROBE_TAIL = r"""
echo "### gimp config"
# GIMP user state is version-keyed under ~/.config/GIMP/<major.minor>/.
# Pick the most-recent version dir (sort -V puts e.g. 2.10 above 2.8).
GIMP_DIR=$(ls -d ~/.config/GIMP/*/ 2>/dev/null | sort -V | tail -1)
if [ -n "$GIMP_DIR" ]; then
  echo "  config dir: $GIMP_DIR"
  if [ -f "$GIMP_DIR/gimprc" ]; then
    echo "  --- gimprc top-level entries (head -40) ---"
    grep -E '^\(' "$GIMP_DIR/gimprc" 2>/dev/null | head -40 | sed 's/^/    /'
  else
    echo "  gimprc: (not present — defaults in effect)"
  fi
else
  echo "  config dir: ~/.config/GIMP/ (none — GIMP not yet launched)"
fi
"""


_VLC_PROBE_TAIL = r"""
echo "### vlc config (~/.config/vlc/vlcrc is INI format — NOT gsettings, NOT dconf)"
VLCRC="$HOME/.config/vlc/vlcrc"
if [ -f "$VLCRC" ]; then
  echo "  path: $VLCRC (exists)"
  echo "  --- active key=value lines (comments and section headers stripped, head -80) ---"
  grep -E '^[a-z][a-z0-9-]*\s*=' "$VLCRC" 2>/dev/null | head -80 | sed 's/^/    /'
else
  echo "  path: $VLCRC (not yet present)"
fi
"""


# canonical domain (lowercased, _-separated) → tail. Aliases (e.g. "vscode")
# are normalised in build_probe_script.
_DOMAIN_PROBE_TAILS = {
    "vs_code":              _VSCODE_PROBE_TAIL,
    "chrome":               _CHROME_PROBE_TAIL,
    "thunderbird":          _THUNDERBIRD_PROBE_TAIL,
    "libreoffice_writer":   _LIBREOFFICE_PROBE_TAIL,
    "libreoffice_calc":     _LIBREOFFICE_PROBE_TAIL,
    "libreoffice_impress":  _LIBREOFFICE_PROBE_TAIL,
    "os":                   _OS_PROBE_TAIL,
    "vlc":                  _VLC_PROBE_TAIL,
    "gimp":                 _GIMP_PROBE_TAIL,
}

_DOMAIN_PROBE_ALIASES = {
    "vscode":         "vs_code",
    "google-chrome":  "chrome",
}


def build_probe_script(domain: Optional[str] = None) -> str:
    """Return the probe script with any domain-specific tail appended.

    domain=None or an unrecognised domain returns the common probe unchanged;
    ``multi_apps`` emits every app tail so a cross-app task sees all configs.
    """
    d = (domain or "").lower().strip()
    d = _DOMAIN_PROBE_ALIASES.get(d, d)
    if d == "multi_apps":
        # Every app tail; dedupe by identity (the LibreOffice tail is shared by
        # three domains).
        seen = set()
        parts = []
        for v in _DOMAIN_PROBE_TAILS.values():
            if id(v) in seen:
                continue
            seen.add(id(v))
            parts.append(v)
        tail = "".join(parts)
    else:
        tail = _DOMAIN_PROBE_TAILS.get(d, "")
    return PROBE_SCRIPT + tail if tail else PROBE_SCRIPT


def probe_environment(
    env: Any,
    *,
    domain: Optional[str] = None,
    timeout_s: int = 10,
) -> Optional[str]:
    """Run the probe script on the VM via env.controller.run_bash_script.

    Returns combined stdout, or None if the env lacks run_bash_script or the run
    failed — caller treats None as "no probe available" (no environment context).
    """
    if not env or not getattr(env, "controller", None):
        logger.info("[Probe] no env / controller; skipping")
        return None
    if not hasattr(env.controller, "run_bash_script"):
        logger.info("[Probe] env.controller has no run_bash_script; skipping")
        return None
    script = build_probe_script(domain)
    try:
        result = env.controller.run_bash_script(script, timeout=timeout_s)
    except Exception as e:
        logger.warning("[Probe] run_bash_script raised: %s", e)
        return None
    if not result or result.get("status") != "success":
        logger.warning(
            "[Probe] runner status != success: %s",
            result.get("status") if result else None,
        )
        return None
    out = (result.get("output") or "").strip()
    if not out:
        logger.info("[Probe] empty output")
        return None
    logger.info("[Probe] captured %d bytes", len(out))
    return out

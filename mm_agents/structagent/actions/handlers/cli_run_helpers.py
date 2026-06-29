"""Helpers for the ``cli_run`` actor action.

Used when a subgoal is best handled by a one-shot shell command: launch a
GUI app on a path (``code <dir>``, ``xdg-open <file>``), line-level ``sed -i``
edits (non-JSON; for JSON prefer ``edit_json``), arbitrary in-VM file
mutations edit_json can't express, and process/system commands a task wants
run "from the command line" (kill, git, pip, gsettings, …).

Wire format (AST-parsed upstream, like edit_json):

    cli_run(command='<bash one-liner>', background=<bool>, wait_seconds=<int>)

Runtime is TERMINAL-AWARE. If an on-screen terminal window is open the
command is run inside it (visible on-screen, recorded in ~/.bash_history —
many evaluators grep the history to confirm work was "done from the
terminal"). Otherwise it falls back to a detached ``/bin/bash -lc``
subprocess: fast, but invisible to the screen and to history. Either way the
result lands in ``/tmp/_last_cli_run.json`` for the planner. background=True
returns immediately (GUI-launch case).

Framework-internal probes (init_ledger probe, lsof, ops-lib upload, verify
scripts) bypass this helper and call run_python_script / run_bash_script
directly. Only the actor's cli_run is terminal-routed.

Safety: throwaway per-task OSWorld VM — arbitrary bash is accepted by design.
"""
from __future__ import annotations
import json


# Static body of the generated VM script. build_runtime_script prepends the
# per-call header (_cmd / _bg / _wait); kept f-string-free to avoid escaping.
_RUNTIME_BODY = r'''
import subprocess, time, json, os, shlex


def _persist(d):
    try:
        os.makedirs('/tmp', exist_ok=True)
        with open('/tmp/_last_cli_run.json', 'w') as _f:
            json.dump(d, _f)
    except Exception:
        pass


def _find_terminal():
    """wmctrl window id of an on-screen terminal window, or None.
    ``wmctrl -lx`` line layout: <id> <desktop> <wm_class> <host> <title>."""
    try:
        _r = subprocess.run(['wmctrl', '-lx'], capture_output=True,
                            text=True, timeout=8)
        for _ln in (_r.stdout or '').splitlines():
            _p = _ln.split(None, 4)
            if len(_p) >= 3 and 'terminal' in _p[2].lower():
                return _p[0]
    except Exception:
        pass
    return None


def _activate(wid):
    try:
        subprocess.run(['wmctrl', '-i', '-a', wid], timeout=8,
                       capture_output=True)
        return True
    except Exception:
        return False


def _type_enter(text):
    """Type a SHORT fixed runner line into the focused window + Enter.
    Only ever used for ``source /tmp/_clirun_<tok>.sh`` — never the
    user's real command (that reaches the VM via reliable file I/O),
    so pyautogui char-by-char reliability is a non-issue here."""
    try:
        import pyautogui
        pyautogui.FAILSAFE = False
        pyautogui.write(text, interval=0.02)
        pyautogui.press('enter')
        return True
    except Exception:
        return False


def _run_detached_fg():
    _r = subprocess.run(['/bin/bash', '-lc', _cmd],
                        capture_output=True, text=True, timeout=30)
    if _r.stdout:
        print('cli_run stdout:\n' + _r.stdout[:2000])
    if _r.stderr:
        print('cli_run stderr:\n' + _r.stderr[:2000])
    print('cli_run [fg]: returncode=' + str(_r.returncode))
    _persist({'command': _cmd, 'mode': 'fg', 'returncode': _r.returncode,
              'stdout': _r.stdout[:4000], 'stderr': _r.stderr[:2000],
              'timestamp': time.time()})


_term = _find_terminal()
if _term is not None and not _activate(_term):
    _term = None   # could not focus it — treat as no terminal

if _term is None:
    # ===== No on-screen terminal — detached subprocess (legacy) =====
    if _bg:
        print('cli_run [bg]: launching: ' + _cmd)
        _p = subprocess.Popen(['/bin/bash', '-lc', _cmd])
        time.sleep(_wait)
        print('cli_run [bg]: pid=' + str(_p.pid) + ' (detached)')
        _persist({'command': _cmd, 'mode': 'bg', 'pid': _p.pid,
                  'returncode': None, 'stdout': '', 'stderr': '',
                  'timestamp': time.time()})
    else:
        print('cli_run [fg]: running: ' + _cmd)
        _run_detached_fg()
else:
    # ===== On-screen terminal — run the command IN it =====
    # The real command is written to a temp .sh (reliable file I/O —
    # NOT typed). Only a short fixed ``source <file>`` runner is typed
    # into the terminal, so pyautogui typing reliability never touches
    # the real (possibly long) command. ``source`` runs it in the
    # interactive shell, so ``history -s`` records the genuine command
    # in ~/.bash_history (bash's own API — task evaluators that grep
    # the shell history then see the real command).
    #
    # The command runs in a ``{ ...; }`` brace group, NOT a ``( ... )``
    # subshell. A brace group executes in the CURRENT interactive
    # shell, so ``cd`` / ``export`` / shell variables set by the
    # command PERSIST to the next cli_run — the terminal is genuinely
    # stateful, exactly like a real user's session. (A ``( )`` subshell
    # would fork and discard those, which silently broke every
    # ``cd <dir>``-then-``<cmd>`` task.) ``history -s/-a`` is emitted
    # BEFORE the command so the genuine command still reaches
    # ~/.bash_history even if the command body contains a stray
    # ``exit``.
    time.sleep(0.6)   # let window focus settle before typing
    _tok = str(int(time.time() * 1000))
    _sh = '/tmp/_clirun_' + _tok + '.sh'
    _out = '/tmp/_clirun_' + _tok + '.out'
    _rc = '/tmp/_clirun_' + _tok + '.rc'
    _hist = shlex.quote(_cmd)
    # First line echoes the real command to the terminal SCREEN (it is
    # outside the output-capture redirect) so the perceiver / planner
    # reading the terminal see what actually ran — otherwise they only
    # see the opaque ``source /tmp/_clirun_<tok>.sh`` runner line and
    # mistake it for the agent doing something irrelevant.
    _echo = "printf '+ cli_run: %s\\n' " + _hist + "\n"
    _hist_lines = 'history -s ' + _hist + '\n' + 'history -a\n'
    if _bg:
        _script = (_echo + _hist_lines
                   + '{ ' + _cmd + ' ; } > ' + _out + ' 2>&1 &\n')
    else:
        _script = (_echo + _hist_lines
                   + '{ ' + _cmd + ' ; } > ' + _out + ' 2>&1\n'
                   'echo $? > ' + _rc + '\n')
    _wrote = True
    try:
        with open(_sh, 'w') as _f:
            _f.write(_script)
    except Exception:
        _wrote = False
    if not _wrote:
        # temp-script write failed — degrade to detached subprocess
        print('cli_run [fg]: temp-script write failed, running detached')
        _run_detached_fg()
    else:
        _type_enter('source ' + _sh)
        if _bg:
            time.sleep(_wait)
            print('cli_run [terminal-bg]: launched in terminal ' + str(_term))
            _persist({'command': _cmd, 'mode': 'terminal-bg',
                      'returncode': None, 'stdout': '', 'stderr': '',
                      'timestamp': time.time()})
            try:
                os.remove(_sh)
            except Exception:
                pass
        else:
            # Poll the returncode sentinel for completion (30s cap).
            _deadline = time.time() + 30
            _rcval = None
            _done = False
            while time.time() < _deadline:
                if os.path.exists(_rc):
                    time.sleep(0.25)   # let the redirect finish flushing
                    _done = True
                    try:
                        _rcval = int(open(_rc).read().strip())
                    except Exception:
                        _rcval = None
                    break
                time.sleep(0.4)
            _stdout = ''
            try:
                with open(_out) as _f:
                    _stdout = _f.read()
            except Exception:
                pass
            print('cli_run [terminal]: ran in terminal ' + str(_term)
                  + ', returncode=' + str(_rcval)
                  + ('' if _done else ' (TIMED OUT after 30s)'))
            if _stdout:
                print('cli_run stdout:\n' + _stdout[:2000])
            _persist({'command': _cmd, 'mode': 'terminal',
                      'returncode': _rcval,
                      'stdout': _stdout[:4000], 'stderr': '',
                      'timestamp': time.time()})
            for _f in (_sh, _out, _rc):
                try:
                    os.remove(_f)
                except Exception:
                    pass
'''


def build_runtime_script(
    command: str,
    *,
    background: bool = False,
    wait_seconds: int = 1,
) -> str:
    """Build the self-contained Python script the VM runs for a ``cli_run``.

    Routes into an on-screen terminal when one is open (visible + recorded in
    ~/.bash_history), else a detached subprocess. Persists the result to
    /tmp/_last_cli_run.json and prints a one-line status.
    """
    if not isinstance(command, str) or not command.strip():
        raise ValueError("cli_run requires a non-empty `command` string")
    if not isinstance(wait_seconds, int) or wait_seconds < 0:
        wait_seconds = 1
    wait_seconds = min(wait_seconds, 30)  # cap
    header = (
        "_cmd = " + json.dumps(command) + "\n"
        "_bg = " + ("True" if background else "False") + "\n"
        "_wait = " + str(wait_seconds) + "\n"
    )
    return header + _RUNTIME_BODY

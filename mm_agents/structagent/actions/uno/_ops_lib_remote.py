"""Upload-once-per-process cache for the per-domain ops libraries.

Why this exists
---------------
``IMPRESS_OPS_LIBRARY`` and ``CALC_OPS_LIBRARY`` are large (each ~100KB)
inline Python blobs prepended to every cli_run_uno / verify-shell call.
Without caching, every HTTP dispatch ships the full library — wasting
bandwidth, bash-parse time, and ``compile()`` time, and bloating the
debug dump files on disk.

This module turns that into a one-shot upload at session start:

1. ``maybe_upload_ops_lib(env, domain)`` is called once at agent
   ``predict()`` entry (when ``env`` and the task domain are known).
2. It writes the library's source to ``/tmp/<modname>_<hash>.py`` on
   the VM via ``env.controller.run_python_script``.
3. Subsequent ``_ops_library_for(domain)`` calls in
   ``cli_run_uno_helpers`` query :func:`get_uploaded_module`. When the
   upload succeeded, the helper emits a tiny ``from <modname> import
   *`` shim instead of inlining the whole library.

The hash is content-derived (sha256 prefix) so:
  - If the library is edited in-place between sessions, the new hash
    triggers a fresh upload (no stale cached module on the VM).
  - Within one session, the cache key is just ``(domain, hash)`` so
    every part of the process sees the same answer without needing
    env access.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Process-global cache. Keyed by domain → uploaded library hash (sha256
# prefix). When a downstream caller asks ``get_uploaded_module(domain)``
# and the cache holds an entry, we know the VM has the module.
_UPLOADED_HASH: dict[str, str] = {}


def _lib_text_for(domain: str) -> Optional[str]:
    """Resolve the library source for a domain. Imports lazily so
    importing this helper doesn't pull in the full ops modules."""
    dom = (domain or "").lower().strip().replace("-", "_")
    if dom == "libreoffice_impress":
        try:
            from mm_agents.structagent.actions.impress.ops_lib import IMPRESS_OPS_LIBRARY
        except Exception as e:
            logger.warning("[ops_lib_remote] impress lib import failed: %s", e)
            return None
        return IMPRESS_OPS_LIBRARY
    if dom == "libreoffice_calc":
        try:
            from mm_agents.structagent.actions.calc.ops_lib import CALC_OPS_LIBRARY
        except Exception as e:
            logger.warning("[ops_lib_remote] calc lib import failed: %s", e)
            return None
        return CALC_OPS_LIBRARY
    if dom == "libreoffice_writer":
        try:
            from mm_agents.structagent.actions.writer.ops_lib import WRITER_OPS_LIBRARY
        except Exception as e:
            logger.warning("[ops_lib_remote] writer lib import failed: %s", e)
            return None
        return WRITER_OPS_LIBRARY
    return None


def _module_prefix(domain: str) -> str:
    dom = (domain or "").lower().strip()
    if "impress" in dom:
        return "_impress_ops_lib"
    if "calc" in dom:
        return "_calc_ops_lib"
    if "writer" in dom:
        return "_writer_ops_lib"
    return "_unknown_ops_lib"


def get_uploaded_module(domain: str) -> Optional[str]:
    """Return the on-VM module name (without ``.py``) if a fresh-hash
    upload has succeeded this process. Returns ``None`` otherwise —
    callers should fall back to inlining the library."""
    h = _UPLOADED_HASH.get((domain or "").lower().strip())
    if not h:
        return None
    return f"{_module_prefix(domain)}_{h}"


def _chunked_upload(runner, mod_name: str, lib_text: str) -> bool:
    """Upload ``lib_text`` to ``/tmp/<mod_name>.py`` on the VM in small
    base64 chunks, via repeated ``run_python_script`` calls.

    Why chunk: the env API intermittently times out on a single large
    payload — a one-shot upload of the impress ops library (~123KB;
    repr-embedded it is larger) hits ``run_python_script``'s hard 90s
    ceiling under load. Each chunk here is ~12KB, comfortably within
    limits; ~15 small sequential calls replace the one fragile big one.

    Returns True only on a SIZE-verified success. On any failure the
    caller falls back to the single-shot upload, then to inlining — so
    a bug here degrades to the old behaviour, it does not break.
    """
    b64 = base64.b64encode(lib_text.encode("utf-8")).decode("ascii")
    staging = f"/tmp/{mod_name}.b64"
    target = f"/tmp/{mod_name}.py"

    def _marker(res, mark: str) -> bool:
        if not isinstance(res, dict):
            return False
        if res.get("status") != "success" and res.get("success") is not True:
            return False
        return mark in (res.get("output") or res.get("message") or "")

    CHUNK = 12000
    n = (len(b64) + CHUNK - 1) // CHUNK
    for i in range(n):
        piece = b64[i * CHUNK:(i + 1) * CHUNK]
        mode = "w" if i == 0 else "a"   # first chunk truncates any stale file
        code = (
            f"with open({staging!r}, {mode!r}, encoding='ascii') as _f:\n"
            f"    _f.write({piece!r})\n"
            "print('CHUNK_OK')\n"
        )
        try:
            if not _marker(runner(code), "CHUNK_OK"):
                return False
        except Exception:
            return False

    # Reassemble: decode the staged base64 → the module file, drop the
    # staging file, and confirm the byte count matches exactly.
    expect = len(lib_text.encode("utf-8"))
    final = (
        "import base64, os\n"
        f"with open({staging!r}, 'r', encoding='ascii') as _f:\n"
        "    _data = base64.b64decode(_f.read())\n"
        f"with open({target!r}, 'wb') as _f:\n"
        "    _f.write(_data)\n"
        f"try:\n    os.remove({staging!r})\nexcept Exception:\n    pass\n"
        f"print('UPLOAD_OK' if (os.path.exists({target!r}) "
        f"and os.path.getsize({target!r}) == {expect}) else 'UPLOAD_FAIL')\n"
    )
    try:
        return _marker(runner(final), "UPLOAD_OK")
    except Exception:
        return False


def maybe_upload_ops_lib(env: Any, domain: str) -> Optional[str]:
    """Upload the ops library for ``domain`` to the VM as a Python
    module, once per (process, content-hash).

    Returns the on-VM module name on success (later importable as
    ``from <name> import *``). Returns ``None`` when the upload can't
    proceed — env has no controller, domain has no library, or the
    HTTP call failed; in any failure case downstream callers fall
    back to inlining.

    NOT a trust-forever cache. OSWorld resets the VM snapshot between
    benchmark tasks, which WIPES ``/tmp`` — so a process-global "we
    uploaded this once" flag goes stale and the shim ``from <mod>
    import *`` then hits ``ModuleNotFoundError``. Every call therefore
    confirms the module file is ACTUALLY present on the VM (a cheap
    ~ms probe); the 120KB write only happens when it's genuinely
    missing (task start, or after a snapshot reset). On any failure
    the domain's cache entry is cleared so ``_ops_library_for`` falls
    back to inlining instead of emitting a shim that can't resolve.
    """
    dom = (domain or "").lower().strip().replace("-", "_")
    lib_text = _lib_text_for(dom)
    if lib_text is None or not env:
        return None
    h = hashlib.sha256(lib_text.encode("utf-8")).hexdigest()[:12]
    mod_name = f"{_module_prefix(dom)}_{h}"
    # Resolve the controller's run_python_script. Both real DesktopEnv
    # (env.controller.run_python_script) and APIDesktopEnv
    # (env.run_python_script) expose it; tolerate either.
    runner = None
    ctrl = getattr(env, "controller", None)
    if ctrl is not None and hasattr(ctrl, "run_python_script"):
        runner = ctrl.run_python_script
    elif hasattr(env, "run_python_script"):
        runner = env.run_python_script
    if runner is None:
        logger.info(
            "[ops_lib_remote] env has no run_python_script; "
            "falling back to inline for %s",
            dom,
        )
        _UPLOADED_HASH.pop(dom, None)
        return None

    def _runner_output(res) -> str:
        if not isinstance(res, dict):
            return ""
        if (res.get("status") == "success"
                or res.get("success") is True):
            return res.get("output") or res.get("message") or ""
        return ""

    # Fast path — cache says we uploaded this hash already. VERIFY the
    # file still exists (snapshot reset between tasks wipes /tmp). The
    # probe payload is tiny, so this is cheap to run every step.
    if _UPLOADED_HASH.get(dom) == h:
        probe_code = (
            "import os\n"
            f"print('PRESENT' if os.path.exists('/tmp/{mod_name}.py') "
            "else 'MISSING')\n"
        )
        try:
            present = "PRESENT" in _runner_output(runner(probe_code))
        except Exception:
            present = False
        if present:
            return mod_name
        logger.info(
            "[ops_lib_remote] %s cached but /tmp file gone "
            "(snapshot reset) — re-uploading", mod_name,
        )
        _UPLOADED_HASH.pop(dom, None)

    # Upload the library to the VM. Chunked path first: the env API
    # intermittently times out on a single ~160KB payload, so the
    # library is base64-chunked into ~12KB pieces (see _chunked_upload).
    # Single-shot is the fallback; if both fail we return None and the
    # caller inlines the library into each cli_run_uno script instead.
    uploaded = False
    try:
        uploaded = _chunked_upload(runner, mod_name, lib_text)
    except Exception as e:
        logger.warning("[ops_lib_remote] chunked upload raised: %s", e)
    if not uploaded:
        logger.info(
            "[ops_lib_remote] chunked upload did not succeed for %s; "
            "trying single-shot", dom,
        )
        # Single-shot fallback: embed the library as a repr()'d string
        # so the VM-side parser needs no content escaping.
        upload_code = (
            "import os\n"
            f"_p = '/tmp/{mod_name}.py'\n"
            "if not os.path.exists(_p):\n"
            "    with open(_p, 'w', encoding='utf-8') as _f:\n"
            f"        _f.write({lib_text!r})\n"
            "print('OK' if os.path.exists(_p) else 'FAIL')\n"
        )
        try:
            uploaded = "OK" in _runner_output(runner(upload_code))
        except Exception as e:
            logger.warning(
                "[ops_lib_remote] single-shot upload raised: %s", e)
    if not uploaded:
        logger.warning(
            "[ops_lib_remote] upload failed for %s — caller will inline",
            dom,
        )
        _UPLOADED_HASH.pop(dom, None)
        return None
    _UPLOADED_HASH[dom] = h
    logger.info(
        "[ops_lib_remote] uploaded %s to /tmp/%s.py (hash=%s, %d bytes)",
        dom, mod_name, h, len(lib_text),
    )
    return mod_name

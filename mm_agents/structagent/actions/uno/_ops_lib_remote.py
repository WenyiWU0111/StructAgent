"""Upload-once-per-process cache for the per-domain ops libraries.

The ops libraries (IMPRESS_OPS_LIBRARY, CALC_OPS_LIBRARY, ...) are ~100KB
inline blobs otherwise prepended to every cli_run_uno / verify-shell call,
wasting bandwidth, parse + compile() time, and disk on debug dumps. This
uploads each once per session instead:

1. ``maybe_upload_ops_lib(env, domain)`` at agent predict() entry writes
   the source to ``/tmp/<modname>_<hash>.py`` on the VM.
2. ``_ops_library_for`` then asks :func:`get_uploaded_module`; on success
   it emits a ``from <modname> import *`` shim instead of inlining.

The hash is content-derived (sha256 prefix): editing the library between
sessions forces a fresh upload, and the (domain, hash) key lets the whole
process agree without env access.
"""
from __future__ import annotations

import base64
import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Process-global cache: domain → uploaded library hash (sha256 prefix).
_UPLOADED_HASH: dict[str, str] = {}


def _lib_text_for(domain: str) -> Optional[str]:
    """Library source for a domain. Lazy import so loading this helper
    doesn't pull in the full ops modules."""
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
    """On-VM module name (no ``.py``) if a fresh-hash upload succeeded this
    process, else None (caller inlines the library)."""
    h = _UPLOADED_HASH.get((domain or "").lower().strip())
    if not h:
        return None
    return f"{_module_prefix(domain)}_{h}"


def _chunked_upload(runner, mod_name: str, lib_text: str) -> bool:
    """Upload ``lib_text`` to ``/tmp/<mod_name>.py`` in ~12KB base64 chunks.

    A one-shot upload (impress lib ~123KB, larger repr-embedded) hits
    run_python_script's 90s ceiling under load; chunking trades it for
    ~15 small calls. Returns True only on size-verified success; on any
    failure the caller falls back to single-shot, then inlining.
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
        mode = "w" if i == 0 else "a"   # first chunk truncates stale file
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

    # Reassemble: decode staged base64 → module file, drop staging, and
    # confirm the byte count matches exactly.
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
    """Upload ``domain``'s ops library to the VM as a Python module, once
    per (process, content-hash). Returns the on-VM module name on success,
    None on any failure (no controller / no library / HTTP error) so the
    caller inlines.

    NOT trust-forever: OSWorld resets the VM snapshot between tasks, wiping
    ``/tmp``, so a cached "uploaded once" flag goes stale and the shim hits
    ModuleNotFoundError. Every call therefore re-probes that the file is
    actually present (cheap ~ms); the 120KB write only happens when it's
    genuinely missing. On failure the cache entry is cleared.
    """
    dom = (domain or "").lower().strip().replace("-", "_")
    lib_text = _lib_text_for(dom)
    if lib_text is None or not env:
        return None
    h = hashlib.sha256(lib_text.encode("utf-8")).hexdigest()[:12]
    mod_name = f"{_module_prefix(dom)}_{h}"
    # run_python_script lives on env.controller (DesktopEnv) or env
    # directly (APIDesktopEnv); tolerate either.
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

    # Fast path: cache says we uploaded this hash — verify the file still
    # exists (snapshot reset wipes /tmp). Probe is tiny, cheap every step.
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

    # Upload. Chunked first (single ~160KB payload times out intermittently;
    # see _chunked_upload); single-shot is the fallback. If both fail we
    # return None and the caller inlines.
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
        # Single-shot fallback: embed the library repr()'d so the VM-side
        # parser needs no content escaping.
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

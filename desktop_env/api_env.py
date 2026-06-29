"""
API-based DesktopEnv that talks to a remote env API server instead of
managing Docker/VM instances directly.  Compatible with the env API server
stack (env_api_manager + env_api_wrapper) used by ZeroGUI training.

Usage:
    env = APIDesktopEnv(
        base_url="http://127.0.0.1",
        manager_port=10001,          # or env_port=10010 for direct
        action_space="pyautogui",
        screen_size=(1920, 1080),
    )
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import List, Optional

import requests

logger = logging.getLogger("desktopenv.api_env")


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _request_api(url, data=None, method="POST", try_max_times=5, timeout=360):
    """HTTP request wrapper with retry logic for the env API server."""
    headers = {"Content-Type": "application/json"}
    for attempt in range(try_max_times):
        try:
            resp = requests.request(
                method=method, url=url, json=data,
                headers=headers, timeout=timeout,
            )
            resp.raise_for_status()
            body = resp.json()
            if not body.get("success"):
                msg = body.get("message", "")
                logger.warning("[API env] server error: %s", msg)
                # Fail fast on known unrecoverable errors
                if isinstance(msg, str) and (
                    "env.reset returned invalid screenshot type" in msg
                    or "env.reset returned None observation" in msg
                ):
                    raise Exception(msg)
            else:
                return body
        except requests.RequestException as e:
            logger.warning("[API env] request error (attempt %d/%d): %s",
                           attempt + 1, try_max_times, e)
        except Exception:
            raise
        time.sleep(1)
    raise Exception(
        f"[API env] request to {url} failed after {try_max_times} attempts. "
        "Check that the OSWorld env API server is running and reachable."
    )


# ── Remote controller (recording proxy) ──────────────────────────────────────

class _RemoteController:
    """Proxies start_recording / end_recording / get_file to the env API server."""

    def __init__(self, base_url: str, port: int):
        self._base = f"{base_url}:{port}"

    def start_recording(self):
        try:
            _request_api(f"{self._base}/start_recording", try_max_times=2)
        except Exception as e:
            logger.warning("[API env] start_recording failed: %s", e)

    def end_recording(self, dest: str):
        try:
            resp = _request_api(f"{self._base}/end_recording", try_max_times=2)
            video_b64 = resp.get("video")
            if video_b64:
                with open(dest, "wb") as f:
                    f.write(base64.b64decode(video_b64))
        except Exception as e:
            logger.warning("[API env] end_recording failed: %s", e)

    def get_file(self, file_path: str) -> Optional[bytes]:
        """Get a file from the VM via the /file endpoint."""
        try:
            import requests as _req
            resp = _req.post(
                f"{self._base}/file",
                json={"file_path": file_path},
                timeout=120,
            )
            if resp.status_code == 200 and len(resp.content) > 0:
                return resp.content
        except Exception as e:
            logger.warning("[API env] get_file(%s) failed: %s", file_path, e)
        return None

    def run_python_script(self, script: str):
        """Execute a Python script on the remote VM via /run_python endpoint."""
        try:
            resp = requests.post(
                f"{self._base}/run_python",
                json={"code": script},
                headers={"Content-Type": "application/json"},
                timeout=90,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"status": "error", "output": "", "error": resp.text}
        except Exception as e:
            logger.error("[API env] run_python_script failed: %s", e)
            return {"status": "error", "output": "", "error": str(e)}

    def execute_python_command(self, command: str):
        """Compat shim for DesktopEnv's ``PythonController.execute_python_command``:
        run a Python snippet on the VM and return a dict with an ``output``
        (stdout) key. Backed by ``/run_python`` so OSWorld getters that call
        ``execute_python_command`` (e.g. get_bookmarks / get_history path lookups)
        work on the API provider too, instead of AttributeError-ing."""
        res = self.run_python_script(command)
        if isinstance(res, dict):
            out = res.get("output")
            if out is None:
                out = res.get("stdout", "")
            return {"output": out or "", "status": res.get("status", "success")}
        return {"output": "", "status": "error"}

    def run_bash_script(self, script: str, timeout: int = 30, working_dir=None):
        """Execute a bash script on the remote VM via /run_bash_script endpoint."""
        try:
            resp = requests.post(
                f"{self._base}/run_bash_script",
                json={"script": script, "timeout": timeout, "working_dir": working_dir},
                headers={"Content-Type": "application/json"},
                timeout=timeout + 100,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                return {"status": "error", "output": "", "error": resp.text, "returncode": -1}
        except Exception as e:
            logger.error("[API env] run_bash_script failed: %s", e)
            return {"status": "error", "output": "", "error": str(e), "returncode": -1}


# ── Main class ────────────────────────────────────────────────────────────────

class APIDesktopEnv:
    """
    Drop-in replacement for DesktopEnv that talks to a remote env API server
    (the same server stack used by ZeroGUI training).
    """

    def __init__(
        self,
        base_url: str,
        env_port: int = None,
        manager_port: int = None,
        action_space: str = "pyautogui",
        screen_size: tuple = (1920, 1080),
        headless: bool = False,
        require_a11y_tree: bool = False,
        os_type: str = "Ubuntu",
    ):
        self.base_url = base_url
        self.env_port = env_port
        self.manager_port = manager_port
        self._use_manager = False
        self.env_id = None
        self.vm_ip = "<api>"
        self.action_history: List[str] = []
        self._last_obs = None

        # If no direct port, ask the manager to create one
        if self.env_port is None:
            assert manager_port is not None, \
                "Either env_port or manager_port must be set"
            self._use_manager = True
            resp = _request_api(
                f"{base_url}:{manager_port}/create_env_api",
                try_max_times=3,
            )
            self.env_id = resp["env_id"]
            self.env_port = resp["port"]
            time.sleep(5)
            logger.info("[API env] manager created env api on port %d",
                        self.env_port)

        # Start the environment
        start_data = {
            "vm_name": "Ubuntu.qcow2",
            "action_space": action_space,
            "screen_width": screen_size[0],
            "screen_height": screen_size[1],
            "headless": headless,
            "require_a11y_tree": require_a11y_tree,
            "os_type": os_type,
        }
        _request_api(
            f"{base_url}:{self.env_port}/start", start_data, try_max_times=3,
        )
        logger.info("[API env] environment started at %s:%d",
                     base_url, self.env_port)

        # Fetch real VM connection info for a11y/CDP access
        try:
            info_resp = _request_api(
                f"{base_url}:{self.env_port}/vm_connection_info",
                method="GET", try_max_times=2,
            )
            conn = info_resp.get("connection_info", {})
            self.vm_ip = conn.get("vm_ip", "<api>")
            self.chromium_port = conn.get("chromium_port", 9222)
            logger.info("[API env] VM connection: ip=%s, chromium_port=%d",
                        self.vm_ip, self.chromium_port)
        except Exception as e:
            logger.warning("[API env] Could not get VM connection info: %s", e)
            self.chromium_port = 9222

        self.controller = _RemoteController(base_url, self.env_port)
        self.controller._vm_ip = getattr(self, 'vm_ip', None)
        # OSWorld's chrome getters read env.server_port (the in-VM HTTP backend,
        # host = vm_ip) for the rare "Chrome died, relaunch it" fallback; the main
        # path is CDP via chromium_port. Expose the standard backend port so those
        # getters don't AttributeError on the API provider; the relaunch fallback
        # is best-effort (it POSTs vm_ip:server_port/setup/launch).
        self.server_port = 5000

    # ── screenshot helpers ─────────────────────────────────────────────────

    @staticmethod
    def _decode_screenshot(b64_str: str) -> bytes:
        """Decode base64 screenshot and validate it as a real image."""
        from PIL import Image
        screenshot_bytes = base64.b64decode(b64_str)
        if not screenshot_bytes:
            raise ValueError("Empty screenshot bytes")
        img = Image.open(BytesIO(screenshot_bytes))
        img.load()  # force full decode to catch truncated images
        return screenshot_bytes

    # ── DesktopEnv-compatible interface ────────────────────────────────────

    def reset(self, task_config):
        logger.info("[API env] resetting...")
        url = f"{self.base_url}:{self.env_port}/reset"
        for attempt in range(3):
            resp = _request_api(url, {"task_config": task_config},
                                try_max_times=1)
            obs = resp["obs"]
            try:
                obs["screenshot"] = self._decode_screenshot(obs["screenshot"])
                self._last_obs = obs
                # Refresh VM connection info (port may change after snapshot revert)
                try:
                    info_resp = _request_api(
                        f"{self.base_url}:{self.env_port}/vm_connection_info",
                        method="GET", try_max_times=1,
                    )
                    conn = info_resp.get("connection_info", {})
                    self.vm_ip = conn.get("vm_ip", self.vm_ip)
                    self.chromium_port = conn.get("chromium_port", self.chromium_port)
                except Exception:
                    pass
                logger.info("[API env] reset done.")
                return obs
            except Exception as e:
                logger.warning(
                    "[API env] reset screenshot invalid (attempt %d/3): %s",
                    attempt + 1, e,
                )
                time.sleep(3)
        raise Exception(
            "[API env] reset failed after 3 attempts (bad screenshots)"
        )

    def _get_obs(self):
        """Return the most recent observation (from reset or step)."""
        return self._last_obs

    def step(self, action, pause=2):
        self.action_history.append(action)
        url = f"{self.base_url}:{self.env_port}/step"
        try:
            resp = _request_api(url, {"action": action, "pause": pause})
            obs = resp["obs"]
            obs["screenshot"] = self._decode_screenshot(obs["screenshot"])
            self._last_obs = obs
            return obs, resp["reward"], resp["done"], resp["info"]
        except Exception as e:
            logger.error("[API env] step failed: %s", e)
            return None, -1, True, None

    def evaluate(self):
        url = f"{self.base_url}:{self.env_port}/evaluate"
        try:
            resp = _request_api(url, method="GET")
            return float(resp["metric"])
        except Exception as e:
            logger.error("[API env] evaluate failed: %s", e)
            return -1.0

    @property
    def vm_platform(self):
        resp = _request_api(
            f"{self.base_url}:{self.env_port}/vm_platform", method="GET",
        )
        return resp["vm_platform"]

    @property
    def vm_screen_size(self):
        resp = _request_api(
            f"{self.base_url}:{self.env_port}/vm_screen_size", method="GET",
        )
        return resp["vm_screen_size"]

    def close(self):
        try:
            _request_api(
                f"{self.base_url}:{self.env_port}/close", try_max_times=2,
            )
        except Exception as e:
            logger.warning("[API env] close failed: %s", e)
        if self._use_manager:
            try:
                _request_api(
                    f"{self.base_url}:{self.manager_port}/terminate_env_api",
                    {"env_id": self.env_id},
                    try_max_times=2,
                )
            except Exception as e:
                logger.warning("[API env] terminate_env_api failed: %s", e)

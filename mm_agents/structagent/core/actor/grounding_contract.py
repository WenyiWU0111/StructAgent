"""Per-model inline-grounding contract: image prep + prompt + coord conversion,
kept consistent per model.

- Claude/Sonnet: resize to a recommended max (XGA/WXGA/FWXGA, <=1366 wide) and
  emit ABSOLUTE pixels in that resized space, then scale back. Hosted APIs
  silently downscale large images, so a 0-1000 grid on the original is lost.
- MiniMax M3: 0-1000 normalized on the original image; map x*W/1000.
- default (Qwen-VL etc.): original image + 0-1000 + lenient auto-detect.
"""
from __future__ import annotations

import logging
from io import BytesIO
from typing import Any, Optional, Tuple

from PIL import Image

logger = logging.getLogger("structagent.actor.grounding_contract")

# Anthropic computer-use recommended max resolutions (w, h): XGA, WXGA, FWXGA.
_CLAUDE_TARGETS = [(1024, 768), (1280, 800), (1366, 768)]
_CLAUDE_FALLBACK_LONG_EDGE = 1366

# Shared tail: which actions must NOT carry coordinates.
_INSTRUCTION_TAIL = (
    "Keep the \"target\" text field too (used for logging). Do NOT add "
    "coordinates to non-spatial actions (type / hotkey / wait / navigate / "
    "cli_run / edit_json / calc_* / impress_* / writer_* / note_write / "
    "extract_info / open_app).\n"
)

# DEFAULT (Qwen-VL etc.) — byte-identical to the historical instruction.
DEFAULT_INSTRUCTION = (
    "[INLINE GROUNDING MODE]\n"
    "For EVERY action that targets a UI element, in ADDITION to "
    "the existing \"target\" text field you MUST include the "
    "element's location read directly from the screenshot, as "
    "coordinates normalized to a 0-1000 grid (top-left = 0,0; "
    "bottom-right = 1000,1000):\n"
    "  - click / left_double / right_single: add \"point\": [x, y]\n"
    "  - drag: add \"start_point\": [x, y] and \"end_point\": [x, y]\n"
    "  - scroll: add \"anchor_point\": [x, y]\n"
    "Keep the \"target\" text field too (used for logging). Do NOT "
    "add coordinates to non-spatial actions (type / hotkey / wait "
    "/ navigate / cli_run / edit_json / calc_* / impress_* / "
    "writer_* / note_write / extract_info / open_app).\n"
    "Example: {\"action\": \"click\", \"target\": \"the Submit "
    "button\", \"point\": [840, 560]}"
)


def _claude_instruction(sent_w: int, sent_h: int) -> str:
    return (
        "[INLINE GROUNDING MODE]\n"
        f"The screenshot you are given is exactly {sent_w}x{sent_h} pixels. "
        "For EVERY action that targets a UI element, in ADDITION to the "
        "\"target\" text field you MUST include the element's location as "
        "ABSOLUTE PIXEL coordinates IN THIS IMAGE (top-left = 0,0; "
        f"bottom-right = {sent_w},{sent_h}):\n"
        "  - click / left_double / right_single: add \"point\": [x, y]\n"
        "  - drag: add \"start_point\": [x, y] and \"end_point\": [x, y]\n"
        "  - scroll: add \"anchor_point\": [x, y]\n"
        + _INSTRUCTION_TAIL +
        "Example: {\"action\": \"click\", \"target\": \"the Submit button\", "
        f"\"point\": [{sent_w // 2}, {sent_h // 2}]}}"
    )


def _m3_instruction() -> str:
    return (
        "[INLINE GROUNDING MODE]\n"
        "For EVERY action that targets a UI element, in ADDITION to the "
        "\"target\" text field you MUST include the element's location as "
        "NORMALIZED integer coordinates in [0, 1000], where (0,0) is the "
        "top-left and (1000,1000) is the bottom-right of the screenshot, "
        "REGARDLESS of screen resolution:\n"
        "  - click / left_double / right_single: add \"point\": [x, y]\n"
        "  - drag: add \"start_point\": [x, y] and \"end_point\": [x, y]\n"
        "  - scroll: add \"anchor_point\": [x, y]\n"
        + _INSTRUCTION_TAIL +
        "Example: {\"action\": \"click\", \"target\": \"the Submit button\", "
        "\"point\": [840, 560]}"
    )


def model_family(model: Optional[str]) -> str:
    """Map a model id/alias to 'claude' | 'm3' | 'default'."""
    m = (model or "").lower()
    if any(k in m for k in ("claude", "sonnet", "anthropic", "opus", "haiku")):
        return "claude"
    if "minimax" in m:
        return "m3"
    return "default"


def _claude_target(w: int, h: int) -> Optional[Tuple[int, int]]:
    """Recommended max matching the aspect ratio and smaller than the source;
    else a long-edge cap. Returns (tw, th) or None (no resize)."""
    if not (w and h):
        return None
    ratio = w / h
    for tw, th in _CLAUDE_TARGETS:
        if abs(tw / th - ratio) < 0.02 and tw < w:
            return (tw, th)
    if max(w, h) > _CLAUDE_FALLBACK_LONG_EDGE:
        r = _CLAUDE_FALLBACK_LONG_EDGE / max(w, h)
        return (round(w * r), round(h * r))
    return None


def prepare_screenshot(model: Optional[str],
                       screenshot_bytes: bytes) -> Tuple[bytes, int, int]:
    """Return (sent_bytes, sent_w, sent_h). Resizes only for Claude."""
    try:
        img = Image.open(BytesIO(screenshot_bytes))
        w, h = img.size
    except Exception as e:  # noqa: BLE001
        logger.warning("prepare_screenshot: cannot read image (%s)", e)
        return screenshot_bytes, 0, 0
    if model_family(model) == "claude":
        tgt = _claude_target(w, h)
        if tgt:
            tw, th = tgt
            try:
                buf = BytesIO()
                img.convert("RGB").resize((tw, th), Image.LANCZOS).save(buf, "PNG")
                return buf.getvalue(), tw, th
            except Exception as e:  # noqa: BLE001
                logger.warning("prepare_screenshot: resize failed (%s)", e)
    return screenshot_bytes, w, h


def grounding_instruction(model: Optional[str], sent_w: int, sent_h: int) -> str:
    """The inline-grounding directive matching this model's coordinate contract."""
    fam = model_family(model)
    if fam == "claude":
        return _claude_instruction(sent_w, sent_h)
    if fam == "m3":
        return _m3_instruction()
    return DEFAULT_INSTRUCTION


def point_to_pixels(pt: Any, model: Optional[str], sent_w: int, sent_h: int,
                    orig_w: int, orig_h: int) -> Optional[Tuple[int, int]]:
    """Convert a model-emitted [x, y] to real-screen pixels (orig_w x orig_h),
    interpreting it per the model's contract. None on missing/malformed point."""
    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
        return None
    try:
        x, y = float(pt[0]), float(pt[1])
    except (TypeError, ValueError):
        return None

    fam = model_family(model)
    if fam == "claude":
        # absolute pixels in the sent (resized) space
        if x <= 1.0 and y <= 1.0:
            xn, yn = x, y
        else:
            sw, sh = (sent_w or orig_w), (sent_h or orig_h)
            xn, yn = (x / sw if sw else 0.0), (y / sh if sh else 0.0)
    elif fam == "m3":
        if x <= 1.0 and y <= 1.0:
            xn, yn = x, y
        elif x <= 1000.0 and y <= 1000.0:
            xn, yn = x / 1000.0, y / 1000.0
        else:
            xn, yn = x / orig_w, y / orig_h
    else:
        if x <= 1.0 and y <= 1.0:
            xn, yn = x, y
        elif x <= 1000.0 and y <= 1000.0:
            xn, yn = x / 1000.0, y / 1000.0
        else:
            xn, yn = x / orig_w, y / orig_h

    xn = max(0.0, min(1.0, xn))
    yn = max(0.0, min(1.0, yn))
    return int(round(xn * orig_w)), int(round(yn * orig_h))

"""Screenshot action overlay — paint actor actions onto screenshots.

Used to make trajectories human-reviewable: each turn's screenshot
gets the action that ran from it overlaid (CLICK marker at coords,
TYPE/HOTKEY/SCROLL labels). The annotated image replaces the raw
screenshot in:
  - the planner's K-frame history
  - actor_history snapshots
  - the ledger boundary-summarizer window
  - the v2 timeline StepRecord
  - lib_run_single's per-step record (rendered into trajectory.html)

So the planner / detector / summarizer / human reviewer all see the
SAME annotated frame for any past turn. Convention: the screenshot is
the PRE-action observation (what the agent saw), with the action
that was decided based on it overlaid.
"""
from __future__ import annotations

import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("desktopenv.screenshot_annotator")

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:  # pragma: no cover
    _PIL_OK = False


# --------------------------------------------------------------------------- #
# Action parsing — pyautogui code → list of (kind, params) tuples
# --------------------------------------------------------------------------- #


_RE_CLICK = re.compile(
    r"pyautogui\.(click|rightClick|right_click|doubleClick|moveTo|moveRel)\s*"
    r"\(\s*([\d.]+)\s*,\s*([\d.]+)"
)
_RE_WRITE = re.compile(
    r"pyautogui\.(?:write|typewrite)\s*\(\s*(?:content\s*=\s*)?['\"]([^'\"]+)['\"]"
)
_RE_HOTKEY = re.compile(r"pyautogui\.hotkey\s*\(\s*([^)]+)\)")
_RE_PRESS  = re.compile(r"pyautogui\.press\s*\(\s*['\"]([^'\"]+)['\"]")
_RE_SCROLL = re.compile(
    r"pyautogui\.scroll\s*\(\s*(-?\d+)"
    r"(?:[^)]*?x\s*=\s*([\d.]+))?"
    r"(?:[^)]*?y\s*=\s*([\d.]+))?"
)


def parse_actions(code: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Find every pyautogui.* call in ``code`` and return ordered list of
    (kind, params) tuples. ``kind`` ∈ {click, doubleclick, rightclick,
    moveto, write, hotkey, press, scroll}. Multi-call code blocks are
    handled (e.g. hotkey('ctrl','a') + write(...) + hotkey('enter'))."""
    if not code:
        return []
    out: List[Tuple[int, str, Dict[str, Any]]] = []  # (pos, kind, params)
    for m in _RE_CLICK.finditer(code):
        v = m.group(1).lower()
        kind = {
            "click": "click",
            "rightclick": "rightclick", "right_click": "rightclick",
            "doubleclick": "doubleclick",
            "moveto": "moveto", "moverel": "moveto",
        }.get(v, v)
        out.append((m.start(), kind, {
            "x": float(m.group(2)), "y": float(m.group(3)),
        }))
    for m in _RE_WRITE.finditer(code):
        out.append((m.start(), "write", {"text": m.group(1)}))
    for m in _RE_HOTKEY.finditer(code):
        keys = re.findall(r"['\"]([^'\"]+)['\"]", m.group(1))
        out.append((m.start(), "hotkey", {"keys": keys}))
    for m in _RE_PRESS.finditer(code):
        out.append((m.start(), "press", {"key": m.group(1)}))
    for m in _RE_SCROLL.finditer(code):
        params: Dict[str, Any] = {"amount": int(m.group(1))}
        if m.group(2):
            params["x"] = float(m.group(2))
        if m.group(3):
            params["y"] = float(m.group(3))
        out.append((m.start(), "scroll", params))
    out.sort(key=lambda r: r[0])
    return [(k, p) for _, k, p in out]


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

# Color palette (RGBA so we can fade)
_C_CLICK       = (220, 30,  30, 255)     # red
_C_RIGHTCLICK  = (200, 30, 200, 255)     # magenta
_C_DOUBLECLICK = (220, 100,  0, 255)     # orange
_C_MOVETO      = ( 30,  90, 220, 255)    # blue
_C_SCROLL      = (130, 30, 200, 255)     # purple
_C_TEXT_BG     = (255, 240, 130, 230)    # yellow
_C_HOTKEY_BG   = (190, 240, 200, 230)    # light green


def _try_font(size: int):
    if not _PIL_OK:
        return None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_with_halo(draw, xy, text, font, fg, halo=(255, 255, 255, 255)):
    x, y = xy
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            if dx == 0 and dy == 0:
                continue
            draw.text((x + dx, y + dy), text, font=font, fill=halo)
    draw.text((x, y), text, font=font, fill=fg)


def _draw_click_marker(draw, x, y, label, color, font):
    r = 16
    # outer ring
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=4)
    # crosshair
    draw.line([x - r - 8, y, x + r + 8, y], fill=color, width=3)
    draw.line([x, y - r - 8, x, y + r + 8], fill=color, width=3)
    # center dot
    draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color)
    # label off to the side
    _text_with_halo(draw, (x + r + 6, y - r - 4), label, font, color)


def _draw_corner_label(draw, line_idx, label, font, bg_rgba, fg_rgba):
    pad = 8
    # measure text
    bbox = draw.textbbox((0, 0), label, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    y = 12 + line_idx * (h + pad * 2 + 6)
    box = [12, y, 12 + w + pad * 2, y + h + pad * 2]
    draw.rectangle(box, fill=bg_rgba, outline=fg_rgba, width=2)
    draw.text((12 + pad, y + pad), label, fill=fg_rgba, font=font)


def annotate_screenshot_with_action(
    screenshot_bytes: bytes,
    action_codes: List[str],
) -> bytes:
    """Overlay action markers on a PNG screenshot. Returns annotated
    PNG bytes. Returns the input unchanged if PIL is unavailable, the
    image fails to decode, or no actions parse out."""
    if not screenshot_bytes:
        return screenshot_bytes
    if not _PIL_OK:
        return screenshot_bytes
    if not action_codes:
        return screenshot_bytes

    # Combine all action codes into one ordered list
    actions: List[Tuple[str, Dict[str, Any]]] = []
    for code in action_codes:
        if not code:
            continue
        actions.extend(parse_actions(code))
    if not actions:
        return screenshot_bytes

    try:
        img = Image.open(BytesIO(screenshot_bytes)).convert("RGBA")
    except Exception as e:
        logger.warning("[Annotator] failed to open screenshot: %s", e)
        return screenshot_bytes

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # font sizes scale roughly with image height
    base = max(14, img.height // 50)
    font_med = _try_font(base)
    font_big = _try_font(base + 4)

    corner_idx = 0
    for kind, p in actions:
        if kind == "click":
            _draw_click_marker(draw, p["x"], p["y"], "CLICK", _C_CLICK, font_med)
        elif kind == "rightclick":
            _draw_click_marker(draw, p["x"], p["y"], "RIGHT_CLICK", _C_RIGHTCLICK, font_med)
        elif kind == "doubleclick":
            _draw_click_marker(draw, p["x"], p["y"], "DOUBLE_CLICK", _C_DOUBLECLICK, font_med)
        elif kind == "moveto":
            _draw_click_marker(draw, p["x"], p["y"], "MOVE", _C_MOVETO, font_med)
        elif kind == "scroll":
            x = p.get("x") or img.width // 2
            y = p.get("y") or img.height // 2
            amount = p["amount"]
            ay = y - amount * 6
            ay = max(60, min(img.height - 30, ay))
            draw.line([(x, y), (x, ay)], fill=_C_SCROLL, width=5)
            # arrowhead pointing at ay
            ah = 10 if ay < y else -10
            draw.polygon(
                [(x - 9, ay + ah), (x + 9, ay + ah), (x, ay)],
                fill=_C_SCROLL,
            )
            _text_with_halo(draw, (x + 14, y - 8),
                            f"SCROLL {amount:+d}", font_med, _C_SCROLL)
        elif kind == "write":
            text = p["text"]
            label = f"TYPE: \"{text[:80]}\""
            _draw_corner_label(draw, corner_idx, label, font_big,
                               _C_TEXT_BG, (180, 0, 0, 255))
            corner_idx += 1
        elif kind == "hotkey":
            keys = p["keys"]
            label = "HOTKEY: " + "+".join(keys)
            _draw_corner_label(draw, corner_idx, label, font_big,
                               _C_HOTKEY_BG, (0, 110, 30, 255))
            corner_idx += 1
        elif kind == "press":
            label = f"PRESS: {p['key']}"
            _draw_corner_label(draw, corner_idx, label, font_big,
                               _C_HOTKEY_BG, (0, 110, 30, 255))
            corner_idx += 1

    img = Image.alpha_composite(img, overlay).convert("RGB")
    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


# --------------------------------------------------------------------------- #
# Convenience: smoke test
# --------------------------------------------------------------------------- #


if __name__ == "__main__":  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("img")
    ap.add_argument("out")
    ap.add_argument("action", nargs="*")
    args = ap.parse_args()
    with open(args.img, "rb") as f:
        b = f.read()
    a = annotate_screenshot_with_action(b, args.action)
    with open(args.out, "wb") as f:
        f.write(a)
    print(f"wrote {args.out}, {len(a)} bytes")

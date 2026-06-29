"""Small runner-side helpers used by ``lib_run_single``.

- ``draw_coordinates``: overlay red crosses on a screenshot (grounding debug).
- ``set_current_result_dir`` / ``get_current_result_dir``: per-process result
  directory context (each multiprocessing worker keeps its own copy).
"""
from __future__ import annotations

import io
import os
from typing import List, Union

from PIL import Image, ImageDraw

# --- per-process result-dir context -----------------------------------------
_context_storage: dict = {}


def set_current_result_dir(example_result_dir: str) -> None:
    _context_storage["current_result_dir"] = example_result_dir


def get_current_result_dir(default=None):
    return _context_storage.get("current_result_dir", default)


# --- grounding-debug overlay -------------------------------------------------
def draw_coordinates(image_bytes: bytes,
                     coordinates: List[Union[int, float]],
                     save_path: str) -> None:
    """Draw a red 'X' at each (x, y) in ``coordinates`` ([x1,y1,x2,y2,...])
    on the screenshot and save it. Best-effort — silently returns on error."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return
    draw = ImageDraw.Draw(image)
    size, color, width = 15, "red", 3
    for i in range(0, len(coordinates) - 1, 2):
        x, y = coordinates[i], coordinates[i + 1]
        draw.line([(x - size, y - size), (x + size, y + size)], fill=color, width=width)
        draw.line([(x + size, y - size), (x - size, y + size)], fill=color, width=width)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image.save(save_path)

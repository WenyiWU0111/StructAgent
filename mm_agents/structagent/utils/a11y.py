"""Accessibility-tree helpers: extract the active tab URL and linearize
the raw a11y tree carried on ``obs``."""

import logging
import re
from typing import Any, Dict, Optional


_logger = logging.getLogger(__name__)


def extract_active_url_from_a11y(a11y_text: Optional[str]) -> Optional[str]:
    """Pull the active tab URL from a linearized a11y tree, or None.

    Handles two Chrome a11y row patterns:
      1. ``entry\\thttps://...\\tenabled`` — text is the URL (older Chrome).
      2. ``entry\\tAddress and search bar (https://...)\\tenabled`` — URL
         in parens (current Chromium on Ubuntu).
    """
    if not a11y_text:
        return None
    for line in a11y_text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        tag, text = parts[0].strip(), parts[1].strip()
        if not text:
            continue
        if tag not in {"entry", "text", "document-web"}:
            continue
        lowered = text.lower()
        if (lowered.startswith("http://")
                or lowered.startswith("https://")
                or lowered.startswith("chrome://")):
            return text.split()[0]
        if "address and search bar" in lowered:
            m = re.search(r"\(([^)]+)\)", text)
            if m:
                candidate = m.group(1).strip()
                return candidate.split()[0]
    return None


def linearize_obs_a11y(obs: Optional[Dict[str, Any]], max_items: int = 300) -> Optional[str]:
    """Linearize obs's a11y XML to tab-separated ``tag \\t text \\t state``
    rows, trimmed to ``max_items``. None if unavailable — callers should
    fall back to pure visual.
    """
    if not obs:
        return None
    raw = obs.get("accessibility_tree") if isinstance(obs, dict) else None
    if not raw:
        return None
    try:
        from mm_agents.structagent.utils.a11y_tree.accessibility_tree_handle import (
            linearize_accessibility_tree, trim_accessibility_tree,
        )
        txt = linearize_accessibility_tree(raw, "Ubuntu")
        if not txt:
            return None
        return trim_accessibility_tree(txt, max_items)
    except Exception as e:
        _logger.warning("[a11y] linearize failed: %s", e)
        return None

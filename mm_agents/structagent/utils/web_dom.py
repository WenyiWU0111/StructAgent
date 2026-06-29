"""Dump checker-source web DOM (CSS class + ARIA state) via CDP.

The OSWorld chrome checker reads the page's web DOM over CDP (``connect_over_cdp``
-> ``page.content()``) and parses BY CSS CLASS — e.g. an active filter chip is the
presence of class ``fT28tf`` / ``filter-selector-link``, an active tab is
``...tab--active``. AT-SPI a11y (the ``accessibility_tree`` we already feed the
perceiver / done-auditor) carries role + name + state but **no CSS class**, so it
cannot tell an active chip from a plain link. This module dumps exactly that
missing layer: every interactive/stateful element's ``class`` + ``aria-*`` state.

Design rules:
  • DOMAIN-AGNOSTIC — never read a task's evaluator config; that would leak the
    answer. We surface raw stateful DOM and let the judge reason itself.
  • COMPLEMENT, don't duplicate — text already lives in a11y; we keep it short and
    lead with the class/aria-state that a11y lacks.
"""
from typing import Any, Callable, Dict, List, Optional

# Run in-page: collect interactive/stateful elements with CSS class + ARIA state.
_COLLECT_JS = r"""() => {
  const ARIA = ['aria-selected','aria-pressed','aria-checked','aria-current',
                'aria-expanded','aria-disabled'];
  const out = [];
  const nodes = document.querySelectorAll(
    'a,button,input,select,option,li,[role],[aria-selected],[aria-pressed],' +
    '[aria-checked],[aria-current]');
  for (const el of nodes) {
    const cls = (typeof el.className === 'string') ? el.className : '';
    const aria = {};
    for (const a of ARIA) { const v = el.getAttribute(a); if (v !== null) aria[a] = v; }
    const role = el.getAttribute('role') || el.tagName.toLowerCase();
    let txt = (el.innerText || el.value || el.getAttribute('aria-label') || '')
                .trim().replace(/\s+/g, ' ').slice(0, 80);
    const interactive =
      ['a','button','input','select','option'].includes(el.tagName.toLowerCase())
      || el.getAttribute('role');
    // keep if it carries a class OR an aria-state, and is interactive/labelled
    if ((cls || Object.keys(aria).length) && interactive
        && (txt || Object.keys(aria).length)) {
      out.push({role, text: txt, cls: cls.slice(0, 160), aria});
    }
  }
  return out;
}"""


def _safe(fn: Callable, default=None):
    try:
        return fn()
    except Exception:
        return default


def _collect_from_page(page, max_elems: int) -> Dict[str, Any]:
    _safe(lambda: page.wait_for_load_state("domcontentloaded", timeout=5000))
    els = _safe(lambda: page.evaluate(_COLLECT_JS), default=None)
    url = _safe(lambda: page.url)
    if els is None:
        return {"url": url, "error": "evaluate failed", "elements": []}
    seen, uniq = set(), []
    for e in els:
        k = (e["role"], e["text"], e["cls"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(e)
    return {"url": url, "elements": uniq[:max_elems]}


def dump_tab_doms(host: str, chromium_port: int, *, max_elems: int = 150,
                  all_tabs: bool = False) -> List[Dict[str, Any]]:
    """Connect over CDP (same entry the checker uses) and return per-tab
    stateful-DOM. Each entry: {url, elements: [{role, text, cls, aria}]}.

    all_tabs=False -> just the front/last page (active-tab heuristic);
    True -> every open page (use when unsure which tab is active)."""
    from playwright.sync_api import sync_playwright
    url = f"http://{host}:{chromium_port}"
    out: List[Dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(url)
        try:
            pages = []
            for ctx in browser.contexts:
                pages.extend(ctx.pages)
            if not pages:
                return out
            targets = pages if all_tabs else [pages[-1]]
            for pg in targets:
                out.append(_collect_from_page(pg, max_elems))
        finally:
            _safe(browser.close)
    return out


import re

# Tailwind / utility class noise to strip — keeps only semantic / state-bearing
# / obfuscated class tokens (e.g. tab-active, filter-selector-link, fT28tf).
_UTIL = re.compile(
    r'[:\[\]!()/]'                       # tailwind variants / arbitrary values
    r'|^(?:text|bg|w|h|min|max|p[xytrbl]?|m[xytrbl]?|flex|grid|gap|space|'
    r'whitespace|rounded|border|shadow|font|leading|tracking|z|top|left|right|'
    r'bottom|inset|absolute|relative|fixed|sticky|block|inline|inline-block|'
    r'hidden|overflow|cursor|opacity|transition|duration|ease|hover|focus|'
    r'group|order|col|row|justify|items|self|content|place|object|truncate|'
    r'no-underline|underline|uppercase|lowercase|capitalize|container)'
    r'(?:-|$)'
)
_STATE_TOKENS = ("active", "selected", "checked", "current")


def _clean_classes(cls: str) -> List[str]:
    out = [t for t in cls.split() if t and not _UTIL.search(t) and len(t) <= 36]
    return out[:4]


def _states(e: Dict[str, Any]) -> List[str]:
    """State that a11y lacks: ARIA true-states + class-encoded state words."""
    st: List[str] = []
    for k, v in (e.get("aria") or {}).items():
        kk = k.replace("aria-", "")
        if v == "true":
            st.append(kk)
        elif kk == "current" and v and v != "false":
            st.append("current")
    for t in _clean_classes(e.get("cls", "")):
        for s in _STATE_TOKENS:
            if (t == s or t.endswith("-" + s) or t.endswith("--" + s)
                    or t.endswith("__" + s)) and s not in st:
                st.append(s)
    return st


def render_web_dom(doms: List[Dict[str, Any]], *, max_chars: int = 2200) -> str:
    """Compact, LLM-friendly view of the page's checker-relevant state — the
    class + ARIA-state layer AT-SPI a11y lacks. Drops CSS utility noise, dedups,
    and LEADS with what is currently active/selected (the decisive signal)."""
    blocks: List[str] = []
    for tab in doms:
        els = tab.get("elements") or []
        if not els:
            continue
        seen, active, other = set(), [], []
        for e in els:
            key = (e.get("role"), e.get("text"))
            if key in seen:
                continue
            seen.add(key)
            role, txt = e.get("role", ""), (e.get("text") or "").strip()
            st = _states(e)
            if not txt and not st:
                continue
            if st:
                cls = _clean_classes(e.get("cls", ""))
                tag = f" .{'.'.join(cls)}" if cls else ""
                active.append(f'  {role} "{txt}" [{", ".join(st)}]{tag}')
            elif txt and role in {
                    "button", "link", "a", "tab", "option", "checkbox", "radio",
                    "menuitem", "menuitemcheckbox", "combobox", "textbox",
                    "searchbox", "switch", "input", "select"}:
                other.append(f'{role} "{txt[:40]}"')
        lines = [f"WEB PAGE: {tab.get('url', '?')}"]
        if active:
            lines.append("active/selected/checked:")
            lines += active
        if other:
            lines.append("other interactive: " + "; ".join(other[:40]))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)[:max_chars]


# back-compat alias (older call sites)
render_stateful_dom = render_web_dom

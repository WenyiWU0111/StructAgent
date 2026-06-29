"""Subgoal-conditioned coarse filter for linearized a11y trees (no LLM).

Two passes:
  1. Embedding similarity — per row, SUM cosine sim over the query set
     ``[subgoal_text] + outcome_evidence_hints``, keep top-N. Sum, not max,
     so a row relevant to many queries beats one strongly relevant to one
     (queries jointly describe the task).
  2. Anomaly carve-out — force-include rows tagged in ``ANOMALY_ROLES``
     (popup/menu/alert) so just-appeared blockers always reach the
     perceiver, even when their labels don't lexically match the subgoal.

Output is the same ``tag\\ttext\\tstate`` format as the input (header +
trailing ``...`` marker preserved) — drop-in for ``trim_accessibility_tree``.

Input is the OUTPUT of ``linearize_accessibility_tree``, which already kills
~96% of nodes (role whitelist, visible/showing, size>0, active app, dedup).
We only do the final intent-based ranking. The crop bbox is NOT computed
here — the perceiver VLM picks that from the full screenshot.

See docs/PROGRESS_LEDGER_V2_CHANGELOG.md §17.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Popup-ish roles that must reach the perceiver regardless of similarity.
# Only ``alert`` and ``menu`` survive linearize_accessibility_tree's funnel
# in real Chrome runtime logs (see tools/probe_a11y_filter_variants.py);
# ``dialog`` / ``alertdialog`` get dropped earlier (size=0 or empty name).
ANOMALY_ROLES = frozenset({
    "alert",
    "menu",
})


# --------------------------------------------------------------------------- #
# Embedding model — lazy-loaded once per process
# --------------------------------------------------------------------------- #

_MODEL = None
_DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_model():
    """Lazy-load the embedding model (~80MB on first use; ~50ms/encode on CPU)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers not installed; run "
                "`pip install sentence-transformers`"
            ) from e
        logger.info("[a11y_prefilter] loading embedding model %s",
                    _DEFAULT_MODEL_NAME)
        _MODEL = SentenceTransformer(_DEFAULT_MODEL_NAME)
    return _MODEL


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _row_tag(row: str) -> str:
    """Leading tag (role) from a 'tag\\ttext\\tstate' row; '' if malformed."""
    tab = row.find("\t")
    return row[:tab] if tab > 0 else ""


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def prefilter_linearized_a11y(
    linearized_text: str,
    subgoal_text: str = "",
    outcome_evidence_hints: Optional[Sequence[str]] = None,
    top_n: int = 100,
) -> str:
    """Subgoal-relevant a11y filter, preserving any appended ``[WEB DOM ...]``
    block verbatim — that block is checker-source web DOM (class + aria-state),
    NOT AT-SPI rows, and must never be embedding-filtered/dropped."""
    marker = "\n\n[WEB DOM"
    web = ""
    if marker in linearized_text:
        linearized_text, _rest = linearized_text.split(marker, 1)
        web = marker + _rest
    return _prefilter_a11y_core(
        linearized_text, subgoal_text, outcome_evidence_hints, top_n) + web


def _prefilter_a11y_core(
    linearized_text: str,
    subgoal_text: str = "",
    outcome_evidence_hints: Optional[Sequence[str]] = None,
    top_n: int = 100,
) -> str:
    """Filter a linearized a11y text down to subgoal-relevant rows.

    Args:
        linearized_text: ``linearize_accessibility_tree`` output; first line
            is the ``tag\\ttext\\tstate`` header.
        subgoal_text: current subgoal (embedding query).
        outcome_evidence_hints: active evidence_hint strings (also queries).
        top_n: target row count; anomaly carve-out adds on top
            (final size = top_n + |anomaly rows|).

    Returns the filtered text in original row order. Returns input unchanged
    when it's empty/header-only, row count <= top_n, or there's no query.
    """
    if not linearized_text:
        return linearized_text

    lines = linearized_text.strip().split("\n")
    if not lines:
        return linearized_text

    header = lines[0]
    body = lines[1:]

    # Strip a trailing "..." marker so we don't embed it (filtering shrinks
    # the set anyway, so we don't re-add it).
    if body and body[-1].strip() == "...":
        body = body[:-1]

    if len(body) <= top_n:
        return linearized_text  # nothing to gain; skip embedding cost

    queries = [q.strip() for q in
               ([subgoal_text] + list(outcome_evidence_hints or []))
               if q and q.strip()]
    if not queries:
        return linearized_text  # no query → no signal → don't filter

    # Pass 2 (cheap, done first): always-keep anomaly rows, by tag.
    anomaly_idx = {
        i for i, row in enumerate(body)
        if _row_tag(row).lower() in ANOMALY_ROLES
    }

    # Pass 1: embedding rank top-N. Skip the must-keep set so the budget
    # goes to genuine ranking.
    rest_idx = [i for i in range(len(body)) if i not in anomaly_idx]
    rest_rows = [body[i] for i in rest_idx]

    try:
        model = _get_model()
        # Normalize → dot product is cosine sim.
        query_emb = model.encode(
            queries, normalize_embeddings=True, show_progress_bar=False
        )
        row_emb = model.encode(
            rest_rows,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=False,
        )
    except Exception as e:
        logger.warning(
            "[a11y_prefilter] embedding failed (%s) — returning unfiltered "
            "(degraded mode)", e,
        )
        return linearized_text

    # Per-row score = SUM of cosine sims over queries (see module docstring).
    sims = (row_emb @ query_emb.T).sum(axis=1)

    n_keep = min(top_n, len(rest_rows))
    if n_keep == 0:
        ranked_idx = set()
    else:
        import numpy as np
        # argpartition is O(n); we only need the top-n boundary.
        top_local = np.argpartition(-sims, n_keep - 1)[:n_keep]
        ranked_idx = {rest_idx[i] for i in top_local}

    # Combine and restore original tree order.
    keep_idx = sorted(anomaly_idx | ranked_idx)
    kept_rows = [body[i] for i in keep_idx]

    if not kept_rows:
        return linearized_text  # defensive: never return a naked header

    out_lines = [header] + kept_rows
    n_dropped = len(body) - len(kept_rows)
    if n_dropped > 0:
        out_lines.append(f"…[prefilter dropped {n_dropped} less-relevant rows]")
    return "\n".join(out_lines)


# --------------------------------------------------------------------------- #
# Diagnostics — quick stats for tests / replays
# --------------------------------------------------------------------------- #

def explain_prefilter(
    linearized_text: str,
    subgoal_text: str = "",
    outcome_evidence_hints: Optional[Sequence[str]] = None,
    top_n: int = 100,
) -> dict:
    """Diagnostics on what the filter would do (for tests / replay tools)."""
    if not linearized_text:
        return {"input_rows": 0, "kept_rows": 0, "anomaly_in_input": 0,
                "filtered_text": linearized_text}
    lines = linearized_text.strip().split("\n")
    body = [r for r in lines[1:] if r.strip() != "..."]
    anomaly = [i for i, r in enumerate(body)
               if _row_tag(r).lower() in ANOMALY_ROLES]
    filtered = prefilter_linearized_a11y(
        linearized_text, subgoal_text, outcome_evidence_hints, top_n,
    )
    out_body = [r for r in filtered.split("\n")[1:] if not r.startswith("…")]
    return {
        "input_rows": len(body),
        "kept_rows": len(out_body),
        "anomaly_in_input": len(anomaly),
        "filtered_text": filtered,
    }

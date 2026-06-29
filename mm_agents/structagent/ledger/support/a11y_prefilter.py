"""Subgoal-conditioned coarse filter for linearized a11y trees.

Two passes, both mechanical (no LLM):
  1. Embedding similarity — for each row, SUM cosine sim over the query
     set ``[subgoal_text] + outcome_evidence_hints``. Keep top-N.
     (Sum aggregation, not max — a row that's medium-relevant to many
      queries should outrank one that's strongly relevant to a single
      query, since the queries together describe the agent's task.)
  2. Anomaly carve-out — force-include rows whose tag is in
     ``ANOMALY_ROLES`` (popup / menu / alert), so just-appeared
     blockers always reach the perceiver and surface as
     ``unexpected_blockers`` — even when their labels don't lexically
     match the subgoal.

The filtered output is the same ``tag\\ttext\\tstate`` format as the
input — drop-in replacement for ``trim_accessibility_tree`` in the
existing pipeline. Header line ``tag\\ttext\\tstate`` and trailing
``...`` truncation marker are preserved.

NOTE — input is the OUTPUT of ``linearize_accessibility_tree``
(autoglm/prompt/accessibility_tree_handle.py), which already filters by
role whitelist + visible+showing + has-content + size>0 + active app
only + dedup. Empirically the funnel kills ~96% of nodes BEFORE this
prefilter sees them. Our job is just the final intent-based ranking.

The visual_focus bbox (which screen region to crop) is NOT computed
here — the perceiver VLM looks at the full screenshot and outputs the
crop bbox itself. This module only ranks text rows for the perceiver's
``relevant_controls`` pool.

See docs/PROGRESS_LEDGER_V2_CHANGELOG.md §17 for design context.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Popup-ish roles that must reach the perceiver regardless of subgoal
# embedding similarity. Empirically determined from real chrome a11y
# dumps (see tools/probe_a11y_filter_variants.py): only ``alert`` and
# ``menu`` actually survive linearize_accessibility_tree's funnel and
# appear in real Chrome runtime logs. ``dialog`` / ``alertdialog`` are
# typically dropped earlier (size=0 or empty name).
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
    """Lazy-load the embedding model. ~80MB download on first use; ~50ms
    per encoding call afterward (CPU)."""
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
    """Extract the leading tag (role) from a 'tag\\ttext\\tstate' row.
    Returns empty string for malformed rows (we keep them as-is)."""
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
        linearized_text: output of ``linearize_accessibility_tree``. First
            line is expected to be the header ``tag\\ttext\\tstate``.
        subgoal_text: the agent's current subgoal (drives embedding query).
        outcome_evidence_hints: list of currently-active outcome
            evidence_hint strings (also embedded as queries).
        top_n: target row count for embedding-ranked rows. Anomaly carve-out
            adds on top of this; final size = top_n + |anomaly rows|.

    Returns:
        Filtered linearized a11y text (header + filtered rows + optional
        truncation marker), preserving the original row order.
        Returns the input unchanged when:
          - input is empty / header-only
          - row count ≤ top_n (no benefit from filtering)
          - both subgoal_text and outcome_evidence_hints are empty (no query)
    """
    if not linearized_text:
        return linearized_text

    lines = linearized_text.strip().split("\n")
    if not lines:
        return linearized_text

    header = lines[0]
    body = lines[1:]

    # Strip a trailing "..." truncation marker if present so we don't try to
    # embed it. We won't re-add it (filtering shrinks the set anyway).
    if body and body[-1].strip() == "...":
        body = body[:-1]

    if len(body) <= top_n:
        # Nothing to gain; return as-is so we don't pay embedding cost.
        return linearized_text

    queries = [q.strip() for q in
               ([subgoal_text] + list(outcome_evidence_hints or []))
               if q and q.strip()]
    if not queries:
        # No query → no semantic signal → don't filter.
        return linearized_text

    # ── Pass 2 (computed first; cheap): anomaly carve-out ──
    # Always-keep set, computed by tag. Cheap; skip embedding for these.
    anomaly_idx = {
        i for i, row in enumerate(body)
        if _row_tag(row).lower() in ANOMALY_ROLES
    }

    # ── Pass 1: embedding rank top-N (SUM aggregation) ──
    # Only embed rows that aren't already in the must-keep set, so we use
    # the budget for genuine ranking work.
    rest_idx = [i for i in range(len(body)) if i not in anomaly_idx]
    rest_rows = [body[i] for i in rest_idx]

    try:
        model = _get_model()
        # Normalize so we can use dot-product as cosine similarity.
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

    # Per-row score = SUM of cosine similarities across all queries.
    # Sum (not max) means rows aligned with MULTIPLE aspects of the task
    # outrank rows aligned with just one — the queries together describe
    # the agent's full intent so we want joint relevance.
    sims = (row_emb @ query_emb.T).sum(axis=1)

    # Take top_n indices into rest_rows by similarity (descending).
    n_keep = min(top_n, len(rest_rows))
    if n_keep == 0:
        ranked_idx = set()
    else:
        import numpy as np
        # argpartition is O(n); we only need the top-n boundary.
        top_local = np.argpartition(-sims, n_keep - 1)[:n_keep]
        ranked_idx = {rest_idx[i] for i in top_local}

    # ── Combine + restore original a11y tree order ──
    keep_idx = sorted(anomaly_idx | ranked_idx)
    kept_rows = [body[i] for i in keep_idx]

    if not kept_rows:
        # Defensive: shouldn't happen, but don't return naked header.
        return linearized_text

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
    """Return diagnostics about what the filter would do — useful for
    tests and replay tools."""
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

"""bBoN — Behavior Best-of-N test-time selection (OFF by default; see DESIGN.md).

A selection layer that sits OUTSIDE the agent: run N diverse rollouts of one task,
render each rollout into a behavior narrative (a turning-point storyboard built
from the agent's own ledger / timeline — FACTS, not the verifier's verdicts), then
a single comparative-MCQ judge picks the most-likely-successful rollout.

The agent core (loop.py / _pa_predict / verifier / memory) is UNTOUCHED. Nothing in
this package is imported on the single-rollout path; it activates only under
ENABLE_BBON, driven by a thin runner wrapper (scripts/python/run_bbon.py).

Modules:
  narrative.py   — turning-point selection + behavior-narrative rendering (read-only)
  judge.py       — comparative-MCQ judge over N narratives
  eval_judge.py  — OFFLINE judge-accuracy harness over a frozen rollout set
"""

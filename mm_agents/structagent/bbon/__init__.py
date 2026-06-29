"""bBoN — Behavior Best-of-N test-time selection (OFF by default; see DESIGN.md).

Selection layer outside the agent: run N diverse rollouts, render each into a
behavior narrative (a turning-point storyboard from the agent's own ledger/timeline
— facts, not verifier verdicts), then a comparative-MCQ judge picks the rollout most
likely to have succeeded.

The agent core is untouched; nothing here is imported on the single-rollout path. It
activates only under ENABLE_BBON via scripts/python/run_bbon.py.

Modules:
  narrative.py   — turning-point selection + narrative rendering (read-only)
  judge.py       — comparative-MCQ judge over N narratives
  eval_judge.py  — offline judge-accuracy harness over a frozen rollout set
"""

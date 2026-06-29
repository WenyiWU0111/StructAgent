"""Perceiver — the agent's perceive step.

Distills raw observation into a focused, agent-friendly form so the
planner reasons over signal instead of noise.

  - perceiver.py        general perceiver: one VLM call per step turns
                        (screenshot + a11y) into a ``SubgoalSnapshot``.
  - slide/sheet/doc_perceiver.py
                        office-domain specializations: a UNO probe of
                        the live LibreOffice document yields a compact
                        structured block (the "SP/SheetPerceiver/DP"
                        view) for Impress / Calc / Writer tasks.
"""

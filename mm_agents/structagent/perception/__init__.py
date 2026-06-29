"""Perceiver — distills raw observation into a focused form for the planner.

  - perceiver.py: general VLM perceiver, (screenshot + a11y) -> ``SubgoalSnapshot``.
  - slide/sheet/doc_perceiver.py: office specializations; a UNO probe of the
    live LibreOffice doc yields a compact structured block for Impress/Calc/Writer.
"""

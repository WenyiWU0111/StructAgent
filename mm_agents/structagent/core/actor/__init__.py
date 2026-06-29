"""Acting stack — the agent's action-producing roles.

  - actor.py            : Actor mixin (decomposer LLM with inline grounding).
  - decomposer_actor.py : decomposer call + inline-grounding protocol.
  - action_compile.py   : ActionCompile mixin (parsed action -> pyautogui).
"""
from .actor import Actor
from .action_compile import ActionCompile

__all__ = ["Actor", "ActionCompile"]

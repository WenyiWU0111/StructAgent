"""WebJudge (Online-Mind2Web) — faithful re-implementation of the OSU-NLP
3-stage LLM grader (arXiv:2504.01382), which grades web-navigation tasks from
task + key-points + key-screenshots + action history and DELIBERATELY EXCLUDES
the agent's final text response (the WebVoyager judge's response-believing
behaviour is a documented false-positive source for navigation tasks).

Stage 1  identify_key_points(task) -> the indispensable checkpoints.
Stage 2  score_image(task, key_points, img) -> 1..5 relevance; keep >= threshold.
Stage 3  final_judgment(task, key_points, action_history, kept imgs+reasons)
         -> "success"/"failure".  No agent answer anywhere.

Reference: OSU-NLP-Group/Online-Mind2Web src/methods/webjudge_online_mind2web.py
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger("desktopenv.webjudge")

# ── Stage 1: key points ───────────────────────────────────────────────
_KEYPOINTS_SYS = (
    "You are an expert tasked with analyzing a given task to identify the key "
    "points explicitly stated in the task description.\n\n"
    "**Objective**: Carefully analyze the task description and extract the key "
    "points that MUST be verified to consider the task successfully completed. "
    "A key point is an indispensable, checkable sub-goal (a navigation target, "
    "a filter applied, an item found, an action taken).\n\n"
    "Return your response in this exact format:\n"
    "**Key Points**:\n- <key point 1>\n- <key point 2>\n- ..."
)

# ── Stage 2: per-screenshot relevance score ───────────────────────────
_IMG_SYS = (
    "You are an expert evaluator tasked with determining whether an image "
    "contains information about the necessary steps to complete a task.\n\n"
    "Score 1-5 how much this single web-page snapshot shows progress toward, or "
    "evidence of, the key points (5 = directly shows a key point achieved; "
    "1 = irrelevant). Output ONLY the integer score."
)
_IMG_USER = (
    "**Task**: {task}\n"
    "**Key Points for Task Completion**: {key_points}\n"
    "The snapshot of the web page is shown in the image."
)

# ── Stage 3: final judgment (NO agent answer) ─────────────────────────
_FINAL_SYS = (
    "You are an expert in evaluating the performance of a web navigation agent. "
    "Given the user's task, the agent's action history, key points for task "
    "completion, and some potentially important web pages in the agent's "
    "trajectory and their reasons, your goal is to determine whether the agent "
    "has completed the task.\n\n"
    "Judge ONLY from the action history and the page snapshots — there is no "
    "self-reported final answer, and you must not assume success from a claim. "
    "Success requires concrete evidence (in the snapshots / actions) that the "
    "key points were achieved.\n\n"
    "Format your response EXACTLY as:\n"
    "Thoughts: <your reasoning over the evidence>\n"
    "Status: success OR failure"
)
_FINAL_USER = (
    "**Task**: {task}\n\n"
    "**Key Points**:\n{key_points}\n\n"
    "**Action History**:\n{action_history}\n\n"
    "{snapshot_block}"
)


def _img_content(b64: str) -> dict:
    return {"type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}}


class WebJudge:
    def __init__(self, client, model: str, score_threshold: int = 3,
                 max_images: int = 50, max_score_images: int = 15):
        self.client = client
        self.model = model
        self.score_threshold = score_threshold
        self.max_images = max_images
        # cap Stage-2 calls per task (evenly sample if a trajectory is huge)
        self.max_score_images = max_score_images

    def _chat(self, system: str, user_content, max_tokens: int = 512) -> str:
        kwargs = {}
        if "qwen3.5" in self.model.lower():
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user_content}],
            temperature=0.0, max_tokens=max_tokens, **kwargs)
        return resp.choices[0].message.content or ""

    def identify_key_points(self, task: str) -> str:
        try:
            out = self._chat(_KEYPOINTS_SYS, f"Task: {task}", max_tokens=400)
            if "**Key Points**:" in out:
                return out.split("**Key Points**:")[1].strip()
            if "Key Points:" in out:
                return out.split("Key Points:")[-1].strip()
            return out.strip()
        except Exception as e:
            logger.warning("[WebJudge] key-points failed: %s", e)
            return task

    def score_image(self, task: str, key_points: str, b64: str) -> int:
        try:
            content = [{"type": "text",
                        "text": _IMG_USER.format(task=task, key_points=key_points)},
                       _img_content(b64)]
            out = self._chat(_IMG_SYS, content, max_tokens=8)
            m = re.findall(r"[1-5]", out)
            return int(m[0]) if m else 1
        except Exception as e:
            logger.warning("[WebJudge] score_image failed: %s", e)
            return 1

    def _select_for_scoring(self, screenshots: List[str]) -> List[int]:
        n = len(screenshots)
        if n <= self.max_score_images:
            return list(range(n))
        # evenly sample, always keep the last few (final state matters most)
        step = n / self.max_score_images
        idxs = sorted(set([int(i * step) for i in range(self.max_score_images)]
                          + [n - 1, n - 2, n - 3]))
        return [i for i in idxs if 0 <= i < n]

    def judge(self, task: str, action_history: List[str],
              screenshots_b64: List[str]) -> Tuple[float, str]:
        """Return (score in {0,1}, raw judgment)."""
        if not task:
            return 0.0, "<no task>"
        key_points = self.identify_key_points(task)

        # Stage 2: relevance-score a (capped) set of screenshots, keep >= thresh
        kept: List[str] = []
        kept_reasons: List[str] = []
        for i in self._select_for_scoring(screenshots_b64):
            s = self.score_image(task, key_points, screenshots_b64[i])
            if s >= self.score_threshold:
                kept.append(screenshots_b64[i])
                kept_reasons.append(f"screenshot #{i + 1} (relevance {s}/5)")
        kept = kept[-self.max_images:]
        kept_reasons = kept_reasons[-self.max_images:]

        ah = "\n".join(f"{i + 1}. {a}" for i, a in enumerate(action_history)) or "(none)"
        if kept:
            reasons = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(kept_reasons))
            snap_block = ("**Important snapshots (with reasons)**:\n" + reasons
                          + "\nThe snapshot images follow.")
            user_content = [{"type": "text", "text": _FINAL_USER.format(
                task=task, key_points=key_points, action_history=ah,
                snapshot_block=snap_block)}]
            for b in kept:
                user_content.append(_img_content(b))
        else:
            snap_block = "**No web page in the trajectory was scored relevant to the key points.**"
            user_content = _FINAL_USER.format(
                task=task, key_points=key_points, action_history=ah,
                snapshot_block=snap_block)

        try:
            verdict = self._chat(_FINAL_SYS, user_content, max_tokens=600)
        except Exception as e:
            logger.warning("[WebJudge] final judgment failed: %s", e)
            return 0.0, f"<final-judge error: {e}>"
        low = verdict.lower()
        m = re.search(r"status\s*[:：]\s*(success|failure)", low)
        success = bool(m and m.group(1) == "success") or (
            not m and "success" in low.split("status")[-1])
        return (1.0 if success else 0.0,
                f"key_points={key_points[:300]} || kept={len(kept)}/"
                f"{len(screenshots_b64)} || {verdict[:500]}")

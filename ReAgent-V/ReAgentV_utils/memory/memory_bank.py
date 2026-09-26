"""
memory_bank.py
==============
ToolMemoryBank — A stateful memory module for the Adaptive Retrieval Loop
of the ReAgent-V CoVR pipeline.

Purpose:
  Instead of performing a single-pass retrieval, the agent now iterates:
  - If Critic Agent score >= reward_threshold  →  accept results, stop.
  - If score < reward_threshold and iterations < max_iterations  →
      analyse critic feedback, adapt the search strategy, and retry.

The memory bank stores:
  - Per-iteration history: tools used, query variants, candidates, scores
  - Set of rejected (negative) video ids to exclude from future searches
  - Adaptive weights (alpha) for balancing Image vs Text in CLIP encoding
"""

from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """Snapshot of one retrieval attempt."""
    iteration: int
    alpha: float                      # Image/Text weight at this iteration
    tools_used: List[str]             # e.g. ['CLIP_coarse', 'DET']
    query_prompt_used: str            # possibly expanded vs the original
    top_candidates: List[str]         # video paths returned as Top-N
    critic_raw: str                   # raw JSON string from Critic Agent
    scalar_reward: float              # extracted scalar reward [0.0, 1.0]
    critique_summary: str = ""        # short natural-language feedback


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ToolMemoryBank:
    """
    Stateful memory and strategy controller for Adaptive Retrieval Loop.

    Usage:
        memory = ToolMemoryBank(max_iterations=3, reward_threshold=0.65)

        for t in range(memory.max_iterations):
            strategy = memory.get_current_strategy()
            # ... run retrieval + reranker with `strategy` ...
            # ... run Critic Agent → get scalar_reward, critic_raw ...
            memory.record_step(t, tools, query, candidates, critic_raw, reward)
            if not memory.should_continue(reward, t):
                break

        best = memory.get_best_result()
    """

    def __init__(
        self,
        max_iterations: int = 3,
        reward_threshold: float = 0.85,
        initial_alpha: float = 0.5,
    ):
        self.max_iterations   = max_iterations
        self.reward_threshold = reward_threshold
        self.history: List[IterationRecord] = []
        self.visited_negatives: set = set()   # video paths confirmed irrelevant
        self.alpha = initial_alpha            # current image/text balance weight

    # ------------------------------------------------------------------
    # Core loop control
    # ------------------------------------------------------------------

    def should_continue(self, scalar_reward: float, iteration: int, can_early_stop: Optional[bool] = None) -> bool:
        """Return True if the loop should attempt another iteration."""
        if iteration + 1 >= self.max_iterations:
            return False  # Reached max budget → stop
        if can_early_stop is not None:
            return not can_early_stop
        if scalar_reward >= self.reward_threshold:
            return False  # Good enough → stop
        return True

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        iteration: int,
        tools_used: List[str],
        query_prompt_used: str,
        top_candidates: List[str],
        critic_raw: str,
        scalar_reward: float,
        safe_candidates: Optional[set] = None,
    ) -> None:
        """Save the outcome of one retrieval + reranking attempt."""
        # Extract short critique summary
        critique_summary = _extract_critique_summary(critic_raw)

        record = IterationRecord(
            iteration=iteration,
            alpha=self.alpha,
            tools_used=tools_used,
            query_prompt_used=query_prompt_used,
            top_candidates=top_candidates,
            critic_raw=critic_raw,
            scalar_reward=scalar_reward,
            critique_summary=critique_summary,
        )
        self.history.append(record)

        # Module 3 Hard Blacklist with Safe Guard:
        # Exclude Top-1 distractor if rejected by Critic (reward < 0.65 with mismatch or reward < 0.50)
        # AND not protected by visual prior / reranker confirmation
        verdict = _extract_verdict(critic_raw)
        is_clear_mismatch = (scalar_reward < 0.65 and verdict in ["MISMATCH", "NO_MATCH"]) or (scalar_reward < 0.50)

        if top_candidates and is_clear_mismatch:
            rejected_id = top_candidates[0]
            if safe_candidates and rejected_id in safe_candidates:
                print(f"[MemoryBank] Safe Blacklist Guard protected candidate: {os.path.basename(rejected_id)} (visual prior/MATCH confirmed, reward={scalar_reward:.3f})")
            else:
                self.visited_negatives.add(rejected_id)
                print(f"[MemoryBank] Hard Blacklist added rejected Top-1: {os.path.basename(rejected_id)} (reward={scalar_reward:.3f}, verdict={verdict})")
        elif top_candidates and (scalar_reward < 0.85):
            print(f"[MemoryBank] Candidate {os.path.basename(top_candidates[0])} preserved from Hard Blacklist (reward={scalar_reward:.3f}).")

        print(
            f"[MemoryBank] Iter {iteration} | reward={scalar_reward:.3f} | "
            f"alpha={self.alpha:.2f} | negatives={len(self.visited_negatives)}"
        )

    # ------------------------------------------------------------------
    # Adaptive strategy
    # ------------------------------------------------------------------

    def get_current_strategy(self) -> Dict[str, Any]:
        """
        Return the retrieval strategy for the next iteration, derived from
        analysing the last critic feedback.

        Returns a dict:
          {
            'alpha': float,                 # Image vs Text weight for CLIP fusion
            'suggested_hybrid_alpha': float,# CLIP vs LLaVA weight in reranker
            'force_ocr': bool,
            'force_det': bool,
            'query_expansion_hint': str,    # Additional keywords for the prompt
            'failure_feedback': str,        # Reason previous candidate failed
            'exclude_videos': set,          # Videos to skip in this iteration
          }
        """
        strategy = {
            "alpha": self.alpha,
            "suggested_hybrid_alpha": 0.55,
            "force_ocr": False,
            "force_det": False,
            "query_expansion_hint": "",
            "failure_feedback": "",
            "exclude_videos": self.visited_negatives.copy(),
        }

        if not self.history:
            return strategy  # First iteration — use defaults

        last = self.history[-1]
        critique = (last.critique_summary + " " + last.critic_raw).lower()
        strategy["failure_feedback"] = last.critique_summary if last.critique_summary else "Previous candidate failed to execute the required visual transformation."

        # If previous iteration had low reward, reduce hybrid_alpha to give more weight to LLaVA visual verification
        if last.scalar_reward < 0.70:
            strategy["suggested_hybrid_alpha"] = 0.35
            print(f"[MemoryBank] Strategy: previous reward {last.scalar_reward:.3f} < 0.70 -> lowering hybrid_alpha to 0.35 (LLaVA-dominant reranking)")

        # --- Rule 1: Visual context mismatch → moderate boost to image weight (safe max 0.60) ---
        if any(kw in critique for kw in ["visual_mismatch", "appearance", "color", "background", "setting", "scene", "not match the reference image"]):
            strategy["alpha"] = min(0.60, self.alpha + 0.10)
            strategy["force_det"] = True
            print("[MemoryBank] Strategy: moderately boosting image weight (safe max 0.60) + DET.")

        # --- Rule 2: Action / temporal mismatch → boost text weight (safe min 0.25) ---
        elif any(kw in critique for kw in ["action_mismatch", "action", "temporal", "motion", "movement", "not demonstrated", "missing"]):
            strategy["alpha"] = max(0.25, self.alpha - 0.15)
            strategy["query_expansion_hint"] = _extract_action_keywords(last.query_prompt_used)
            print("[MemoryBank] Strategy: boosting text weight (temporal/action mismatch, safe min 0.25).")

        # --- Rule 3: Missing text / signs → force OCR ---
        elif any(kw in critique for kw in ["text_mismatch", "text", "sign", "written", "ocr", "read", "label"]):
            strategy["force_ocr"] = True
            print("[MemoryBank] Strategy: enabling forced OCR (text content mismatch).")

        # --- Rule 4: General low score → target-focused fusion (alpha=0.35) ---
        else:
            strategy["alpha"] = 0.35
            strategy["force_det"] = True
            print("[MemoryBank] Strategy: target-focused fusion (alpha=0.35) (generic low score).")

        self.alpha = strategy["alpha"]  # persist for next call
        return strategy

    # ------------------------------------------------------------------
    # Result retrieval
    # ------------------------------------------------------------------

    def get_best_result(self) -> Optional[IterationRecord]:
        """Return the iteration record with the highest scalar reward."""
        if not self.history:
            return None
        return max(self.history, key=lambda r: r.scalar_reward)

    def summary(self) -> str:
        """Return a human-readable summary of all iterations."""
        lines = ["=== Memory Bank Summary ==="]
        for r in self.history:
            lines.append(
                f"  Iter {r.iteration}: reward={r.scalar_reward:.3f} | "
                f"alpha={r.alpha:.2f} | tools={r.tools_used} | "
                f"critique='{r.critique_summary[:80]}'"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_llava_output(raw: str) -> str:
    """
    Strip LLaVA chat-template artifacts and markdown code fences so that
    the remaining string is plain JSON that json.loads() can parse.

    Handles patterns like:
      - "system\n...\nassistant\n{...}"
      - "```json\n{...}\n```"
      - Leading/trailing whitespace
    """
    if not raw:
        return raw
    text = raw.strip()
    # Remove chat-template prefix: keep only content after last 'assistant\n'
    if "assistant\n" in text:
        text = text.split("assistant\n")[-1].strip()
    # Strip markdown code fences  ```json ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    return text.strip()


def _extract_scalar_reward(critic_raw: str) -> float:
    """Parse scalar_reward from the Critic Agent JSON output."""
    cleaned = _strip_llava_output(critic_raw)
    # Attempt 1: full JSON parse
    try:
        data = json.loads(cleaned)
        return float(data.get("scalar_reward", 0.5))
    except Exception:
        pass
    # Attempt 2: regex on cleaned text
    m = re.search(r'"scalar_reward"\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # Attempt 3: regex on raw text (fallback)
    m = re.search(r'"scalar_reward"\s*:\s*([0-9]*\.?[0-9]+)', critic_raw)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    return 0.5


def _extract_critique_summary(critic_raw: str) -> str:
    """Extract the structured_feedback field from critic JSON."""
    cleaned = _strip_llava_output(critic_raw)
    try:
        data = json.loads(cleaned)
        return data.get("structured_feedback", "")
    except Exception:
        pass
    m = re.search(r'"structured_feedback"\s*:\s*"([^"]+)"', cleaned)
    if m:
        return m.group(1)
    # Fallback: first 200 chars of cleaned output
    return cleaned[:200]


def _extract_action_keywords(prompt: str) -> str:
    """
    Simple heuristic to extract action-related keywords from a modification prompt
    for query expansion. E.g. 'replace cow with goat' → 'goat cow animal farm'.
    """
    stop_words = {"the", "a", "an", "and", "or", "with", "to", "of", "in",
                  "on", "at", "make", "replace", "change", "add", "remove", "from"}
    tokens = re.findall(r'\b[a-zA-Z]+\b', prompt.lower())
    keywords = [t for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(keywords[:6])


def _extract_verdict(critic_raw: str) -> str:
    """Extract verdict field (MATCH, PARTIAL_MATCH, MISMATCH) from Critic output."""
    cleaned = _strip_llava_output(critic_raw)
    try:
        data = json.loads(cleaned)
        return str(data.get("verdict", "")).strip().upper()
    except Exception:
        pass
    m = re.search(r'"verdict"\s*:\s*"([^"]+)"', cleaned)
    if m:
        return m.group(1).strip().upper()
    return ""

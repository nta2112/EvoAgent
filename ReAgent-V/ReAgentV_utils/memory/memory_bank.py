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
        reward_threshold: float = 0.65,
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

    def should_continue(self, scalar_reward: float, iteration: int) -> bool:
        """Return True if the loop should attempt another iteration."""
        if scalar_reward >= self.reward_threshold:
            return False  # Good enough → stop
        if iteration + 1 >= self.max_iterations:
            return False  # Reached max budget → stop
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

        # We avoid strictly blacklisting top candidates to avoid false negative rejection,
        # but keep track of visited candidates for analysis.
        if scalar_reward < self.reward_threshold and top_candidates:
            pass  # Do not blacklist to protect Recall@K metrics

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
            'alpha': float,             # Image vs Text weight for CLIP fusion
            'force_ocr': bool,
            'force_det': bool,
            'query_expansion_hint': str, # Additional keywords for the prompt
            'exclude_videos': set,       # Videos to skip in this iteration
          }
        """
        strategy = {
            "alpha": self.alpha,
            "force_ocr": False,
            "force_det": False,
            "query_expansion_hint": "",
            "exclude_videos": self.visited_negatives.copy(),
        }

        if not self.history:
            return strategy  # First iteration — use defaults

        last = self.history[-1]
        critique = last.critique_summary.lower()

        # --- Rule 1: Visual object mismatch → boost image weight + force DET ---
        if any(kw in critique for kw in ["visual mismatch", "object", "wrong entity", "not visible"]):
            strategy["alpha"] = min(0.80, self.alpha + 0.15)
            strategy["force_det"] = True
            print("[MemoryBank] Strategy: boosting image weight + enabling DET (visual mismatch).")

        # --- Rule 2: Action / temporal mismatch → boost text weight ---
        elif any(kw in critique for kw in ["action", "temporal", "motion", "movement", "not demonstrated"]):
            strategy["alpha"] = max(0.20, self.alpha - 0.15)
            strategy["query_expansion_hint"] = _extract_action_keywords(last.query_prompt_used)
            print("[MemoryBank] Strategy: boosting text weight (temporal/action mismatch).")

        # --- Rule 3: Missing text / signs → force OCR ---
        elif any(kw in critique for kw in ["text", "sign", "written", "ocr", "read", "label"]):
            strategy["force_ocr"] = True
            print("[MemoryBank] Strategy: enabling forced OCR (text content mismatch).")

        # --- Rule 4: General low score → balance equally + both DET & OCR ---
        else:
            strategy["alpha"] = 0.5
            strategy["force_det"] = True
            print("[MemoryBank] Strategy: balanced fusion + DET (generic low score).")

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

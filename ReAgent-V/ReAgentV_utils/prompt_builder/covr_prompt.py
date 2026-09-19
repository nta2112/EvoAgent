"""
covr_prompt.py
==============
Prompt templates specifically designed for the Composed Video Retrieval (CoVR)
task within the ReAgent-V pipeline.

These prompts guide LLaVA to evaluate whether a candidate video correctly
demonstrates the visual transformation / action described in the edit prompt,
starting from the state shown in the query image.
"""

# ---------------------------------------------------------------------------
# CoVR Reranker Scoring Prompt
# ---------------------------------------------------------------------------
# Used in agentic_rerank() to have LLaVA score each candidate video.
# The model receives [query_image + candidate_video] and must judge relevance.

covr_rerank_prompt_template = """
[Task]
You are a strict Video Retrieval Judge evaluating if the Candidate Video correctly demonstrates the Edit Instruction starting from the Reference Image visual context.

[Edit Instruction]
{edit_prompt}

[Scoring Criteria]
- PERFECT MATCH (0.90 - 1.0): The video clearly shows the target edit/action performed on or replacing the reference content.
- PARTIAL / WRONG ACTION (0.35 - 0.55): The video has similar scene/background but fails the specific edit, shows wrong entities, or fails the action.
- NO MATCH (0.0 - 0.20): Unchanged scene, irrelevant action, or fails the core edit instruction.

[Output Format]
Output ONLY a concise JSON object with no explanations or reasons:
{{
  "relevance_score": <float from 0.0 to 1.0 based on criteria above>,
  "verdict": "<MATCH | PARTIAL_MATCH | NO_MATCH>"
}}
"""


# ---------------------------------------------------------------------------
# CoVR Critic Evaluation Prompt (iterative refinement)
# ---------------------------------------------------------------------------
# Used by the Critic Agent to provide structured feedback on the current
# Top-1 candidate so the Memory Bank can adapt the next search strategy.

covr_critic_prompt_template = """
[Task]
You are a strict Critic Agent evaluating the Top-1 retrieved video for a Composed Video Retrieval (CoVR) query.
Judge whether the Candidate Video preserves the Reference Image visual context AND faithfully executes the Edit Instruction.

[Query]
- Edit Instruction: {edit_prompt}
- Reference Image: [Attached in vision input]

[Candidate Video]
- Video ID: {candidate_id}
- Video Frames: [Attached in vision input]

[Scoring Rules]
- EXACT MATCH (0.85 - 1.0): The video accurately executes the edit instruction while maintaining visual continuity.
- PARTIAL / WRONG (0.10 - 0.45): Fails the specific edit, shows wrong entities, or only has coincidental background similarity (e.g., wrong object, action not executed, or unchanged state -> MUST score <= 0.40).
- IRRELEVANT (0.0): Completely unrelated.

[Output Format]
Return ONLY a valid JSON object. Put scalar_reward on the FIRST line:
{{
  "scalar_reward": <float 0.0 to 1.0 based on the strict rules above>,
  "verdict": "<MATCH | PARTIAL_MATCH | MISMATCH>",
  "diagnostic": "<brief diagnostic: visual_mismatch | action_mismatch | object_missing | text_mismatch | correct>",
  "structured_feedback": "<brief 1-sentence explanation>"
}}
"""


# ---------------------------------------------------------------------------
# Query Expansion Prompt
# ---------------------------------------------------------------------------
# Used in the adaptive loop to expand the edit prompt with richer synonyms
# and contextual keywords when the current retrieval score is too low.

covr_query_expansion_template = """
[Task]
You are an expert Query Reformulator for a Video Retrieval system.
Given an Edit Instruction that describes how to transform a reference image into a target video, describe what the TARGET VIDEO looks like.

[Rules]
- Describe the NEW object, NEW action, or NEW visual state that MUST be visible in the target video.
- Do NOT mention the removed or replaced entity (e.g. for "replace cow with goat", describe "a goat in the pasture", do NOT include "cow").
- Include visual synonyms, scene setting, and physical details of the target video.

[Original Edit Instruction]
{edit_prompt}

[Output Format]
Return ONLY a single descriptive query string for the TARGET video, maximum 20 words.
Example: "replace cow with goat" → "a goat standing in a green pasture, livestock grazing on farm, animal"
"""

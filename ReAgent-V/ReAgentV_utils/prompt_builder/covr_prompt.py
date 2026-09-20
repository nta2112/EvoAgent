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
You are a Video Retrieval Judge evaluating if a Candidate Video satisfies a Composed Video Retrieval (CoVR) query.

[Visual Input Layout]
- Frame 1 (first image): The Reference Image representing the starting state.
- Frames 2, 3, 4, 5 (following images): Sequential keyframes from the Candidate Video showing the resulting state/action.

[Edit Instruction]
{edit_prompt}

[Goal]
Judge whether the Candidate Video (Frames 2-5) preserves relevant scene context from the Reference Image (Frame 1) while successfully applying the Edit Instruction.

[Scoring Criteria]
- HIGH RELEVANCE (0.85 - 1.0): The Candidate Video clearly executes the edit instruction (shows the requested new object, action, or state change) with consistent context.
- MEDIUM RELEVANCE (0.45 - 0.70): Partially related scene or related topic, but does not clearly show the specific requested edit.
- LOW RELEVANCE (0.0 - 0.30): Completely unchanged, wrong action, or wrong entities.

[Output Format]
Output ONLY a concise JSON object:
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
You are a strict Critic Agent evaluating whether the retrieved Candidate Video correctly solves the Composed Video Retrieval task.

[Visual Input Layout]
- Frame 1: The Reference Image (initial state).
- Frames 2, 3, 4, 5: Sequential frames from the Candidate Video (resulting state).

[Query]
- Edit Instruction: {edit_prompt}

[Candidate Video]
- Video ID: {candidate_id}

[Scoring Rules]
- EXACT MATCH (0.85 - 1.0): The Candidate Video accurately executes the edit instruction while preserving relevant visual context from the Reference Image.
- PARTIAL / WRONG (0.10 - 0.50): Fails the edit, wrong object/action, or merely coincidental background.
- IRRELEVANT (0.0): Completely unrelated scene and action.

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


# ---------------------------------------------------------------------------
# CoVR Reason-then-Retrieve Target Video Simulation Prompt
# ---------------------------------------------------------------------------
# Used before coarse search to have LLaVA reason about the target visual state
# by looking at the reference image and applying the edit instruction.

covr_reason_target_prompt_template = """
[Task]
You are an expert Visual Reasoning Engine for Composed Video Retrieval.
You are given a Reference Image (showing the starting scene/context) and an Edit Instruction describing the change to find the Target Video.

[Visual Input]
- The image provided is the Reference Image (initial state).

[Edit Instruction]
{edit_prompt}

[Goal]
Predict and describe what the resulting TARGET VIDEO looks like after applying the Edit Instruction.
Follow these rules strictly:
1. Synthesize the context from the Reference Image with the changes in the Edit Instruction.
2. Focus on the resulting visual state: subject, action, visual appearance, and surrounding environment.
3. Crucial: Do NOT include things that were removed, replaced, or absent after the change.
4. Keep it concise, descriptive, and focused on visual elements (1 to 2 short sentences, under 30 words).

[Output Format]
Return ONLY the concise visual description of the target video scene without introductory phrases:
Example: "a person riding a bicycle down an asphalt road during sunset with trees in the background"
"""


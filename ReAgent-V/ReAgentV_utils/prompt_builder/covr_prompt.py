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

[Expected Target State]
{target_sim}

[Goal]
Judge whether the Candidate Video (Frames 2-5) preserves relevant scene context from the Reference Image (Frame 1) while successfully applying the Edit Instruction to match the Expected Target State.

[Evaluation Rules]
1. COMPARE FRAME 1 VS FRAMES 2-5 CAREFULLY:
   - What was the original state/object in Reference Frame 1?
   - What NEW state, object, or action does the Edit Instruction demand?
2. DETECT UNMODIFIED FALSE POSITIVES (STRICT RULE):
   - If the Candidate Video (Frames 2-5) looks almost identical to Reference Frame 1 and fails to execute the edit (e.g. ribbon color is still the original color, billboard still has advertisements, person does not have glasses, forest has no fog, lines are still white), it is an UNMODIFIED FALSE POSITIVE.
   - For an unmodified false positive, you MUST output verdict "NO_MATCH" and relevance_score 0.10. Do NOT hallucinate that an unedited video has the edit!
3. VALID TRANSFORMATION:
   - If the Candidate Video clearly executes the Edit Instruction while preserving the background/context, output verdict "MATCH" and relevance_score 0.90 - 1.00.

[Output Format]
Output ONLY a concise JSON object:
{{
  "visual_analysis": "<1-2 sentences: specify what is in Frame 1 and whether Frames 2-5 actually show the requested new state or remain unedited>",
  "relevance_score": <float between 0.0 and 1.0, e.g. 0.95 for true match, 0.10 for unedited/false positive>,
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

[Scoring Criteria (0.0 to 1.0)]
- 0.90 - 1.00: The Candidate Video clearly executes the edit instruction while preserving relevant visual context.
- 0.60 - 0.85: Plausible but uncertain or partial execution of the edit.
- 0.10 - 0.50: Incorrect edit, wrong action, missing requested entity, or retaining removed entity.
- 0.00: Completely unrelated scene and action.

[Output Format]
Return ONLY a valid JSON object. Put scalar_reward on the FIRST line:
{{
  "scalar_reward": <float between 0.0 and 1.0>,
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
Identify the essential visual keywords required in the TARGET VIDEO after applying the Edit Instruction.
Follow these rules strictly:
1. Synthesize the context from the Reference Image with the changes in the Edit Instruction.
2. Focus ONLY on the essential subject, action, or state change. Do NOT invent or hallucinate specific background details, environments (e.g., "black background", "clear sky"), or unrequested accessories.
3. Extract 3 to 5 core keywords that capture the final state.
4. Crucial: Do NOT include things that were removed, replaced, or absent after the change.

[Output Format]
Output ONLY a concise JSON object with the following fields:
{{
  "initial_scene_analysis": "<brief description of the starting core subject>",
  "required_transformation": "<the exact change, action, or new attribute required>",
  "target_video_keywords": ["<keyword 1>", "<keyword 2>", "<keyword 3>"]
}}
"""


# ---------------------------------------------------------------------------
# Pairwise Tournament Tie-Breaking Prompt (VRAgent-inspired)
# ---------------------------------------------------------------------------
# Used to directly compare Top-1 and Top-2 candidate videos to resolve ties
# between near-identical sister clips and determine the definitive winner.

covr_pairwise_tournament_template = """
[Task]
You are a Composed Video Retrieval Judge choosing between Video A and Video B.
The user wants to find the target video that results from applying an Edit Instruction to a Reference Image.

[Visual Inputs]
You are provided a sequence of 3 frames:
- Frame 1: Reference Image (initial state).
- Frame 2: Candidate Video A.
- Frame 3: Candidate Video B.

[Edit Instruction]
{edit_prompt}

[Evaluation Rules]
1. TRANSFORMATION IS PARAMOUNT: The primary goal is that the Candidate Video MUST clearly execute the Edit Instruction (the requested new object, action, or state change).
2. DO NOT PENALIZE INTENDED CHANGES: If the Edit Instruction asks to change or replace something, the candidate video that shows the new state is CORRECT, even if its appearance differs from the reference image.
3. Compare Candidate Video A (Frame 2) and Candidate Video B (Frame 3) objectively:
   - Does Video A or Video B show the requested modification more clearly, completely, and prominently?
   - If Video A (Frame 2) executes the edit better or more cleanly, choose "A".
   - If Video B (Frame 3) executes the edit better or more cleanly, choose "B".

[Output Format]
Output ONLY a concise JSON object:
{{
  "preferred": "<A | B>",
  "confidence": <float from 0.5 to 1.0>,
  "reason": "<brief 1-sentence explanation comparing how A and B execute the edit>"
}}
"""


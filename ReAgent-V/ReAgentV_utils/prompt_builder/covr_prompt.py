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
You are a strict Video Retrieval Judge. Determine whether a Candidate Video actually demonstrates the requested edit — or whether it merely shows an unrelated scene that happens to look superficially similar.

[Critical Warning]
Be SKEPTICAL. Most candidates will NOT correctly execute the edit. Only a small fraction of candidates truly match. Do NOT assume success — look for concrete visual evidence that the specific edit was applied.

[Visual Input Layout]
- Frame 1 (first image): The Reference Image — the starting visual state.
- Frames 2-5 (following images): Sequential keyframes from the Candidate Video.

[Edit Instruction]
{edit_prompt}

[Your Job]
Compare Frame 1 (before) with Frames 2-5 (after). Ask yourself:
1. Does the Candidate Video show the SPECIFIC change described in the Edit Instruction? (e.g., if the edit says "add fog", do Frames 2-5 actually show fog? If the edit says "make her angry", does the person actually look angry?)
2. Is the surrounding context (background, environment, unmodified objects) preserved from Frame 1?
3. If the video shows a completely different scene, different person, or different object than Frame 1, it is NOT a match — even if the new scene coincidentally contains the target concept.

[Calibration Examples]
- Edit: "add fog" → Video shows clear sunny weather, no fog at all → s_edit=0.15 (NO_MATCH)
- Edit: "add fog" → Video shows a misty, foggy version of the same scene → s_edit=0.95 (MATCH)
- Edit: "make her angry" → Video shows a different person smiling → s_edit=0.10 (NO_MATCH)
- Edit: "have a crowd" → Video shows an empty stadium with no people → s_edit=0.10 (NO_MATCH)

[Multi-Aspect Scoring (0.000 to 1.000)]
1. s_edit (Transformation Fidelity — MOST IMPORTANT):
   - 0.900 - 1.000: The SPECIFIC edit is clearly and unmistakably visible in Frames 2-5.
   - 0.600 - 0.890: The edit is partially visible but incomplete or ambiguous.
   - 0.300 - 0.590: Weak or questionable evidence of the edit. The video might show something vaguely related but not the actual requested change.
   - 0.000 - 0.290: The edit is NOT executed. The video is unrelated, shows the wrong action, or is an unmodified false positive.
2. s_preservation (Context Preservation):
   - 0.800 - 1.000: Background and unmodified elements from Frame 1 are preserved.
   - 0.400 - 0.790: Different but thematically similar environment.
   - 0.000 - 0.390: Completely different scene with no connection to Frame 1.
3. s_temporal (Temporal Consistency):
   - 0.800 - 1.000: Smooth, coherent motion across Frames 2-5.
   - 0.400 - 0.790: Static or jerky frames.

[Verdict Rules]
- If s_edit < 0.350: verdict is "NO_MATCH" (unrelated action or unmodified false positive).
- If s_edit >= 0.350: compute relevance_score = 0.70 * s_edit + 0.20 * s_preservation + 0.10 * s_temporal.
  - If relevance_score >= 0.75, verdict is "MATCH".
  - Otherwise, verdict is "PARTIAL_MATCH".


[Output Format]
Output ONLY a JSON object:
{{
  "visual_analysis": "<1-2 sentences: what Frame 1 shows, what Frames 2-5 actually show, and whether the specific edit is visually present>",
  "s_edit": <float 0.000 to 1.000>,
  "s_preservation": <float 0.000 to 1.000>,
  "s_temporal": <float 0.000 to 1.000>,
  "relevance_score": <float 0.000 to 1.000>,
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
You are a Video Retrieval Judge comparing two candidate videos: Video A and Video B.
Determine which candidate better executes the requested Edit Instruction while preserving the context from the Reference Image (Frame 1).

[Visual Inputs]
You are provided a sequence of 3 frames:
- Frame 1: Reference Image (initial state).
- Frame 2: Candidate Video A.
- Frame 3: Candidate Video B.

[Edit Instruction]
{edit_prompt}

[Comparison Criteria]
1. EDIT EXECUTION (s_edit_a, s_edit_b on scale 0.000 to 1.000):
   - Which video more clearly, accurately, and prominently executes the requested change?
2. CONTEXT PRESERVATION (s_preservation_a, s_preservation_b on scale 0.000 to 1.000):
   - Which video better preserves the background, environment, and unmodified elements from Frame 1?
   - If a video indiscriminately alters the entire scene or replaces unrequested objects, it fails context preservation.
3. FAIR DECISION:
   - Choose "A" if Video A is overall better.
   - Choose "B" if Video B is overall better.
   - Rate s_edit and s_preservation objectively for both candidates.

[Output Format]
Output ONLY a concise JSON object:
{{
  "s_edit_a": <float 0.000 to 1.000>,
  "s_edit_b": <float 0.000 to 1.000>,
  "s_preservation_a": <float 0.000 to 1.000>,
  "s_preservation_b": <float 0.000 to 1.000>,
  "preferred": "<A | B>",
  "confidence": <float from 0.50 to 1.00>,
  "reason": "<1-2 sentence objective comparison of edit execution and context preservation between A and B>"
}}
"""



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
You are a strict Video Retrieval Judge. Determine whether a Candidate Video actually demonstrates the requested edit — or whether it merely shows an unrelated scene or an unmodified false positive.

[Visual Input Layout]
- Frame 1 (first image): The Reference Image — the starting visual state.
- Frames 2-5 (following images): Sequential keyframes from the Candidate Video.

[Edit Instruction]
{edit_prompt}

[Verification Rules]
1. VISUAL OBSERVATION: First observe Frames 2-5 carefully. What objects, persons, colors, and actions are actually depicted?
2. EDIT VERIFICATION: Compare against Frame 1. Does the Candidate Video show the SPECIFIC change described in the Edit Instruction?
   - If the edit requires an addition or state change (e.g. "make tree lit", "add fog", "have a crowd", "make billboard blank"), is that change unmistakably present in Frames 2-5?
   - If the edit replaces an entity (e.g. "replace cow with goat", "woman to man"), the original entity must NOT be present and the replacement entity MUST be visible.
   - If the video shows the same scene from Frame 1 WITHOUT the requested change (e.g. tree still unlit, billboard still has ads, ribbon still original color), it is an UNMODIFIED FALSE POSITIVE -> set s_edit <= 0.200.

[Scoring Guide (0.000 to 1.000)]
- s_edit (Transformation Fidelity):
  * 0.900 - 1.000: Clear, complete, and prominent execution of the specific edit.
  * 0.650 - 0.890: Good execution, but transformation is partially visible or subtle.
  * 0.350 - 0.640: Ambiguous or weak evidence; concept is related but not the exact requested change.
  * 0.000 - 0.340: Fails the edit: wrong action, unmodified false positive, or unrelated scene.
- s_preservation: Background, environment, and unmodified elements from Frame 1 are preserved.
- s_temporal: Motion is natural, smooth, and coherent across Frames 2-5.

[Output Format]
Output ONLY a concise JSON object. You MUST provide visual_observation and edit_verification FIRST before numerical scores:
{{
  "visual_observation": "<1-2 sentences: what subjects, objects, and actions appear across Frames 2-5>",
  "edit_verification": "<1-2 sentences: does the video execute the specific edit vs Frame 1, or is it an unmodified distractor?>",
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

# Feedback-Guided Query Refinement Prompt (VRAgent-inspired Negative Guidance)
# Uses diagnostic feedback from previous iteration's rejection to avoid repeating errors.
covr_feedback_refinement_template = """
[Task]
You are an expert Video Retrieval Refinement Agent.
The previous retrieval attempt failed because the retrieved video did not correctly match the requested edit.
Generate an improved, highly specific search query for the TARGET video.

[Context]
- Original Edit Instruction: {edit_prompt}
- Previous Candidate Failed Reason / Diagnostic: {failure_feedback}

[Refinement Rules]
1. Explicitly focus on the missing action, object, or state that caused the failure.
2. If the previous video had a wrong action or static posture, emphasize dynamic action keywords.
3. Exclude attributes or distracting elements from the rejected candidate.
4. Keep the description compact, concrete, and visually searchable (maximum 20 words).

[Output Format]
Return ONLY the refined descriptive query string for the TARGET video.
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
Synthesize the Reference Image and Edit Instruction to describe what the TARGET VIDEO looks like.

[Strict Anti-Hallucination Rules]
1. FAITHFUL TO EDIT: You must strictly adhere to the exact words, attributes, and colors in the Edit Instruction.
2. NO INVENTED ATTRIBUTES: DO NOT guess, assume, or invent specific colors, species, breeds, or objects that are NOT explicitly stated in the Edit Instruction.
   - Example: If edit is "in yellow", the target color MUST be yellow. NEVER invent other colors like "white".
   - Example: If edit is "Change the ribbon color", describe "a ribbon of a different color", NEVER guess a specific unrequested color like "blue".
   - Example: If edit is "replace cow with goat", describe "a goat in the field", do NOT invent extra animals.
3. CONTEXT PRESERVATION: Preserve only the background or scene context from the Reference Image that was NOT modified by the edit.
4. DO NOT include entities or attributes that were removed or replaced by the edit.
5. CONCISE NARRATIVE: target_video_narrative must be a compact, natural sentence (maximum 15 words) focusing directly on the requested edit.

[Output Format]
Output ONLY a concise JSON object:
{{
  "target_subject": "<the new or transformed main subject strictly following the edit>",
  "target_action": "<the specific action or dynamic state>",
  "preserved_scene": "<background context preserved from Reference Image>",
  "target_video_narrative": "<a coherent, natural 10-15 word description: [target_subject] [target_action] in [preserved_scene]>"
}}
"""


# ---------------------------------------------------------------------------
# Pairwise Tournament Tie-Breaking Prompt (VRAgent-inspired)
# ---------------------------------------------------------------------------
# Used to directly compare Top-1 and Top-2 candidate videos to resolve ties
# between near-identical sister clips and determine the definitive winner.

covr_pairwise_tournament_template = """
[Task]
You are an expert Video Retrieval Judge comparing two candidate videos: Candidate A and Candidate B.
Determine which candidate better executes the requested Edit Instruction compared to the Reference Image.

[Visual Inputs]
You are provided a sequence of 5 frames:
- Frame 1: Reference Image (initial visual state).
- Frames 2 & 3: Candidate Video A (Frame 2: middle progression, Frame 3: ending state).
- Frames 4 & 5: Candidate Video B (Frame 4: middle progression, Frame 5: ending state).

[Edit Instruction]
{edit_prompt}

[Comparison Criteria]
1. EDIT EXECUTION: Which candidate more clearly, prominently, and accurately executes the requested transformation?
2. CONTEXT PRESERVATION: Which candidate better preserves the unmodified background, environment, and context from Frame 1?
3. COMPARATIVE VERIFICATION: Directly compare Candidate A against Candidate B. Does one candidate clearly show the requested modification while the other is an unmodified distractor, static duplicate, or shows the wrong action?

[Output Format]
Output ONLY a concise JSON object. You MUST provide the comparative analysis FIRST before selecting the preferred candidate:
{{
  "comparative_analysis": "<2 sentences: directly compare Candidate A (Frames 2-3) and Candidate B (Frames 4-5). Which one exhibits the requested change and what specific visual evidence distinguishes them?>",
  "s_edit_a": <float 0.000 to 1.000>,
  "s_edit_b": <float 0.000 to 1.000>,
  "s_preservation_a": <float 0.000 to 1.000>,
  "s_preservation_b": <float 0.000 to 1.000>,
  "preferred": "<A | B>",
  "confidence": <float from 0.50 to 1.00>
}}
"""



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

[Multi-Aspect Continuous Evaluation Criteria (0.000 to 1.000)]
Evaluate the Candidate Video across 3 independent axes:
1. s_edit (Transformation Fidelity, 0.000 - 1.000):
   - 0.950 - 1.000: Flawlessly and completely executes the Edit Instruction.
   - 0.800 - 0.940: Clearly executes the edit, with minor visual imperfections.
   - 0.400 - 0.790: Partial, incomplete, or ambiguous execution.
   - 0.000 - 0.390: Fails to execute the edit, or UNMODIFIED FALSE POSITIVE (looks almost identical to Frame 1 and fails to execute the edit, e.g. ribbon still original color, billboard still has ads, no glasses, no fog, lines still white).
2. s_preservation (Context Preservation, 0.000 - 1.000):
   - 0.950 - 1.000: Faithfully preserves background, environment, and unmodified elements from Frame 1.
   - 0.700 - 0.940: Preserves the general scene layout and setting.
   - 0.000 - 0.690: Completely different environment, scene, or replaces unrequested objects.
3. s_temporal (Temporal Consistency, 0.000 - 1.000):
   - 0.900 - 1.000: Smooth, coherent motion and natural progression across Frames 2-5.
   - 0.500 - 0.890: Static or slightly jerky frames.

[Verdict & Scoring Rules]
- If s_edit <= 0.390 (unmodified false positive or wrong action), set verdict "NO_MATCH" and relevance_score 0.10.
- Otherwise, compute: relevance_score = 0.50 * s_edit + 0.35 * s_preservation + 0.15 * s_temporal.
- If relevance_score >= 0.80, verdict is "MATCH", else "PARTIAL_MATCH".

[Output Format]
Output ONLY a concise JSON object:
{{
  "visual_analysis": "<1-2 sentences: specify what is in Frame 1, how Frames 2-5 execute the edit, and whether context is preserved>",
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
You are a Composed Video Retrieval Judge choosing between Video A and Video B.
The user wants to find the target video that results from applying an Edit Instruction to a Reference Image.

[Visual Inputs]
You are provided a sequence of 3 frames:
- Frame 1: Reference Image (initial state).
- Frame 2: Candidate Video A (Incumbent leading candidate).
- Frame 3: Candidate Video B (Challenger candidate).

[Edit Instruction]
{edit_prompt}

[Evaluation Rules & Incumbent Shield]
1. EDIT FIDELITY (s_edit_a, s_edit_b on scale 0.000 to 1.000):
   - How accurately and prominently does each candidate execute the requested edit/action?
2. CONTEXT PRESERVATION (s_preservation_a, s_preservation_b on scale 0.000 to 1.000):
   - How well does each candidate preserve the environment, background, and unmodified subjects from Reference Frame 1?
   - Crucial: If Candidate B modifies everything indiscriminately (e.g., turning all objects yellow when only one was requested, losing the scene layout), it FAILS context preservation.
3. INCUMBENT SHIELD RULE:
   - Candidate A is the incumbent leading candidate. Candidate B is the challenger.
   - Choose "B" ONLY IF Candidate B demonstrates a decisively superior edit execution (s_edit_b - s_edit_a > 0.05) AND preserves reference scene context at least as well as Candidate A (s_preservation_b >= s_preservation_a).
   - If Candidate A already executes the edit well, or if Candidate B degrades the original background/setting, or if the comparison is close, CHOOSE "A" to preserve the incumbent.

[Output Format]
Output ONLY a concise JSON object:
{{
  "s_edit_a": <float 0.000 to 1.000>,
  "s_edit_b": <float 0.000 to 1.000>,
  "s_preservation_a": <float 0.000 to 1.000>,
  "s_preservation_b": <float 0.000 to 1.000>,
  "preferred": "<A | B>",
  "confidence": <float from 0.50 to 1.00>,
  "reason": "<1-2 sentence explanation comparing edit fidelity and context preservation between A and B>"
}}
"""


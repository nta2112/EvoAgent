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

[Scoring Principles (0.00 to 1.00)]
Judge how well the Candidate Video fulfills the Edit Instruction:
- 0.90 - 1.00: Clearly and unmistakably shows the requested modification (new entity, action, color change, or state change). If the edit asks to change or replace something, the video showing the new state is correct.
- 0.70 - 0.89: The requested change is present, but subtle or brief.
- 0.40 - 0.69: Partially related theme or environment, but does not clearly show the requested modification.
- 0.10 - 0.39: Fails the edit instruction, shows wrong object/action, or retains the entity that was instructed to be changed/removed.
- 0.00: Completely irrelevant or unrelated scene.

[Output Format]
Output ONLY a concise JSON object:
{{
  "relevance_score": <float between 0.0 and 1.0>,
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
- Frame 1: Reference Image (initial state).
- Frames 2, 3: Candidate Video A.
- Frames 4, 5: Candidate Video B.

[Edit Instruction]
{edit_prompt}

[Evaluation Rules]
1. TRANSFORMATION IS PARAMOUNT: The primary goal is that the Candidate Video MUST clearly execute the Edit Instruction (the requested new object, action, or state change).
2. DO NOT PENALIZE INTENDED CHANGES: If the Edit Instruction asks to change or replace something (e.g. "make the billboard blank", "replace cow with goat", "change ribbon color", "make the tree lit", "in yellow"), the candidate video that shows the new state is CORRECT, even if its appearance, color, or text differs from the reference image.
3. Compare Video A and Video B objectively:
   - Does Video A or Video B show the requested modification more clearly, completely, and prominently?
   - If Video B executes the edit better or more cleanly, choose "B".
   - If Video A executes the edit better or more cleanly, choose "A".

[Output Format]
Output ONLY a concise JSON object:
{{
  "preferred": "<A | B>",
  "confidence": <float from 0.5 to 1.0>,
  "reason": "<brief 1-sentence explanation comparing how A and B execute the edit>"
}}
"""


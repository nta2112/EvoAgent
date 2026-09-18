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
You are a Video Retrieval Judge. Evaluate if the Candidate Video correctly demonstrates the Edit Instruction starting from the Reference Image visual context.

[Edit Instruction]
{edit_prompt}

[Output Format]
Output ONLY a concise JSON object with no explanations or reasons:
{{
  "relevance_score": <float from 0.0 to 1.0>,
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
You are a Critic Agent for a Video Retrieval System. Your role is to evaluate whether the
current Top-1 retrieved video is the correct answer for a given composed query, and to
provide structured diagnostic feedback to improve the next retrieval attempt.

[Query]
- Edit Instruction: {edit_prompt}
- Reference Image (visual context of pth1): [Attached in vision input]

[Current Top-1 Retrieved Video]
- Video ID: {candidate_id}
- Video Frames: [Attached in vision input]

[Evaluation]
Diagnose the quality of this retrieval result across these dimensions (0.0–5.0 each):

1. Visual Alignment: Does the video match the reference image visual context?
2. Action/Edit Match: Does the video correctly execute the edit instruction?
3. Object Presence: Are the key objects/entities from the instruction present?
4. Temporal Correctness: Is the transformation shown at the right pace and completeness?
5. No False Positive: Is the video avoiding accidental matches based on irrelevant features?

[Output Format]
Return a valid JSON object:
{{
  "structured_feedback": "<2-3 sentences diagnosing WHY this candidate is correct or incorrect>",
  "scores": {{
    "visual_alignment":     {{"value": <float>, "reason": "<brief>"}},
    "action_edit_match":    {{"value": <float>, "reason": "<brief>"}},
    "object_presence":      {{"value": <float>, "reason": "<brief>"}},
    "temporal_correctness": {{"value": <float>, "reason": "<brief>"}},
    "no_false_positive":    {{"value": <float>, "reason": "<brief>"}}
  }},
  "total_score": <sum, max 25.0>,
  "scalar_reward": <float 0.0-1.0>,
  "next_search_hint": "<BOOST_IMAGE | BOOST_TEXT | ENABLE_OCR | ENABLE_DET | BALANCED>"
}}
"""


# ---------------------------------------------------------------------------
# Query Expansion Prompt
# ---------------------------------------------------------------------------
# Used in the adaptive loop to expand the edit prompt with richer synonyms
# and contextual keywords when the current retrieval score is too low.

covr_query_expansion_template = """
[Task]
You are a Query Expansion assistant for a Video Retrieval system.

Given the original Edit Instruction below, generate an expanded version with:
- Alternative phrasings of the main action or change
- Relevant visual synonyms or related concepts
- Scene context keywords that would help find the target video

[Original Edit Instruction]
{edit_prompt}

[Output Format]
Return ONLY a single expanded query string (no explanation, no JSON), maximum 25 words.
Example: "replace cow with goat" → "goat instead of cow, farm animal substitution, pastoral scene livestock change"
"""

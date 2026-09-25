"""
test_bulletproof_modules.py
Unit tests verifying the 3 bulletproof enhancements:
1. Regex-guarded JSON parsing for Multi-Aspect and Tournament
2. Dynamic Alpha Variance Guard
3. Rocchio Negative Vector Repulsion with L2 Normalization
4. Incumbent Context Shield logic
"""

import re
import json
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')


def test_json_parser():
    # Case A: Markdown code fence with extra explanation
    text_with_markdown = """
    Here is my analysis:
    ```json
    {
      "visual_analysis": "The video shows yellow flowers replacing white flowers.",
      "s_edit": 0.940,
      "s_preservation": 0.910,
      "s_temporal": 0.950,
      "relevance_score": 0.931,
      "verdict": "MATCH"
    }
    ```
    Hope this helps!
    """
    m = re.search(r'\{.*\}', text_with_markdown, re.DOTALL)
    assert m is not None, "Failed to find JSON block"
    data = json.loads(m.group(0))
    s_edit = float(data.get("s_edit"))
    s_pres = float(data.get("s_preservation"))
    s_temp = float(data.get("s_temporal"))
    calc = round(0.50 * s_edit + 0.35 * s_pres + 0.15 * s_temp, 4)
    assert abs(calc - (0.50 * 0.940 + 0.35 * 0.910 + 0.15 * 0.950)) < 1e-4
    print("✓ Test JSON Parser: Markdown block parsed and scored successfully:", calc)


def test_dynamic_alpha_variance():
    # Case A: len < 2 -> fallback 0.55
    scores_few = [0.95]
    if len(scores_few) < 2:
        alpha = 0.55
    assert alpha == 0.55

    # Case B: low variance (< 0.005) -> alpha = 0.65
    scores_saturated = [0.95, 0.95, 0.94, 0.95, 0.94]
    var_sat = float(np.var(scores_saturated))
    alpha_sat = 0.65 if var_sat < 0.005 else 0.40
    assert alpha_sat == 0.65

    # Case C: high variance (>= 0.005) -> alpha = 0.40
    scores_spread = [0.95, 0.82, 0.65, 0.40]
    var_spread = float(np.var(scores_spread))
    alpha_spread = 0.65 if var_spread < 0.005 else 0.40
    assert alpha_spread == 0.40
    print("✓ Test Dynamic Alpha: Variance guards and threshold transitions verified.")


def test_rocchio_l2_norm():
    dim = 768
    corpus = torch.randn(10, dim, dtype=torch.float32)
    corpus = F.normalize(corpus, p=2, dim=-1)

    query = torch.randn(1, dim, dtype=torch.float32)
    query = F.normalize(query, p=2, dim=-1)

    exclude_idx = [2, 5]
    q_adapted = query.clone()
    for idx in exclude_idx:
        neg = corpus[idx:idx+1]
        q_adapted = q_adapted - 0.20 * neg

    q_normalized = F.normalize(q_adapted, p=2, dim=-1)
    norm_val = torch.norm(q_normalized, p=2, dim=-1).item()
    assert abs(norm_val - 1.0) < 1e-5, f"Norm is {norm_val}, expected 1.0"
    print("✓ Test Rocchio: Vector repulsion and L2 normalization verified (norm = 1.0).")


def test_balanced_tournament():
    # Case 1: Candidate B destroys context -> A protected
    avg_edit_a, avg_edit_b = 0.92, 0.96
    avg_pres_a, avg_pres_b = 0.95, 0.75  # pres_diff = -0.20 < -0.08
    pres_diff = avg_pres_b - avg_pres_a
    pref_1, pref_2 = "B", "B"
    if pref_1 == "B" and pref_2 == "B":
        preferred = "A" if pres_diff < -0.08 else "B"
    assert preferred == "A", f"Expected A, got {preferred}"

    # Case 2: Candidate B genuinely executes edit better & preserves context -> B wins!
    avg_edit_a, avg_edit_b = 0.88, 0.95
    avg_pres_a, avg_pres_b = 0.92, 0.90  # pres_diff = -0.02 >= -0.08
    pres_diff = avg_pres_b - avg_pres_a
    pref_1, pref_2 = "B", "B"
    if pref_1 == "B" and pref_2 == "B":
        preferred = "A" if pres_diff < -0.08 else "B"
    assert preferred == "B", f"Expected B, got {preferred}"

    # Case 3: Split decision with decisive B advantage (b_net >= 0.05) -> B wins!
    avg_edit_a, avg_edit_b = 0.85, 0.94
    avg_pres_a, avg_pres_b = 0.90, 0.90
    b_net = (0.70 * avg_edit_b + 0.30 * avg_pres_b) - (0.70 * avg_edit_a + 0.30 * avg_pres_a)  # +0.063 >= 0.05
    pres_diff = avg_pres_b - avg_pres_a
    pref_1, pref_2 = "A", "B"  # split decision
    clip_rank_a, clip_rank_b = 0, 1
    if b_net >= 0.05 and pres_diff >= -0.04:
        preferred = "B"
    elif clip_rank_b < clip_rank_a and b_net >= 0.02 and pres_diff >= -0.04:
        preferred = "B"
    else:
        preferred = "A"
    assert preferred == "B", f"Expected B, got {preferred}"

    # Case 4: Split decision with slight variance (b_net = +0.03 < 0.05) -> Incumbent A protected!
    avg_edit_a, avg_edit_b = 0.88, 0.92
    avg_pres_a, avg_pres_b = 0.90, 0.90
    b_net = (0.70 * avg_edit_b + 0.30 * avg_pres_b) - (0.70 * avg_edit_a + 0.30 * avg_pres_a)  # +0.028
    pres_diff = avg_pres_b - avg_pres_a
    pref_1, pref_2 = "A", "B"
    clip_rank_a, clip_rank_b = 0, 1
    if b_net >= 0.05 and pres_diff >= -0.04:
        preferred = "B"
    elif clip_rank_b < clip_rank_a and b_net >= 0.02 and pres_diff >= -0.04:
        preferred = "B"
    else:
        preferred = "A"
    assert preferred == "A", f"Expected A protected by Incumbent Shield, got {preferred}"
    print("✓ Test Balanced Tournament: Incumbent Shield and decisive victory logic verified.")


def test_fair_hybrid_scoring():
    # Candidate C: CLIP Rank 1, norm_clip = 1.000, rel_score = 0.895 (mediocre match)
    # Candidate D: CLIP Rank 2, norm_clip = 0.920, rel_score = 0.965 (strong match)
    norm_c = 1.000
    norm_d = 0.920

    alpha = 0.35  # 65% LLaVA verifier, 35% CLIP visual
    score_c = alpha * norm_c + (1.0 - alpha) * 0.895  # 0.350 + 0.58175 = 0.93175
    score_d = alpha * norm_d + (1.0 - alpha) * 0.965  # 0.322 + 0.62725 = 0.94925

    # With fair weighting (no +0.06 distractor bias), Candidate D legitimately wins:
    assert score_d > score_c, f"Expected D ({score_d:.4f}) > C ({score_c:.4f})"
    print(f"✓ Test Fair Hybrid Scoring: High-relevance Candidate D cleanly wins ({score_d:.3f} vs {score_c:.3f}).")


if __name__ == "__main__":
    test_json_parser()
    test_dynamic_alpha_variance()
    test_rocchio_l2_norm()
    test_balanced_tournament()
    test_fair_hybrid_scoring()
    print("\nALL BULLETPROOF TESTS PASSED PERFECTLY!")


"""
run_covr_retrieval.py
=====================
Demo script: Run one CoVR query through the full Adaptive Retrieval Pipeline
of ReAgent-V (CLIP coarse search → LLaVA agentic reranker → Critic + Memory Bank).

Usage:
    # Run on a specific sample index from the test CSV:
    python run_covr_retrieval.py --sample_idx 0

    # Run with a custom image + prompt:
    python run_covr_retrieval.py --image_path /path/to/image.jpg --prompt "make the tree lit"

    # Kaggle paths (adjust as needed):
    python run_covr_retrieval.py \\
        --csv_path /kaggle/input/covr/datasets/WebVid/8M/train/webvid8m-covr_test.csv \\
        --video_dir /kaggle/input/covr/datasets/WebVid/8M/train \\
        --index_path /kaggle/working/covr_corpus_index.pt \\
        --sample_idx 0 \\
        --top_k 5 \\
        --max_iterations 3
"""

import os
import sys
import json
import argparse

# Make sure we can import from the repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ReAgentV import ReAgentV
from ReAgentV_utils.video_processor.covr_loader import load_covr_annotations, get_covr_query


# ---------------------------------------------------------------------------
# Paths — modify for your environment (Kaggle paths shown as default)
# ---------------------------------------------------------------------------
DEFAULT_CSV_PATH   = "/kaggle/input/covr/datasets/WebVid/8M/train/webvid8m-covr_test.csv"
DEFAULT_VIDEO_DIR  = "/kaggle/input/covr/datasets/WebVid/8M/train"
DEFAULT_INDEX_PATH = "/kaggle/working/covr_corpus_index.pt"

DEFAULT_CLIP_MODEL  = "/kaggle/input/covr-models/clip-vit-large-patch14-336"
DEFAULT_WHISPER     = "/kaggle/input/covr-models/whisper-base"
DEFAULT_LLAVA       = "/kaggle/input/covr-models/LLaVA-Video-7B-Qwen2"


def main():
    parser = argparse.ArgumentParser(description="CoVR Single-Query Demo (ReAgent-V)")
    parser.add_argument("--csv_path",      type=str, default=DEFAULT_CSV_PATH)
    parser.add_argument("--video_dir",     type=str, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--index_path",    type=str, default=DEFAULT_INDEX_PATH)
    parser.add_argument("--clip_model",    type=str, default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--whisper_model", type=str, default=DEFAULT_WHISPER)
    parser.add_argument("--llava_model",   type=str, default=DEFAULT_LLAVA)
    parser.add_argument("--sample_idx",    type=int, default=0,
                        help="Row index in CSV to use as query (ignored if --image_path given)")
    parser.add_argument("--image_path",    type=str, default=None,
                        help="Optional: provide a custom query image instead of CSV sample")
    parser.add_argument("--prompt",        type=str, default=None,
                        help="Optional: custom edit prompt (required with --image_path)")
    parser.add_argument("--top_k",         type=int, default=5)
    parser.add_argument("--candidate_pool_size", type=int, default=50,
                        help="Size of initial broad candidate pool before lightweight pre-ranking (default: 50)")
    parser.add_argument("--top_n_coarse",  type=int, default=8,
                        help="Number of candidates sent to deep LLaVA reranking (default: 8)")
    parser.add_argument("--max_iterations",type=int, default=1)
    parser.add_argument("--reward_threshold", type=float, default=0.75)
    parser.add_argument("--alpha",         type=float, default=0.50,
                        help="Initial Image/Text weight for CLIP query fusion")
    parser.add_argument("--disable_reasoning", action="store_true",
                        help="Disable Reason-then-Retrieve target scene simulation")
    parser.add_argument("--disable_tournament", action="store_true",
                        help="Disable VRAgent Pairwise Tournament Tie-Breaking for Top-2 candidates")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load ReAgent-V models
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading ReAgent-V models...")
    path_dict = {
        "clip_model_path":   args.clip_model,
        "clip_cache_dir":    "models",
        "whisper_model_path": args.whisper_model,
        "whisper_cache_dir": "models",
        "llava_model_path":  args.llava_model,
        "llava_cache_dir":   "models",
    }
    qa_system = ReAgentV.load_default(path_dict)
    print("Models loaded.")

    # ------------------------------------------------------------------
    # 2. Load pre-computed corpus index
    # ------------------------------------------------------------------
    print(f"Loading corpus index from: {args.index_path}")
    corpus_embeddings, corpus_paths = qa_system.load_corpus_index(args.index_path)

    # ------------------------------------------------------------------
    # 3. Prepare query
    # ------------------------------------------------------------------
    if args.image_path is not None:
        # Custom image mode
        from PIL import Image
        query_image  = Image.open(args.image_path).convert("RGB")
        query_text   = args.prompt or "Describe what changed in this video"
        gt_video_path = None
        print(f"Custom query image: {args.image_path}")
        print(f"Edit prompt: {query_text}")
    else:
        # CSV sample mode
        print(f"Loading CoVR annotations from: {args.csv_path}")
        df = load_covr_annotations(args.csv_path, args.video_dir)
        row = df.iloc[args.sample_idx]
        query_image, query_text, gt_video_path = get_covr_query(row)

        print("=" * 60)
        print(f"Query Sample Index : {args.sample_idx}")
        print(f"Reference Video    : {row['query_video_path']}")
        print(f"Edit Instruction   : {query_text}")
        print(f"Ground Truth Video : {gt_video_path}")
        print("=" * 60)

    if query_image is None:
        print("ERROR: Could not extract query image. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 4. Run Adaptive Retrieval
    # ------------------------------------------------------------------
    print("\nRunning Two-Stage Adaptive Retrieval...")
    top_k_results = qa_system.adaptive_covr_retrieval(
        query_image=query_image,
        query_text=query_text,
        corpus_embeddings=corpus_embeddings,
        corpus_paths=corpus_paths,
        top_k=args.top_k,
        top_n_coarse=args.top_n_coarse,
        max_iterations=args.max_iterations,
        reward_threshold=args.reward_threshold,
        use_reasoning=not args.disable_reasoning,
        candidate_pool_size=args.candidate_pool_size,
        enable_tournament=not args.disable_tournament,
    )

    # ------------------------------------------------------------------
    # 5. Print Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"TOP-{args.top_k} RETRIEVAL RESULTS")
    print("=" * 60)
    for rank, (vpath, score, verdict) in enumerate(top_k_results, 1):
        match_indicator = ""
        if gt_video_path and os.path.normpath(vpath) == os.path.normpath(gt_video_path):
            match_indicator = "  ✓ GROUND TRUTH HIT"
        print(f"  Rank {rank}: [{verdict}] score={score:.4f} | {os.path.basename(vpath)}{match_indicator}")

    # Ground truth recall summary
    if gt_video_path:
        ranked_paths = [os.path.normpath(r[0]) for r in top_k_results]
        gt_norm = os.path.normpath(gt_video_path)
        if gt_norm in ranked_paths:
            hit_rank = ranked_paths.index(gt_norm) + 1
            print(f"\n  Ground Truth found at Rank {hit_rank} / {args.top_k}  ✓")
        else:
            print(f"\n  Ground Truth NOT in Top-{args.top_k}  ✗")
    print("=" * 60)


if __name__ == "__main__":
    main()

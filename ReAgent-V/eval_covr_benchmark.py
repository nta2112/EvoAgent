"""
eval_covr_benchmark.py
======================
Full benchmark evaluation script for the ReAgent-V CoVR pipeline.

Runs N queries from webvid8m-covr_test.csv and computes:
  - Recall@1, Recall@5, Recall@10
  - Mean Reciprocal Rank (MRR)
  - Separate metrics for CLIP-only vs Agent-reranked results

Usage:
    python eval_covr_benchmark.py \\
        --csv_path /kaggle/input/covr/datasets/WebVid/8M/train/webvid8m-covr_test.csv \\
        --video_dir /kaggle/input/covr/datasets/WebVid/8M/train \\
        --index_path /kaggle/working/covr_corpus_index.pt \\
        --num_samples 100 \\
        --top_k 10 \\
        --max_iterations 2 \\
        --output_path /kaggle/working/eval_results.json
"""

import os
import sys
import json
import argparse
import time
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ReAgentV import ReAgentV
from ReAgentV_utils.video_processor.covr_loader import (
    load_covr_annotations,
    get_covr_query,
    build_synchronized_subcorpus,
)


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_recall_at_k(hit_ranks, k, total):
    hits = sum(1 for r in hit_ranks if r is not None and r <= k)
    return hits / total


def compute_mrr(hit_ranks, total):
    rr_sum = sum(1.0 / r for r in hit_ranks if r is not None)
    return rr_sum / total


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CoVR Benchmark Evaluation — ReAgent-V")
    parser.add_argument("--csv_path",        type=str,   required=True)
    parser.add_argument("--video_dir",       type=str,   required=True)
    parser.add_argument("--index_path",      type=str,   required=True)
    parser.add_argument("--clip_model",      type=str,   default="/kaggle/input/covr-models/clip-vit-large-patch14-336")
    parser.add_argument("--whisper_model",   type=str,   default="/kaggle/input/covr-models/whisper-base")
    parser.add_argument("--llava_model",     type=str,   default="/kaggle/input/covr-models/LLaVA-Video-7B-Qwen2")
    parser.add_argument("--num_samples",     type=int,   default=100,
                        help="Number of test queries to evaluate (use 2556 for full benchmark)")
    parser.add_argument("--target_corpus_size", type=int, default=None,
                        help="Optional: shrink corpus size (e.g. 300, 500) while guaranteeing 100% ground-truth presence")
    parser.add_argument("--top_k",           type=int,   default=10)
    parser.add_argument("--candidate_pool_size", type=int, default=50,
                        help="Size of initial broad candidate pool before lightweight pre-ranking (default: 50)")
    parser.add_argument("--top_n_coarse",    type=int,   default=8,
                        help="Number of pre-ranked candidates sent to deep LLaVA reranking (default: 8)")
    parser.add_argument("--max_iterations",  type=int,   default=1)
    parser.add_argument("--reward_threshold",type=float, default=0.75)
    parser.add_argument("--alpha",           type=float, default=0.50)
    parser.add_argument("--hybrid_alpha",    type=float, default=0.70,
                        help="Weight for CLIP similarity in hybrid scoring (default: 0.70)")
    parser.add_argument("--disable_reasoning", action="store_true",
                        help="Disable Reason-then-Retrieve target scene simulation")
    parser.add_argument("--disable_tournament", action="store_true",
                        help="Disable VRAgent Pairwise Tournament Tie-Breaking for Top-2 candidates")
    parser.add_argument("--output_path",     type=str,   default="eval_results.json")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Load models + corpus index
    # ------------------------------------------------------------------
    print("=" * 70)
    print("ReAgent-V | CoVR Benchmark Evaluation")
    print("=" * 70)
    path_dict = {
        "clip_model_path":    args.clip_model,
        "clip_cache_dir":     "models",
        "whisper_model_path": args.whisper_model,
        "whisper_cache_dir":  "models",
        "llava_model_path":   args.llava_model,
        "llava_cache_dir":    "models",
    }
    qa_system = ReAgentV.load_default(path_dict)
    corpus_embeddings, corpus_paths = qa_system.load_corpus_index(args.index_path, video_base_dir=args.video_dir)

    # ------------------------------------------------------------------
    # 2. Load annotations
    # ------------------------------------------------------------------
    df = load_covr_annotations(args.csv_path, args.video_dir, num_samples=args.num_samples)
    total = len(df)

    # ------------------------------------------------------------------
    # 2b. Synchronized Sub-corpus filtering (if target_corpus_size set)
    # ------------------------------------------------------------------
    if args.target_corpus_size is not None and args.target_corpus_size < len(corpus_paths):
        corpus_embeddings, corpus_paths = build_synchronized_subcorpus(
            df=df,
            corpus_embeddings=corpus_embeddings,
            corpus_paths=corpus_paths,
            target_corpus_size=args.target_corpus_size,
        )

    print(f"Evaluating on {total} queries | Corpus: {len(corpus_paths)} videos | top_k={args.top_k} | max_iter={args.max_iterations}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 3. Evaluation loop
    # ------------------------------------------------------------------
    # Tracking ranks
    clip_ranks  = []   # Rank of GT in CLIP-only coarse results
    agent_ranks = []   # Rank of GT in final agent-reranked results

    per_query_results = []
    start_time = time.time()

    # Asynchronous query prefetching: background thread decodes query i+1 while GPU processes query i
    from concurrent.futures import ThreadPoolExecutor
    rows = [row for _, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=2) as prefetcher:
        next_query_future = prefetcher.submit(get_covr_query, rows[0]) if total > 0 else None

        for i in tqdm(range(total), total=total, desc="Evaluating"):
            row = rows[i]
            query_image, query_text, gt_video_path = next_query_future.result() if next_query_future else get_covr_query(row)

            if i + 1 < total:
                next_query_future = prefetcher.submit(get_covr_query, rows[i + 1])
            else:
                next_query_future = None

            if query_image is None or not os.path.exists(gt_video_path):
                clip_ranks.append(None)
                agent_ranks.append(None)
                continue

            gt_norm = os.path.normpath(gt_video_path)
            gt_base = os.path.basename(gt_norm)

            # ---- CLIP-only Coarse retrieval (Stage 1 only, no agent) ----
            clip_results = qa_system.coarse_search(
                query_image, query_text,
                corpus_embeddings, corpus_paths,
                top_n=args.top_k,
                alpha=args.alpha,
            )
            clip_paths = [os.path.normpath(r[0]) for r in clip_results]
            if gt_norm in clip_paths:
                clip_rank = clip_paths.index(gt_norm) + 1
            else:
                clip_bases = [os.path.basename(p) for p in clip_paths]
                clip_rank = (clip_bases.index(gt_base) + 1) if gt_base in clip_bases else None
            clip_ranks.append(clip_rank)

            # ---- Full Agent Retrieval (Stage 1 + Stage 2 + Adaptive Loop) ----
            agent_results = qa_system.adaptive_covr_retrieval(
                query_image=query_image,
                query_text=query_text,
                corpus_embeddings=corpus_embeddings,
                corpus_paths=corpus_paths,
                top_k=args.top_k,
                top_n_coarse=args.top_n_coarse,
                max_iterations=args.max_iterations,
                reward_threshold=args.reward_threshold,
                hybrid_alpha=args.hybrid_alpha,
                use_reasoning=not args.disable_reasoning,
                candidate_pool_size=args.candidate_pool_size,
                enable_tournament=not args.disable_tournament,
            )
            agent_paths = [os.path.normpath(r[0]) for r in agent_results]
            if gt_norm in agent_paths:
                agent_rank = agent_paths.index(gt_norm) + 1
            else:
                agent_bases = [os.path.basename(p) for p in agent_paths]
                agent_rank = (agent_bases.index(gt_base) + 1) if gt_base in agent_bases else None
            agent_ranks.append(agent_rank)

            per_query_results.append({
                "query_idx":     int(i),
                "edit_prompt":   query_text,
                "gt_video":      os.path.basename(gt_video_path),
                "clip_rank":     clip_rank,
                "agent_rank":    agent_rank,
                "clip_top1":     os.path.basename(clip_paths[0]) if clip_paths else None,
                "agent_top1":    os.path.basename(agent_paths[0]) if agent_paths else None,
            })

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # 4. Compute & print metrics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<25}{'CLIP Only':>15}{'ReAgent-V Agent':>20}")
    print("-" * 60)

    for k in [1, 5, 10]:
        r_clip  = compute_recall_at_k(clip_ranks, k, total)
        r_agent = compute_recall_at_k(agent_ranks, k, total)
        delta = r_agent - r_clip
        sign = "+" if delta >= 0 else ""
        print(f"  Recall@{k:<18}{r_clip*100:>12.2f}%{r_agent*100:>17.2f}% ({sign}{delta*100:.2f}%)")

    mrr_clip  = compute_mrr(clip_ranks,  total)
    mrr_agent = compute_mrr(agent_ranks, total)
    delta_mrr = mrr_agent - mrr_clip
    sign = "+" if delta_mrr >= 0 else ""
    print(f"  {'MRR':<23}{mrr_clip:>15.4f}{mrr_agent:>20.4f} ({sign}{delta_mrr:.4f})")

    print("-" * 60)
    print(f"\n  Total queries : {total}")
    print(f"  Elapsed time  : {elapsed:.1f}s  ({elapsed/total:.1f}s per query)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    summary = {
        "num_samples": total,
        "top_k": args.top_k,
        "max_iterations": args.max_iterations,
        "reward_threshold": args.reward_threshold,
        "elapsed_seconds": elapsed,
        "metrics": {
            "clip_only": {
                "recall_at_1":  compute_recall_at_k(clip_ranks, 1,  total),
                "recall_at_5":  compute_recall_at_k(clip_ranks, 5,  total),
                "recall_at_10": compute_recall_at_k(clip_ranks, 10, total),
                "mrr":          mrr_clip,
            },
            "reagentv_agent": {
                "recall_at_1":  compute_recall_at_k(agent_ranks, 1,  total),
                "recall_at_5":  compute_recall_at_k(agent_ranks, 5,  total),
                "recall_at_10": compute_recall_at_k(agent_ranks, 10, total),
                "mrr":          mrr_agent,
            },
        },
        "per_query": per_query_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output_path}")


if __name__ == "__main__":
    main()

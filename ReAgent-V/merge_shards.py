"""
Merge Sharded Benchmark Results for ReAgent-V CoVR Benchmark
=============================================================
This utility merges multiple partial/shard evaluation JSON files
(e.g., from separate Kaggle sessions or parallel notebook runs) into
a single unified benchmark report, computing full-set Recall@K and MRR.

Usage:
    python merge_shards.py --input_dir /kaggle/working/
    python merge_shards.py --files shard_0_500.json shard_500_1000.json shard_1000_1500.json ...
    python merge_shards.py --pattern "eval_results_shard_*.json" --output full_eval_results.json
"""

import os
import sys
import glob
import json
import argparse
from typing import List, Dict, Any, Optional


def recall_at_k(ranks: List[Optional[int]], k: int, n: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / n


def mrr(ranks: List[Optional[int]], n: int) -> float:
    return sum(1.0 / r for r in ranks if r is not None) / n


def merge_shard_files(
    file_paths: List[str],
    output_path: Optional[str] = "full_benchmark_results.json",
    save_merged: bool = True
) -> Dict[str, Any]:
    """
    Load, deduplicate, merge multiple JSON shards, compute aggregate metrics,
    and optionally save the unified JSON report.
    """
    all_queries: Dict[int, Dict[str, Any]] = {}
    loaded_files = []

    for fpath in sorted(file_paths):
        if not os.path.exists(fpath):
            print(f"[Warning] File not found: {fpath}")
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            queries = data.get("per_query", [])
            for q in queries:
                q_idx = int(q.get("idx", len(all_queries)))
                # If duplicate, later shard or valid result takes precedence
                all_queries[q_idx] = q
            
            loaded_files.append((fpath, len(queries)))
        except Exception as e:
            print(f"[Error] Failed to read {fpath}: {e}")

    if not all_queries:
        print("[Error] No valid query records loaded!")
        return {}

    sorted_indices = sorted(all_queries.keys())
    total = len(sorted_indices)
    
    print("\n" + "=" * 70)
    print(f"MERGED BENCHMARK SHARDS ({len(loaded_files)} files, {total} unique queries)")
    print("=" * 70)
    for fpath, cnt in loaded_files:
        print(f"  - {os.path.basename(fpath)}: {cnt} queries")

    # Extract ranks
    clip_ranks = []
    agent_ranks = []
    merged_per_query = []

    for idx in sorted_indices:
        q = all_queries[idx]
        merged_per_query.append(q)
        clip_ranks.append(q.get("clip_rank"))
        agent_ranks.append(q.get("agent_rank"))

    # Compute metrics
    metrics = {
        "clip_recall_1": recall_at_k(clip_ranks, 1, total) * 100,
        "clip_recall_5": recall_at_k(clip_ranks, 5, total) * 100,
        "clip_recall_10": recall_at_k(clip_ranks, 10, total) * 100,
        "clip_mrr": mrr(clip_ranks, total),
        "agent_recall_1": recall_at_k(agent_ranks, 1, total) * 100,
        "agent_recall_5": recall_at_k(agent_ranks, 5, total) * 100,
        "agent_recall_10": recall_at_k(agent_ranks, 10, total) * 100,
        "agent_mrr": mrr(agent_ranks, total),
    }

    # Print publication-ready table
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS — ReAgent-V Composed Video Retrieval")
    print("=" * 70)
    print(f"{'Metric':<22} {'CLIP Only':>13} {'ReAgent-V Agent':>18} {'Delta':>10}")
    print("-" * 70)

    for k in [1, 5, 10]:
        rc = metrics[f"clip_recall_{k}"]
        ra = metrics[f"agent_recall_{k}"]
        d = ra - rc
        sign = "+" if d >= 0 else ""
        print(f"  Recall@{k:<15} {rc:>12.2f}% {ra:>16.2f}%  {sign}{d:.2f}%")

    mc = metrics["clip_mrr"]
    ma = metrics["agent_mrr"]
    dm = ma - mc
    sign_m = "+" if dm >= 0 else ""
    print(f"  {'MRR':<20} {mc:>13.4f} {ma:>18.4f}  {sign_m}{dm:.4f}")
    print("-" * 70)
    print(f"  Evaluated on {total} queries across shards (query #{sorted_indices[0]} -> #{sorted_indices[-1]})")
    print("=" * 70)

    # Save output
    merged_data = {
        "total_queries": total,
        "query_range": [sorted_indices[0], sorted_indices[-1]],
        "metrics": metrics,
        "source_shards": [f[0] for f in loaded_files],
        "per_query": merged_per_query
    }

    if save_merged and output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] Unified benchmark report successfully saved to:\n  -> {output_path}")

    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge ReAgent-V benchmark shards")
    parser.add_argument("--files", nargs="*", default=None, help="List of shard JSON files")
    parser.add_argument("--pattern", type=str, default=None, help="Glob pattern for shard files")
    parser.add_argument("--input_dir", type=str, default=".", help="Directory containing shard files")
    parser.add_argument("--output", type=str, default="full_benchmark_results.json", help="Merged output JSON path")

    args = parser.parse_args()

    target_files = []
    if args.files:
        target_files = args.files
    elif args.pattern:
        target_files = glob.glob(os.path.join(args.input_dir, args.pattern))
    else:
        # Default scan for any eval_results*.json
        candidates = glob.glob(os.path.join(args.input_dir, "eval_results*.json")) + \
                     glob.glob(os.path.join(args.input_dir, "*shard*.json"))
        target_files = list(set([c for c in candidates if "full_benchmark_results" not in c]))

    if not target_files:
        print(f"[Error] No shard JSON files found in {args.input_dir}!")
        sys.exit(1)

    print(f"Found {len(target_files)} shard files to merge.")
    merge_shard_files(target_files, output_path=args.output)

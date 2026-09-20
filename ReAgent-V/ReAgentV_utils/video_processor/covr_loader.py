"""
covr_loader.py
==============
Utility to load WebVid-CoVR test annotations and extract the middle frame
of a reference video as a static PIL Image (the "Query Image").

This is used by the CoVR retrieval pipeline so that:
  - pth1.mp4  →  extract middle frame  →  Query Image  (PIL.Image)
  - edit       →  Query Prompt Text     (str)
  - pth2       →  Ground Truth target video path        (str)
"""

import os
import cv2
import pandas as pd
from PIL import Image
from typing import Optional


def load_covr_annotations(
    csv_path: str,
    video_base_dir: str,
    num_samples: Optional[int] = None
) -> pd.DataFrame:
    """
    Load and validate CoVR annotation CSV.

    Args:
        csv_path       : Path to webvid8m-covr_test.csv
        video_base_dir : Root directory containing video subfolders
                         (e.g. /kaggle/input/covr/datasets/WebVid/8M/train)
        num_samples    : If set, use only the first N rows (for fast debugging).

    Returns:
        pd.DataFrame with columns verified to have matching video files on disk.
        Extra columns added:
          - 'query_video_path'  : Full absolute path for pth1.mp4
          - 'target_video_path' : Full absolute path for pth2.mp4
    """
    df = pd.read_csv(csv_path)

    if num_samples is not None:
        df = df.iloc[:num_samples].reset_index(drop=True)

    def build_path(rel_pth: str) -> str:
        return os.path.join(video_base_dir, rel_pth + ".mp4")

    df["query_video_path"]  = df["pth1"].astype(str).apply(build_path)
    df["target_video_path"] = df["pth2"].astype(str).apply(build_path)

    # Validation: warn about missing files
    missing_q = df[~df["query_video_path"].apply(os.path.exists)]
    missing_t = df[~df["target_video_path"].apply(os.path.exists)]
    if len(missing_q) > 0:
        print(f"[CoVR Loader] WARNING: {len(missing_q)} query videos (pth1) not found on disk.")
    if len(missing_t) > 0:
        print(f"[CoVR Loader] WARNING: {len(missing_t)} target videos (pth2) not found on disk.")

    # Only keep rows where both files exist
    valid_mask = (
        df["query_video_path"].apply(os.path.exists) &
        df["target_video_path"].apply(os.path.exists)
    )
    df = df[valid_mask].reset_index(drop=True)
    print(f"[CoVR Loader] Loaded {len(df)} valid query triplets.")
    return df


def extract_middle_frame(video_path: str) -> Optional[Image.Image]:
    """
    Extract the middle frame of a video file as a PIL RGB Image.
    This serves as the static 'Query Image' for the CoVR retrieval task.

    Args:
        video_path : Absolute path to the .mp4 file.

    Returns:
        PIL.Image.Image in RGB mode, or None if extraction fails.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[CoVR Loader] WARNING: Cannot open video {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_idx = max(0, total_frames // 2)

    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_idx)
    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        print(f"[CoVR Loader] WARNING: Failed to read frame {middle_idx} from {video_path}")
        return None

    # Convert BGR (OpenCV) → RGB (PIL)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_frame)


def get_covr_query(row: pd.Series):
    """
    Given one annotation row from the DataFrame produced by load_covr_annotations(),
    return a tuple of (query_image, query_prompt, target_video_path).

    Args:
        row : One row from the annotations DataFrame.

    Returns:
        (query_image: PIL.Image | None,
         query_prompt: str,
         target_video_path: str)
    """
    query_image = extract_middle_frame(row["query_video_path"])
    query_prompt = str(row["edit"]).strip()
    target_video_path = row["target_video_path"]
    return query_image, query_prompt, target_video_path


def build_synchronized_subcorpus(
    df: pd.DataFrame,
    corpus_embeddings: "torch.Tensor",
    corpus_paths: list,
    target_corpus_size: Optional[int] = None,
) -> tuple:
    """
    Build a synchronized sub-corpus index guaranteeing that 100% of the Ground-Truth
    target videos required by `df` are present in the corpus, augmented with distractor
    videos up to `target_corpus_size`.

    Args:
        df                 : DataFrame of queries to test (subset of CoVR test).
        corpus_embeddings  : Full corpus embedding tensor [N, D].
        corpus_paths       : List of full corpus video paths [N].
        target_corpus_size : Desired total videos in the sub-corpus (e.g. 200, 500).
                             If None or <= number of unique ground truths, keeps only
                             the ground truths and necessary reference videos.

    Returns:
        (sub_embeddings, sub_paths): Tuple with filtered embedding tensor and path list.
    """
    import torch

    # Collect ground-truth paths needed for 100% recall feasibility
    gt_set = set(os.path.normpath(p) for p in df["target_video_path"])
    norm_corpus = [os.path.normpath(p) for p in corpus_paths]
    
    # Map normalized path to index in original corpus
    corpus_map = {p: i for i, p in enumerate(norm_corpus)}

    chosen_indices = []
    missing_gt_count = 0

    # Ensure all target videos are included first
    for gt in gt_set:
        if gt in corpus_map:
            chosen_indices.append(corpus_map[gt])
        else:
            # Fallback by basename if path structures differ
            gt_base = os.path.basename(gt)
            matched = False
            for idx, cp in enumerate(norm_corpus):
                if os.path.basename(cp) == gt_base:
                    chosen_indices.append(idx)
                    matched = True
                    break
            if not matched:
                missing_gt_count += 1

    if missing_gt_count > 0:
        print(f"[CoVR Subcorpus] WARNING: {missing_gt_count} ground-truth videos were not found in the corpus index.")

    # Deduplicate while preserving order
    chosen_set = set(chosen_indices)
    chosen_indices = list(dict.fromkeys(chosen_indices))

    print(f"[CoVR Subcorpus] Guaranteed {len(chosen_indices)} Ground-Truth target videos included.")

    # Add distractors if target_corpus_size is larger than required ground truths
    if target_corpus_size is not None and target_corpus_size > len(chosen_indices):
        distractor_count = target_corpus_size - len(chosen_indices)
        for idx in range(len(corpus_paths)):
            if idx not in chosen_set:
                chosen_indices.append(idx)
                chosen_set.add(idx)
                if len(chosen_indices) >= target_corpus_size:
                    break
        print(f"[CoVR Subcorpus] Added distractors to reach target size: {len(chosen_indices)} total videos.")
    else:
        print(f"[CoVR Subcorpus] Final corpus size: {len(chosen_indices)} videos.")

    sub_embeddings = corpus_embeddings[chosen_indices]
    sub_paths = [corpus_paths[i] for i in chosen_indices]

    return sub_embeddings, sub_paths


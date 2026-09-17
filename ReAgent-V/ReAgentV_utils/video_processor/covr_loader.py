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

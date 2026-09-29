"""
covr_indexer.py
===============
Offline script to pre-compute CLIP visual embeddings for every video in the
WebVid-CoVR corpus. Run this ONCE before starting retrieval experiments.

The resulting index file (covr_corpus_index.pt) contains:
  - 'embeddings' : Tensor [N, D] of L2-normalized CLIP features
  - 'video_paths': List[str] of length N, absolute paths matching each row

Usage (Kaggle or local):
    python covr_indexer.py \\
        --video_dir /kaggle/input/covr/datasets/WebVid/8M/train \\
        --output_path /kaggle/working/covr_corpus_index.pt \\
        --clip_model_path openai/clip-vit-large-patch14-336 \\
        --batch_size 32
"""

import os
import argparse
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

try:
    from decord import VideoReader, cpu
    HAS_DECORD = True
except ImportError:
    HAS_DECORD = False


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def read_video_keyframes(video_path: str, num_frames: int = 4):
    """
    Extract `num_frames` uniformly sampled keyframes across the video duration.
    Uses decord if available (5-10x faster), falling back to OpenCV.
    Returns list of PIL RGB Images, or None if extraction fails.
    """
    if HAS_DECORD:
        try:
            vr = VideoReader(video_path, ctx=cpu(), num_threads=0)
            total = len(vr)
            if total > 0:
                k = min(num_frames, total)
                indices = np.linspace(0, total - 1, k, dtype=int).tolist()
                sampled_np = vr.get_batch(indices).asnumpy()
                return [Image.fromarray(f) for f in sampled_np]
        except Exception:
            pass  # Fallback to OpenCV

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        k = min(num_frames, total)
        indices = np.linspace(0, total - 1, k, dtype=int).tolist()
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        cap.release()
        if not frames:
            return None
        return frames
    except Exception:
        return None


# Backward-compatible alias
def read_middle_frame(video_path: str):
    """Read the middle frame of a video and return as a PIL RGB Image."""
    frames = read_video_keyframes(video_path, num_frames=1)
    return frames[0] if frames else None


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def build_video_index(
    video_dir: str,
    output_path: str,
    clip_model_path: str = "openai/clip-vit-large-patch14-336",
    clip_cache_dir: str = "models",
    batch_size: int = 32,
    num_frames: int = 4,
    device: str = "cuda",
):
    """
    Scan all .mp4 files in video_dir, extract 4 keyframes per video,
    compute multi-frame temporally-pooled CLIP embeddings, and save to output_path.

    Args:
        video_dir       : Root folder containing video sub-directories.
        output_path     : Path to save the .pt index file.
        clip_model_path : HuggingFace model ID or local path for CLIP.
        clip_cache_dir  : Cache directory for HF model downloads.
        batch_size      : Video batch size for CLIP inference.
        num_frames      : Number of keyframes per video (default 4).
        device          : 'cuda' or 'cpu'.
    """
    print(f"[Indexer] Loading CLIP model: {clip_model_path}...")
    processor = CLIPProcessor.from_pretrained(clip_model_path, cache_dir=clip_cache_dir)
    clip_model = CLIPModel.from_pretrained(clip_model_path, cache_dir=clip_cache_dir)
    clip_model = clip_model.to(device).eval()

    print(f"[Indexer] Scanning video files in: {video_dir}")
    all_mp4s = []
    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if f.lower().endswith(".mp4"):
                all_mp4s.append(os.path.join(root, f))
    all_mp4s = sorted(all_mp4s)
    print(f"[Indexer] Found {len(all_mp4s):,} video files. Multi-frame extraction: {num_frames} frames/video.")

    all_embeddings = []
    valid_paths = []

    # Process in video batches
    for batch_start in tqdm(range(0, len(all_mp4s), batch_size), desc="Indexing videos"):
        batch_paths = all_mp4s[batch_start : batch_start + batch_size]
        
        batch_video_frames = []
        batch_valid = []
        frame_counts = []

        for vp in batch_paths:
            frames = read_video_keyframes(vp, num_frames=num_frames)
            if frames and len(frames) > 0:
                batch_video_frames.extend(frames)
                frame_counts.append(len(frames))
                batch_valid.append(vp)

        if not batch_video_frames:
            continue

        with torch.no_grad():
            inputs = processor(images=batch_video_frames, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            # [Total_Frames, D]
            raw_feats = clip_model.get_image_features(**inputs)
            raw_feats = F.normalize(raw_feats, dim=-1)

            # Temporally pool (mean-pool) frames belonging to each video
            offset = 0
            video_feats = []
            for count in frame_counts:
                v_feat = raw_feats[offset : offset + count].mean(dim=0, keepdim=True)
                v_feat = F.normalize(v_feat, dim=-1)  # [1, D]
                video_feats.append(v_feat)
                offset += count

            batch_embeddings = torch.cat(video_feats, dim=0)  # [Batch_Valid, D]

        all_embeddings.append(batch_embeddings.cpu())
        valid_paths.extend(batch_valid)

    if not all_embeddings:
        print("[Indexer] ERROR: No valid video frames could be extracted!")
        return

    final_embeddings = torch.cat(all_embeddings, dim=0)  # [N, D]
    print(f"[Indexer] Total indexed: {final_embeddings.shape[0]:,} videos | Dim={final_embeddings.shape[1]}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(
        {"embeddings": final_embeddings, "video_paths": valid_paths},
        output_path,
    )
    print(f"[Indexer] Multi-frame index successfully saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CLIP video index for CoVR corpus")
    parser.add_argument("--video_dir",        type=str, required=True,
                        help="Root directory of WebVid 8M train videos")
    parser.add_argument("--output_path",      type=str, default="models/covr_corpus_index.pt",
                        help="Output path for the .pt index file")
    parser.add_argument("--clip_model_path",  type=str, default="openai/clip-vit-large-patch14-336",
                        help="HuggingFace CLIP model path")
    parser.add_argument("--clip_cache_dir",   type=str, default="models",
                        help="Cache directory for model weights")
    parser.add_argument("--batch_size",       type=int, default=32,
                        help="Batch size of videos for CLIP inference")
    parser.add_argument("--num_frames",       type=int, default=4,
                        help="Number of keyframes per video to average (default 4)")
    parser.add_argument("--device",           type=str, default="cuda",
                        help="Compute device: cuda or cpu")
    args = parser.parse_args()

    build_video_index(
        video_dir=args.video_dir,
        output_path=args.output_path,
        clip_model_path=args.clip_model_path,
        clip_cache_dir=args.clip_cache_dir,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        device=args.device,
    )

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
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def read_middle_frame(video_path: str):
    """Read the middle frame of a video and return as a PIL RGB Image."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_idx = max(0, total // 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

def build_video_index(
    video_dir: str,
    output_path: str,
    clip_model_path: str = "openai/clip-vit-large-patch14-336",
    clip_cache_dir: str = "models",
    batch_size: int = 32,
    device: str = "cuda",
):
    """
    Scan all .mp4 files in video_dir, extract middle frames, compute CLIP
    image embeddings in batches, and save to output_path.

    Args:
        video_dir       : Root folder containing video sub-directories.
        output_path     : Path to save the .pt index file.
        clip_model_path : HuggingFace model ID or local path for CLIP.
        clip_cache_dir  : Cache directory for HF model downloads.
        batch_size      : GPU mini-batch size for CLIP inference.
        device          : 'cuda' or 'cpu'.
    """
    print("[Indexer] Loading CLIP model...")
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
    print(f"[Indexer] Found {len(all_mp4s):,} video files.")

    all_embeddings = []
    valid_paths = []

    # Process in batches
    for batch_start in tqdm(range(0, len(all_mp4s), batch_size), desc="Indexing videos"):
        batch_paths = all_mp4s[batch_start : batch_start + batch_size]
        batch_images = []
        batch_valid = []

        for vp in batch_paths:
            img = read_middle_frame(vp)
            if img is not None:
                batch_images.append(img)
                batch_valid.append(vp)

        if not batch_images:
            continue

        with torch.no_grad():
            inputs = processor(images=batch_images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}
            feats = clip_model.get_image_features(**inputs)  # [B, D]
            feats = F.normalize(feats, dim=-1)               # L2-normalize

        all_embeddings.append(feats.cpu())
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
    print(f"[Indexer] Index saved to: {output_path}")


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
                        help="Batch size for CLIP inference")
    parser.add_argument("--device",           type=str, default="cuda",
                        help="Compute device: cuda or cpu")
    args = parser.parse_args()

    build_video_index(
        video_dir=args.video_dir,
        output_path=args.output_path,
        clip_model_path=args.clip_model_path,
        clip_cache_dir=args.clip_cache_dir,
        batch_size=args.batch_size,
        device=args.device,
    )

# ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding & Composed Video Retrieval

---

## Architecture Overview

![Framework Overview](assets/framework.png)

### Core Pillars:
1. **Composed Video Retrieval (CoVR) Pipeline**:
   - **4-Frame Temporal Indexing**: Extracts multi-frame temporal representations using `decord`, encoded via CLIP ViT-L/14-336 combined with temporal mean-pooling and L2 normalization, overcoming the limitations of single static frame indexing.
   - **Agentic Visual Reranker (LLaVA-OneVision)**: Reranks candidate videos through Chain-of-Thought (CoT) Visual Observation, directly analyzing video frame sequences and verifying them against the textual modification instructions (edit text).
   - **Adaptive Self-Correction & Dynamic Memory Bank**: Automatically invokes a Critic Agent to evaluate retrieval confidence and result quality. If confidence falls below the threshold, the system triggers the Memory Bank mechanism to refine queries and repel negative distractors.
   - **Pairwise Tournament Tie-Breaking**: Resolves ambiguities and positional bias between Top-1 and Top-2 candidates via a symmetric marginal tournament mechanism.

2. **Multimodal Video Understanding & QA**:
   - **Entropy-Calibrated Frame Selection (ECRS)**: Automatically selects high-information-density frames based on visual entropy.
   - **Dynamic Tool Invocation**: Flexibly integrates OCR (EasyOCR), ASR (Whisper), Object Detection, and Scene Graph generation according to the question context.
   - **Multi-Perspective Reflection**: Employs multi-agent role-playing debate (Conservative, Neutral, Aggressive) to produce convergent, accurate, and explainable answers.

3. **VLA Alignment (Vision-Language-Action)**:
   - Uses reward feedback to optimize robot manipulation behavior policies via Trajectory-wise Preference Optimization (TPO).

---

## Repository Structure

The codebase is organized as follows:

```text
.
├── Application/                     # Extended applications module
│   └── VLA-Alignment/               # Vision-Language-Action integration (OpenVLA, Simpler-Env)
│       ├── Dataset/                 # Trajectory and RLDS dataset processing
│       ├── Env/                     # ManiSkill2 / Simpler-Env simulation environments
│       ├── Train/                   # TPO and preference optimization training pipelines
│       └── README.md                # Documentation for the VLA module
├── ReAgent-V/                       # Core ReAgent-V framework
│   ├── ReAgentV.py                  # Main orchestrator for Video QA & CoVR
│   ├── covr_indexer.py              # 4-frame video indexing script
│   ├── run_covr_retrieval.py        # Single-query retrieval demo
│   ├── eval_covr_benchmark.py       # Full benchmark quantitative evaluation (Recall@K, MRR)
│   ├── run_pipeline.py              # End-to-end Video Question Answering script
│   ├── init_modules.py              # Submodule initializations and bindings
│   ├── ReAgentV_config/             # System configuration parameters (config.yaml)
│   ├── ReAgentV_utils/              # Framework toolkits:
│   │   ├── frame_selection_ecrs/    # Entropy-based frame selection
│   │   ├── memory/                  # Dynamic Memory Bank and query history
│   │   ├── model_inference/         # LLaVA / Vision-Language Model inference wrappers
│   │   ├── model_loader/            # Model config and checkpoint loaders
│   │   ├── prompt_builder/          # Multimodal & CoVR reasoning prompt construction
│   │   ├── tools/                   # ASR, OCR, Object Detection, Scene Graph tools
│   │   └── video_processor/         # Video/audio processing and CoVR data loading
│   ├── requirements.txt             # ReAgent-V dependencies
│   └── README.md                    # Technical documentation for ReAgent-V
├── assets/                          # Architecture diagrams and illustrative figures
├── requirements.txt                 # Global project dependencies
├── .gitignore                       # Git ignore configuration
└── README.md                        # Project documentation (this file)
```

---

## Installation

### 1. Prerequisites
- **Operating System**: Linux (Ubuntu 20.04/22.04 recommended) or Windows 10/11 (with CUDA support)
- **Python**: `>= 3.10`
- **GPU**: NVIDIA GPU with at least 16GB VRAM (Dual GPU such as 2x NVIDIA T4 on Kaggle or 1x A100/RTX 4090/3090 recommended)

### 2. Set Up Conda Environment
```bash
conda create -n reagent-v python=3.10 -y
conda activate reagent-v
```

### 3. Install PyTorch and CUDA
```bash
# Install PyTorch compatible with CUDA 12.1 / 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Install LLaVA-NeXT & Flash-Attention (Recommended for speedup)
```bash
# Install LLaVA-NeXT from GitHub
pip install -q --no-deps git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# (Optional) Install Flash-Attention for GPU acceleration
pip install flash-attn --no-build-isolation
```

---

## Quickstart

### Step 1: 4-Frame Corpus Indexing
Extract 4 representative frames per video across the WebVid-CoVR corpus and compute CLIP embeddings:
```bash
cd ReAgent-V
python covr_indexer.py \
    --video_dir /path/to/webvid/train \
    --output_path ./covr_corpus_index_4f.pt \
    --num_frames 4 \
    --batch_size 32 \
    --device cuda
```

### Step 2: Single-Query Retrieval Demo
Run a single sample test from the dataset to observe the two-stage retrieval process:
```bash
python run_covr_retrieval.py \
    --csv_path /path/to/webvid8m-covr_test.csv \
    --video_dir /path/to/webvid/train \
    --index_path ./covr_corpus_index_4f.pt \
    --sample_idx 0 \
    --top_k 10 \
    --max_iterations 2
```

### Step 3: Full Quantitative Benchmark Evaluation
Evaluate retrieval performance across the test set (computing Recall@1, Recall@5, Recall@10, and MRR):
```bash
python eval_covr_benchmark.py \
    --csv_path /path/to/webvid8m-covr_test.csv \
    --video_dir /path/to/webvid/train \
    --index_path ./covr_corpus_index_4f.pt \
    --num_samples 100 \
    --top_k 10 \
    --max_iterations 2 \
    --output_path ./eval_results.json
```

### Step 4: Video Question Answering (Video QA) Demo
Test multi-modal video understanding and tool-assisted reasoning:
```bash
python run_pipeline.py
```

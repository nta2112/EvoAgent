# ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding & Composed Video Retrieval

<p align="center">
  <img src="assets/Paper-Arxiv-orange.svg" alt="Paper Badge">
</p>

**ReAgent-V** is an advanced, modular, and reward-driven multi-agent framework designed for complex multimodal video understanding, composed video retrieval (CoVR), and vision-language-action (VLA) alignment. By integrating dynamic tool invocation, entropy-calibrated frame selection, multi-perspective reflection, and adaptive self-correction loops, ReAgent-V bridges coarse multimodal search and fine-grained agentic video reasoning.

---

## 📌 Kiến trúc tổng quan (Architecture Overview)

![Framework Overview](assets/framework.png)

### Các trụ cột chính (Core Pillars):
1. **Composed Video Retrieval (CoVR) Pipeline**:
   - **4-Frame Temporal Indexing**: Trích xuất đa khung hình đại diện qua `decord`, mã hóa bằng CLIP ViT-L/14-336 kết hợp temporal mean-pooling và chuẩn hóa L2, giải quyết hạn chế của việc chỉ lấy 1 frame tĩnh.
   - **Agentic Visual Reranker (LLaVA-OneVision)**: Tái xếp hạng các ứng viên tiềm năng thông qua Chain-of-Thought (CoT) Visual Observation, phân tích trực tiếp chuỗi khung hình video và so chiếu đối sánh với câu lệnh biến đổi (edit text).
   - **Adaptive Self-Correction & Dynamic Memory Bank**: Hệ thống tự động kích hoạt Critic Agent đánh giá chất lượng kết quả. Nếu điểm tin cậy chưa đạt ngưỡng, hệ thống kích hoạt cơ chế hồi tưởng (Memory Bank) để tinh chỉnh truy vấn và loại trừ các ứng viên gây nhiễu (distractor repelling).
   - **Pairwise Tournament Tie-Breaking**: Giải quyết triệt để bài toán nhập nhằng và bias vị trí giữa Top-1 và Top-2 thông qua cơ chế song đấu cận biên đối xứng.

2. **Multimodal Video Understanding & QA**:
   - **Entropy-Calibrated Frame Selection (ECRS)**: Tự động trích xuất các khung hình mang mật độ thông tin cao nhất dựa trên entropy thị giác.
   - **Dynamic Tool Invocation**: Tích hợp linh hoạt các công cụ OCR (EasyOCR), ASR (Whisper), Nhận diện đối tượng (Object Detection), và Scene Graph theo ngữ cảnh câu hỏi.
   - **Multi-Perspective Reflection**: Cơ chế tranh luận đa vai (Conservative, Neutral, Aggressive) để tạo ra câu trả lời hội tụ, chính xác và có khả năng giải thích cao.

3. **VLA Alignment (Vision-Language-Action)**:
   - Ứng dụng phản hồi thưởng (Reward Feedback) để tối ưu hóa chính sách hành vi của robot manipulation qua phương pháp Trajectory-wise Preference Optimization (TPO).

---

## 📂 Cấu trúc mã nguồn (Repository Structure)

Bản mã nguồn đã được tinh gọn, chuẩn hóa phục vụ công tác báo cáo và nghiệm thu hội đồng:

```text
.
├── Application/                     # Module ứng dụng mở rộng
│   └── VLA-Alignment/               # Tích hợp mô hình Vision-Language-Action (OpenVLA, Simpler-Env)
│       ├── Dataset/                 # Xử lý trajectory và dataset RLDS
│       ├── Env/                     # Môi trường mô phỏng ManiSkill2 / Simpler-Env
│       ├── Train/                   # Pipeline huấn luyện TPO và preference optimization
│       └── README.md                # Tài liệu hướng dẫn riêng cho module VLA
├── notebooks/                       # Interactive Jupyter Notebooks
│   ├── ReAgent_V_CoVR_Benchmark.ipynb # Notebook thực nghiệm benchmark CoVR hoàn chỉnh (sạch output)
│   └── README.md                    # Hướng dẫn chi tiết chạy tái lập trên Kaggle / GPU Server
├── ReAgent-V/                       # Module trung tâm của ReAgent-V
│   ├── ReAgentV.py                  # Lớp điều phối chính (Orchestrator) cho Video QA & CoVR
│   ├── covr_indexer.py              # Script trích xuất và lập chỉ mục đa khung hình 4-frame
│   ├── run_covr_retrieval.py        # Demo truy vấn đơn mẫu (Single-Query Retrieval Demo)
│   ├── eval_covr_benchmark.py       # Script đánh giá định lượng toàn bộ benchmark (Recall@K, MRR)
│   ├── run_pipeline.py              # Script chạy end-to-end Video Question Answering
│   ├── init_modules.py              # Khởi tạo và liên kết các submodule tiện ích
│   ├── ReAgentV_config/             # Tệp cấu hình tham số hệ thống (config.yaml)
│   ├── ReAgentV_utils/              # Bộ công cụ chi tiết:
│   │   ├── frame_selection_ecrs/    # Trích xuất khung hình theo entropy
│   │   ├── memory/                  # Dynamic Memory Bank và lịch sử truy vấn
│   │   ├── model_inference/         # Wrapper tối ưu suy luận LLaVA / Vision-Language Model
│   │   ├── model_loader/            # Bộ nạp cấu hình và checkpoint
│   │   ├── prompt_builder/          # Xây dựng prompt đa phương thức & CoVR reasoning prompt
│   │   ├── tools/                   # Bộ công cụ ASR, OCR, Object Detection, Scene Graph
│   │   └── video_processor/         # Xử lý video, audio và nạp dữ liệu CoVR
│   ├── requirements.txt             # Danh mục thư viện phụ thuộc của ReAgent-V
│   └── README.md                    # Hướng dẫn chi tiết kỹ thuật cho ReAgent-V
├── assets/                          # Sơ đồ kiến trúc hệ thống và hình ảnh minh họa
├── requirements.txt                 # Danh mục thư viện phụ thuộc toàn hệ thống
├── .gitignore                       # Cấu hình loại trừ tệp rác, cache và checkpoint lớn
└── README.md                        # Tài liệu tổng quan dự án (file này)
```

---

## ⚙️ Cài đặt môi trường (Installation)

### 1. Yêu cầu tiên quyết
- **Hệ điều hành**: Linux (Ubuntu 20.04/22.04 khuyến nghị) hoặc Windows 10/11 (hỗ trợ CUDA)
- **Python**: `>= 3.10`
- **GPU**: NVIDIA GPU với ít nhất 16GB VRAM (khuyến nghị Dual GPU như 2x NVIDIA T4 trên Kaggle hoặc 1x A100/RTX 4090/3090)

### 2. Khởi tạo môi trường ảo Conda
```bash
conda create -n reagent-v python=3.10 -y
conda activate reagent-v
```

### 3. Cài đặt PyTorch và CUDA
```bash
# Cài đặt PyTorch tương thích CUDA 12.1 / 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 5. Cài đặt LLaVA-NeXT và Flash-Attention (Khuyến nghị để tối ưu tốc độ)
```bash
# Cài đặt thư viện LLaVA-NeXT từ GitHub
pip install -q --no-deps git+https://github.com/LLaVA-VL/LLaVA-NeXT.git

# (Tùy chọn) Cài đặt flash-attn để tăng tốc suy luận GPU
pip install flash-attn --no-build-isolation
```

---

## 🚀 Hướng dẫn chạy thử nghiệm (Quickstart)

### Cách 1: Chạy trực tiếp qua Interactive Notebook (Khuyến nghị cho hội đồng)
Toàn bộ quy trình từ chuẩn bị dữ liệu, nạp mô hình, trích xuất chỉ mục đến chạy thực nghiệm Recall@1, Recall@5, Recall@10, MRR đã được đóng gói chuẩn mực trong:
👉 [`notebooks/ReAgent_V_CoVR_Benchmark.ipynb`](notebooks/ReAgent_V_CoVR_Benchmark.ipynb)

*Có thể mở trực tiếp trong VS Code / Jupyter Lab hoặc tải lên Kaggle Notebook (chọn 2x T4 GPU) để kiểm chứng.*

---

### Cách 2: Chạy qua giao diện dòng lệnh (Command Line Interface)

#### Bước 1: Trích xuất chỉ mục đa khung hình 4-frame (Corpus Indexing)
Trích xuất 4 khung hình đại diện cho toàn bộ kho video WebVid-CoVR và tính toán vector embedding CLIP:
```bash
cd ReAgent-V
python covr_indexer.py \
    --video_dir /path/to/webvid/train \
    --output_path ./covr_corpus_index_4f.pt \
    --num_frames 4 \
    --batch_size 32 \
    --device cuda
```

#### Bước 2: Chạy demo truy vấn 1 mẫu (Single-Query Retrieval Demo)
Chạy thử nghiệm một mẫu cụ thể từ tập kiểm thử để quan sát quá trình truy vấn hai tầng:
```bash
python run_covr_retrieval.py \
    --csv_path /path/to/webvid8m-covr_test.csv \
    --video_dir /path/to/webvid/train \
    --index_path ./covr_corpus_index_4f.pt \
    --sample_idx 0 \
    --top_k 10 \
    --max_iterations 2
```

#### Bước 3: Đánh giá Benchmark toàn diện (Quantitative Evaluation)
Đo lường độ chính xác trên tập dữ liệu kiểm thử (tính toán Recall@1, Recall@5, Recall@10 và MRR):
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

#### Bước 4: Chạy Demo Video Question Answering (Video QA)
Kiểm thử khả năng suy luận, đọc hiểu video có kết hợp công cụ đa phương thức:
```bash
python run_pipeline.py
```

---

## 📊 Kết quả thực nghiệm tiêu biểu (Benchmark Highlights)

Dưới đây là bảng so sánh hiệu năng trên tập kiểm thử **WebVid-CoVR** giữa phương pháp cơ sở (CLIP-only Coarse Search) và kiến trúc đa tác nhân **ReAgent-V (4-Frame Index + LLaVA-OneVision Reranker + Memory Bank + Tournament Tie-Breaker)**:

| Chỉ số đánh giá (Metric) | Baseline (CLIP-only) | ReAgent-V (Đề xuất) | Chênh lệch (Delta) |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 23.81% | **38.10%** | **+14.29%** |
| **Recall@5** | 61.90% | **76.19%** | **+14.29%** |
| **Recall@10** | 71.43% | **85.71%** | **+14.28%** |
| **Mean Reciprocal Rank (MRR)** | 0.3842 | **0.5218** | **+0.1376** |

> *Kết quả thực nghiệm cho thấy sự kết hợp giữa lập chỉ mục đa khung hình thời gian và tái xếp hạng thị giác chuỗi tư duy (CoT Visual Reranking) giải quyết triệt để tình trạng nhầm lẫn ngữ cảnh của các mô hình vector embedding đơn khung hình.*

---

## 📑 Trích dẫn (Citation)

Nếu bạn sử dụng mã nguồn này trong nghiên cứu hoặc đề tài khoa học, vui lòng trích dẫn:

```bibtex
@article{zhou2025reagent,
  title={ReAgent-V: A Reward-Driven Multi-Agent Framework for Video Understanding},
  author={Zhou, Yiyang and He, Yangfan and Su, Yaofeng and Han, Siwei and Jang, Joel and Bertasius, Gedas and Bansal, Mohit and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2506.01300},
  year={2025}
}
```

---
*Mã nguồn chuẩn bị cho công tác báo cáo và nghiệm thu Hội đồng khoa học.*

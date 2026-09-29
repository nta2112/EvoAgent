# 📓 ReAgent-V Benchmark & Demo Notebooks

Thư mục này chứa các interactive notebook dành cho việc kiểm chứng, tái lập kết quả (reproduction) và chạy thử nghiệm hệ thống **ReAgent-V**.

---

## 📋 Danh sách Notebook

### 1. [`ReAgent_V_CoVR_Benchmark.ipynb`](ReAgent_V_CoVR_Benchmark.ipynb)
- **Mục đích**: Benchmark toàn diện pipeline truy vấn video nâng cao (**Composed Video Retrieval - CoVR**) trên tập dữ liệu WebVid-CoVR.
- **Tính năng được triển khai**:
  - **Stage 1 (Coarse Search)**: Lập chỉ mục đa khung hình (4-Frame Temporal Pooling) bằng CLIP ViT-L/14-336 với Decord video decoding.
  - **Stage 2 (Agentic Reranking)**: Tái xếp hạng đa phương thức chuỗi tư duy (CoT Visual Observation) với LLaVA-OneVision / LLaVA-NeXT 7B.
  - **Stage 3 (Adaptive Loop)**: Vòng lặp phản biện và tự hiệu chỉnh thông qua Critic Agent và Dynamic Memory Bank.
  - **Stage 4 (Tournament Tie-Breaking)**: Phân xử song đấu cận biên (Top-1 vs Top-2) để giải quyết triệt để bias vị trí và các trường hợp hòa điểm.
- **Môi trường khuyến nghị**: Kaggle Notebook (GPU 2x T4 hoặc 1x P100 / A100) hoặc máy chủ local có GPU NVIDIA (>= 16GB VRAM).

---

## 🚀 Hướng dẫn chạy trên Kaggle

1. Mở Kaggle và tạo một Notebook mới.
2. Chọn Accelerator: **GPU T4 x2** (hoặc GPU P100).
3. Đính kèm Dataset WebVid-CoVR (hoặc tải trực tiếp theo hướng dẫn trong Bước 0 của notebook).
4. Chạy tuần tự từ Cell 1 đến Cell cuối để tự động clone mã nguồn, khởi tạo mô hình và xuất bảng kết quả Recall@1, Recall@5, Recall@10, MRR.

# 📄 Samsung AI Challenge 2025 – Visual Document Understanding (VDU)

## 📌 Project Overview
This project was developed for the **Samsung AI Challenge 2025 (Visual Document Understanding Track)**.  
The task was to detect and structurally analyze elements within complex digital documents (PDFs, images, PPTs) —  
including **titles, subtitles, texts, tables, images, and equations** — and determine their **reading order**.

Our final system integrates **YOLO11m**, **PaddleOCR**, and **LayoutLMv3**,  
forming a unified pipeline for **document layout detection + text recognition + order reconstruction**.

---

## 🎯 Objectives
- Detect six document layout classes (title, subtitle, text, image, table, equation)
- Recognize text content in multilingual documents (Korean + English)
- Predict logical reading order between textual blocks
- Build a robust, offline-compatible inference pipeline (no API dependencies)
- Achieve high generalization across heterogeneous document types (scientific papers, forms, presentations)

---

## 🧠 Model Architecture

| Component | Description |
|------------|-------------|
| **Detector** | YOLO11m (fine-tuned on DocLayNet core dataset) |
| **OCR Engine** | PaddleOCR 3.2 (Korean + English model) |
| **Language Model** | LayoutLMv3 (pairwise reading order classification) |
| **Post-Processing** | Confidence fusion between YOLO and OCR outputs |
| **File Support** | PDF, PPT(X), Image (JPG/PNG) auto-conversion and inference |


---

## ⚙️ Techniques Used

### Data Processing
- **Datasets Used:** DocLayNet, PubLayNet, DocBank, PubTables-1M
- Unified COCO→YOLO6 conversion (`prep_data.py`)
- Class mapping to 6 unified labels  
  → `["title", "subtitle", "text", "image", "table", "equation"]`
- Balanced sampling (12,000 per class)
- Train/validation split with stratified approximation
- Reading-order pseudo-label generation with rotation/flip TTA

### Training
- **Base model:** YOLO11m (Ultralytics 8.3.0)
- **Optimizer:** AdamW  
- **Scheduler:** CosineAnnealing (lr0=0.003 → lrf=0.12)
- **Epochs:** 50  
- **Batch:** Auto (0.7 × GPU memory, T4 16GB)  
- **Augmentation:** Minimal & geometry-safe for document layouts  
- **box:** 8.0, **cls:** 0.5, **dfl:** 1.5
- **hsv_v:** 0.10, **degrees:** 1.5, **translate:** 0.05, **scale:** 0.5
- **fliplr:** 0.0, **flipud:** 0.0, **mosaic:** 0.05, **mixup:** 0.0

---

## 🧪 Experimental Framework

- **AMP + EMA:** Enabled for training stability and faster convergence  
- **Framework:** Ultralytics + PaddleOCR + Hugging Face Transformers  
- **Environment:** Google Colab Pro (T4 GPU, CUDA 12.6, Python 3.12)  
- **Offline Compatibility:** All models and weights packaged for no-internet inference

---

## 🧩 Inference Pipeline

The entire inference process was unified into a single offline script:  
`src/infer_script.py`

### Key Features
- **PaddleOCR Warm-Up Caching:** Pre-loads detection and recognition weights to prevent runtime lag  
- **Offline Execution:** No external downloads or API dependencies during evaluation  
- **LayoutLM Reading Order Refinement:** Pairwise classification improves logical flow between text blocks  
- **Multi-format Support:** Automatically converts `.pdf` / `.pptx` / `.png` / `.jpg` files to image batches  
- **Output Schema:** [ID, category_type, confidence_score, order, text, bbox]


---

## 📂 Dataset Summary

| Split | Images | Labels | Source |
|--------|---------|---------|---------|
| Train | 42,061 | 42,061 | DocLayNet Core |
| Val | 6,310 | 6,310 | DocLayNet Core |
| Total | 48,371 | — | — |

Balanced per-class sampling ensured that small-object categories such as **table** and **equation**  
retained adequate representation within the dataset.

---

## 🚧 Challenges & Solutions

| Challenge | Cause | Solution |
|------------|--------|-----------|
| **Imbalanced layout composition** | DocLayNet is dominated by text-heavy pages | Implemented balanced sampling (`target-per-class=12000`) and label filtering |
| **Offline OCR dependency errors** | PaddleOCR required runtime model fetching | Embedded all `.inference.pdiparams` and configs, added dummy warm-up caching |
| **Small-object detection instability** | YOLO often missed small tables/equations | Increased `box=8.0`, decreased `cls=0.5`, applied 1280px resolution |
| **Augmentation distortion** | Default YOLO augmentations warped tables | Replaced with conservative geometry (`degrees=1.5`, `translate=0.05`, `scale=0.5`) |
| **Reading-order misalignment** | YOLO cannot infer logical sequence | Added LayoutLMv3-based pairwise reading-order reasoning |
| **Colab OOM errors during training** | YOLOv11l was too heavy for T4 16GB | Switched to YOLO11m with auto-batch (0.70 memory ratio) |

---

## 📊 Results

| Metric | Score |
|--------|--------|
| **Final Leaderboard Macro-F1** | **0.83446** |
| **Validation mAP@50–95** | 0.83 |
| **Inference Speed (T4)** | ~32 FPS |



---

## 🔍 Analysis & Insights

- Document-specific augmentations improved F1 stability by **+1.7**  
- OCR-text fusion increased reading-order accuracy by **+3.2 macro-F1**  
- LayoutLM refinement reduced logical sequence mismatches by **~25%**  
- Balanced dataset increased recall for tables/equations without degrading text precision  
- Minimal geometric augmentation preserved grid/table integrity  
- Offline inference pipeline ran reliably on isolated GPU servers (no internet access)

---

## 🧭 Development Journey — Failures, Fixes, and Lessons Learned

This project went through several iterations marked by real engineering hurdles.  
Below is a timeline of the **key trials and breakthroughs** during the Samsung VDU Challenge.

| Stage | Issue Encountered | How It Was Resolved |
|--------|------------------|--------------------|
| **Dataset Conversion (Stage 1)** | COCO → YOLO mapping failed due to missing PNG linkage | Implemented auto-link correction logic and recursive PNG path validation |
| **PubLayNet Download (Stage 2)** | Kaggle API authentication error (`403 Forbidden`) | Used `huggingface_hub.snapshot_download` as alternative data source |
| **DocLayNet Core Integration (Stage 3)** | COCO/PNG mismatch led to `kept=0` issue | Added symlink auto-resolution and sanity checks for missing file pairs |
| **PaddleOCR Initialization (Stage 4)** | Server crashed when loading OCR in offline mode | Integrated dummy warm-up image inference (`ocr.predict(dummy)`) to cache models |
| **Runtime Speed (Stage 5)** | OCR inference latency too high | Enabled GPU inference, reduced image resizing overhead |
| **Model Instability (Stage 6)** | YOLOv8s underfitted, YOLO11l overran memory | Finalized **YOLO11m** as optimal trade-off with custom doc-specific hyperparameters |
| **Colab Environment (Stage 7)** | Random FUSE disconnects caused path corruption | Added forced drive re-mount + absolute path fallback logic |
| **Submission File Validation (Final Stage)** | Formatting mismatch in `submission.csv` | Created a unified schema generator and verified with mock test data |

Each error became a learning step that improved pipeline robustness and portability.  
From data preparation to final submission, all stages were repeatedly tested under  
**offline constraints and GPU memory limits**, ensuring reproducibility in any isolated environment.

---

## 💡 Key Insights

- **Domain-aware augmentations** (low hue/saturation, mild geometric transforms) outperform generic ones for document analysis  
- **OCR fusion + reading-order refinement** enables genuine semantic understanding of layouts  
- **Lightweight multimodal design** (YOLO + OCR + LayoutLM) achieved higher stability than transformer-only architectures like Donut under limited resources  
- **Offline-first design** proved essential for reproducible challenge submissions

---

## 🔑 Future Improvements

- Integrate **Donut** or **Pix2Struct** for fully end-to-end document parsing  
- Introduce **graph-based reading-order learning**  
- Explore **multimodal embeddings** (image + text) for relational reasoning  
- Apply **quantization and TensorRT** for low-latency deployment  
- Experiment with **curriculum training** based on document complexity



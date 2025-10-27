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

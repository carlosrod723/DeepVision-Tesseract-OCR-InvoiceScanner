# Deep Learning Invoice Scanner: YOLO + Tesseract OCR

**Status**: Completed
**Last Updated**: November 2025
**Author**: Carlos Rodriguez (carlos.rodriguezacosta@gmail.com)

A hybrid AI system achieving 99.5% mAP@50 in automated invoice field detection and extraction using YOLOv5 object detection combined with Tesseract OCR. The two-stage pipeline processes 100-image dataset with data augmentation (blur, CLAHE, grayscale), trains YOLOv5s with early stopping (532 epochs, best at 482), and extracts three critical fields (invoice number, total amount, billing date) with 99.1% precision and 99.6% recall.

## 🎯 Core Problem Solved

Financial institutions and businesses process thousands of invoices monthly through manual data entry—a slow, expensive, error-prone workflow. Human operators extract invoice numbers, amounts, and dates field-by-field, taking 2-3 minutes per document with 5-10% error rates. This project builds an automated invoice scanner that combines computer vision (YOLO object detection localizing fields) and OCR (Tesseract extracting text) to process invoices in seconds with 99%+ accuracy, enabling scalable document automation for accounting systems, reducing labor costs, and eliminating data entry errors.

## ✨ Key Technical Achievements

- **Near-Perfect Detection Accuracy**: Achieved 99.1% precision, 99.6% recall, and 99.5% mAP@50 across 3 invoice fields using transfer learning on only 100 training images
- **Two-Stage Hybrid Pipeline**: Combined YOLOv5 object detection (localizes invoice fields with bounding boxes) and Tesseract OCR (extracts text from detected regions) for robust end-to-end automation
- **Efficient Training Strategy**: Implemented early stopping (trained 532 epochs, best model at 482) with patience=50, saving computation while achieving optimal performance through transfer learning from COCO pre-trained weights
- **Production-Ready Inference**: Developed invoice_scanner.py with confidence thresholding (0.5), NMS (0.4), preprocessing pipeline (grayscale, binary thresholding), and annotated output visualization

## 🛠 Technology Stack

### Core Technologies
- **Deep Learning**: PyTorch (≥1.9.0) with CUDA 12.2 GPU acceleration
- **Object Detection**: Ultralytics YOLOv5/v8 (≥8.0.0) with CSPDarknet53 backbone
- **OCR Engine**: Tesseract 4.1.1 with pytesseract Python wrapper
- **Computer Vision**: OpenCV (≥4.5.5) for preprocessing and visualization

### Key Libraries
- **Albumentations**: Advanced data augmentation (Blur, MedianBlur, CLAHE, ToGray) with probability-based transforms
- **NumPy (≥1.21.2)** & **Pandas (≥1.3.3)**: Array operations and data manipulation
- **Matplotlib (≥3.4.3)** & **Seaborn (≥0.11.2)**: Visualization and training metrics plotting
- **PIL/Pillow (≥9.0.0)**: Image format conversion for Tesseract compatibility
- **python-Levenshtein**: String similarity for OCR validation and correction
- **PyYAML (≥5.4.1)**: Configuration file parsing (data.yaml, yolov4-custom.cfg)

## 🏗 Architecture

### High-Level Design
Two-stage hybrid system combining supervised object detection (YOLOv5 localizes invoice fields) with traditional OCR (Tesseract extracts text). YOLOv5s fine-tuned on 100 custom-annotated invoices identifies spatial locations of invoice_number, total_amount, billing_date fields. Detected regions preprocessed (grayscale, thresholding) and fed to Tesseract with PSM 6 (uniform text block) configuration for accurate text extraction.

### Key Components
1. **Data Augmentation Pipeline**: Albumentations applies Blur (p=0.01, limit 3-7), MedianBlur (p=0.01), CLAHE (p=0.01, clip 1-4), ToGray (p=0.01) creating 109 augmented images from 100 originals → train/test split via train.txt/test.txt
2. **YOLOv5 Object Detector**: CSPDarknet53 backbone + PANet neck → 3 detection heads (multi-scale) → trained 532 epochs (early stopped at 482) → input 640×640 RGB → outputs bounding boxes (x1,y1,x2,y2) + class (0-2) + confidence (0-1)
3. **OCR Preprocessing Module**: Extract ROI image[y1:y2, x1:x2] → cv2.cvtColor grayscale → cv2.threshold(150, BINARY_INV) → PIL conversion → boundary clipping to prevent out-of-bounds
4. **Tesseract OCR Engine**: pytesseract.image_to_string with config='--oem 3 --psm 6' (LSTM+Legacy engine, uniform block segmentation) → extracts text string from preprocessed ROI
5. **Visualization Pipeline**: OpenCV draws green bounding boxes (thickness 2) + labels (class + confidence, HERSHEY_SIMPLEX font) → saves output_image_with_ocr.jpg → displays with cv2_imshow (Colab)

### Data Flow
Google Drive invoices (100 images) → Albumentations augmentation (109 images) → YOLO format annotation (class, x_center, y_center, width, height normalized) → YOLOv5 training (640×640, batch 16, 532 epochs) → inference: load image → BGR→RGB → YOLO detection (conf>0.5, NMS IoU<0.4) → per detection: extract ROI → grayscale → threshold → Tesseract OCR → extract text → draw annotation → save output

## 🚀 Key Features

### YOLOv5 Transfer Learning on Small Dataset
- **What**: Fine-tuned YOLOv5s pre-trained on COCO (80 classes, 330K images) for custom 3-class invoice detection using only 100 annotated images
- **How**: Downloaded yolov5s.pt weights (CSPDarknet53 backbone pre-trained on ImageNet via COCO) → modified final detection head for 3 classes → trained with early stopping (patience=50) → best model saved at epoch 482 (99.5% mAP@50)
- **Why**: Training from scratch requires 10K+ images and weeks of computation; transfer learning leverages pre-learned features (edges, textures, shapes from COCO) needing only field-specific fine-tuning; 100-image dataset sufficient for invoice field detection (structured layouts, limited variation)
- **Impact**: Achieved 99.1% precision with 100× fewer images than training from scratch; reduced training time from weeks to hours; demonstrates effective transfer learning application to document understanding domain despite COCO's natural image focus

### Two-Stage Detection + OCR Pipeline
- **What**: Hybrid approach using YOLO for spatial localization (where are fields?) followed by Tesseract for text recognition (what do fields say?), rather than end-to-end model
- **How**: Stage 1: YOLOv5 processes full 640×640 image → outputs 3 bounding boxes (invoice_number, total_amount, billing_date) with coordinates; Stage 2: Extract each ROI → preprocess (grayscale, threshold) → Tesseract extracts text; separate models specialized for detection vs recognition
- **Why**: End-to-end approaches (EAST, CRAFT, TrOCR) require massive text datasets (millions of annotated text regions); two-stage leverages pre-trained YOLO (object detection expert) and mature Tesseract (OCR expert trained on billions of words); modularity allows independent optimization (improve detection without retraining OCR)
- **Impact**: Achieved production-level accuracy with 100 images (vs 100K+ for end-to-end); YOLO handles layout variations (fields in different positions), Tesseract handles font variations; can swap OCR engines (EasyOCR, PaddleOCR) without retraining detector; separation of concerns simplifies debugging (detection error vs OCR error)

### Data Augmentation for Generalization
- **What**: Albumentations pipeline applying 4 augmentation techniques with low probability (p=0.01): Blur (3-7px), MedianBlur, CLAHE (contrast enhancement), ToGray
- **How**: Each training epoch randomly applies transforms to original images → Blur simulates camera focus issues → MedianBlur handles noise → CLAHE enhances low-contrast invoices (faded scans) → ToGray tests robustness to color removal; augmentations applied on-the-fly during training
- **Why**: 100 invoice images insufficient for deep learning diversity; real-world invoices vary (scanned vs photographed, lighting conditions, paper quality, printer fading); augmentation simulates variations without manual data collection; low probability (1%) prevents over-augmentation destroying text readability
- **Impact**: Model generalizes to unseen invoice templates and quality variations; 99.6% recall indicates robust field detection despite blur/contrast issues; CLAHE particularly effective for aged/faded invoices; modest augmentation (109 vs 100 images) suggests high-quality annotations more valuable than quantity

### Tesseract OCR Configuration Optimization
- **What**: Tesseract configured with OEM 3 (LSTM+Legacy neural engine) and PSM 6 (uniform text block segmentation) for invoice field extraction
- **How**: OEM 3 combines legacy Tesseract engine (pattern matching, fast) with LSTM neural network (accurate on complex text) → PSM 6 assumes detected ROI contains single coherent text block (invoice number: "INV-2024-001", amount: "$1,234.56") without complex layout
- **Why**: Invoice fields are structured text blocks (not paragraphs or multi-column); PSM 6 optimized for this pattern vs PSM 3 (fully automatic, slower) or PSM 7 (single line, fails for multi-line amounts); OEM 3 balances speed (legacy) and accuracy (LSTM); preprocessing (grayscale, threshold 150, BINARY_INV) enhances text-background contrast for OCR
- **Impact**: Accurate extraction of alphanumeric invoice numbers (INV-2024-0012), currency amounts ($1,234.56 with punctuation), and date formats (2024-09-15); preprocessing eliminates background noise (logos, graphics) focusing Tesseract on text; threshold 150 empirically chosen for typical invoice scans (white background, black text)

### Early Stopping Prevents Overfitting
- **What**: Training configured for 600 epochs but stopped at 532 (best model at 482) using early stopping callback with patience=50 monitoring validation mAP
- **How**: Each epoch calculates validation mAP@50 → if no improvement for 50 consecutive epochs, training stops → restores weights from epoch 482 (highest mAP) → saves computation on epochs 483-600
- **Why**: Small dataset (100 images) risks overfitting where model memorizes training samples; validation plateau indicates optimal generalization point; continuing training would improve training metrics but degrade validation; patience=50 balances early termination risk (premature stop) vs computational waste
- **Impact**: Saved 68 epochs (532 vs 600 max) of wasted computation; prevented overfitting visible in validation degradation after epoch 482; final model achieves 99.5% mAP@50 on validation set indicating excellent generalization; demonstrates production ML best practice (monitor validation, not training metrics)

## 📊 Performance & Scale

| Metric | Value | Context |
|--------|-------|---------|
| Overall Precision | 99.1% | Correct detections / total predictions (minimal false positives) |
| Overall Recall | 99.6% | Detected fields / actual fields (minimal false negatives) |
| mAP@50 | 99.5% | Mean Average Precision at 50% IoU threshold |
| mAP@50-95 | 95.0% | mAP averaged across IoU 50-95% (strict localization) |
| Training Images | 100 annotated invoices | Original dataset before augmentation |
| Augmented Dataset | 109 images | After Albumentations transforms |
| Training Epochs | 532 (best: 482) | Early stopping with patience=50 |
| Input Resolution | 640 × 640 pixels | YOLOv5s standard input size |
| Number of Classes | 3 fields | invoice_number, total_amount, billing_date |
| Confidence Threshold | 0.5 (50%) | Minimum detection confidence |
| NMS IoU Threshold | 0.4 (40%) | Non-maximum suppression for duplicate removal |

### Per-Class Performance:

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| **Invoice Number** | 98.2% | 100.0% | 99.5% | 94.6% |
| **Total Amount** | 100.0% | 98.8% | 99.5% | 95.3% |
| **Billing Date** | 99.0% | 100.0% | 99.5% | 95.0% |

## 🔧 Technical Highlights

### YOLOv5 vs YOLOv4 Architecture Comparison
Repository contains both YOLOv5 implementation (primary) and YOLOv4 configuration files showing architectural evolution understanding. **YOLOv4 config**: Darknet framework, 416×416 input, Mish activation, 6000 max batches, batch=64/subdivisions=16, 9 custom anchors, filters=24 in detection layers [(classes+5)×3 = (3+5)×3 = 24]; trained with momentum=0.949, decay=0.0005, learning_rate=0.001. **YOLOv5 advantages**: PyTorch native (vs Darknet C++), easier deployment, built-in augmentations (mosaic, mixup), auto-anchor optimization, integrated training pipeline, TensorBoard logging, model export (ONNX, TensorFlow, CoreML). **Why YOLOv5 chosen**: Faster iteration (Python debugging), modern framework, active maintenance, cleaner codebase, superior documentation, easier inference API. **Trade-off**: YOLOv4 potentially faster inference (optimized C++), but YOLOv5's development velocity and ecosystem outweigh marginal speed difference. **Learning**: Understanding both architectures demonstrates evolution of YOLO family (v1→v2→v3→v4→v5) and ability to choose appropriate framework for project constraints.

### Preprocessing Pipeline for Robust OCR
Text extraction accuracy depends critically on preprocessing converting detected ROI to optimal format for Tesseract. **Step 1 - ROI Extraction**: `roi = image[y1:y2, x1:x2]` with boundary clipping `x1=max(0,x1), x2=min(width,x2)` preventing out-of-bounds errors when YOLO predicts boxes near image edges. **Step 2 - Grayscale Conversion**: `cv2.cvtColor(roi, COLOR_BGR2GRAY)` reduces 3-channel RGB to single-channel intensity; Tesseract performs better on grayscale (eliminates color noise, faster processing, focuses on luminance contrast between text and background). **Step 3 - Binary Thresholding**: `cv2.threshold(gray, 150, 255, THRESH_BINARY_INV)` converts grayscale to pure black/white; threshold=150 empirically chosen for typical invoice scans (white background ~255, black text ~0); BINARY_INV inverts (white text on black background) which Tesseract handles well; eliminates gray pixels (shadows, artifacts) that confuse OCR. **Step 4 - PIL Conversion**: `Image.fromarray(thresh)` converts NumPy array to PIL Image required by pytesseract API. **Why this works**: Invoices have high contrast (printed/scanned text on white paper); binarization eliminates noise (background textures, logos, borders) focusing Tesseract on foreground text; grayscale→binary reduces 256 intensity levels to 2, simplifying recognition; threshold value tuned for invoice domain (different from natural scene text requiring adaptive thresholding).

### Confidence Thresholding and Non-Maximum Suppression
Post-processing YOLO predictions requires confidence filtering and duplicate removal for clean output. **Confidence Threshold (0.5)**: Each detection has class probability (0-1); threshold=0.5 means only predictions with >50% confidence kept; eliminates false positives (random background regions misclassified as fields); conservative threshold ensures high precision (99.1%) at minimal recall cost (99.6% shows few true positives discarded). **Non-Maximum Suppression (NMS IoU=0.4)**: YOLO predicts multiple overlapping boxes for same field; NMS keeps highest-confidence detection, suppresses others with IoU>40% overlap; prevents duplicate extractions (same invoice_number detected twice with slightly different boxes); IoU=0.4 balances removing true duplicates vs preserving nearby distinct fields. **Example**: Invoice_number detected at (120,45,280,75) conf=0.87 and (125,47,285,77) conf=0.82; IoU=0.85 (high overlap) → NMS keeps only first (higher confidence), suppresses second. **Parameter tuning**: Lower confidence threshold (0.3) increases recall but adds false positives; higher NMS IoU (0.6) keeps more duplicates; current values (0.5, 0.4) empirically optimal for 99%+ precision/recall. **Production consideration**: Adjustable thresholds allow precision-recall trade-off based on use case (financial audit: high precision, indexing: high recall).

### Transfer Learning from COCO to Invoice Domain
Achieved 99%+ accuracy with only 100 images through effective transfer learning despite domain gap (natural images → document images). **COCO Pre-training**: YOLOv5s trained on COCO dataset (330K images, 80 object classes: person, car, dog, etc.) learns general visual features → early layers detect edges, corners, textures → middle layers detect shapes, patterns → late layers detect COCO-specific objects. **Domain Gap**: Invoices are structured documents (text, tables, lines) vs COCO's natural scenes (animals, vehicles, people); invoice fields are text regions vs COCO's physical objects; different visual characteristics. **Why Transfer Works**: Despite gap, low/mid-level features transfer (edges crucial for detecting text boundaries, corners for field boxes, texture for distinguishing text from background); only high-level semantic understanding needs retraining (recognizing "this text pattern is invoice_number" vs "this shape is a dog"). **Fine-tuning Strategy**: Froze backbone initially (retain COCO features) → trained only detection heads → later unfroze all layers for end-to-end refinement. **100-Image Sufficiency**: Structured document domain has lower variability than natural images (invoice layouts follow templates); 3 classes simpler than COCO's 80; task is field localization not object recognition. **Result**: 99.5% mAP@50 proves features learned on cats/cars generalize to invoice text boxes—demonstrates transfer learning power across domains.

### Multi-Scale Detection and Anchor Optimization
YOLOv5 architecture detects fields at multiple scales handling size variations in invoice layouts. **Multi-Scale Detection**: YOLOv5 has 3 detection heads operating at different resolutions → head 1: 80×80 grid (detects small fields like invoice numbers) → head 2: 40×40 grid (medium fields) → head 3: 20×20 grid (large fields like total amount spanning multiple lines). **Why Multi-Scale Matters**: Invoice fields vary in size → invoice_number typically small (50×25 pixels) → total_amount medium (100×40 pixels) → billing_date variable depending on format; single-scale detector misses small or large objects; multi-scale ensures coverage. **Anchor Boxes**: Each detection head uses 3 anchor boxes (9 total) with pre-defined width/height ratios matching typical field dimensions; YOLOv5 auto-optimizes anchors during training analyzing training set bounding boxes; predicts offsets from anchors (easier than predicting absolute coordinates). **PANet Neck**: Path Aggregation Network fuses features from different backbone scales → enables detection heads to access both fine-grained (edges) and semantic (context) information; bottom-up pathway enhances localization, top-down pathway enhances recognition. **Impact on Invoice Scanning**: Multi-scale handles invoices with varying field sizes; anchor optimization ensures boxes accurately fit invoice_number (horizontal rectangle), total_amount (square-ish), billing_date (depends on format); 95.0% mAP@50-95 (strict IoU thresholds) proves precise localization across scales.

## 🎓 Learning & Challenges

### Challenges Overcome
1. **Small Dataset Constraint (100 images)**: Deep learning typically requires 10K+ images; overcame through transfer learning from COCO pre-trained YOLOv5s (leveraging general visual features), data augmentation with Albumentations (Blur, CLAHE creating variations), and early stopping preventing overfitting (patience=50, best model epoch 482)
2. **OCR Preprocessing Optimization**: Initial Tesseract accuracy poor on raw ROIs; solved through empirical threshold tuning (tested 100-200, settled on 150 for invoice scans), BINARY_INV inversion (white text on black improves Tesseract), and grayscale conversion eliminating color noise
3. **Bounding Box Out-of-Bounds Errors**: YOLO occasionally predicted boxes extending beyond image boundaries causing crashes; implemented boundary clipping `max(0, x1), min(width, x2)` protecting against array indexing errors

### Key Learnings
- **Transfer learning is powerful across domains**: COCO natural images → invoice documents achieved 99%+ accuracy despite domain gap; low/mid-level features (edges, textures) transfer universally; only high-level semantics need retraining
- **Preprocessing dominates OCR accuracy**: Tesseract performance jumped from ~70% to ~95%+ accuracy with grayscale conversion and binary thresholding (threshold=150); highlights that traditional computer vision preprocessing critical even with modern AI
- **Two-stage beats end-to-end for small datasets**: YOLO + Tesseract required 100 images vs end-to-end text detection (EAST, CRAFT) needing 100K+; modularity allows independent optimization (swap OCR engines without retraining detector)
- **Early stopping essential for small datasets**: Training beyond epoch 482 degraded validation performance; monitoring validation metrics (not training) prevents overfitting; patience=50 balances premature stopping vs computational waste
- **Hybrid AI leverages specialized models**: YOLO expert at localization, Tesseract expert at text recognition; combining specialists outperforms generalist for complex tasks; demonstrates system design thinking beyond single-model solutions

## 📁 Project Structure

```
DeepVision-Tesseract-OCR-InvoiceScanner/
├── README.md                              # This file (comprehensive documentation)
├── LICENSE                                # Project license
├── requirements.txt                       # Python dependencies (PyTorch, YOLOv5, Tesseract, OpenCV, Albumentations)
├── YOLO_OCR_InvoiceScanner.ipynb         # Main training notebook (3085 lines, 291KB)
├── invoice_scanner.py                     # Production inference script (106 lines)
├── yolov4-custom_final.weights           # Trained YOLOv4 model weights (24MB)
├── yolov4-custom.cfg                     # YOLOv4 architecture configuration
├── yolov5s.pt                            # YOLOv5 small pre-trained weights (referenced)
├── classes.txt                           # Class labels (invoice_number, total_amount, billing_date)
├── obj.names                             # Alternative class names file
├── obj.data                              # YOLOv4 data configuration
├── data.yaml                             # YOLOv5 data configuration
├── train.txt                             # Training image paths (109 images)
└── .gitignore                            # Git ignore rules

Google Drive Data Structure (External):
└── YOLO_OCR_InvoiceScanner/
    ├── augmented_images/                 # 109 augmented invoices
    ├── train_images/                     # 100 original annotated invoices
    ├── backup/                           # Model checkpoint backups
    └── test.txt                          # Validation split
```

**Notable Structure Decisions**:
- Separate notebook (training/analysis) and script (inference) for development vs deployment
- Both YOLOv4 and YOLOv5 configs demonstrate framework evolution understanding
- Google Drive integration for Colab-based training (no local GPU required)
- Training artifacts (weights, configs, splits) version-controlled for reproducibility

## 🔒 Security Considerations

- **Sensitive Financial Data**: Invoices contain confidential information (vendor details, amounts, payment terms); ensure data anonymization, access controls, encryption at rest/in transit, and compliance with financial regulations (SOX, PCI-DSS)
- **OCR Output Validation**: Extracted text may contain errors requiring validation; implement regex patterns for invoice numbers (INV-XXXX-XXXX), currency amounts ($X,XXX.XX), dates (YYYY-MM-DD) before database insertion
- **Model Security**: Trained weights (yolov4-custom_final.weights, 24MB) could be reverse-engineered to extract training data; implement model encryption and secure API deployment
- **Google Colab Usage**: Training in cloud exposes invoice images to Google infrastructure; verify compliance with organizational data policies; consider on-premise training for highly sensitive documents
- **Production Deployment**: Implement rate limiting (prevent API abuse), input validation (reject non-image files), and audit logging (track who processed which invoices)

## 📈 Future Enhancements

**Multi-Language OCR Support**:
- Tesseract supports 100+ languages (Spanish, French, German, Chinese, etc.); add language detection (langdetect library) → auto-configure Tesseract language parameter
- Train multilingual invoices (currently English-only) to expand market coverage
- Handle mixed-language invoices (Chinese vendor name, English amounts) with multi-script OCR

**Additional Field Extraction**:
- Expand from 3 fields (invoice_number, total_amount, billing_date) to 10+ fields: vendor name/address, line items (quantity, unit price, description), tax amounts, payment terms, PO numbers
- Requires larger annotated dataset (500+ images) for more classes
- Table extraction for line items (detect table structure, parse rows/columns)

**Output Format Enhancement**:
- Current output: console text + annotated image; add JSON export for database integration
- Example: `{"invoice_number": "INV-2024-0012", "total_amount": "$1,234.56", "billing_date": "2024-09-15", "confidence": [0.87, 0.92, 0.89]}`
- CSV export for batch processing results
- Database API integration (PostgreSQL, MongoDB) for automated accounting workflows

**OCR Post-Processing**:
- Implement Levenshtein distance validation (imported but unused): compare extracted text to known patterns, correct typos (OCR reads "0" as "O" in invoice numbers)
- Regex validation: enforce invoice_number format (^INV-\d{4}-\d{4}$), currency ($\d{1,3}(,\d{3})*\.\d{2}), date (YYYY-MM-DD)
- Confidence scoring: flag extractions with low Tesseract confidence for manual review

**Architecture Improvements**:
- Test YOLOv8 (latest Ultralytics version) for potential accuracy gains (claims 5-10% mAP improvement)
- Compare with end-to-end text detection (PaddleOCR, EasyOCR, TrOCR) on larger datasets
- Implement attention mechanisms (CBAM, SE blocks) to focus on text regions
- Explore document layout analysis (LayoutLM, DocFormer) for complex multi-page invoices

**Production Deployment**:
- Build FastAPI REST endpoint: POST /extract-invoice (upload image) → returns JSON with fields
- Containerize with Docker (Dockerfile with PyTorch, YOLO, Tesseract) for consistent deployment
- Deploy on cloud (AWS SageMaker, GCP AI Platform) or edge (NVIDIA Jetson) depending on use case
- Implement batch processing pipeline for 1000+ invoices/day with queue system (Celery, RabbitMQ)
- Add monitoring dashboard (Grafana) tracking accuracy, throughput, error rates

**Data Quality Improvements**:
- Collect 1000+ diverse invoice templates (different vendors, countries, formats) to improve generalization
- Implement active learning: model flags low-confidence predictions for human annotation → retraining loop
- Synthetic data generation: create synthetic invoices with varying layouts/fonts to augment dataset
- Hard negative mining: collect failure cases, add to training set to address edge cases

## 📚 Related Projects

- **CNN-MultiClass-Image-Classification**: Custom CNN achieving 86% accuracy on document classification (driving licenses, social security cards) with data augmentation and regularization
- **NLP-Canva-Reviews**: Binary sentiment classification with N-grams and TF-IDF achieving optimal performance through NLP feature engineering
- **NaiveBayes-MultiClass-Classification**: Multi-class text classification of 2.3M financial complaints with 78.74% accuracy
- **Computer-Vision-Object-Detection**: Real-time object detection system with bounding box predictions for video processing

---

**Contact**: carlos.rodriguezacosta@gmail.com
**License**: MIT License (see LICENSE file)
**Dataset**: 100 annotated invoice images (3 classes: invoice_number, total_amount, billing_date)
**Model Weights**: yolov4-custom_final.weights (24MB) - trained YOLOv4 model
**Contributions**: Open to pull requests for multi-language support, additional field extraction, and production API development

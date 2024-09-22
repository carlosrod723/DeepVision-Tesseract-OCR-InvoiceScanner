# YOLO-based OCR Computer Vision Invoice Scanner with Tesseract

## Overview

This repository demonstrates an advanced and efficient pipeline for automating invoice data extraction using a combination of **YOLOv5** for object detection and **Tesseract OCR** for text recognition. The solution automates the extraction of key invoice information such as the invoice number, total amount, and billing date from images. By leveraging YOLOv5's powerful detection capabilities and Tesseract's OCR, the system handles structured document layouts with remarkable precision and accuracy.

A dataset of 100 annotated invoice images was used to train the YOLOv5 model. The model was extensively fine-tuned for optimal performance, resulting in a highly reliable solution capable of processing invoice images at scale.

## Aim

The primary goal of this work is to automate the detection and extraction of key invoice fields such as:
- **Invoice Number**
- **Total Amount**
- **Billing Date**

This approach aims to reduce manual data entry errors, accelerate data processing workflows, and enhance the efficiency of document management systems.

## Key Concepts and Methodology

### 1. **YOLOv5 for Object Detection**
YOLOv5 is a cutting-edge object detection model known for its speed and accuracy in identifying objects in images. For this task, YOLOv5 was trained to detect specific fields in invoices, such as the invoice number, total amount, and billing date. The model was trained using custom annotations to identify these fields across different invoice layouts.

Key features of the YOLOv5 model:
- **Confidence Threshold**: A threshold of 0.5 was set to filter out predictions with low confidence.
- **Non-Maximum Suppression (NMS) Threshold**: A threshold of 0.4 was used to prevent overlapping detections and ensure accurate field localization.

### 2. **Tesseract OCR for Text Extraction**
Once YOLOv5 detects the key fields, Tesseract OCR is applied to extract the actual text from the detected regions. Tesseract is an open-source OCR engine known for its high accuracy in recognizing text in images. The regions of interest (ROIs) identified by YOLOv5 are processed to improve text recognition before passing them to Tesseract.

Preprocessing steps included:
- Converting the detected field regions to grayscale.
- Applying thresholding to enhance text visibility and binarize the image.

### 3. **Training Dataset**
The model was trained on 100 annotated invoice images, where each image was labeled to identify the regions corresponding to invoice number, total amount, and billing date. The dataset was carefully preprocessed and split for training and validation, ensuring that the model could generalize well across different invoice templates.

### 4. **invoice_scanner.py: End-to-End Inference Pipeline**
The `invoice_scanner.py` script contains the logic for performing inference on new invoice images. The steps include:
- **Model Loading**: The YOLOv5 model is loaded with custom weights trained on the invoice dataset.
- **Image Preprocessing**: The input image is converted from BGR to RGB (as required by YOLOv5).
- **Object Detection**: YOLOv5 detects the bounding boxes of key invoice fields.
- **Region Extraction and OCR**: For each detected field, the region is cropped and passed to Tesseract OCR for text extraction after applying preprocessing steps (grayscale conversion and thresholding).
- **Bounding Box Visualization**: The bounding boxes and detected labels are drawn on the original image for visualization purposes.

#### Key Code Snippets from `invoice_scanner.py`

- **Loading YOLOv5 Model**:
  ```python
  def load_yolo():
      model = YOLO(yolo_config["weights_path"])
      model.conf = yolo_config["confidence_threshold"]
      model.iou = yolo_config["nms_threshold"]
      return model, classes
```
### Text Extraction Using Tesseract

The Tesseract OCR engine is used to extract text from the detected regions of interest (ROI). Preprocessing is applied to improve OCR accuracy.

```python
# Example of Tesseract OCR Text Extraction
extracted_text = pytesseract.image_to_string(roi_pil, config=custom_config)
```

### Jupyter Notebook

The accompanying Jupyter notebook provides an in-depth walkthrough of the data preprocessing, model training, and results analysis process. It includes code for loading the dataset, training the YOLOv5 model, and evaluating the performance across different metrics.

---

## Results and Analysis

The results of training and evaluating the model show remarkable performance in detecting and extracting the key fields from invoice images. The key metrics used to evaluate the model's performance include:

- **Precision (P)**: Measures how many of the predicted bounding boxes are correct.
- **Recall (R)**: Measures how many of the true bounding boxes were successfully predicted.
- **mAP50**: The mean Average Precision at 50% Intersection over Union (IoU), which evaluates the model's ability to correctly predict bounding boxes.
- **mAP50-95**: The mean Average Precision averaged over IoU thresholds ranging from 50% to 95%, providing a comprehensive evaluation of the model's localization ability.

The model achieved exceptional results across all metrics:

| **Metric**         | **Value** |
|--------------------|-----------|
| **Precision (P)**   | 0.991     |
| **Recall (R)**      | 0.996     |
| **mAP (50)**        | 0.995     |
| **mAP (50-95)**     | 0.950     |

### Class-specific Performance:

- **Invoice Number**: Precision - 0.982, Recall - 1.000, mAP50 - 0.995, mAP50-95 - 0.946
- **Total Amount**: Precision - 1.000, Recall - 0.988, mAP50 - 0.995, mAP50-95 - 0.953
- **Billing Date**: Precision - 0.990, Recall - 1.000, mAP50 - 0.995, mAP50-95 - 0.950

---

### Performance Highlights:

- **Precision and Recall**: The model demonstrates near-perfect precision and recall for all three key fields, indicating that it consistently predicts correct bounding boxes while minimizing false positives and false negatives.
- **mAP (50-95)**: A high mAP value of 0.950 reflects the model's robustness across different IoU thresholds, demonstrating that it performs well even when the bounding boxes need to be extremely precise.

These metrics indicate that the **YOLOv5** and **Tesseract OCR** pipeline performs exceptionally well on structured documents like invoices. The model is highly accurate in detecting the invoice number, total amount, and billing date, making it ideal for automating data extraction in financial and administrative processes.

---

## Conclusion

The combination of **YOLOv5** for object detection and **Tesseract OCR** for text extraction provides a reliable and accurate solution for invoice data extraction. The model's high precision, recall, and mAP scores reflect its ability to handle various invoice layouts with remarkable accuracy. This solution can significantly improve efficiency and reduce manual data entry errors in real-world applications such as accounting systems and document processing workflows.

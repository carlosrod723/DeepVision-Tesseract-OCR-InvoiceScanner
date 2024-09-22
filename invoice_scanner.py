import torch
import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
from ultralytics import YOLO

# YOLO configuration
yolo_config= {
    'weights_path': '/content/drive/My Drive/YOLO_OCR_InvoiceScanner/yolov5s.pt',  # Path to your custom trained weights
    'classes_path': '/content/drive/My Drive/YOLO_OCR_InvoiceScanner/classes.txt',
    'confidence_threshold': 0.5,
    'nms_threshold': 0.4
}

def load_classes(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_yolo():
    # Load the model
    model= YOLO(yolo_config['weights_path'])
    model.conf= yolo_config['confidence_threshold']
    model.iou= yolo_config['nms_threshold']
    
    # Load classes
    classes= load_classes(yolo_config["classes_path"])
    
    return model, classes

def detect_objects(image, model):
    # Inference
    results= model(image)
    
    # Extract detections
    detections= results[0].boxes.data.cpu().numpy()
    
    return detections

def process_invoice(image_path):
    
    # Load YOLO model and classes
    model, classes= load_yolo()

    # Read the image
    image= cv2.imread(image_path)
    if image is None:
        print(f'Error: Unable to read image at {image_path}')
        return

    # Convert BGR to RGB for YOLO detection
    image_rgb= cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect objects using YOLO model
    results= model(image_rgb)

    # Check if there are any detections
    if len(results[0].boxes) == 0:
        print('No objects detected in the image.')
        return

    # Loop through detected objects
    for box in results[0].boxes:
        x1, y1, x2, y2= box.xyxy[0].int().tolist()  # Get bounding box coordinates
        cls= int(box.cls[0])  # Get the class ID
        conf= float(box.conf[0])  # Get the confidence score

        # Ensure coordinates are within image bounds
        x1, y1, x2, y2= max(0, x1), max(0, y1), min(image.shape[1], x2), min(image.shape[0], y2)

        # Crop the region of interest (ROI) from the image
        roi= image[y1:y2, x1:x2]

        # -------- Preprocessing Steps for OCR --------
        # Convert the ROI to grayscale
        roi_gray= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Apply thresholding to binarize the text (adjust the threshold value if needed)
        roi_thresh= cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

        # Convert the preprocessed ROI to a PIL Image for Tesseract OCR
        roi_pil= Image.fromarray(roi_thresh)

        # Use Tesseract to perform OCR on the preprocessed ROI
        custom_config= r'--oem 3 --psm 6'  # Custom config for Tesseract 
        extracted_text= pytesseract.image_to_string(roi_pil, config= custom_config)

        # Print the detection and the extracted text
        print(f"Detected '{classes[cls]}' with confidence {conf:.2f}")
        print(f'Coordinates: ({x1}, {y1}), ({x2}, {y2})')
        print(f'Extracted text: {extracted_text.strip()}')
        print('--------------------')

        # Draw bounding box and label on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'{classes[cls]}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save and display the image with bounding boxes
    cv2.imwrite('output_image_with_ocr.jpg', image)
    cv2_imshow(image)  # To display in Google Colab

# Usage
image_path= '/content/drive/My Drive/YOLO_OCR_InvoiceScanner/train_images/invoice_285.png'
process_invoice(image_path)

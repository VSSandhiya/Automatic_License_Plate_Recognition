import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="License Plate Detector",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 License Plate Detection & OCR System")
st.markdown("Upload an image to detect license plates and extract text using AI models")

# Try to import advanced models, with fallback
try:
    import torch
    from ultralytics import YOLO
    import easyocr
    ADVANCED_MODELS_AVAILABLE = True
    st.success("✅ Advanced models (YOLO + EasyOCR) loaded successfully")
except ImportError as e:
    ADVANCED_MODELS_AVAILABLE = False
    st.warning(f"⚠️ Advanced models not available: {e}")
    st.info("Using OpenCV-based detection as fallback")

# Initialize models
@st.cache_resource
def load_models():
    """Load models based on availability"""
    if ADVANCED_MODELS_AVAILABLE:
        try:
            # Load YOLO model
            yolo_model = YOLO('indian_license_plate_best.pt')
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])
            
            return yolo_model, reader, True
        except Exception as e:
            st.error(f"Error loading advanced models: {e}")
            return None, None, False
    else:
        return None, None, False

def detect_license_plates_opencv(image):
    """Fallback license plate detection using OpenCV"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Find edges
        edged = cv2.Canny(filtered, 30, 200)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area, keep only large ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        
        license_plate_contours = []
        
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # If the contour has 4 vertices, it might be a license plate
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio (license plates are typically rectangular)
                aspect_ratio = w / h
                if 2.0 < aspect_ratio < 6.0:  # Typical license plate aspect ratio
                    license_plate_contours.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': 0.7,  # Fixed confidence for OpenCV detection
                        'contour': approx
                    })
        
        return license_plate_contours
    except Exception as e:
        st.error(f"Error in OpenCV license plate detection: {e}")
        return []

def detect_license_plate_yolo(image, model):
    """Detect license plates in the image using YOLO model"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run YOLO detection
        results = model(img_array)
        
        # Extract bounding boxes
        plates = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                plates.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence
                })
        
        return plates
    except Exception as e:
        st.error(f"Error in YOLO license plate detection: {e}")
        return []

def extract_text_from_plate_easyocr(plate_image, reader):
    """Extract text from license plate using EasyOCR"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image)
        
        # Run OCR
        results = reader.readtext(plate_array)
        
        # Extract and combine text
        text_list = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Filter low confidence results
                text_list.append(text)
        
        return ' '.join(text_list) if text_list else "No text detected"
    except Exception as e:
        st.error(f"Error in EasyOCR text extraction: {e}")
        return "Error extracting text"

def extract_text_from_plate_opencv(plate_image):
    """Fallback text extraction using OpenCV"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_array, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours for potential characters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on size and aspect ratio
        char_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Typical character properties
            if 0.2 < aspect_ratio < 1.0 and area > 50:
                char_contours.append((x, y, w, h))
        
        # Sort characters left to right
        char_contours.sort(key=lambda x: x[0])
        
        if len(char_contours) > 0:
            return f"Detected {len(char_contours)} potential characters (OpenCV)"
        else:
            return "No characters detected"
            
    except Exception as e:
        st.error(f"Error in OpenCV text extraction: {e}")
        return "Error extracting text"

def draw_bounding_boxes(image, plates):
    """Draw bounding boxes on the original image"""
    img_array = np.array(image.copy())
    
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = plate['bbox']
        confidence = plate['confidence']
        
        # Draw rectangle
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label = f"Plate {i+1}: {confidence:.2f}"
        cv2.putText(img_array, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

# Load models
yolo_model, ocr_reader, use_advanced = load_models()

# Main UI
st.sidebar.header("📤 Upload Image")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
)

# Model selection
if ADVANCED_MODELS_AVAILABLE:
    use_advanced_models = st.sidebar.checkbox(
        "Use Advanced Models (YOLO + EasyOCR)",
        value=use_advanced
    )
else:
    use_advanced_models = False
    st.sidebar.info("Only OpenCV-based detection available")

# Main content area
if uploaded_file is None:
    st.info("👆 Please upload an image to get started")
    
    # Display model information
    if ADVANCED_MODELS_AVAILABLE:
        st.markdown("""
        ### Available Detection Methods:
        - **YOLO + EasyOCR**: Advanced AI models for high accuracy
        - **OpenCV**: Traditional computer vision (fallback)
        """)
    else:
        st.markdown("""
        ### Detection Method:
        - **OpenCV**: Traditional computer vision detection
        """)
    
    st.markdown("""
    ### How to use:
    1. Upload an image containing a vehicle with a license plate
    2. Choose detection method (if available)
    3. Click "Detect License Plates"
    4. View results with bounding boxes and extracted text
    
    ### Supported formats:
    - JPG, JPEG, PNG, BMP, TIFF
    """)
else:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)
    
    # Process button
    if st.button("🔍 Detect License Plates", type="primary"):
        with st.spinner("🔄 Processing image..."):
            if use_advanced_models and use_advanced and yolo_model and ocr_reader:
                # Use YOLO + EasyOCR
                plates = detect_license_plate_yolo(image, yolo_model)
                detection_method = "YOLO + EasyOCR"
            else:
                # Use OpenCV fallback
                plates = detect_license_plates_opencv(image)
                detection_method = "OpenCV"
            
            if plates:
                # Draw bounding boxes
                image_with_boxes = draw_bounding_boxes(image, plates)
                
                with col2:
                    st.subheader(f"🎯 Detection Results ({detection_method})")
                    st.image(image_with_boxes, use_column_width=True)
                
                # Extract text from each detected plate
                st.subheader("📝 Extracted Text")
                
                for i, plate in enumerate(plates):
                    x1, y1, x2, y2 = plate['bbox']
                    
                    # Crop the license plate region
                    plate_image = image.crop((x1, y1, x2, y2))
                    
                    # Extract text based on available method
                    if use_advanced_models and use_advanced and ocr_reader:
                        extracted_text = extract_text_from_plate_easyocr(plate_image, ocr_reader)
                    else:
                        extracted_text = extract_text_from_plate_opencv(plate_image)
                    
                    # Display results
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.write(f"**Plate {i+1}:**")
                    
                    with col2:
                        st.write(f"`{extracted_text}`")
                    
                    with col3:
                        st.write(f"Confidence: `{plate['confidence']:.2f}`")
                    
                    # Display cropped plate
                    st.image(plate_image, caption=f"License Plate {i+1}", width=200)
                    
                    st.divider()
            else:
                st.warning("🚫 No license plates detected in the image")
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(image, use_column_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** For best results, use clear images with good lighting and visible license plates")

# Model information
with st.expander("🔧 Model Information"):
    if ADVANCED_MODELS_AVAILABLE:
        st.markdown("""
        **Advanced Models (when enabled):**
        - **Detection:** YOLOv8 trained on Indian license plates
        - **OCR:** EasyOCR with English language support
        - **Confidence Threshold:** 0.5
        
        **Fallback Method:**
        - **Detection:** OpenCV contour-based detection
        - **OCR:** OpenCV character detection
        """)
    else:
        st.markdown("""
        **Detection Method:** OpenCV-based
        - Contour detection with aspect ratio filtering
        - Character detection using thresholding
        - Suitable for basic license plate detection
        """)

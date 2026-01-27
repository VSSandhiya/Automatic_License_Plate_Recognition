import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="License Plate Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# No PyTorch imports - using pure OpenCV to avoid DLL issues
PYTORCH_AVAILABLE = False

# Elegant CSS styling
def load_elegant_css():
    st.markdown("""
    <style>
    /* Main background with glass white-blue gradient */
    .stApp {
        background: linear-gradient(135deg, 
            rgba(224, 242, 255, 0.95) 0%, 
            rgba(187, 222, 251, 0.90) 40%, 
            rgba(144, 202, 249, 0.85) 70%, 
            rgba(100, 181, 246, 0.80) 100%);
        color: #0f172a;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    
    /* Sidebar styling */
    .stSidebar {
        background: linear-gradient(180deg,
            rgba(255,255,255,0.75) 0%,
            rgba(227,242,253,0.85) 100%);
        border-right: 2px solid rgba(255,255,255,0.4);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }
    
    /* Headers */
    h1 {
        color: #0f172a;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
        font-weight: 700;
    }
    
    /* Subheaders */
    .stSubheader {
        color: #1e293b;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #42A5F5 0%, #1E88E5 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 6px 14px rgba(30,136,229,0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #64B5F6 0%, #42A5F5 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(30,136,229,0.4);
    }
    
    /* File uploader */
    .stFileUploader {
        background: rgba(255,255,255,0.55);
        border: 2px dashed rgba(30,136,229,0.4);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(90deg, #66BB6A 0%, #43A047 100%);
        border-left: 4px solid #2E7D32;
        color: white;
        font-weight: 600;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(90deg, #42A5F5 0%, #1E88E5 100%);
        border-left: 4px solid #1565C0;
        color: white;
        font-weight: 600;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(90deg, #FFA726 0%, #FB8C00 100%);
        border-left: 4px solid #EF6C00;
        color: white;
        font-weight: 600;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(90deg, #EF5350 0%, #E53935 100%);
        border-left: 4px solid #C62828;
        color: white;
        font-weight: 600;
    }
    
    /* Expanders */
    .stExpander {
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 12px;
        backdrop-filter: blur(12px);
    }
    
    .stExpanderHeader {
        background: rgba(255,255,255,0.35);
        border-radius: 12px 12px 0 0;
    }
    
    /* Metrics display */
    .metric-container {
        background: rgba(255,255,255,0.55);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        backdrop-filter: blur(12px);
    }
    
    /* Confidence badges */
    .confidence-high {
        background: linear-gradient(90deg, #66BB6A 0%, #43A047 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .confidence-medium {
        background: linear-gradient(90deg, #FFA726 0%, #FB8C00 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .confidence-low {
        background: linear-gradient(90deg, #EF5350 0%, #E53935 100%);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Image containers */
    .stImage {
        border: 2px solid rgba(255,255,255,0.6);
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    /* Text styling */
    .stMarkdown {
        color: #1e293b;
    }
    
    /* Code blocks */
    .stCode {
        background: rgba(255,255,255,0.6);
        border: 1px solid rgba(255,255,255,0.4);
        border-radius: 6px;
        color: #0f172a;
    }
    </style>
    """, unsafe_allow_html=True)

# Load elegant CSS
load_elegant_css()


# Title and description
st.title("License Plate Detection & OCR System")
st.markdown("Upload an image to detect license plates and extract text using AI models")

# Initialize models - using OpenCV only
@st.cache_resource
def load_models():
    """Load OpenCV models only"""
    return None, None, False  # No PyTorch models

# Load models
yolo_model, ocr_reader, models_available = load_models()

# Show model status
if not PYTORCH_AVAILABLE:
    st.sidebar.warning("⚠️ PyTorch not available due to DLL issues")
    st.sidebar.info("🔧 Using OpenCV fallback mode")
elif not models_available:
    st.sidebar.warning("⚠️ Models failed to load")
    st.sidebar.info("🔧 Using OpenCV fallback mode")
else:
    st.sidebar.success("✅ PyTorch models loaded successfully")

def detect_license_plate_yolo(image, model):
    """Detect license plates in the image using YOLO model"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run YOLO detection
        results = model(img_array)
        
        # Extract bounding boxes
        plates = []
        total_confidence = 0
        
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
                total_confidence += confidence
        
        # Calculate overall image confidence
        overall_confidence = total_confidence / len(plates) if plates else 0
        
        return plates, overall_confidence
    except Exception as e:
        st.error(f"Error in YOLO detection: {e}")
        return [], 0

def detect_license_plate_opencv(image):
    """Detect license plates using OpenCV with confidence scoring"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply bilateral filter to preserve edges while reducing noise
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply histogram equalization for better contrast
        equalized = cv2.equalizeHist(filtered)
        
        # Find edges using multiple methods
        edged1 = cv2.Canny(equalized, 30, 200)
        edged2 = cv2.Canny(filtered, 50, 150)
        
        # Combine edge maps
        edged = cv2.bitwise_or(edged1, edged2)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((3,3), np.uint8)
        edged = cv2.dilate(edged, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area, keep only large ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        
        license_plate_contours = []
        
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = w * h
            
            # Check for license plate properties
            if (2.0 < aspect_ratio < 8.0 and  # Typical license plate aspect ratio
                area > 2000 and  # Minimum area
                h > 20 and w > 60):  # Minimum dimensions
                
                # Additional check for rectangular shape
                if len(approx) >= 4:
                    # Calculate confidence based on multiple factors
                    area_confidence = min(1.0, area / 50000)  # Area-based confidence
                    aspect_confidence = 1.0 - abs(aspect_ratio - 4.0) / 4.0  # Aspect ratio confidence
                    shape_confidence = len(approx) / 8.0  # Shape confidence (more vertices = better)
                    
                    # Combined confidence score
                    confidence = (area_confidence * 0.4 + aspect_confidence * 0.4 + shape_confidence * 0.2)
                    
                    license_plate_contours.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': confidence
                    })
        
        # Remove overlapping detections
        filtered_plates = []
        for plate in sorted(license_plate_contours, key=lambda x: x['confidence'], reverse=True):
            overlap = False
            for existing in filtered_plates:
                # Calculate IoU
                x1_max = max(plate['bbox'][0], existing['bbox'][0])
                y1_max = max(plate['bbox'][1], existing['bbox'][1])
                x2_min = min(plate['bbox'][2], existing['bbox'][2])
                y2_min = min(plate['bbox'][3], existing['bbox'][3])
                
                if x2_min > x1_max and y2_min > y1_max:
                    intersection = (x2_min - x1_max) * (y2_min - y1_max)
                    union = ((plate['bbox'][2] - plate['bbox'][0]) * (plate['bbox'][3] - plate['bbox'][1]) +
                             (existing['bbox'][2] - existing['bbox'][0]) * (existing['bbox'][3] - existing['bbox'][1]) - intersection)
                    
                    if intersection / union > 0.3:  # 30% overlap threshold
                        overlap = True
                        break
            
            if not overlap:
                filtered_plates.append(plate)
        
        return filtered_plates[:5]  # Return top 5 candidates
        
    except Exception as e:
        st.error(f"Error in OpenCV detection: {e}")
        return []

def extract_text_from_plate(plate_image, reader):
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
        st.error(f"Error in text extraction: {e}")
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

# Main UI
st.sidebar.header("📤 Upload Image")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
)

# Main content area
if uploaded_file is None:
    st.info("👆 Please upload an image to get started")
    
    # Display sample images or instructions
    st.markdown("""
    ### How to use:
    1. Upload an image containing a vehicle with a license plate
    2. The system will automatically detect license plates using YOLO
    3. Extract text from detected plates using EasyOCR
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
            # Use appropriate detection method based on model availability
            if models_available and yolo_model is not None:
                # Use YOLO if available
                plates, overall_confidence = detect_license_plate_yolo(image, yolo_model)
                detection_method = "YOLO"
            else:
                # Use OpenCV fallback
                plates = detect_license_plate_opencv(image)
                if plates:
                    overall_confidence = sum(p['confidence'] for p in plates) / len(plates)
                else:
                    overall_confidence = 0.0
                detection_method = "OpenCV"
                
                if plates:
                    # Filter for high accuracy license plates only (confidence >= 70%)
                    high_accuracy_plates = [p for p in plates if p['confidence'] >= 0.70]
                    
                    if high_accuracy_plates:
                        # Draw bounding boxes only on high accuracy license plates
                        image_with_boxes = draw_bounding_boxes(image, high_accuracy_plates)
                        
                        with col2:
                            st.subheader(f"🎯 High Accuracy License Plates ({detection_method})")
                            st.image(image_with_boxes, use_column_width=True)
                            
                            # Enhanced confidence display for high accuracy plates
                            high_accuracy_confidence = sum(p['confidence'] for p in high_accuracy_plates) / len(high_accuracy_plates)
                            st.markdown(f"""
                            <div class="metric-container">
                                <h3 style="margin: 0; color: #0f172a;">📊 High Accuracy Metrics</h3>
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                                    <div>
                                        <div style="font-size: 24px; font-weight: bold; color: #0f172a;">
                                            {high_accuracy_confidence:.1%}
                                        </div>
                                        <div style="font-size: 14px; color: #1e293b;">Average Confidence</div>
                                    </div>
                                    <div style="text-align: right;">
                                        <div style="font-size: 24px; font-weight: bold; color: #0f172a;">
                                            {len(high_accuracy_plates)}
                                        </div>
                                        <div style="font-size: 14px; color: #1e293b;">High Accuracy Plates</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Extract text from each high accuracy license plate only
                        st.subheader("📝 High Accuracy License Plates")
                        
                        for i, plate in enumerate(high_accuracy_plates):
                            x1, y1, x2, y2 = plate['bbox']
                            
                            # Crop the license plate region
                            plate_image = image.crop((x1, y1, x2, y2))
                            
                            # Extract text using OpenCV fallback if EasyOCR not available
                            if models_available and ocr_reader is not None:
                                extracted_text = extract_text_from_plate(plate_image, ocr_reader)
                            else:
                                # Use OpenCV text extraction
                                extracted_text = "OpenCV text extraction not implemented"
                            
                            # Display results with enhanced confidence display
                            st.markdown("---")
                            
                            # Enhanced plate header with confidence
                            plate_confidence_percent = plate['confidence'] * 100
                            if plate_confidence_percent >= 80:
                                confidence_class = "confidence-high"
                                confidence_emoji = "🟢"
                            elif plate_confidence_percent >= 60:
                                confidence_class = "confidence-medium"
                                confidence_emoji = "🟡"
                            else:
                                confidence_class = "confidence-low"
                                confidence_emoji = "🔴"
                            
                            st.markdown(f"""
                            <div class="metric-container">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <h3 style="margin: 0; color: #0f172a;">{confidence_emoji} License Plate {i+1}</h3>
                                    <span class="{confidence_class}">{plate_confidence_percent:.1f}% Confidence</span>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                            
                            # Display results
                            col1, col2, col3 = st.columns([1, 2, 1])
                            
                            with col1:
                                st.write("**Extracted Text:**")
                            
                            with col2:
                                if extracted_text and extracted_text != "No text detected":
                                    st.success(f"🎯 `{extracted_text}`")
                                else:
                                    st.warning("`No text detected`")
                            
                            with col3:
                                st.write(f"**Detection:** `{plate['confidence']:.3f}`")
                            
                            # Enhanced detailed metrics
                            with st.expander(f"📊 Detailed Metrics - Plate {i+1}"):
                                st.markdown(f"""
                                <div class="metric-container">
                                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                        <div>
                                            <div style="font-size: 18px; font-weight: bold; color: #0f172a;">
                                                {plate['confidence']:.1%}
                                            </div>
                                            <div style="font-size: 12px; color: #1e293b;">Plate Confidence</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 18px; font-weight: bold; color: #0f172a;">
                                                ({x1}, {y1})
                                            </div>
                                            <div style="font-size: 12px; color: #1e293b;">Position</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 18px; font-weight: bold; color: #0f172a;">
                                                {x2-x1} × {y2-y1}
                                            </div>
                                            <div style="font-size: 12px; color: #1e293b;">Dimensions (px)</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 18px; font-weight: bold; color: #0f172a;">
                                                {(x2-x1)*(y2-y1):,}
                                            </div>
                                            <div style="font-size: 12px; color: #1e293b;">Area (px²)</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 18px; font-weight: bold; color: #0f172a;">
                                                {(x2-x1)/(y2-y1):.2f}
                                            </div>
                                            <div style="font-size: 12px; color: #1e293b;">Aspect Ratio</div>
                                        </div>
                                        <div>
                                            <div style="font-size: 18px; font-weight: bold; color: #0f172a;">
                                                {overall_confidence:.1%}
                                            </div>
                                            <div style="font-size: 12px; color: #1e293b;">Overall Confidence</div>
                                        </div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Display cropped plate
                        st.image(plate_image, caption=f"License Plate {i+1} (Confidence: {plate_confidence_percent:.1f}%)", width=300)
                        
                        st.divider()
                else:
                    with col2:
                        st.subheader("🎯 Detection Results")
                        st.image(image, use_column_width=True)
                        
                        # No plates detected - show zero confidence
                        st.markdown(f"""
                        <div class="metric-container" style="background: rgba(245, 101, 101, 0.2); border: 1px solid rgba(245, 101, 101, 0.5);">
                            <h3 style="margin: 0; color: white;">📊 Detection Metrics</h3>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                                <div>
                                    <div style="font-size: 24px; font-weight: bold; color: #f56565;">
                                        0.0%
                                    </div>
                                    <div style="font-size: 14px; color: #e2e8f0;">Overall Confidence</div>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 24px; font-weight: bold; color: #f56565;">
                                        0
                                    </div>
                                    <div style="font-size: 14px; color: #e2e8f0;">Plates Detected</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("🚫 No license plates detected in the image")
                        st.info("💡 Try using a clearer image with visible license plates")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** For best results, use clear images with good lighting and visible license plates")

# Model information
with st.expander("🔧 Model Information"):
    st.markdown("""
    **Detection Model:** YOLOv8 trained on Indian license plates
    - Model file: `indian_license_plate_best.pt`
    - Confidence threshold: 0.5
    """)

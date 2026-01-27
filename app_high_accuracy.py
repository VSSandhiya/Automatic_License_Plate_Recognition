import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="High Accuracy License Plate Detector",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
st.title("🚗 High Accuracy License Plate Detector")
st.markdown("Upload an image to detect high accuracy license plates with confidence levels")

def detect_high_accuracy_plates_opencv(image):
    """Detect high accuracy license plates using OpenCV with confidence scoring"""
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
            
            # Check for license plate properties (stricter criteria for high accuracy)
            if (2.5 < aspect_ratio < 6.0 and  # Stricter aspect ratio range
                area > 3000 and  # Higher minimum area
                h > 25 and w > 80):  # Larger minimum dimensions
                
                # Additional check for rectangular shape
                if len(approx) >= 4:
                    # Calculate confidence based on multiple factors
                    area_confidence = min(1.0, area / 60000)  # Area-based confidence
                    aspect_confidence = 1.0 - abs(aspect_ratio - 4.0) / 4.0  # Aspect ratio confidence
                    shape_confidence = len(approx) / 8.0  # Shape confidence (more vertices = better)
                    
                    # Combined confidence score
                    confidence = (area_confidence * 0.4 + aspect_confidence * 0.4 + shape_confidence * 0.2)
                    
                    # Only include if confidence is high enough
                    if confidence >= 0.70:  # 70% minimum confidence
                        license_plate_contours.append({
                            'bbox': (x, y, x + w, y + h),
                            'confidence': confidence,
                            'contour': contour,
                            'area': area,
                            'aspect_ratio': aspect_ratio
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
        
        return filtered_plates[:5]  # Return top 5 high accuracy candidates
        
    except Exception as e:
        st.error(f"Error in license plate detection: {e}")
        return []

def extract_text_opencv(plate_image):
    """Extract text using OpenCV"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image)
        
        # Check if image is already grayscale
        if len(plate_array.shape) == 3:
            gray = cv2.cvtColor(plate_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = plate_array
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Noise removal
        kernel = np.ones((1,1), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours for potential characters
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on character properties
        char_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # Character properties: reasonable aspect ratio and size
            if 0.1 < aspect_ratio < 1.0 and 20 < area < 2000 and h > 10:
                char_contours.append((x, y, w, h))
        
        # Sort characters left to right
        char_contours.sort(key=lambda x: x[0])
        
        if len(char_contours) > 0:
            # Try to estimate if this looks like a license plate
            if len(char_contours) >= 4:  # Most license plates have multiple characters
                return f"Detected {len(char_contours)} characters (License Plate Format)"
            else:
                return f"Detected {len(char_contours)} potential characters"
        else:
            return "No clear characters detected"
            
    except Exception as e:
        st.error(f"Error in text extraction: {e}")
        return "Error extracting text"

def draw_bounding_boxes(image, plates):
    """Draw bounding boxes on the original image"""
    img_array = np.array(image.copy())
    
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = plate['bbox']
        confidence = plate['confidence']
        
        # Draw rectangle with thicker lines for high accuracy
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label with confidence
        label = f"Plate {i+1}: {confidence:.1%}"
        cv2.putText(img_array, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return Image.fromarray(img_array)

# Main UI
st.sidebar.header("📤 Upload Image")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
)

# Detection parameters
st.sidebar.subheader("🔧 Detection Parameters")
confidence_threshold = st.sidebar.slider("Minimum Confidence", 0.70, 1.0, 0.70, 0.05)
enhance_images = st.sidebar.checkbox("Enhance Plate Images", value=True)

st.sidebar.success("✅ OpenCV High Accuracy Detection: Available")
st.sidebar.info("🎯 Focus: License plates with ≥70% confidence")

# Main content area
if uploaded_file is None:
    st.info("👆 Please upload an image to get started")
    
    st.markdown("""
    ### 🎯 High Accuracy License Plate Detection
    
    This version focuses exclusively on high accuracy license plates:
    - **Strict Filtering**: Only plates with ≥70% confidence
    - **Enhanced Detection**: Improved OpenCV algorithms
    - **Boundary Boxes**: Applied only to high accuracy plates
    - **Confidence Display**: Each plate shows accuracy level
    - **No PyTorch**: Pure OpenCV implementation - no DLL issues
    """)
else:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)
    
    # Process button
    if st.button("🔍 Detect High Accuracy License Plates", type="primary"):
        with st.spinner("🔄 Processing image with OpenCV..."):
            # Detect high accuracy license plates
            plates = detect_high_accuracy_plates_opencv(image)
            
            # Filter by user-specified confidence threshold
            filtered_plates = [p for p in plates if p['confidence'] >= confidence_threshold]
            
            if filtered_plates:
                # Calculate average confidence
                avg_confidence = sum(p['confidence'] for p in filtered_plates) / len(filtered_plates)
                
                # Draw bounding boxes only on high accuracy plates
                image_with_boxes = draw_bounding_boxes(image, filtered_plates)
                
                with col2:
                    st.subheader("🎯 High Accuracy License Plates")
                    st.image(image_with_boxes, use_column_width=True)
                    
                    # Enhanced confidence display
                    st.markdown(f"""
                    <div class="metric-container">
                        <h3 style="margin: 0; color: #0f172a;">📊 High Accuracy Metrics</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                            <div>
                                <div style="font-size: 24px; font-weight: bold; color: #0f172a;">
                                    {avg_confidence:.1%}
                                </div>
                                <div style="font-size: 14px; color: #1e293b;">Average Confidence</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 24px; font-weight: bold; color: #0f172a;">
                                    {len(filtered_plates)}
                                </div>
                                <div style="font-size: 14px; color: #1e293b;">High Accuracy Plates</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Extract text from each high accuracy license plate
                st.subheader("📝 High Accuracy License Plates")
                
                for i, plate in enumerate(filtered_plates):
                    x1, y1, x2, y2 = plate['bbox']
                    
                    # Crop the license plate region
                    plate_image = image.crop((x1, y1, x2, y2))
                    
                    # Extract text using OpenCV
                    extracted_text = extract_text_opencv(plate_image)
                    
                    # Display results with enhanced confidence display
                    st.markdown("---")
                    
                    # Enhanced plate header with confidence
                    plate_confidence_percent = plate['confidence'] * 100
                    if plate_confidence_percent >= 80:
                        confidence_class = "confidence-high"
                        confidence_emoji = "🟢"
                    elif plate_confidence_percent >= 70:
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
                        st.write(f"`{extracted_text}`")
                    
                    with col3:
                        st.write(f"**Detection:** `{plate['confidence']:.3f}`")
                    
                    # Display cropped plate
                    st.image(plate_image, caption=f"License Plate {i+1} (Confidence: {plate_confidence_percent:.1f}%)", width=300)
                    
                    st.divider()
            else:
                st.warning("🚫 No high accuracy license plates detected")
                st.info("💡 Try lowering the confidence threshold or use a clearer image")
                
                with col2:
                    st.subheader("🎯 Detection Results")
                    st.image(image, use_column_width=True)
                    
                    # Show zero confidence
                    st.markdown(f"""
                    <div class="metric-container" style="background: rgba(239, 83, 80, 0.2); border: 1px solid rgba(239, 83, 80, 0.5);">
                        <h3 style="margin: 0; color: #0f172a;">📊 Detection Metrics</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px;">
                            <div>
                                <div style="font-size: 24px; font-weight: bold; color: #f44336;">
                                    0.0%
                                </div>
                                <div style="font-size: 14px; color: #1e293b;">High Accuracy Plates</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 24px; font-weight: bold; color: #f44336;">
                                    0
                                </div>
                                <div style="font-size: 14px; color: #1e293b;">Plates Found</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Pure OpenCV high accuracy detection - no PyTorch dependencies, focuses on license plates with ≥70% confidence")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="License Plate Detector",
    page_icon="🚗",
    layout="wide"
)

# Dark mode CSS
def get_dark_mode_css():
    return """
    <style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stMarkdown {
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2d2d2d;
    }
    .stButton > button {
        background-color: #4a4a4a;
        color: #ffffff;
        border: 1px solid #555555;
    }
    .stButton > button:hover {
        background-color: #5a5a5a;
        border-color: #666666;
    }
    .stSelectbox > div > div > select {
        background-color: #3a3a3a;
        color: #ffffff;
    }
    .stSlider > div > div > div {
        background-color: #3a3a3a;
    }
    .stFileUploader {
        background-color: #3a3a3a;
    }
    .stAlert {
        background-color: #3a3a3a;
        color: #ff6b6b;
    }
    
    .stAlert[data-testid="stAlert"] {
        color: #ff6b6b !important;
    }
    
    .stAlert .element-container {
        color: #ff6b6b !important;
    }
    
    .st-emotion-cache-10trblm {
        color: #000000 !important;
    }
    
    .st-emotion-cache-1j6rxz7 {
        color: #000000 !important;
    }
    
    .st-emotion-cache-nahz7x {
        color: #000000 !important;
    }
    
    p[data-component-name="<p />"] {
        color: #000000 !important;
    }
    
    h1 {
        color: #000000 !important;
    }
    
    .stMarkdown {
        color: #000000 !important;
    }
    .stExpander {
        background-color: #2d2d2d;
        border: 1px solid #444444;
    }
    </style>
    """

def get_light_mode_css():
    return """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    .stAlert[data-testid="stAlert"] {
        color: #d32f2f !important;
    }
    
    .stAlert .element-container {
        color: #d32f2f !important;
    }
    
    .st-emotion-cache-10trblm {
        color: #000000 !important;
    }
    
    .st-emotion-cache-1j6rxz7 {
        color: #000000 !important;
    }
    
    .st-emotion-cache-nahz7x {
        color: #000000 !important;
    }
    
    p[data-component-name="<p />"] {
        color: #000000 !important;
    }
    
    h1 {
        color: #000000 !important;
    }
    
    .stMarkdown {
        color: #000000 !important;
    }
    </style>
    """

# Initialize session state for theme
if 'theme_mode' not in st.session_state:
    st.session_state.theme_mode = 'light'

# Theme toggle function
def toggle_theme():
    if st.session_state.theme_mode == 'light':
        st.session_state.theme_mode = 'dark'
    else:
        st.session_state.theme_mode = 'light'

# Apply theme CSS
if st.session_state.theme_mode == 'dark':
    st.markdown(get_dark_mode_css(), unsafe_allow_html=True)
else:
    st.markdown(get_light_mode_css(), unsafe_allow_html=True)

# Theme toggle in sidebar
st.sidebar.markdown("---")
theme_toggle = st.sidebar.button(
    f"🌙 {'Light' if st.session_state.theme_mode == 'dark' else 'Dark'} Mode",
    on_click=toggle_theme
)

# Title and description
st.title("🚗 License Plate Detection & OCR System")
st.markdown("Upload an image to detect license plates and extract text using computer vision and OCR")

def detect_license_plates_opencv(image):
    """License plate detection using OpenCV"""
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
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
        
        license_plate_contours = []
        
        for contour in contours:
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)
            
            # If the contour has 4 vertices, it might be a license plate
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                
                # Check aspect ratio and size (license plates are typically rectangular)
                aspect_ratio = w / h
                area = w * h
                
                # Typical license plate properties
                if 2.0 < aspect_ratio < 6.0 and area > 1000:
                    license_plate_contours.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': min(0.9, area / 10000),  # Confidence based on size
                        'contour': approx
                    })
        
        # If no rectangular contours found, try with relaxed criteria
        if not license_plate_contours:
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = w * h
                
                # More relaxed criteria
                if 1.5 < aspect_ratio < 8.0 and area > 500:
                    license_plate_contours.append({
                        'bbox': (x, y, x + w, y + h),
                        'confidence': min(0.7, area / 15000),
                        'contour': None
                    })
        
        return license_plate_contours[:5]  # Return top 5 candidates
        
    except Exception as e:
        st.error(f"Error in license plate detection: {e}")
        return []

def extract_text_from_plate_opencv(plate_image):
    """Text extraction using OpenCV"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image)
        
        # Check if image is already grayscale
        if len(plate_array.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_array, cv2.COLOR_RGB2GRAY)
        else:
            # Already grayscale
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

@st.cache_resource
def load_models():
    """Load YOLO model for detection and EasyOCR for text extraction"""
    models = {}
    
    # Load YOLO model for license plate detection
    try:
        from ultralytics import YOLO
        yolo_model = YOLO('indian_license_plate_best.pt')
        models['yolo'] = yolo_model
        st.success("✅ YOLO model loaded successfully")
    except ImportError:
        st.warning("⚠️ Ultralytics not installed. Using OpenCV fallback.")
    except Exception as e:
        st.warning(f"⚠️ YOLO model loading failed: {e}")
    
    # Load EasyOCR for text extraction
    try:
        import easyocr
        st.write("🔄 Loading EasyOCR model...")
        reader = easyocr.Reader(['en'], gpu=False)
        models['easyocr'] = reader
        st.success("✅ EasyOCR loaded successfully")
    except ImportError:
        st.warning("⚠️ EasyOCR not installed. Install with: pip install easyocr")
    except Exception as e:
        st.warning(f"⚠️ EasyOCR loading failed: {e}")
        st.info("💡 Try installing: pip install easyocr")
    
    return models

def detect_license_plates_yolo(image, yolo_model):
    """Detect license plates using YOLO model"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Run YOLO detection
        results = yolo_model(img_array)
        
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
        st.error(f"Error in YOLO detection: {e}")
        return []

def extract_text_easyocr(plate_image, easyocr_reader):
    """Extract text using EasyOCR"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image)
        
        # Run EasyOCR with detailed output
        results = easyocr_reader.readtext(plate_array, detail=1, paragraph=False)
        
        # Extract and clean text with debugging
        text_list = []
        confidences = []
        
        for (bbox, text, confidence) in results:
            # Debug output
            st.write(f"EasyOCR detected: '{text}' with confidence: {confidence:.3f}")
            
            if confidence > 0.3:  # Lower threshold for better detection
                clean_text = ''.join(c for c in text if c.isalnum()).upper()
                if len(clean_text) >= 2:
                    text_list.append(clean_text)
                    confidences.append(confidence)
        
        if text_list:
            avg_confidence = sum(confidences) / len(confidences)
            final_text = ' '.join(text_list)
            st.success(f"EasyOCR Result: '{final_text}' (Avg Confidence: {avg_confidence:.3f})")
            return final_text
        else:
            st.warning("EasyOCR: No valid text detected above confidence threshold")
            return "No text detected"
            
    except Exception as e:
        st.error(f"Error in EasyOCR text extraction: {e}")
        return "Error extracting text"

def extract_text_from_plate_opencv_enhanced(plate_image):
    """Enhanced OpenCV text extraction"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image)
        
        # Check if image is already grayscale
        if len(plate_array.shape) == 3:
            gray = cv2.cvtColor(plate_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = plate_array
        
        # Multiple preprocessing techniques
        methods = []
        
        # Method 1: Adaptive threshold
        binary1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append(binary1)
        
        # Method 2: Otsu threshold
        _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(binary2)
        
        # Method 3: Enhanced contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        _, binary3 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(binary3)
        
        best_result = ""
        
        for i, binary in enumerate(methods):
            # Noise removal
            kernel = np.ones((1,1), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter and sort characters
            char_contours = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                area = cv2.contourArea(contour)
                
                # Enhanced character detection
                if 0.2 < aspect_ratio < 1.5 and 15 < area < 3000 and h > 8:
                    char_contours.append((x, y, w, h))
            
            # Sort left to right
            char_contours.sort(key=lambda x: x[0])
            
            if len(char_contours) >= 4:
                # This looks like a license plate
                current_result = f"Detected {len(char_contours)} characters"
                if len(current_result) > len(best_result):
                    best_result = current_result
        
        return best_result if best_result else "No clear characters detected"
        
    except Exception:
        return "Error extracting text"

def enhance_plate_image(plate_image):
    """Enhance license plate image for better processing"""
    try:
        # Convert to numpy array
        plate_array = np.array(plate_image)
        
        # Check if image is already grayscale
        if len(plate_array.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(plate_array, cv2.COLOR_RGB2GRAY)
        else:
            # Already grayscale
            gray = plate_array
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
        
        # Convert back to PIL (ensure RGB format)
        return Image.fromarray(blurred, mode='L').convert('RGB')
        
    except Exception as e:
        st.error(f"Error enhancing image: {e}")
        return plate_image

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

def create_annotated_plate_image(plate_image, ocr_results):
    """Create annotated plate image with text overlay"""
    try:
        # Convert PIL image to numpy array
        plate_array = np.array(plate_image.copy())
        
        # Get the best text result (prefer EasyOCR > Keras-OCR > Tesseract > OpenCV)
        best_text = ""
        best_method = ""
        
        if 'easyocr' in ocr_results and ocr_results['easyocr'] != "Processing failed" and ocr_results['easyocr'] != "No text detected":
            best_text = ocr_results['easyocr']
            best_method = "EasyOCR"
        elif 'keras_ocr' in ocr_results and ocr_results['keras_ocr'] != "Processing failed" and ocr_results['keras_ocr'] != "No text detected":
            best_text = ocr_results['keras_ocr']
            best_method = "Keras-OCR"
        elif 'tesseract' in ocr_results and ocr_results['tesseract'] != "Processing failed" and ocr_results['tesseract'] != "No text detected":
            best_text = ocr_results['tesseract']
            best_method = "Tesseract"
        elif 'opencv' in ocr_results:
            best_text = ocr_results['opencv']
            best_method = "OpenCV"
        
        if best_text and best_text != "No text detected" and "Detected" not in best_text:
            # Add text overlay
            height, width = plate_array.shape[:2]
            
            # Add semi-transparent background for text
            overlay = plate_array.copy()
            cv2.rectangle(overlay, (5, height-40), (width-5, height-5), (0, 0, 0), -1)
            
            # Blend overlay
            alpha = 0.7
            plate_array = cv2.addWeighted(overlay, alpha, plate_array, 1 - alpha, 0)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Split text if too long
            if len(best_text) > 15:
                words = best_text.split()
                mid = len(words) // 2
                text1 = ' '.join(words[:mid])
                text2 = ' '.join(words[mid:])
                
                # Add first line
                cv2.putText(plate_array, text1, (10, height-25), font, font_scale, (0, 255, 0), thickness)
                # Add second line
                cv2.putText(plate_array, text2, (10, height-8), font, font_scale, (0, 255, 0), thickness)
            else:
                # Add single line text
                cv2.putText(plate_array, best_text, (10, height-15), font, font_scale, (0, 255, 0), thickness)
            
            # Add method label
            cv2.putText(plate_array, f"[{best_method}]", (10, 20), font, 0.4, (255, 255, 255), 1)
        
        return Image.fromarray(plate_array)
        
    except Exception as e:
        # Return original image if annotation fails
        return plate_image

# Load models
models = load_models()

# Main UI
st.sidebar.header("📤 Upload Image")

# File uploader
uploaded_file = st.sidebar.file_uploader(
    "Choose an image...",
    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
)

# Detection parameters
st.sidebar.subheader("🔧 Detection Parameters")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
enhance_images = st.sidebar.checkbox("Enhance Plate Images", value=True)

# Show available models
if 'yolo' in models:
    st.sidebar.success("✅ YOLO Detection: Available")
else:
    st.sidebar.warning("⚠️ YOLO Detection: Not Available (Using OpenCV)")

if 'easyocr' in models:
    st.sidebar.success("✅ EasyOCR: Available")
else:
    st.sidebar.warning("⚠️ EasyOCR: Not Available")

# Main content area
if uploaded_file is None:
    st.info("👆 Please upload an image to get started")
    
    st.markdown("""
    ### 🔍 Detection Method: OpenCV Computer Vision
    
    This version uses traditional computer vision techniques:
    - **Edge Detection**: Canny edge detection to find boundaries
    - **Contour Analysis**: Find rectangular shapes with license plate proportions
    - **Aspect Ratio Filtering**: Filter for typical license plate dimensions
    - **Character Detection**: Identify potential characters within detected regions
    
    ### How to use:
    1. Upload an image containing a vehicle with a license plate
    2. Adjust detection parameters if needed
    3. Click "Detect License Plates"
    4. View results with bounding boxes and character analysis
    
    ### Supported formats:
    - JPG, JPEG, PNG, BMP, TIFF
    
    ### Tips for better results:
    - Use clear, well-lit images
    - Ensure license plates are visible and not heavily skewed
    - Higher resolution images work better
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
            # Use YOLO if available, otherwise fallback to OpenCV
            if 'yolo' in models:
                plates = detect_license_plates_yolo(image, models['yolo'])
                detection_method = "YOLO"
            else:
                plates = detect_license_plates_opencv(image)
                detection_method = "OpenCV"
            
            # Filter by confidence threshold
            filtered_plates = [p for p in plates if p['confidence'] >= confidence_threshold]
            
            if filtered_plates:
                # Draw bounding boxes
                image_with_boxes = draw_bounding_boxes(image, filtered_plates)
                
                with col2:
                    st.subheader(f"🎯 Detection Results ({detection_method})")
                    st.image(image_with_boxes, use_column_width=True)
                
                # Process each detected plate
                st.subheader("📝 License Plate Analysis")
                
                for i, plate in enumerate(filtered_plates):
                    x1, y1, x2, y2 = plate['bbox']
                    
                    # Crop the license plate region
                    plate_image = image.crop((x1, y1, x2, y2))
                    
                    # Enhance if requested
                    if enhance_images:
                        plate_image = enhance_plate_image(plate_image)
                        enhanced_label = " (Enhanced)"
                    else:
                        enhanced_label = ""
                    
                    # Extract text using EasyOCR if available
                    if 'easyocr' in models:
                        st.write(f"**🔤 Processing Plate {i+1} with EasyOCR...**")
                        extracted_text = extract_text_easyocr(plate_image, models['easyocr'])
                        ocr_method_used = "EasyOCR"
                    else:
                        st.write(f"**🔤 Processing Plate {i+1} with OpenCV...**")
                        extracted_text = extract_text_from_plate_opencv(plate_image)
                        ocr_method_used = "OpenCV"
                    
                    # Display results in a more prominent way
                    st.markdown("---")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.write(f"**Plate {i+1}{enhanced_label}:**")
                    
                    with col2:
                        if ocr_method_used == "EasyOCR" and extracted_text != "No text detected" and extracted_text != "Error extracting text":
                            st.success(f"🎯 **{extracted_text}**")
                        else:
                            st.write(f"`{extracted_text}`")
                    
                    with col3:
                        st.write(f"Confidence: `{plate['confidence']:.2f}`")
                        st.write(f"Method: `{ocr_method_used}`")
                    
                    # Create annotated plate image with text overlay
                    ocr_results = {ocr_method_used.lower(): extracted_text}
                    annotated_plate = create_annotated_plate_image(plate_image, ocr_results)
                    
                    # Display cropped plate with text overlay
                    st.image(annotated_plate, caption=f"License Plate {i+1}{enhanced_label} - {ocr_method_used}", width=300)
                    
                    # Additional analysis
                    with st.expander(f"📊 Detailed Analysis - Plate {i+1}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Detection Method:** {detection_method}")
                            st.write(f"**OCR Method:** {ocr_method_used}")
                            st.write(f"**Dimensions:** {x2-x1} × {y2-y1} pixels")
                            st.write(f"**Area:** {(x2-x1)*(y2-y1)} pixels²")
                        with col2:
                            st.write(f"**Position:** ({x1}, {y1})")
                            st.write(f"**Detection Confidence:** {plate['confidence']:.2f}")
                            st.write(f"**Enhanced:** {'Yes' if enhance_images else 'No'}")
                    
                    st.divider()
            else:
                st.warning("🚫 No license plates detected with current confidence threshold")
                st.info("💡 Try lowering the confidence threshold or use a clearer image")
                
                with col2:
                    st.subheader(f"🎯 Detection Results ({detection_method})")
                    st.image(image, use_column_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Uses YOLO for detection and EasyOCR for text extraction with OpenCV fallback")

# Technical details
with st.expander("🔧 Technical Details"):
    st.markdown("""
    **Detection Pipeline:**
    - **YOLO Model**: `indian_license_plate_best.pt` for license plate detection
    - **Fallback**: OpenCV-based detection when YOLO is unavailable
    - **Confidence Filtering**: Adjustable threshold for detection accuracy
    
    **Text Extraction:**
    - **EasyOCR**: Deep learning-based OCR with high accuracy for license plates
    - **Text Cleaning**: Alphanumeric filtering and uppercase conversion
    - **Confidence Threshold**: 0.5 for reliable text detection
    - **Fallback**: OpenCV character detection when EasyOCR unavailable
    
    **Image Processing:**
    - **Cropping**: Extract detected license plate regions
    - **Enhancement**: Optional histogram equalization and noise reduction
    - **Annotation**: Text overlay on cropped images with method labels
    
    **Performance:**
    - YOLO Detection: ~1-2 seconds per image
    - EasyOCR Processing: ~1-3 seconds per plate
    - Overall: ~2-5 seconds per image with enhancement
    
    **Model Requirements:**
    - YOLO: `indian_license_plate_best.pt` in project directory
    - EasyOCR: Automatic download of recognition models
    - Dependencies: ultralytics, easyocr, opencv-python
    """)

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="License Plate Detector", page_icon="🚗", layout="wide")

st.title("🚗 License Plate Detector (No PyTorch)")
st.markdown("Pure OpenCV solution - no PyTorch dependencies")

def detect_plates_opencv(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(filtered, 30, 200)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    plates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        area = w * h
        
        if 2.0 < aspect_ratio < 8.0 and area > 2000:
            confidence = min(0.9, area / 50000)
            plates.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': confidence
            })
    
    return plates[:5]

def extract_text_opencv(plate_image):
    plate_array = np.array(plate_image)
    if len(plate_array.shape) == 3:
        gray = cv2.cvtColor(plate_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = plate_array
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 0.2 < w/h < 2.0 and 20 < w*h < 2000 and h > 10:
            chars.append((x, y, w, h))
    
    chars.sort(key=lambda x: x[0])
    return f"Detected {len(chars)} characters" if len(chars) >= 4 else "No clear text"

def draw_boxes(image, plates):
    img_array = np.array(image.copy())
    for i, plate in enumerate(plates):
        x1, y1, x2, y2 = plate['bbox']
        cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_array, f"Plate {i+1}: {plate['confidence']:.2f}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(img_array)

uploaded_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original")
        st.image(image, use_column_width=True)
    
    if st.button("Detect License Plates"):
        with st.spinner("Processing..."):
            plates = detect_plates_opencv(image)
            filtered = [p for p in plates if p['confidence'] > 0.3]
            
            if filtered:
                result_image = draw_boxes(image, filtered)
                
                with col2:
                    st.subheader("Detection Results")
                    st.image(result_image, use_column_width=True)
                
                st.subheader("Text Extraction")
                for i, plate in enumerate(filtered):
                    x1, y1, x2, y2 = plate['bbox']
                    plate_img = image.crop((x1, y1, x2, y2))
                    text = extract_text_opencv(plate_img)
                    
                    st.write(f"**Plate {i+1}:** `{text}`")
                    st.image(plate_img, caption=f"License Plate {i+1}", width=300)
                    st.divider()
            else:
                st.warning("No license plates detected")
                with col2:
                    st.subheader("Results")
                    st.image(image, use_column_width=True)

st.info("✅ Pure OpenCV - No PyTorch dependencies required!")

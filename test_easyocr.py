import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr

# Set page configuration
st.set_page_config(
    page_title="EasyOCR Test",
    page_icon="🔤",
    layout="wide"
)

st.title("🔤 EasyOCR Test - Debug Version")

# Load EasyOCR
@st.cache_resource
def load_easyocr():
    try:
        st.write("🔄 Loading EasyOCR...")
        reader = easyocr.Reader(['en'], gpu=False)
        st.success("✅ EasyOCR loaded successfully!")
        return reader
    except Exception as e:
        st.error(f"❌ Error loading EasyOCR: {e}")
        return None

# Initialize EasyOCR
reader = load_easyocr()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)
    
    if reader:
        if st.button("🔍 Extract Text with EasyOCR"):
            with st.spinner("Processing..."):
                try:
                    # Convert to numpy array
                    img_array = np.array(image)
                    
                    st.write("🔍 Running EasyOCR...")
                    
                    # Run EasyOCR with maximum detail
                    results = reader.readtext(img_array, detail=1, paragraph=False)
                    
                    st.write(f"📊 Found {len(results)} text regions:")
                    
                    if results:
                        # Display each result
                        for i, (bbox, text, confidence) in enumerate(results):
                            st.write(f"**Result {i+1}:**")
                            st.write(f"  Text: '{text}'")
                            st.write(f"  Confidence: {confidence:.4f}")
                            st.write(f"  Bounding Box: {bbox}")
                            
                            # Clean the text
                            clean_text = ''.join(c for c in text if c.isalnum()).upper()
                            st.write(f"  Cleaned: '{clean_text}'")
                            st.write("---")
                        
                        # Extract all valid text
                        all_text = []
                        for bbox, text, confidence in results:
                            if confidence > 0.1:  # Very low threshold for testing
                                clean_text = ''.join(c for c in text if c.isalnum()).upper()
                                if len(clean_text) >= 1:
                                    all_text.append(clean_text)
                        
                        if all_text:
                            final_result = ' '.join(all_text)
                            st.success(f"🎯 **FINAL RESULT: '{final_result}'**")
                            st.balloons()
                        else:
                            st.warning("⚠️ No valid text found after cleaning")
                    
                    else:
                        st.warning("⚠️ No text detected by EasyOCR")
                        
                except Exception as e:
                    st.error(f"❌ Error during OCR: {e}")
                    st.write(f"Error details: {str(e)}")
    else:
        st.error("❌ EasyOCR not available")

# Test with a simple text image
st.markdown("---")
st.subheader("🧪 Create Test Image")

if st.button("Create Test Text Image"):
    # Create a simple test image with text
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255  # White background
    
    # Add text using OpenCV
    cv2.putText(img, 'TEST123', (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    
    # Convert to PIL
    test_image = Image.fromarray(img)
    
    st.subheader("📷 Test Image")
    st.image(test_image, use_column_width=True)
    
    if reader:
        with st.spinner("Testing EasyOCR on test image..."):
            try:
                img_array = np.array(test_image)
                results = reader.readtext(img_array, detail=1, paragraph=False)
                
                st.write(f"📊 Test Results: {len(results)} text regions found")
                
                for i, (bbox, text, confidence) in enumerate(results):
                    st.write(f"**Test Result {i+1}:** '{text}' (Confidence: {confidence:.4f})")
                
                if results:
                    st.success("✅ EasyOCR is working correctly!")
                else:
                    st.warning("⚠️ EasyOCR did not detect text in test image")
                    
            except Exception as e:
                st.error(f"❌ Error testing EasyOCR: {e}")

# Installation instructions
st.markdown("---")
st.subheader("📦 Installation Instructions")
st.markdown("""
If EasyOCR is not working, try these commands:

```bash
pip install easyocr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Or install CPU-only PyTorch:
```bash
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
""")

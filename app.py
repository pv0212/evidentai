import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import streamlit.components.v1 as components

# --- CONFIGURATION ---
# PASTE YOUR HUGGING FACE MODEL ID HERE
# Example: "johndoe/deepfake-detector"
MODEL_ID = "InferenceEngineer/evidentai" 
# ---------------------

st.set_page_config(page_title="Deepfake Detector")

# 1. Load HTML (Frontend)
# We do this FIRST so the UI appears even if the model is loading
try:
    with open("app.html", "r") as f:
        components.html(f.read(), height=200, scrolling=True)
except:
    st.title("Deepfake Detection Tool")

st.write("---")

# 2. Load Model (The Fix is here)
@st.cache_resource
def load_model():
    try:
        # Load Processor
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        
        # Load Model with ignore_mismatched_sizes=True
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID,
            ignore_mismatched_sizes=True  # <--- THIS FIXES YOUR ERROR
        )
        return processor, model
    except Exception as e:
        st.error(f"CRITICAL ERROR: {e}")
        return None, None

with st.spinner("Loading model... please wait..."):
    processor, model = load_model()

# 3. File Uploader UI
uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your Image", use_column_width=True)
    
    with col2:
        if model:
            st.write("Analyzing...")
            
            # Predict
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_class_idx = logits.argmax(-1).item()
            
            # Get Label
            # If your config has id2label, use it. Otherwise default to index.
            if hasattr(model.config, 'id2label') and model.config.id2label:
                label = model.config.id2label[predicted_class_idx]
            else:
                label = f"Class {predicted_class_idx}"
            
            # Display Result with big text
            if "fake" in str(label).lower():
                st.error(f"Result: {label.upper()}")
            else:
                st.success(f"Result: {label.upper()}")
        else:
            st.error("Model failed to load. Check logs.")

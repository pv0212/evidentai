import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import streamlit.components.v1 as components

# --- CONFIGURATION ---
# PASTE YOUR HUGGING FACE MODEL ID HERE
MODEL_ID = "InferenceEngineer/evidentai" 
# ---------------------

st.set_page_config(page_title="Deepfake Detector")

# Load HTML (if it exists)
try:
    with open("app.html", "r") as f:
        components.html(f.read(), height=200)
except:
    st.title("Deepfake Detector")

@st.cache_resource
def load_model():
    try:
        # Load from Hugging Face Cloud directly
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        return processor, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

processor, model = load_model()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)
    
    # Predict
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
    label = model.config.id2label[predicted_class_idx]
    st.success(f"Result: {label}")
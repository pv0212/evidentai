import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import streamlit.components.v1 as components

# --- CONFIGURATION ---
# PASTE YOUR HUGGING FACE MODEL ID HERE
MODEL_ID = "InferenceEngineer/evidentai" 
# ---------------------

# 1. Full Screen Layout
st.set_page_config(
    page_title="Evident AI - Deepfake Detector",
    layout="wide",  # <--- This makes it full screen
    initial_sidebar_state="collapsed"
)

# 2. Custom CSS to hide uploader text and maximize width
st.markdown("""
    <style>
        /* Make the main container wider */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }
        
        /* Hides 'Drag and drop file here' */
        div[data-testid='stFileUploader'] section > div > div > span {
            display: none;
        }
        
        /* Hides 'Limit 200MB per file' */
        div[data-testid='stFileUploader'] section > div > div > small {
            display: none;
        }

        /* Optional: Style the browse button to look better */
        div[data-testid='stFileUploader'] button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Load HTML (Frontend)
try:
    with open("app.html", "r") as f:
        # We increase height to accommodate full screen visuals if needed
        components.html(f.read(), height=250, scrolling=False)
except:
    st.title("Deepfake Detection Tool")

# 4. Load Model (With the previous Fix applied)
@st.cache_resource
def load_model():
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            id2label={0: "Fake", 1: "Real"}, # Adjust if your results are swapped
            label2id={"Fake": 0, "Real": 1},
            ignore_mismatched_sizes=True 
        )
        return processor, model
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

processor, model = load_model()

# 5. UI Layout
st.write("---")

# Using columns to center the uploader slightly if it's TOO wide
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # Display logic
    image = Image.open(uploaded_file).convert("RGB")
    
    # Create two columns for Result and Image
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.image(image, use_column_width=True)
    
    with c2:
        if model:
            st.info("Scanning image structure...")
            inputs = processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_idx = logits.argmax(-1).item()
                confidence = probs[0][predicted_class_idx].item() * 100
            
            # Label Logic
            label = model.config.id2label[predicted_class_idx]
            
            # Display Result
            if "fake" in str(label).lower():
                st.error(f"🚨 DETECTED: FAKE IMAGE\nConfidence: {confidence:.2f}%")
            else:
                st.success(f"✅ VERIFIED: REAL IMAGE\nConfidence: {confidence:.2f}%")

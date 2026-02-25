import streamlit as st
from transformers import AutoModelForImageClassification, AutoImageProcessor
import torch
from PIL import Image
import streamlit.components.v1 as components

# --- CONFIGURATION ---
MODEL_ID = "InferenceEngineer/evidentai" 
# ---------------------

# 1. Full Screen Layout
st.set_page_config(
    page_title="Evident AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. FORCE FULL WIDTH CSS
# This CSS removes the default "white space" around the app
st.markdown("""
    <style>
        /* Remove padding from the main block */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
            padding-left: 1rem;
            padding-right: 1rem;
            max-width: 100%;
        }
        
        /* Remove top header line */
        header {visibility: hidden;}
        
        /* Hide 'Limit 200MB' and 'Drag and drop' text */
        div[data-testid='stFileUploader'] section > div > div > span {display: none;}
        div[data-testid='stFileUploader'] section > div > div > small {display: none;}
        
        /* Make the uploader button span full width if needed */
        div[data-testid='stFileUploader'] button {
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)

# 3. Load HTML (Frontend) - Full Width
try:
    with open("app.html", "r") as f:
        html_content = f.read()
        # width=None makes it fill the container
        components.html(html_content, height=300, scrolling=False)
except:
    st.title("Deepfake Detection Tool")

# 4. Load Model (Cached)
@st.cache_resource
def load_model():
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(
            MODEL_ID,
            num_labels=2,
            id2label={0: "Fake", 1: "Real"}, 
            label2id={"Fake": 0, "Real": 1},
            ignore_mismatched_sizes=True 
        )
        return processor, model
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

processor, model = load_model()

# 5. File Uploader - DIRECTLY on the page (No Columns)
# This will make it stretch across the whole screen
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Use columns ONLY for the result display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, use_column_width=True)
    
    with col2:
        if model:
            st.info("Analyzing...")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_idx = logits.argmax(-1).item()
                confidence = probs[0][predicted_class_idx].item() * 100
            
            label = model.config.id2label[predicted_class_idx]
            
            if "fake" in str(label).lower():
                st.error(f"🚨 FAKE IMAGE DETECTED ({confidence:.1f}%)")
            else:
                st.success(f"✅ REAL IMAGE VERIFIED ({confidence:.1f}%)")

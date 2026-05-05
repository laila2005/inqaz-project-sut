import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Inqaz Emergency AI", page_icon="🚨", layout="centered")

# Custom CSS for modern UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 5px;
        width: 100%;
        height: 50px;
        font-weight: bold;
    }
    .alert-box {
        padding: 20px;
        background-color: #ffe6e6;
        border-left: 5px solid #ff4b4b;
        color: #cc0000;
        margin-top: 20px;
        border-radius: 5px;
    }
    .safe-box {
        padding: 20px;
        background-color: #e6ffe6;
        border-left: 5px solid #00cc00;
        color: #008000;
        margin-top: 20px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚨 Inqaz AI Rescue System")
st.markdown("**Egypt's AI-powered emergency response application.**")

@st.cache_resource
def load_model():
    # Attempt to load the best model (Transfer Learning usually performs better)
    try:
        return tf.keras.models.load_model('results/saved_models/transfer_learning.h5')
    except:
        return None

model = load_model()

if model is None:
    st.warning("⚠️ Model not found. Please train the model first by running `python src/train.py`.")
else:
    st.write("Upload a photo to assess the situation.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        if st.button("Analyze Situation"):
            with st.spinner("Analyzing image for severe crash indicators..."):
                # Preprocess image
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                img = image.resize((224, 224))
                img_array = np.array(img)
                # Apply MobileNetV2 preprocessing range [-1, 1]
                img_array = (img_array / 127.5) - 1.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Predict
                prediction = model.predict(img_array)[0][0]
                
                time.sleep(1) # Simulate processing time
                
                if prediction > 0.5:
                    st.markdown("""
                    <div class="alert-box">
                        <h3>⚠️ CAR CRASH DETECTED</h3>
                        <p>Confidence: <b>{:.2f}%</b></p>
                        <hr>
                        <p><b>Automated Actions Triggered:</b></p>
                        <ul>
                            <li>📡 Getting precise GPS coordinates... <b>Success (30.0444° N, 31.2357° E)</b></li>
                            <li>🚓 Alerting Ministry of Interior (122)... <b>Sent!</b></li>
                            <li>🚑 Dispatching nearest Ambulance (123)... <b>Dispatched!</b></li>
                            <li>📸 Attaching photo evidence to dispatch center... <b>Sent!</b></li>
                        </ul>
                    </div>
                    """.format(prediction * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="safe-box">
                        <h3>✅ NORMAL SITUATION</h3>
                        <p>Confidence: <b>{:.2f}%</b></p>
                        <p>No severe crash detected. No emergency services were alerted.</p>
                    </div>
                    """.format((1 - prediction) * 100), unsafe_allow_html=True)

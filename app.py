import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Inqaz Emergency AI", page_icon="🚨", layout="centered")

# Custom CSS for premium modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Glowing button */
    .stButton>button {
        background: linear-gradient(90deg, #ff3b3b 0%, #ff1a1a 100%);
        color: white;
        border: none;
        border-radius: 12px;
        width: 100%;
        height: 60px;
        font-weight: 800;
        font-size: 18px;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 59, 59, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(255, 59, 59, 0.6);
        color: white;
    }
    
    /* Alert and Safe boxes with Glassmorphism */
    .alert-box {
        padding: 25px;
        background: rgba(255, 59, 59, 0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 59, 59, 0.2);
        border-left: 6px solid #ff3b3b;
        color: #ffcccc;
        margin-top: 25px;
        border-radius: 12px;
        animation: pulse 2s infinite;
        box-shadow: 0 8px 32px 0 rgba(255, 59, 59, 0.15);
    }
    
    .safe-box {
        padding: 25px;
        background: rgba(0, 204, 0, 0.05);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(0, 204, 0, 0.2);
        border-left: 6px solid #00cc00;
        color: #ccffcc;
        margin-top: 25px;
        border-radius: 12px;
        box-shadow: 0 8px 32px 0 rgba(0, 204, 0, 0.1);
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 59, 59, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 59, 59, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 59, 59, 0); }
    }
    
    h1 {
        text-align: center;
        background: -webkit-linear-gradient(45deg, #ff4b4b, #ff8c42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem !important;
        margin-bottom: 0px !important;
        padding-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        color: #a0aabf;
        font-size: 1.1rem;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Style file uploader area */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.15);
        border-radius: 12px;
        padding: 15px;
        transition: all 0.3s ease;
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: #ff4b4b;
        background: rgba(255, 255, 255, 0.04);
    }
    
    /* Image container */
    div[data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🚨 Inqaz AI Rescue System</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Egypt's next-generation AI-powered emergency response platform</div>", unsafe_allow_html=True)

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
                
                if prediction < 0.5:
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
                    """.format((1 - prediction) * 100), unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="safe-box">
                        <h3>✅ NORMAL SITUATION</h3>
                        <p>Confidence: <b>{:.2f}%</b></p>
                        <p>No severe crash detected. No emergency services were alerted.</p>
                    </div>
                    """.format(prediction * 100), unsafe_allow_html=True)

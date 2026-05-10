import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import cv2

# Must be the very first Streamlit command
st.set_page_config(page_title="Inqaz AI Command Center", page_icon="🚨", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------------------------------------------------
# Premium UI CSS Injection
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

    /* Global Dark Mode Enforcement and Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #050505 !important;
        color: #E2E8F0 !important;
    }
    
    /* App background override for Streamlit */
    .stApp {
        background: radial-gradient(circle at 50% -20%, #1a0b14 0%, #050505 80%) !important;
    }

    /* Hide standard Streamlit header and footer */
    header[data-testid="stHeader"], footer, #MainMenu {visibility: hidden;}

    /* Premium Header Typography */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        background: linear-gradient(90deg, #ff0044 0%, #ff6a00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: -60px;
        margin-bottom: 0px;
        letter-spacing: -1px;
    }
    
    .sub-title {
        color: #8892b0;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 30px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 1px solid #1f1f1f;
    }
    .sidebar-header {
        font-family: 'JetBrains Mono', monospace;
        color: #ff0044;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #00ff66;
        box-shadow: 0 0 10px #00ff66;
        margin-right: 8px;
    }

    /* Cards and Containers */
    .glass-card {
        background: rgba(20, 20, 20, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    
    .glow-border {
        position: relative;
    }
    .glow-border::before {
        content: "";
        position: absolute;
        top: -2px; left: -2px; right: -2px; bottom: -2px;
        background: linear-gradient(45deg, #ff0044, #ff6a00, #ff0044);
        z-index: -1;
        border-radius: 18px;
        background-size: 400%;
        animation: glowing 20s linear infinite;
        opacity: 0;
        transition: opacity 0.3s ease-in-out;
    }
    .glow-border.active::before {
        opacity: 1;
    }

    @keyframes glowing {
        0% { background-position: 0 0; }
        50% { background-position: 400% 0; }
        100% { background-position: 0 0; }
    }

    /* Analyze Button */
    .stButton>button {
        background: #ff0044;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 15px 0;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 0, 68, 0.4);
        font-family: 'JetBrains Mono', monospace;
    }
    .stButton>button:hover {
        background: #e6003d;
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 0, 68, 0.6);
    }

    /* Heatmap Legend */
    .heatmap-legend-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #8892b0;
    }
    .gradient-bar {
        height: 10px;
        width: 100%;
        margin: 0 15px;
        border-radius: 5px;
        background: linear-gradient(90deg, #000080, #00ffff, #00ff00, #ffff00, #ff0000, #800000);
    }

    /* Streamlit Metric Overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        color: #8892b0 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Result text */
    .critical-text {
        color: #ff0044;
        font-weight: 800;
        font-size: 1.5rem;
        margin-top: 10px;
    }
    .safe-text {
        color: #00ff66;
        font-weight: 800;
        font-size: 1.5rem;
        margin-top: 10px;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-header'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    st.markdown("<p><span class='status-indicator'></span><b>AI CORE:</b> ONLINE</p>", unsafe_allow_html=True)
    st.markdown("<p style='color: #8892b0; font-family: \"JetBrains Mono\"; font-size: 0.85rem;'>Model: MobileNetV2 Base<br>Architecture: Transfer Learning<br>Version: 2.1.0-prod</p>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='sidebar-header'>DATA INGESTION</div>", unsafe_allow_html=True)
    
    input_mode = st.radio("Select input source:", ["Upload Image", "Live Camera Feed"])
    
    st.markdown("---")
    st.markdown("<p style='color: #4a5568; font-size: 0.8rem; text-align: center;'>Ministry of Interior - Automated Traffic Incident Response System</p>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>INQAZ AI ENGINE</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Advanced Vehicle Damage Assessment & Triage System</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Core Functions
# -----------------------------------------------------------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model('results/saved_models/transfer_learning.h5')
    except:
        return None

def make_gradcam_heatmap(img_array, full_model, last_conv_layer_name="out_relu"):
    base_model = full_model.layers[0]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    
    grad_model = tf.keras.models.Model(
        [base_model.inputs], [last_conv_layer.output, base_model.output]
    )
    
    head_model_input = tf.keras.Input(shape=base_model.output.shape[1:])
    x = head_model_input
    for layer in full_model.layers[1:]:
        x = layer(x)
    head_model = tf.keras.Model(head_model_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, base_output = grad_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = head_model(base_output)
        class_channel = preds[:, 0]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.5):
    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    
    img = img.resize((224, 224))
    img_array = np.array(img)
    
    jet = cv2.resize(jet, (img_array.shape[1], img_array.shape[0]))
    
    superimposed_img = jet * alpha + img_array * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    return Image.fromarray(superimposed_img)

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
model = load_model()

if model is None:
    st.error("SYSTEM FAILURE: Neural Network weights not found in `results/saved_models/transfer_learning.h5`.")
    st.stop()

# Two-column layout for modern dashboard feel
col_input, col_results = st.columns([1, 1.2], gap="large")

with col_input:
    st.markdown("### 1. Visual Telemetry Feed")
    
    if input_mode == "Live Camera Feed":
        image_source = st.camera_input("Initialize Camera")
    else:
        image_source = st.file_uploader("Upload Scene Evidence", type=["jpg", "jpeg", "png"])
    
    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption='RAW INPUT ACQUIRED', use_container_width=True)
        
        analyze_clicked = st.button("EXECUTE AI ANALYSIS", use_container_width=True)

with col_results:
    st.markdown("### 2. Neural Network Output")
    
    if image_source is None:
        st.info("Awaiting visual data in the telemetry feed. Please upload an image or start the camera.")
    elif 'analyze_clicked' in locals() and analyze_clicked:
        
        # --- LOADING ANIMATION SEQUENCE ---
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.markdown("<p style='font-family:\"JetBrains Mono\"; color:#8892b0;'>[1/3] Standardizing resolution to 224x224...</p>", unsafe_allow_html=True)
        time.sleep(0.4)
        progress_bar.progress(30)
        
        status_text.markdown("<p style='font-family:\"JetBrains Mono\"; color:#8892b0;'>[2/3] Extracting Deep Spatial Features...</p>", unsafe_allow_html=True)
        time.sleep(0.5)
        progress_bar.progress(65)
        
        status_text.markdown("<p style='font-family:\"JetBrains Mono\"; color:#8892b0;'>[3/3] Generating Explainable AI Heatmap...</p>", unsafe_allow_html=True)
        
        # --- PREPROCESSING & PREDICTION ---
        if image.mode != 'RGB':
            image = image.convert('RGB')
        img_resized = image.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = (img_array / 127.5) - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)[0][0]
        
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status_text.empty()
        
        # --- DISPLAY RESULTS ---
        st.markdown("<div class='glass-card glow-border active'>", unsafe_allow_html=True)
        
        if prediction < 0.5:
            confidence = (1 - prediction) * 100
            st.markdown("<div class='critical-text'>CRITICAL DAMAGE DETECTED</div>", unsafe_allow_html=True)
            st.markdown("Automated emergency protocols have been flagged for activation.")
            
            # Metrics Row
            m1, m2 = st.columns(2)
            m1.metric("Crash Probability", f"{confidence:.2f}%", "- SEVERE")
            m2.metric("Status", "DISPATCH REQ", "- EMS ALERT")
            
            st.markdown("---")
            st.markdown("#### Explainable AI (Grad-CAM)")
            st.markdown("<p style='color: #8892b0; font-size: 0.9rem;'>The heatmap visualizes the neural network's attention mechanism. It isolates the structural deformation patterns that led to the crash classification.</p>", unsafe_allow_html=True)
            
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                cam_image = display_gradcam(image, heatmap)
                st.image(cam_image, use_container_width=True)
                
                # Custom Legend
                st.markdown("""
                <div class='heatmap-legend-container'>
                    <span>LOW ATTENTION (Background)</span>
                    <div class='gradient-bar'></div>
                    <span style='color: #ff0044; font-weight: bold;'>HIGH ATTENTION (Damage Focus)</span>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"XAI Module Error: {e}")
                
        else:
            confidence = prediction * 100
            st.markdown("<div class='safe-text'>SCENE CLEAR</div>", unsafe_allow_html=True)
            st.markdown("Visual signature indicates nominal traffic conditions.")
            
            # Metrics Row
            m1, m2 = st.columns(2)
            m1.metric("Clear Probability", f"{confidence:.2f}%", "+ SAFE")
            m2.metric("Status", "STANDBY", "+ NORMAL")

        st.markdown("</div>", unsafe_allow_html=True)

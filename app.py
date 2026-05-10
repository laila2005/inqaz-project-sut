import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time
import cv2

st.set_page_config(page_title="Inqaz AI Command Center", page_icon="🚨", layout="centered", initial_sidebar_state="collapsed")

# -----------------------------------------------------------------------------
# Premium UI CSS Injection
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap');

    /* Global Dark Mode Enforcement and Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #0B0E14 !important;
        color: #E2E8F0 !important;
    }
    
    /* App background override for Streamlit */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1A1F2C 0%, #0B0E14 100%) !important;
    }

    /* Hide standard Streamlit header and footer */
    header[data-testid="stHeader"], footer, #MainMenu {visibility: hidden;}

    /* Premium Header Typography */
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        text-align: center;
        background: linear-gradient(135deg, #FF3366, #FF9933);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem !important;
        font-weight: 800;
        margin-top: -50px !important;
        margin-bottom: 5px !important;
        letter-spacing: -1px;
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        margin-bottom: 40px;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* Container Styling for Uploaders & Interactive Elements */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(255, 51, 102, 0.5);
        background: rgba(255, 51, 102, 0.03);
        box-shadow: 0 10px 30px rgba(255, 51, 102, 0.15);
        transform: translateY(-2px);
    }

    /* Premium Glowing Button */
    .stButton>button {
        background: linear-gradient(135deg, #FF3366 0%, #E60039 100%);
        color: white;
        border: none;
        border-radius: 12px;
        width: 100%;
        height: 60px;
        font-weight: 600;
        font-size: 1.1rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(255, 51, 102, 0.3);
        font-family: 'Space Grotesk', sans-serif;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(255, 51, 102, 0.6);
        color: white;
    }
    
    .stButton>button:active {
        transform: translateY(1px);
    }

    /* Glassmorphism Results Cards */
    .result-card {
        padding: 30px;
        border-radius: 20px;
        margin-top: 30px;
        animation: slideUp 0.6s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }

    .alert-card {
        background: linear-gradient(180deg, rgba(255, 26, 26, 0.1) 0%, rgba(204, 0, 0, 0.05) 100%);
        border: 1px solid rgba(255, 77, 77, 0.3);
        box-shadow: 0 20px 40px rgba(255, 26, 26, 0.15), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-left: 6px solid #FF3366;
    }

    .safe-card {
        background: linear-gradient(180deg, rgba(0, 204, 102, 0.1) 0%, rgba(0, 153, 77, 0.05) 100%);
        border: 1px solid rgba(0, 255, 128, 0.3);
        box-shadow: 0 20px 40px rgba(0, 204, 102, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-left: 6px solid #00E676;
    }

    .result-card h3 {
        font-family: 'Space Grotesk', sans-serif;
        margin-top: 0;
        font-size: 1.8rem;
        letter-spacing: -0.5px;
    }

    .alert-card h3 { color: #FF6680; }
    .safe-card h3 { color: #33FF99; }

    .result-card p, .result-card ul {
        color: #CBD5E1;
        font-size: 1.05rem;
        line-height: 1.6;
    }

    .result-card hr {
        border-color: rgba(255, 255, 255, 0.1);
        margin: 20px 0;
    }

    /* Image Styling */
    div[data-testid="stImage"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 20px 0;
    }

    @keyframes slideUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Custom Checkbox */
    .stCheckbox label {
        color: #94A3B8 !important;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Pulse animation for severe alert */
    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.4); }
        70% { box-shadow: 0 0 0 15px rgba(255, 51, 102, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0); }
    }
    .pulse-alert {
        animation: pulse-border 2s infinite;
    }

</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Application Header
# -----------------------------------------------------------------------------
st.markdown("<h1>INQAZ.AI</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ministry of Interior • Advanced Threat Assessment Matrix</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Model Loading & Processing Functions
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
# Main Application Logic
# -----------------------------------------------------------------------------
model = load_model()

if model is None:
    st.error("⚠️ Neural Network offline. Please compile the model first by running `python src/train.py`.")
else:
    # Interaction controls
    col1, col2 = st.columns([3, 1])
    with col2:
        use_camera = st.checkbox("🟢 Live Feed", help="Toggle hardware camera input")
    
    if use_camera:
        image_source = st.camera_input("Establish Visual Link")
    else:
        image_source = st.file_uploader("Ingest Visual Data", type=["jpg", "jpeg", "png"])
    
    if image_source is not None:
        image = Image.open(image_source)
        st.image(image, caption='SOURCE FEED SECURED', use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Initialize Threat Analysis"):
            with st.spinner("Processing visual telemetry through Deep Neural Network..."):
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
                time.sleep(1.2) # Dramatic processing pause
                
                if prediction < 0.5:
                    st.markdown("""
                    <div class="result-card alert-card pulse-alert">
                        <h3>CRITICAL INCIDENT DETECTED</h3>
                        <p>Confidence Matrix: <b>{:.2f}%</b> Match for Severe Collision</p>
                        <hr>
                        <p><b>Automated Dispatch Protocols Engaged:</b></p>
                        <ul>
                            <li>🚓 <b>Traffic Division:</b> Alert routed to nearest patrol units.</li>
                            <li>🚑 <b>Medical Response:</b> EMS units (123) deployed to sector.</li>
                            <li>📸 <b>Visual Intel:</b> Forensic analysis distributed to field teams.</li>
                        </ul>
                    </div>
                    """.format((1 - prediction) * 100), unsafe_allow_html=True)
                    
                    st.markdown("<br><h4 style='color:#94A3B8; font-family:\"Space Grotesk\", sans-serif;'>Forensic XAI Mapping</h4>", unsafe_allow_html=True)
                    st.write("Grad-CAM overlay isolates areas of catastrophic structural damage.")
                    
                    try:
                        heatmap = make_gradcam_heatmap(img_array, model)
                        cam_image = display_gradcam(image, heatmap)
                        st.image(cam_image, caption='Structural Damage Isolation Matrix', use_container_width=True)
                    except Exception as e:
                        st.error(f"XAI Module Error: {e}")
                else:
                    st.markdown("""
                    <div class="result-card safe-card">
                        <h3>SCENE CLEAR</h3>
                        <p>Analysis Confidence: <b>{:.2f}%</b></p>
                        <hr>
                        <p>Visual signature indicates nominal traffic conditions. No structural deformation detected. Standby for next input.</p>
                    </div>
                    """.format(prediction * 100), unsafe_allow_html=True)

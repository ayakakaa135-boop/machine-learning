import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time
import gdown
import os


@st.cache_resource
def load_engine():
    file_id = st.secrets["GOOGLE_DRIVE_FILE_ID"]
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model_from_drive.h5'

    if not os.path.exists(output):
        with st.spinner('Downloading AI Model from Cloud... Please wait'):
            gdown.download(url, output, quiet=False)
    
    try:
        # تحميل الملف الذي تم تنزيله من الدرايف
        return tf.keras.models.load_model(output)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# استدعاء الدالة (مرة واحدة فقط)
model = load_engine()

# --- 2. إعدادات الصفحة ---
st.set_page_config(
    page_title="DermAI | Precision Diagnostic Suite",
    page_icon="🧬",
    layout="wide"
)

# --- 3. تصميم الـ CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f0f2f6; }
    .report-card {
        background: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        border-top: 8px solid #1e3c72;
    }
    .stButton>button {
        width: 100%; border-radius: 12px; height: 3.5em;
        background-image: linear-gradient(to right, #1e3c72, #2a5298);
        color: white; font-weight: bold; border: none; transition: 0.3s;
    }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

# --- 4. معلومات الأمراض والبيانات النصية ---
disease_info = {
    'Actinic keratoses': "Precancerous scaly patches caused by sun damage.",
    'Basal cell carcinoma': "Slow-growing skin cancer, rarely spreads but needs treatment.",
    'Benign keratosis': "Non-cancerous skin growth common in older adults.",
    'Dermatofibroma': "Common non-cancerous skin growth, typically small and firm.",
    'Melanoma': "Dangerous form of skin cancer. Requires immediate medical attention.",
    'Nevus': "A common mole. Usually benign but monitor for changes.",
    'Vascular lesions': "Skin growths like angiomas or birthmarks."
}
classes = list(disease_info.keys())

localizations = ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen',
                 'face', 'chest', 'foot', 'unknown', 'neck', 'scalp', 'hand',
                 'ear', 'genital', 'acral']

# --- 5. واجهة المستخدم ---
with st.sidebar:
    st.markdown("# 🧬 **DermAI Suite**")
    st.divider()
    st.success("● Neural Engine: ACTIVE")
    st.info("● Model Source: Google Drive")

st.title("DermAI | Hybrid Diagnostic Interface")

tab1, tab2 = st.tabs(["🚀 Diagnostic Terminal", "📖 Disease Library"])

with tab1:
    col1, col2 = st.columns([1, 1.5], gap="large")

    with col1:
        st.markdown("### **1. Patient Demographics**")
        age = st.slider("Patient Age", 0, 100, 30)
        sex = st.selectbox("Biological Sex", ["male", "female"])
        loc = st.selectbox("Lesion Location", localizations)

        st.markdown("### **2. Image Acquisition**")
        file = st.file_uploader("Upload Macro Image", type=['jpg', 'jpeg', 'png'])

        if file:
            img = Image.open(file)
            st.image(img, use_container_width=True, caption="Target Scan")

    with col2:
        st.markdown("### **3. AI Processing & Report**")
        if file and model is not None:
            if st.button("EXECUTE ANALYSIS"):
                progress_bar = st.progress(0, "Analyzing cellular structures...")
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                meta_input = np.zeros((1, 19))
                meta_input[0, 0] = age / 100.0 

                if sex == "male": meta_input[0, 1] = 1
                else: meta_input[0, 2] = 1

                if loc in localizations:
                    loc_idx = localizations.index(loc) + 3
                    if loc_idx < 19: meta_input[0, loc_idx] = 1

                preds = model.predict([img_array, meta_input])[0]
                idx = np.argmax(preds)
                confidence = preds[idx] * 100

                st.markdown(f"""
                <div class="report-card">
                    <h4 style="margin:0; color:#666;">Clinical Finding:</h4>
                    <h1 style="color:#1e3c72; margin-bottom:10px;">{classes[idx]}</h1>
                    <p style="font-size:1.2rem;"><b>Confidence Score:</b> {confidence:.2f}%</p>
                    <hr>
                    <p style="color:#444;">{disease_info[classes[idx]]}</p>
                </div>
                """, unsafe_allow_html=True)

                fig = go.Figure(go.Bar(x=preds * 100, y=classes, orientation='h', marker=dict(color='#1e3c72')))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Medical Knowledge Base")
    for d, info in disease_info.items():
        with st.expander(f"Learn about {d}"):
            st.write(info)

st.divider()
st.caption("© 2026 DermAI Medical Systems | For Educational Use Only")


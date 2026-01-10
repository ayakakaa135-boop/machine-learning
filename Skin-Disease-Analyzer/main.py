import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time

# 1. Advanced Page Configuration
st.set_page_config(
    page_title="DermAI | Precision Diagnostic Suite",
    page_icon="🧬",
    layout="wide"
)

# 2. Ultra-Professional Custom CSS
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


# 3. Core Engine & Metadata Setup
@st.cache_resource
def load_engine():
    # Loading your saved model
    return tf.keras.models.load_model('skin_disease_best_model.h5')


model = load_engine()

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

# List of localizations exactly as used in your training
localizations = ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen',
                 'face', 'chest', 'foot', 'unknown', 'neck', 'scalp', 'hand',
                 'ear', 'genital', 'acral']

# 4. Sidebar Branding
with st.sidebar:
    st.markdown("# 🧬 **DermAI Suite**")
    st.divider()
    st.markdown("### **System Diagnostics**")
    st.success("● Neural Engine: ACTIVE")
    st.info("● Class Mapping: 19-Feature Vector")
    st.divider()
    st.caption("This system uses a multi-input CNN + Meta architecture.")

# 5. Main Application Logic
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
        if file:
            if st.button("EXECUTE ANALYSIS"):
                # Simulation UX
                progress_bar = st.progress(0, "Analyzing cellular structures...")
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # --- PREPROCESSING (The Secret Sauce) ---
                # A) Image Preprocessing
                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                # B) Meta Preprocessing (Mapping the 19 columns)
                # Structure: [Age, Sex_Male, Sex_Female, Loc1, Loc2, ... Loc15]
                meta_input = np.zeros((1, 19))
                meta_input[0, 0] = age / 100.0  # Normalized Age

                # Sex Mapping
                if sex == "male":
                    meta_input[0, 1] = 1
                else:
                    meta_input[0, 2] = 1

                # Location Mapping (Starting from index 3 to 18)
                if loc in localizations:
                    loc_idx = localizations.index(loc) + 3
                    if loc_idx < 19:
                        meta_input[0, loc_idx] = 1

                # C) Model Execution
                preds = model.predict([img_array, meta_input])[0]
                idx = np.argmax(preds)
                confidence = preds[idx] * 100

                # D) Visual Report Card
                st.markdown(f"""
                <div class="report-card">
                    <h4 style="margin:0; color:#666;">Clinical Finding:</h4>
                    <h1 style="color:#1e3c72; margin-bottom:10px;">{classes[idx]}</h1>
                    <p style="font-size:1.2rem;"><b>Confidence Score:</b> {confidence:.2f}%</p>
                    <hr>
                    <p style="color:#444;">{disease_info[classes[idx]]}</p>
                </div>
                """, unsafe_allow_html=True)

                # Plotly Chart
                fig = go.Figure(go.Bar(
                    x=preds * 100, y=classes, orientation='h',
                    marker=dict(color='#1e3c72')
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=20, b=0), xaxis_title="Confidence %")
                st.plotly_chart(fig, use_container_width=True)

                if classes[idx] == 'Melanoma' and confidence > 50:
                    st.error("🚨 **High Alert:** Potential malignancy detected. Biopsy is mandatory.")

with tab2:
    st.header("Medical Knowledge Base")
    for d, info in disease_info.items():
        with st.expander(f"Learn about {d}"):
            st.write(info)

st.divider()
st.caption("© 2026 DermAI Medical Systems | For Educational Use Only")
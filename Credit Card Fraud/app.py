import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gdown  
import os


st.set_page_config(page_title="Fraud Detector Pro", layout="wide")


DRIVE_URL = 'https://drive.google.com/uc?id=1gy-YoMoiqleY0G3Ijif39_7X8wzf-Buh'
MODEL_PATH = 'best_fraud_detector_model.pkl'

# تحميل النموذج والمقياس والبيانات
@st.cache_resource
def load_resources():
    # التحقق من وجود الملف محلياً، وإذا لم يوجد يتم تحميله من الدرايف
    if not os.path.exists(MODEL_PATH):
        with st.spinner('Downloading model from Google Drive...'):
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load('main_scaler.pkl')
    
    # تحميل عينة من البيانات
    try:
        df_sample = pd.read_csv('creditcard.csv').sample(1000)
    except FileNotFoundError:
        st.error("ملف creditcard.csv غير موجود. يرجى التأكد من رفعه لتشغيل ميزة العينة العشوائية.")
        df_sample = pd.DataFrame()
        
    return model, scaler, df_sample

model, scaler, df_sample = load_resources()

# باقي الكود الخاص بواجهة المستخدم كما هو
st.title("🛡️ Credit Card Fraud Detection System")
st.sidebar.header("Control Panel")

if st.sidebar.button("🎲 Load Random Transaction"):
    if not df_sample.empty:
        random_row = df_sample.sample(1)
        st.session_state.v17 = random_row['V17'].values[0]
        st.session_state.v14 = random_row['V14'].values[0]
        st.session_state.v12 = random_row['V12'].values[0]
        st.session_state.v10 = random_row['V10'].values[0]
        st.session_state.v16 = random_row['V16'].values[0]
        st.session_state.v3 = random_row['V3'].values[0]
        st.session_state.v7 = random_row['V7'].values[0]
        st.session_state.v11 = random_row['V11'].values[0]
        st.session_state.v4 = random_row['V4'].values[0]
        st.session_state.actual_class = random_row['Class'].values[0]

# إنشاء واجهة المدخلات
st.subheader("Transaction Features")
col1, col2, col3 = st.columns(3)

def get_val(key):
    return st.session_state.get(key, 0.0)

with col1:
    v17 = st.number_input("V17", value=get_val('v17'))
    v14 = st.number_input("V14", value=get_val('v14'))
    v12 = st.number_input("V12", value=get_val('v12'))

with col2:
    v10 = st.number_input("V10", value=get_val('v10'))
    v16 = st.number_input("V16", value=get_val('v16'))
    v3 = st.number_input("V3", value=get_val('v3'))

with col3:
    v7 = st.number_input("V7", value=get_val('v7'))
    v11 = st.number_input("V11", value=get_val('v11'))
    v4 = st.number_input("V4", value=get_val('v4'))

if st.button("🔍 Analyze Transaction"):
    input_data = np.array([[v17, v14, v12, v10, v16, v3, v7, v11, v4]])
    prediction = model.predict(input_data)
    is_fraud = prediction[0] == -1

    st.markdown("---")
    if is_fraud:
        st.error("🚨 **RESULT: POTENTIAL FRAUD DETECTED!**")
    else:
        st.success("✅ **RESULT: TRANSACTION APPEARS SECURE.**")

    if 'actual_class' in st.session_state:
        actual = "Fraud" if st.session_state.actual_class == 1 else "Normal"
        st.info(f"**Actual Ground Truth (from dataset):** {actual}")

st.sidebar.markdown("""
---
**Model Info:**
- Algorithm: Isolation Forest
- Features: Top 9 (Correlation Based)
- F1-Score: 0.62
""")

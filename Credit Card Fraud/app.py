import streamlit as st
import pandas as pd
import joblib
import numpy as np
import gdown
import os

# إعدادات الصفحة
st.set_page_config(page_title="Fraud Detector Pro", layout="wide")

# رابط النموذج من درايف (الرابط المباشر)
DRIVE_URL = 'https://drive.google.com/uc?id=1gy-YoMoiqleY0G3Ijif39_7X8wzf-Buh'
MODEL_PATH = 'main_scaler.pkl'

@st.cache_resource
def load_resources():
    # 1. تحميل النموذج من درايف إذا لم يكن موجوداً
    if not os.path.exists(MODEL_PATH):
        try:
            gdown.download(DRIVE_URL, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Error downloading model: {e}")

    model = joblib.load(MODEL_PATH)
    
    # 2. تحميل المقياس (يجب أن ترفعه مع الكود على جيت هوب لأنه صغير)
    scaler = None
    if os.path.exists('main_scaler.pkl'):
        scaler = joblib.load('main_scaler.pkl')
    
    # 3. محاولة تحميل عينة بيانات صغيرة (اختياري)
    df_sample = None
    # حاول تحميل الملف سواء كان في المجلد الرئيسي أو الفرعي
    paths_to_check = ['creditcard_sample.csv', 'Credit Card Fraud/creditcard_sample.csv']
    for p in paths_to_check:
        if os.path.exists(p):
            df_sample = pd.read_csv(p)
            break
            
    return model, scaler, df_sample

model, scaler, df_sample = load_resources()

st.title("🛡️ Credit Card Fraud Detection System")

# القائمة الجانبية
st.sidebar.header("Control Panel")

if df_sample is not None:
    if st.sidebar.button("🎲 Load Random Sample"):
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
        if 'Class' in random_row:
            st.session_state.actual_class = random_row['Class'].values[0]
else:
    st.sidebar.warning("Sample data not found. Manual input only.")

# واجهة المدخلات
st.subheader("Transaction Features")
col1, col2, col3 = st.columns(3)

def get_val(key): return st.session_state.get(key, 0.0)

with col1:
    v17 = st.number_input("V17", value=get_val('v17'), format="%.4f")
    v14 = st.number_input("V14", value=get_val('v14'), format="%.4f")
    v12 = st.number_input("V12", value=get_val('v12'), format="%.4f")
with col2:
    v10 = st.number_input("V10", value=get_val('v10'), format="%.4f")
    v16 = st.number_input("V16", value=get_val('v16'), format="%.4f")
    v3 = st.number_input("V3", value=get_val('v3'), format="%.4f")
with col3:
    v7 = st.number_input("V7", value=get_val('v7'), format="%.4f")
    v11 = st.number_input("V11", value=get_val('v11'), format="%.4f")
    v4 = st.number_input("V4", value=get_val('v4'), format="%.4f")

if st.button("🔍 Analyze Transaction"):
    # تجهيز البيانات
    input_data = np.array([[v17, v14, v12, v10, v16, v3, v7, v11, v4]])
    
    # التوقع
    prediction = model.predict(input_data)
    is_fraud = prediction[0] == -1
    
    st.markdown("---")
    if is_fraud:
        st.error("🚨 **RESULT: POTENTIAL FRAUD DETECTED!**")
    else:
        st.success("✅ **RESULT: TRANSACTION APPEARS SECURE.**")

    if 'actual_class' in st.session_state:
        actual = "Fraud" if st.session_state.actual_class == 1 else "Normal"
        st.info(f"Actual Label in Data: **{actual}**")

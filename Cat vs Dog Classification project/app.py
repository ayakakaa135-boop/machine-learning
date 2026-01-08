import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


st.title("🐱 تصنيف القطط والكلاب باستخدام الذكاء الاصطناعي 🐶")
st.write("قم برفع صورة لتعرف هل هي قطة أم كلب!")



@st.cache_resource  # لتسريع التطبيق وعدم تحميل النموذج في كل مرة
def load_my_model():
    return tf.keras.models.load_model('best_cnn_model.keras')


model = load_my_model()

# 3. أداة رفع الصور
uploaded_file = st.file_uploader("اختر صورة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة المرفوعة
    image = Image.open(uploaded_file)
    st.image(image, caption='الصورة المرفوعة', use_column_width=True)
    st.write("جاري التحليل...")

    # 4. المعالجة المسبقة (Preprocessing)
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 5. التوقع
    prediction = model.predict(img_array)

    # 6. إظهار النتيجة بشكل جذاب
    if prediction[0] > 0.5:
        st.success(f"هذا **كلب**! نسبة الثقة: {prediction[0][0] * 100:.2f}%")
    else:
        st.info(f"هذه **قطة**! نسبة الثقة: {(1 - prediction[0][0]) * 100:.2f}%")
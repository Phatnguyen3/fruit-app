# app.py
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Tải mô hình
model = load_model('best_fruit_model.h5')
# Map id -> tên
class_names = ['Apple', 'Banana', 'Orange']

# Thông tin dinh dưỡng mẫu
nutrition_info = {
    'Apple': "Calories: 52 kcal/100g, Vitamin C, Fiber",
    'Banana': "Calories: 89 kcal/100g, Potassium, Vitamin B6",
    'Orange': "Calories: 47 kcal/100g, Vitamin C, Fiber"
}

# Gợi ý món ăn mẫu
recipes = {
    'Apple': ["Apple pie", "Fruit salad", "Baked apple"],
    'Banana': ["Banana smoothie", "Banana bread", "Pancake with banana"],
    'Orange': ["Orange juice", "Orange cake", "Salad with orange"]
}

st.title("🍎🍌🍊 Fruit Classifier App")

uploaded_file = st.file_uploader("Upload an image of fruit", type=['jpg','jpeg','png'])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(100,100))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    
    st.subheader(f"✅ Prediction: {pred_class}")
    st.write(f"**Nutritional info:** {nutrition_info[pred_class]}")
    st.write(f"**Suggested recipes:**")
    for r in recipes[pred_class]:
        st.write(f"- {r}")
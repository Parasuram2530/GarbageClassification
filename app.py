import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import altair as alt
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('GarbageClassifier.h5')

model = load_model()
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Title & Description
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Classification using EfficientNetV2B2")
st.markdown("Upload an image or use your **webcam** to classify garbage into 6 categories.")

# Sidebar
with st.sidebar:
    st.header("🧠 Model Info")
    st.markdown("""
    - **Architecture**: EfficientNetV2B2  
    - **Input Size**: 224x224x3  
    - **Dataset**: Garbage Classification (Kaggle)  
    - **Accuracy**: ~90%  
    - **Classes**: cardboard, glass, metal, paper, plastic, trash
    """)

# Input method
option = st.radio("📸 Select Image Source:", ["Upload Image", "Use Webcam"])

image = None
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Use Webcam":
    camera_file = st.camera_input("Take a picture")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

# Prediction
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)
    st.markdown("### Predicting...")

    with st.spinner("Analyzing the image..."):
        resized = image.resize((224, 224))
        img_array = preprocess_input(np.array(resized))
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        class_idx = np.argmax(predictions)
        predicted_class = class_names[class_idx]
        confidence = np.max(predictions)

    st.success("Prediction Complete!")

    # Show prediction
    st.markdown(f"### 🏷️ Prediction: `{predicted_class.capitalize()}`")
    st.markdown(f"### 📊 Confidence: `{confidence:.2f}`")

    # Probability bar chart
    df = pd.DataFrame({
        'Class': class_names,
        'Probability': predictions
    })

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Probability:Q', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('Class:N', sort='-x'),
        color=alt.Color('Class:N', legend=None),
        tooltip=['Class', 'Probability']
    ).properties(height=300)

    st.altair_chart(chart, use_container_width=True)

# Footer
st.markdown("---")
st.caption("🔬 Built with EfficientNetV2B2 • By Parasuu 💚")

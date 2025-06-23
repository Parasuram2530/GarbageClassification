import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


model = tf.keras.models.load_model('GarbageClassifier.h5')

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

st.title("Garbage Classifier")
st.write("Upload the image for the garbage")
file_uploaded = st.file_uploader("Choose an Image...", type=['png', 'jpg', 'jpeg'])

if file_uploaded is not None:
    image = Image.open(file_uploaded).convert('RGB')
    st.image(image=image, caption='Uploaded Image', use_column_width=True)

    img = image.resize((224, 224))
    # img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = preprocess_input(np.array(img))
    img_array = np.expand_dims(img_array, axis=0)
    

    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    predicted_class = class_names[class_idx]
    confidence = np.max(predictions)

    st.subheader("Debug Info")
    st.write("Raw predictions:", predictions)
    st.write("Predicted class index:", class_idx)
    st.write("Class names list:", class_names)

    st.markdown(f"### Prediction: {predicted_class.capitalize()}")
    st.markdown(f"### Confidence: {confidence:.2f}")



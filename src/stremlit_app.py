import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
MODEL_PATH = "model_weights/vgg19_model_02.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define image size
IMG_SIZE = 128

# Class labels
CLASS_NAMES = ['Normal', 'Pneumonia']

def preprocess_image(img):
    """Preprocesses the uploaded image to match model input requirements."""
    img = img.convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img

def predict_pneumonia(img):
    """Makes a prediction on the uploaded image."""
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    return CLASS_NAMES[predicted_class], confidence

# Streamlit UI
st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to predict whether pneumonia is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))
    
    # Display the uploaded image
    st.image(img, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Make a prediction
    label, confidence = predict_pneumonia(img)
    
    # Display result
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}")

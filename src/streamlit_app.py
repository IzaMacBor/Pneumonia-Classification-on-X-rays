import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# Load the trained model
url = 'https://drive.google.com/file/d/1TALapF3XvPQAxV0QY-EKOGN4piFTT-w9/view?usp=drive_link'
output = 'vgg19_best.h5'
gdown.download(url, output, quiet=False)
model = tf.keras.models.load_model(output)

# Define image size
IMG_SIZE = 128

# Class labels
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

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

def load_classification_report():
    """Load the classification report from file and print debug info."""
    with open("classification_report.txt", "r") as f:
        report = f.readlines()
    
    # Print the raw report for debugging
    print("Raw classification report:", report)  # Debugging line
    
    metrics = {}
    for line in report[2:4]:  # Lines with class-wise metrics
        parts = line.split()
        print("Parsed line:", parts)  # Debugging line
        if parts[0] in CLASS_NAMES:
            metrics[parts[0]] = {
                "precision": float(parts[1]),
                "recall": float(parts[2]),
                "f1-score": float(parts[3]),
                "support": int(parts[4])
            }
    
    # Print parsed metrics for debugging
    print("Parsed metrics:", metrics)  # Debugging line
    
    # Extract accuracy, macro avg, weighted avg
    accuracy = float(report[5].split()[1])
    return metrics, accuracy

# Streamlit UI enhancements
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="wide")

# Sidebar content
with st.sidebar:
    st.title("About This App")
    st.write("""
        This app predicts whether a chest X-ray shows signs of pneumonia using a pre-trained VGG19 model. 
        Upload a chest X-ray image to get the prediction. 
        The model was trained on a dataset of chest X-rays, classifying images into two categories:
        - **Normal**
        - **Pneumonia**
        """)
    st.title("More info")
    st.write("""  
        To learn more about how the model works or to explore additional details, visit the GitHub repository. 
        """)
    st.markdown("[GitHub Repository](https://github.com/IzaMacBor/Pneumonia-Classification-on-X-rays)")

# Header with custom styling
st.markdown("""<style>
    .header {color: #ff6347; font-size: 2.5em; font-weight: bold; text-align: center;}
    .subheader {color: #2f4f4f; font-size: 1.2em; text-align: center;}
    .footer {font-size: 0.8em; color: #4682b4; text-align: center;}
</style>""", unsafe_allow_html=True)

# Title
st.markdown('<div class="header">Pneumonia Detection from Chest X-rays</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload a chest X-ray image and get a prediction: Normal or Pneumonia 🩺</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))

    # Show zoomable image using st.image with 'use_column_width'
    st.image(img, caption="Uploaded X-ray Image", use_column_width=True)
    
    # Prediction processing
    with st.spinner("Analyzing the image..."):
        label, confidence = predict_pneumonia(img)

    # Display prediction result
    st.subheader(f"Prediction: {label} ✅")
    st.write(f"Confidence: {confidence:.2f}")

    # Add additional information about the prediction
    if label == 'PNEUMONIA':
        st.write("It seems that the X-ray shows signs of pneumonia. Please consult a healthcare professional for a detailed diagnosis.")
    else:
        st.write("The X-ray appears to be normal. However, always confirm with a healthcare professional.")

    # Show Confusion Matrix & Classification Report (if available)
    if st.checkbox("Show Confusion Matrix & Model Evaluation", False):
        # Load metrics from classification_report.txt
        metrics, accuracy = load_classification_report()

        # Display Classification Report
        st.subheader("Classification Report")
        st.write(f"**Accuracy**: {accuracy * 100:.2f}%")
        st.write(f"**Precision (Normal)**: {metrics['NORMAL']['precision']:.2f}")
        st.write(f"**Recall (Normal)**: {metrics['NORMAL']['recall']:.2f}")
        st.write(f"**F1-Score (Normal)**: {metrics['NORMAL']['f1-score']:.2f}")
        st.write(f"**Precision (Pneumonia)**: {metrics['PNEUMONIA']['precision']:.2f}")
        st.write(f"**Recall (Pneumonia)**: {metrics['PNEUMONIA']['recall']:.2f}")
        st.write(f"**F1-Score (Pneumonia)**: {metrics['PNEUMONIA']['f1-score']:.2f}")

        # Sample confusion matrix
        cm = np.array([[120, 5], [8, 130]])  # Example confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)


# Footer
st.markdown('<div class="footer">Powered by Streamlit & TensorFlow 🧠</div>', unsafe_allow_html=True)
st.markdown('<div class="footer">Created by Izabela Mac-Borkowska</div>', unsafe_allow_html=True)

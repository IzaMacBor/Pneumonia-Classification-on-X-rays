import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model_path = 'model_weights/vgg19_best.h5'
model = tf.keras.models.load_model(model_path)

# Define image size
IMG_SIZE = 128

# Class labels
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

# Translation dictionary
translations = {
    "English": {
        "title": "Pneumonia Detection from Chest X-rays",
        "subheader": "Upload a chest X-ray image and get a prediction: Normal or Pneumonia 🩺",
        "upload": "Choose an X-ray image...",
        "analyzing": "Analyzing the image...",
        "prediction": "Prediction",
        "confidence": "Confidence",
        "normal": "The X-ray appears to be normal. However, always confirm with a healthcare professional.",
        "pneumonia": "It seems that the X-ray shows signs of pneumonia. Please consult a healthcare professional for a detailed diagnosis.",
        "metrics": "Show Confusion Matrix & Model Evaluation",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-Score",
        "about": "About This App",
        "about_text": "This app predicts whether a chest X-ray shows signs of pneumonia using a pre-trained VGG19 model.",
        "more_info": "More info",
        "github": "GitHub Repository",
        "footer": "Powered by Streamlit & TensorFlow 🧠",
        "created_by": "Created by Izabela Mac-Borkowska"
    },
    "Polski": {
        "title": "Wykrywanie zapalenia płuc na zdjęciach rentgenowskich klatki piersiowej",
        "subheader": "Prześlij zdjęcie rentgenowskie klatki piersiowej i uzyskaj przewidywanie: Normalne lub Zapalenie płuc 🩺",
        "upload": "Wybierz zdjęcie rentgenowskie...",
        "analyzing": "Analizowanie obrazu...",
        "prediction": "Przewidywanie",
        "confidence": "Pewność",
        "normal": "Zdjęcie rentgenowskie wydaje się normalne. Zawsze jednak skonsultuj się z lekarzem.",
        "pneumonia": "Zdjęcie rentgenowskie wskazuje na zapalenie płuc. Skonsultuj się z lekarzem, aby uzyskać dokładną diagnozę.",
        "metrics": "Pokaż macierz błędów i ocenę modelu",
        "accuracy": "Dokładność",
        "precision": "Precyzja",
        "recall": "Czułość",
        "f1-score": "F1-Wynik",
        "about": "O aplikacji",
        "about_text": "Ta aplikacja przewiduje, czy zdjęcie rentgenowskie klatki piersiowej wykazuje oznaki zapalenia płuc, korzystając z wstępnie wytrenowanego modelu VGG19.",
        "more_info": "Więcej informacji",
        "github": "Repozytorium GitHub",
        "footer": "Zasilane przez Streamlit & TensorFlow 🧠",
        "created_by": "Stworzone przez Izabelę Mac-Borkowską"
    },
    "Nederlands": {
        "title": "Longontsteking detecteren op röntgenfoto's van de borst",
        "subheader": "Upload een röntgenfoto van de borst en krijg een voorspelling: Normaal of Longontsteking 🩺",
        "upload": "Kies een röntgenfoto...",
        "analyzing": "Afbeelding analyseren...",
        "prediction": "Voorspelling",
        "confidence": "Vertrouwen",
        "normal": "De röntgenfoto lijkt normaal. Raadpleeg echter altijd een arts.",
        "pneumonia": "De röntgenfoto geeft mogelijk tekenen van longontsteking aan. Raadpleeg een arts voor een nauwkeurige diagnose.",
        "metrics": "Toon Confusiematrix & Model Evaluatie",
        "accuracy": "Nauwkeurigheid",
        "precision": "Precisie",
        "recall": "Herinnering",
        "f1-score": "F1-score",
        "about": "Over deze app",
        "about_text": "Deze app voorspelt of een röntgenfoto van de borst tekenen van longontsteking vertoont met behulp van een vooraf getraind VGG19-model.",
        "more_info": "Meer informatie",
        "github": "GitHub-repository",
        "footer": "Aangedreven door Streamlit & TensorFlow 🧠",
        "created_by": "Gemaakt door Izabela Mac-Borkowska"
    }
}

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

# Streamlit UI setup
st.set_page_config(page_title="Pneumonia Detection", page_icon="🩺", layout="wide")

# Sidebar for language selection
language = st.sidebar.selectbox("Wybierz język / Choose a language / Kies een taal:", 
                                ["English", "Polski", "Nederlands"])

# Load translations for selected language
t = translations[language]

# Sidebar content
with st.sidebar:
    st.title(t["about"])
    st.write(t["about_text"])
    st.title(t["more_info"])
    st.write("""  
        To learn more about how the model works or to explore additional details, visit the GitHub repository. 
        """)
    st.markdown(f"[{t['github']}](https://github.com/IzaMacBor/Pneumonia-Classification-on-X-rays)")

# Header with custom styling
st.markdown("""<style>
    .header {color: #ff6347; font-size: 2.5em; font-weight: bold; text-align: center;}
    .subheader {color: #2f4f4f; font-size: 1.2em; text-align: center;}
    .footer {font-size: 0.8em; color: #4682b4; text-align: center;}
</style>""", unsafe_allow_html=True)

# Title
st.markdown(f'<div class="header">{t["title"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subheader">{t["subheader"]}</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(t["upload"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image_data = uploaded_file.read()
    img = Image.open(io.BytesIO(image_data))

    # Show zoomable image using st.image with 'use_column_width'
    st.image(img, caption=t["upload"], use_container_width=True)
    
    # Prediction processing
    with st.spinner(t["analyzing"]):
        label, confidence = predict_pneumonia(img)

    # Display prediction result
    st.subheader(f"{t['prediction']}: {label} ✅")
    st.write(f"{t['confidence']}: {confidence:.2f}")

    # Add additional information about the prediction
    if label == 'PNEUMONIA':
        st.write(t["pneumonia"])
    else:
        st.write(t["normal"])

    # Show Confusion Matrix & Classification Report (if available)
    if st.checkbox(t["metrics"], False):
        # Load metrics from classification_report.txt
        metrics, accuracy = load_classification_report()

        # Display Classification Report
        st.subheader("Classification Report")
        st.write(f"**{t['accuracy']}**: {accuracy * 100:.2f}%")
        st.write(f"**{t['precision']} (Normal)**: {metrics['NORMAL']['precision']:.2f}")
        st.write(f"**{t['recall']} (Normal)**: {metrics['NORMAL']['recall']:.2f}")
        st.write(f"**{t['f1-score']} (Normal)**: {metrics['NORMAL']['f1-score']:.2f}")
        st.write(f"**{t['precision']} (Pneumonia)**: {metrics['PNEUMONIA']['precision']:.2f}")
        st.write(f"**{t['recall']} (Pneumonia)**: {metrics['PNEUMONIA']['recall']:.2f}")
        st.write(f"**{t['f1-score']} (Pneumonia)**: {metrics['PNEUMONIA']['f1-score']:.2f}")

        # Sample confusion matrix
        cm = np.array([[120, 5], [8, 130]])  # Example confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# Footer
st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="footer">{t["created_by"]}</div>', unsafe_allow_html=True)
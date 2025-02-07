import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

class ModelVisualizer:
    def __init__(self, model_path, data_path, img_size=128):
        """
        Initializes the model visualizer.
        
        Args:
            model_path (str): Path to the saved model.
            data_path (str): Path to the test dataset.
            img_size (int): Image size for model input.
        """
        self.model = load_model(model_path)
        self.data_path = data_path
        self.img_size = img_size
        
        # Configure the data generator
        self.test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = self.test_datagen.flow_from_directory(
            os.path.join(data_path, 'test'),
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        # Class names
        self.class_names = list(self.test_generator.class_indices.keys())
    
    def plot_training_history(self, history_path, save_path='training_history.png'):
        """
        Visualizes the training history.
        
        Args:
            history_path (str): Path to the training history JSON file.
            save_path (str): Path to save the plot.
        """
        with open(history_path, "r") as f:
            history = json.load(f)
        
        plt.figure(figsize=(12, 4))
        
        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def compute_predictions(self):
        """
        Computes predictions for the test set.
        
        Returns:
            tuple: Predicted labels and true labels.
        """
        self.test_generator.reset()
        predictions = self.model.predict(self.test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = self.test_generator.classes
        
        return predicted_classes, true_classes
    
    def plot_confusion_matrix(self, save_path='confusion_matrix.png'):
        """
        Plots the confusion matrix.
        
        Args:
            save_path (str): Path to save the plot.
        """
        predicted_classes, true_classes = self.compute_predictions()
        
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, 
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
    def generate_classification_report(self):
        """
        Generates a classification report.
        
        Returns:
            str: The classification report.
        """
        predicted_classes, true_classes = self.compute_predictions()
        
        report = classification_report(
            true_classes, 
            predicted_classes, 
            target_names=self.class_names
        )
        
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        
        return report
    
    def visualize_sample_predictions(self, num_samples=5, save_path='sample_predictions.png'):
        """
        Visualizes sample predictions.
        
        Args:
            num_samples (int): Number of samples to visualize.
            save_path (str): Path to save the plot.
        """
        self.test_generator.reset()
        
        # Get sample images
        x_batch, y_batch = next(self.test_generator)
        predictions = self.model.predict(x_batch)
        
        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i+1)
            plt.imshow(x_batch[i])
            true_class = self.class_names[np.argmax(y_batch[i])]
            pred_class = self.class_names[np.argmax(predictions[i])]
            plt.title(f'True: {true_class}\nPredicted: {pred_class}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

# Paths - adjust based on your environment
model_path = 'model_weights/vgg19_model_02.h5'
data_path = 'chest_xrays/chest_xray'
history_path = 'model_weights/vgg19_model_02_history.json'
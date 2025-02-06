import os
import cv2
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD
from typing import Tuple, List

class XRayDataGenerator:
    """Generates image datasets with augmentation."""
    def __init__(self, base_dir: str, img_size: int = 128, batch_size: int = 32):
        self.base_dir = base_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.setup_generators()
    
    def setup_generators(self):
        self.train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=40, horizontal_flip=True, vertical_flip=True,
            shear_range=0.2, width_shift_range=0.4, height_shift_range=0.4, fill_mode="nearest"
        )
        self.valid_datagen = ImageDataGenerator(rescale=1./255)
        self.test_datagen = ImageDataGenerator(rescale=1./255)
    
    def get_generator(self, subset: str):
        return self.train_datagen.flow_from_directory(
            os.path.join(self.base_dir, subset),
            batch_size=self.batch_size,
            target_size=(self.img_size, self.img_size),
            class_mode='categorical',
            shuffle=True,
            seed=42,
            color_mode='rgb'
        )

class VGG19Classifier:
    """Builds and trains VGG19-based classifier."""
    def __init__(self, model_name, input_shape=(128, 128, 3), weights_path=None):
        self.model_name = model_name
        self.input_shape = input_shape
        self.weights_path = weights_path
        self.model = self.build_model()
        if self.weights_path:
            self.load_weights(self.weights_path)
    
    def build_model(self):
        base_model = VGG19(include_top=False, input_shape=self.input_shape, weights="imagenet")
        for layer in base_model.layers:
            layer.trainable = False  # Domyślnie zamrażamy warstwy
        x = Flatten()(base_model.output)
        x = Dense(4608, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1152, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)
        return Model(base_model.input, output)
    
    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)
    
    def set_trainable_layers(self, trainable_layers: List[str] = None):
        """Unfreezes specific layers for fine-tuning."""
        for layer in self.model.layers:
            layer.trainable = False  # Domyślnie wszystkie zamrożone
        
        if trainable_layers:
            for layer in self.model.layers:
                if layer.name in trainable_layers:
                    layer.trainable = True
                    print(f"Unfreezing: {layer.name}")
    
    def compile_model(self, learning_rate=0.0001):
        optimizer = SGD(learning_rate=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    
    def train_model(self, train_generator, valid_generator, steps_per_epoch, epochs, callbacks):
        return self.model.fit(
            train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs,
            callbacks=callbacks, validation_data=valid_generator
        )
    
    def evaluate_model(self, valid_generator, test_generator):
        val_loss, val_acc = self.model.evaluate(valid_generator)
        test_loss, test_acc = self.model.evaluate(test_generator)
        print(f"Validation Loss: {val_loss}, Accuracy: {val_acc}")
        print(f"Test Loss: {test_loss}, Accuracy: {test_acc}")
    
    def save_model(self, save_path):
        os.makedirs("model_weights", exist_ok=True)
        self.model.save(filepath=save_path, overwrite=True)

class XRayTrainer:
    """Manages training workflow."""
    def __init__(self, base_dir: str, img_size: int = 128, batch_size: int = 32, epochs: int = 20):
        self.data_generator = XRayDataGenerator(base_dir, img_size, batch_size)
        self.epochs = epochs
        self.callbacks = [
            EarlyStopping(monitor="val_loss", patience=4, verbose=1, mode="min"),
            ModelCheckpoint("model_weights/vgg19_best.h5", monitor="val_loss", save_best_only=True),
            ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-6)
        ]
    
    def train_model_variant(self, model_name, weights_path=None, trainable_layers=None, epochs=10, steps_per_epoch=50):
        model = VGG19Classifier(model_name, weights_path=weights_path)
        
        if trainable_layers:
            model.set_trainable_layers(trainable_layers)
        
        model.compile_model()
        train_generator = self.data_generator.get_generator("train")
        valid_generator = self.data_generator.get_generator("val")
        test_generator = self.data_generator.get_generator("test")
        
        history = model.train_model(train_generator, valid_generator, steps_per_epoch, epochs, self.callbacks)
        model.save_model(f"model_weights/{model_name}.h5")
        model.evaluate_model(valid_generator, test_generator)

        history_path = f"model_weights/{model_name}_history.json"
        with open(history_path, "w") as f:
            json.dump(history.history, f)

        print(f"Training history saved to {history_path}")

        return history
    
        

# Example usage
trainer = XRayTrainer(base_dir="chest_xrays/chest_xray")

# Train initial model
history_01 = trainer.train_model_variant("vgg19_model_01", epochs=20)

# Fine-tune with last two convolutional layers unfrozen
history_02 = trainer.train_model_variant("vgg19_model_02", weights_path="model_weights/vgg19_model_01.h5", trainable_layers=["block5_conv3", "block5_conv4"], epochs=10)

# Train another version from pre-trained weights with different parameters
history_03 = trainer.train_model_variant("vgg19_model_03", weights_path="model_weights/vgg19_model_01.h5", epochs=5, steps_per_epoch=100)

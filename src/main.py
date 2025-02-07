from train import XRayTrainer
from visuals import ModelVisualizer

# Path to dataset
DATA_PATH = "chest_xrays/chest_xray"
VISUALS_PATH = "visuals"

# Ensure the visuals directory exists
import os
os.makedirs(VISUALS_PATH, exist_ok=True)

# Initialize trainer
trainer = XRayTrainer(base_dir=DATA_PATH)

# Train the initial model
print("Training initial model...")
trainer.train_model_variant("vgg19_model_01", epochs=20)

# Fine-tune the model by unfreezing certain layers
print("Fine-tuning model...")
trainer.train_model_variant(
    "vgg19_model_02",
    weights_path="model_weights/vgg19_model_01.h5",
    trainable_layers=["block5_conv3", "block5_conv4"],
    epochs=10
)

# Initialize visualizer
visualizer = ModelVisualizer(
    model_path='model_weights/vgg19_model_02.h5',
    data_path=DATA_PATH
)

# Generate and display visualizations
print("Generating visualizations...")
visualizer.plot_training_history('model_weights/vgg19_model_02_history.json', save_path=f'{VISUALS_PATH}/training_history.png')
visualizer.plot_confusion_matrix(save_path=f'{VISUALS_PATH}/confusion_matrix.png')
visualizer.visualize_sample_predictions(save_path=f'{VISUALS_PATH}/sample_predictions.png')
report = visualizer.generate_classification_report()

print("Classification Report:")
print(report)

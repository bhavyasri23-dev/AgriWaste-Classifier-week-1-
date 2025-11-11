"""
AgriWaste Classifier - Prediction Script
Load the trained model and predict on a single image.

Usage: python predict.py <image_path>
Example: python predict.py data/test/freshapples/image.jpg
"""
import os
import sys
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

# Suppress Python warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from PIL import Image

# Configuration (must match training)
IMG_SIZE = 64
MODEL_PATH = 'best_model.keras'

# Class names (from training, hardcoded for consistency)
class_names = [
    'freshapples', 'freshbanana', 'freshbittergroud', 'freshcapsicum', 'freshcucumber',
    'freshokra', 'freshoranges', 'freshpotato', 'freshtomato', 'rottenapples',
    'rottenbanana', 'rottenbittergroud', 'rottencapsicum', 'rottencucumber',
    'rottenokra', 'rottenoranges', 'rottenpotato', 'rottentomato'
]

def load_and_preprocess_image(image_path):
    """Load and preprocess a single image."""
    # Load image
    img = Image.open("C:\\Users\\BHAVYA\\PycharmProjects\\AgricultureWasteClassifier\\data\\Test\\freshbanana\\b_f001.png").convert('RGB')
    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))
    # Convert to array and rescale
    img_array = np.array(img) / 255.0
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image_path):
    """Predict the class of a single image."""
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found! Train the model first.")
        return None

    # Load model
    print(f"📁 Loading model from '{MODEL_PATH}'...")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")

    # Preprocess image
    img_array = load_and_preprocess_image(image_path)

    # Predict
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]

    return predicted_class_name, confidence

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_path}' not found!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("🔍 AGRIWASTE CLASSIFIER - PREDICTION")
    print("=" * 60)
    print(f"Image: {image_path}")
    print("=" * 60)

    result = predict_image(image_path)

    if result:
        predicted_class, confidence = result
        print(f"   Predicted Class: {predicted_class}")
        print(f"   Confidence:      {confidence * 100:.2f}%")
        print("=" * 60 + "\n")
    else:
        sys.exit(1)
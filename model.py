"""
AgriWaste Classifier - CPU OPTIMIZED VERSION
Lightweight model that trains fast on CPU (2-5 minutes per epoch)
KERAS 3 FULLY COMPATIBLE

Usage: python model.py
"""
import os
import warnings

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Changed from '2' to '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Explicitly enable (suppresses info message)

# Suppress Python warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Reduce TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CPU-OPTIMIZED Configuration
IMG_SIZE = 64  # Reduced from 128
BATCH_SIZE = 32  # Reduced from 64
EPOCHS = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

# Dataset paths
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

# Check if data directories exist
if not os.path.exists(TRAIN_DIR):
    print(f"❌ Error: Training directory '{TRAIN_DIR}' not found!")
    exit(1)

print("\n" + "=" * 60)
print("🚀 AGRIWAVE CLASSIFIER - CPU OPTIMIZED TRAINING")
print("=" * 60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")
print(f"Image Size: {IMG_SIZE}x{IMG_SIZE} (smaller = faster)")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Max Epochs: {EPOCHS}")
print("⚠️  Training on CPU - Each epoch: ~2-5 minutes")
print("=" * 60 + "\n")

# Keras 3 compatible data loading using tf.keras.utils.image_dataset_from_directory
print("📂 Loading dataset...")

# Load training and validation data
train_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Get class names from training data
class_names = train_ds.class_names
num_classes = len(class_names)

# Load test data without specifying class_names to avoid mismatch
test_ds = keras.utils.image_dataset_from_directory(
    TEST_DIR,
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
)

test_class_names = test_ds.class_names
test_num_classes = len(test_class_names)

# Define corrections for misspelled test class names to match training
corrections = {
    'freshpatato': 'freshpotato',
    'freshtamto': 'freshtomato',
    'rottenpatato': 'rottenpotato',
    'rottentamto': 'rottentomato',
    # Add more if needed based on your directory names
}

# Map test classes to training indices
test_to_train_index = {}
for test_class in test_class_names:
    corrected_class = corrections.get(test_class, test_class)
    if corrected_class in class_names:
        test_to_train_index[test_class] = class_names.index(corrected_class)
    else:
        print(f"⚠️ Warning: Test class '{test_class}' (corrected to '{corrected_class}') not found in training classes. It will be skipped in evaluation.")

print(f"Test classes mapped: {test_to_train_index}")

# Create class indices mapping (for compatibility with your other scripts)
class_indices = {name: idx for idx, name in enumerate(class_names)}

print("\n📊 DATASET INFORMATION:")
print(f"   Number of Classes (Training): {num_classes}")
print(f"   Class Names (Training): {class_names}")
print(f"   Class Mapping: {class_indices}")
print(f"   Number of Classes (Test): {test_num_classes}")
print(f"   Test Class Names: {test_class_names}")

# Calculate dataset sizes
train_size = tf.data.experimental.cardinality(train_ds).numpy() * BATCH_SIZE
val_size = tf.data.experimental.cardinality(val_ds).numpy() * BATCH_SIZE
test_size = tf.data.experimental.cardinality(test_ds).numpy() * BATCH_SIZE

print(f"   Training Samples: ~{train_size}")
print(f"   Validation Samples: ~{val_size}")
print(f"   Test Samples: ~{test_size}\n")

# Data augmentation layer (Keras 3 style)
data_augmentation = keras.Sequential([
    layers.Rescaling(1. / 255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Normalization only (for validation and test)
normalization = keras.Sequential([
    layers.Rescaling(1. / 255)
])

# Apply augmentation to training data
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (normalization(x, training=False), y))
test_ds = test_ds.map(lambda x, y: (normalization(x, training=False), y))

# Remap test labels to match training classes (18 classes)
def remap_labels(x, y):
    # y is shape (batch, test_num_classes), one-hot
    new_y = tf.zeros((tf.shape(y)[0], num_classes), dtype=y.dtype)
    for i in range(test_num_classes):
        test_class = test_class_names[i]
        if test_class in test_to_train_index:
            train_idx = test_to_train_index[test_class]
            mask = y[:, i] == 1
            one_hot_train = tf.one_hot(train_idx, num_classes, dtype=y.dtype)
            new_y = tf.where(tf.expand_dims(mask, -1), one_hot_train, new_y)
    return x, new_y

test_ds = test_ds.map(remap_labels)

# Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Define the lightweight CNN model (CPU-optimized)
model = keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Check if model is already trained (saved)
model_path = 'best_model.keras'
if os.path.exists(model_path):
    print(f"📁 Loading pre-trained model from '{model_path}'...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully. Skipping training.\n")
else:
    print("🔄 No pre-trained model found. Starting training...\n")
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)
    ]

    # KERAS 3 COMPATIBLE - Clean fit call without workers/multiprocessing
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

# Evaluate on test set
print("\n" + "=" * 60)
print("📊 FINAL EVALUATION ON TEST SET")
print("=" * 60)

test_results = model.evaluate(test_ds, verbose=1)

print(f"\n{'=' * 60}")
print("🎉 FINAL TEST RESULTS")
print(f"{'=' * 60}")
print(f"   Test Accuracy:  {test_results[1] * 100:.2f}%")
print(f"   Test Loss:      {test_results[0]:.4f}")
print(f"{'=' * 60}\n")

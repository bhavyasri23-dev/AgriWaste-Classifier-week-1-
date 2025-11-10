# AgriWaste-Classifier: Automated Detection of Spoiled Agricultural Produce
# Complete implementation for Google Colab

# ============================================================================
# SECTION 1: INSTALLATION & IMPORTS
# ============================================================================

# Install required packages

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")

# ============================================================================
# SECTION 2: DATA PREPARATION
# ==========================================




# Set dataset paths - MODIFY THESE PATHS according to your folder structure # Change this path

# Alternative: If uploading directly
# from google.colab import files
# uploaded = files.upload()

# Configuration
IMG_SIZE = 96
BATCH_SIZE = 82
EPOCHS = 15

LEARNING_RATE = 0.001
CLASS_NAMES = ['fresh', 'rotten']  # Update if you have borderline class


# ============================================================================
# SECTION 3: DATA LOADING & AUGMENTATION
# ============================================================================
import zipfile
import os

zip_path = "data.zip"
extract_dir = "data"

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    z.extractall(extract_dir)

print("Dataset extracted to:", extract_dir)

def create_data_generators():
    """Create data generators with augmentation"""

    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    # Test data (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        extract_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # Load validation data
    val_generator = train_datagen.flow_from_directory(
        extract_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        extract_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator


# Create generators
print("\n📊 Loading and preparing data...")
train_gen, val_gen, test_gen = create_data_generators()

print(f"\n✅ Data loaded successfully!")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")
print(f"Test samples: {test_gen.samples}")
print(f"Classes: {train_gen.class_indices}")


# ============================================================================
# SECTION 4: MODEL ARCHITECTURE
# ============================================================================

def create_efficientnet_model(num_classes=2):
    """Create EfficientNetB0-based model"""

    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_resnet_model(num_classes=2):
    """Create ResNet50-based model"""

    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_custom_cnn(num_classes=2):
    """Create custom CNN model"""

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


# Select model (choose one)
print("\n🧠 Creating model...")
MODEL_TYPE = 'efficientnet'  # Options: 'efficientnet', 'resnet', 'custom'

if MODEL_TYPE == 'efficientnet':
    model = create_efficientnet_model(len(CLASS_NAMES))
    print("✅ EfficientNetB0 model created")
elif MODEL_TYPE == 'resnet':
    model = create_resnet_model(len(CLASS_NAMES))
    print("✅ ResNet50 model created")
else:
    model = create_custom_cnn(len(CLASS_NAMES))
    print("✅ Custom CNN model created")

# Compile model
model.compile(

    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)




# Display model summary
print("\n📋 Model Architecture:")
model.summary()

# ============================================================================
# SECTION 5: MODEL TRAINING
# ============================================================================

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Train model
print("\n🚀 Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training completed!")

# ============================================================================
# SECTION 6: MODEL EVALUATION
# ============================================================================

print("\n📊 Evaluating model on test data...")

# Evaluate on test set
test_loss, test_acc, test_precision, test_recall = model.evaluate(test_gen)

print(f"\n🎯 Test Results:")
print(f"  Accuracy:  {test_acc:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1-Score:  {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")

# Generate predictions
print("\n🔮 Generating predictions...")
test_gen.reset()
predictions = model.predict(test_gen, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes

# Classification report
print("\n📈 Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))


# ============================================================================
# SECTION 7: VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation metrics"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_sample_predictions(generator, model, num_samples=9):
    """Plot sample predictions"""

    generator.reset()
    images, labels = next(generator)
    predictions = model.predict(images[:num_samples])

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.ravel()

    for i in range(num_samples):
        axes[i].imshow(images[i])
        true_label = CLASS_NAMES[np.argmax(labels[i])]
        pred_label = CLASS_NAMES[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100

        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                          color=color, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()


# Generate all visualizations
print("\n📊 Generating visualizations...")
plot_training_history(history)
plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)
plot_sample_predictions(test_gen, model)


# ============================================================================
# SECTION 8: ANALYTICS & STATISTICS
# ============================================================================

def generate_analytics(y_true, y_pred, classes):
    """Generate comprehensive analytics"""

    # Calculate statistics
    total_samples = len(y_true)
    fresh_count = np.sum(y_true == 0)
    rotten_count = np.sum(y_true == 1)

    fresh_pred_count = np.sum(y_pred == 0)
    rotten_pred_count = np.sum(y_pred == 1)

    spoilage_rate = (rotten_count / total_samples) * 100

    # Create analytics dataframe
    analytics_data = {
        'Metric': [
            'Total Samples',
            'Fresh Samples',
            'Rotten Samples',
            'Spoilage Rate (%)',
            'Predicted Fresh',
            'Predicted Rotten',
            'Correctly Classified',
            'Misclassified'
        ],
        'Value': [
            total_samples,
            fresh_count,
            rotten_count,
            f'{spoilage_rate:.2f}%',
            fresh_pred_count,
            rotten_pred_count,
            np.sum(y_true == y_pred),
            np.sum(y_true != y_pred)
        ]
    }

    df_analytics = pd.DataFrame(analytics_data)

    print("\n📊 ANALYTICS REPORT")
    print("=" * 50)
    print(df_analytics.to_string(index=False))
    print("=" * 50)

    # Visualize analytics
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Pie chart - Actual distribution
    sizes = [fresh_count, rotten_count]
    colors = ['#90EE90', '#FF6B6B']
    explode = (0.05, 0.05)

    axes[0].pie(sizes, explode=explode, labels=classes, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
    axes[0].set_title('Actual Produce Distribution', fontsize=14, fontweight='bold')

    # Bar chart - Classification comparison
    x = np.arange(len(classes))
    width = 0.35

    axes[1].bar(x - width / 2, [fresh_count, rotten_count], width,
                label='Actual', color='#4CAF50', alpha=0.8)
    axes[1].bar(x + width / 2, [fresh_pred_count, rotten_pred_count], width,
                label='Predicted', color='#2196F3', alpha=0.8)

    axes[1].set_xlabel('Category', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(classes)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('analytics_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    return df_analytics


# Generate analytics
df_analytics = generate_analytics(y_true, y_pred, CLASS_NAMES)


# ============================================================================
# SECTION 9: PREDICTION FUNCTION
# ============================================================================

def predict_single_image(image_path, model):
    """Predict freshness of a single image"""

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx] * 100

    # Display result
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.axis('off')

    result_text = f"{CLASS_NAMES[class_idx].upper()}\nConfidence: {confidence:.2f}%"
    color = 'green' if class_idx == 0 else 'red'

    plt.title(result_text, fontsize=16, fontweight='bold', color=color, pad=20)
    plt.tight_layout()
    plt.show()

    # Composting recommendation
    if class_idx == 1:  # Rotten
        print("\n♻️ COMPOSTING RECOMMENDATION:")
        print("✓ This produce is spoiled and suitable for composting")
        print("✓ Add to compost bin with brown materials (dry leaves, paper)")
        print("✓ Maintain 3:1 ratio of brown to green materials")
        print("✓ Turn compost regularly for faster decomposition")
    else:
        print("\n✅ QUALITY STATUS:")
        print("✓ This produce is fresh and safe for consumption")
        print("✓ Store properly to maintain freshness")

    return CLASS_NAMES[class_idx], confidence


# Test prediction function

# ============================================================================
# SECTION 10: SAVE MODEL & ARTIFACTS
# ============================================================================

print("\n💾 Saving model and artifacts...")

# Save model
model.save('agri_waste_classifier_model.h5')
print("✅ Model saved as 'agri_waste_classifier_model.h5'")

# Save model in TFLite format (for mobile deployment)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('agri_waste_classifier.tflite', 'wb') as f:
    f.write(tflite_model)
print("✅ TFLite model saved as 'agri_waste_classifier.tflite'")

# Save analytics
df_analytics.to_csv('analytics_report.csv', index=False)
print("✅ Analytics saved as 'analytics_report.csv'")

# Save class names
with open('class_names.txt', 'w') as f:
    for name in CLASS_NAMES:
        f.write(f"{name}\n")
print("✅ Class names saved as 'class_names.txt'")

print("\n" + "=" * 70)
print("🎉 AGRI-WASTE CLASSIFIER TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📦 Generated Files:")
print("  1. best_model.h5 - Best performing model")
print("  2. agri_waste_classifier_model.h5 - Final trained model")
print("  3. agri_waste_classifier.tflite - Mobile deployment model")
print("  4. training_history.png - Training metrics visualization")
print("  5. confusion_matrix.png - Confusion matrix")
print("  6. sample_predictions.png - Sample predictions")
print("  7. analytics_dashboard.png - Analytics dashboard")
print("  8. analytics_report.csv - Detailed analytics")
print("  9. class_names.txt - Class labels")
print("\n🚀 Next Steps:")
print("  → Download the model files")
print("  → Integrate with Streamlit/Flask web app")
print("  → Deploy on cloud platform")
print("  → Test with real-world produce images")
print("=" * 70)







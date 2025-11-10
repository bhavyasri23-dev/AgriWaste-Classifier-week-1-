"""
AgriWaste Classifier - Standalone Prediction Script
Usage: python predict.py --image path/to/image.jpg
"""
from predict import predict_image

import argparse
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Configuration
IMG_SIZE = 224
CLASS_NAMES = ['fresh', 'rotten']
CONFIDENCE_THRESHOLD = 75.0
MODEL_PATH = 'agri_waste_classifier_improved.h5'


def load_model(model_path=MODEL_PATH):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("✅ Model loaded successfully!")
    return model


def preprocess_image(image_path, img_size=IMG_SIZE):
    """Load and preprocess image for prediction"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img_resized = cv2.resize(img, (img_size, img_size))

    # Normalize
    img_normalized = img_resized / 255.0

    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)

    return img, img_batch


def predict_image(model, image_path, show_plot=True, save_result=False):
    """
    Make prediction on a single image

    Args:
        model: Trained Keras model
        image_path: Path to image file
        show_plot: Whether to display visualization
        save_result: Whether to save prediction result

    Returns:
        dict: Prediction results
    """
    print(f"\n🔍 Analyzing image: {image_path}")

    # Preprocess
    original_img, processed_img = preprocess_image(image_path)

    # Predict
    print("Making prediction...")
    prediction = model.predict(processed_img, verbose=0)

    # Extract results
    class_idx = np.argmax(prediction[0])
    confidence = prediction[0][class_idx] * 100

    fresh_prob = prediction[0][0] * 100
    rotten_prob = prediction[0][1] * 100

    predicted_class = CLASS_NAMES[class_idx]
    is_certain = confidence >= CONFIDENCE_THRESHOLD

    # Print results
    print("\n" + "=" * 60)
    print("📊 PREDICTION RESULTS")
    print("=" * 60)
    print(f"Predicted Class: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Fresh Probability: {fresh_prob:.2f}%")
    print(f"Rotten Probability: {rotten_prob:.2f}%")
    print(f"Certainty: {'High ✓' if is_certain else 'Low ⚠'}")
    print("=" * 60)

    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if predicted_class == 'rotten':
        print("❌ This produce appears to be ROTTEN")
        print("♻️  Composting recommended:")
        print("   • Add to compost bin with brown materials")
        print("   • Maintain 3:1 ratio of brown to green materials")
        print("   • Turn compost regularly")
        print("   • DO NOT CONSUME")
    else:
        if is_certain:
            print("✅ This produce appears to be FRESH")
            print("   • Safe for consumption")
            print("   • Store properly to maintain freshness")
            print("   • Refrigerate if necessary")
        else:
            print("⚠️  UNCERTAIN - Manual inspection recommended")
            print("   • The model confidence is below threshold")
            print("   • Inspect manually for signs of spoilage")
            print("   • Check smell, texture, and color")
            print("   • When in doubt, discard to be safe")

    # Visualization
    if show_plot:
        visualize_prediction(
            original_img,
            predicted_class,
            confidence,
            fresh_prob,
            rotten_prob,
            is_certain,
            image_path
        )

    # Save results
    if save_result:
        save_prediction_results(
            image_path,
            predicted_class,
            confidence,
            fresh_prob,
            rotten_prob,
            is_certain
        )

    return {
        'class': predicted_class,
        'confidence': confidence,
        'fresh_probability': fresh_prob,
        'rotten_probability': rotten_prob,
        'is_certain': is_certain
    }


def visualize_prediction(img, pred_class, confidence, fresh_prob, rotten_prob, is_certain, image_path):
    """Create visualization of prediction"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Display image
    axes[0].imshow(img)
    axes[0].axis('off')

    # Determine color
    if is_certain:
        color = 'green' if pred_class == 'fresh' else 'red'
        status = pred_class.upper()
        emoji = '✅' if pred_class == 'fresh' else '❌'
    else:
        color = 'orange'
        status = 'UNCERTAIN'
        emoji = '⚠️'

    axes[0].set_title(
        f'{emoji} {status}\nConfidence: {confidence:.2f}%',
        fontsize=16,
        fontweight='bold',
        color=color,
        pad=15
    )

    # Bar chart for probabilities
    categories = ['Fresh', 'Rotten']
    probabilities = [fresh_prob, rotten_prob]
    colors = ['#28a745', '#dc3545']

    bars = axes[1].bar(categories, probabilities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1].set_ylabel('Probability (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Classification Probabilities', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    axes[1].grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.,
            height + 2,
            f'{prob:.1f}%',
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

    # Add threshold line
    axes[1].axhline(y=CONFIDENCE_THRESHOLD, color='red', linestyle='--', linewidth=2,
                    label=f'Threshold ({CONFIDENCE_THRESHOLD}%)')
    axes[1].legend()

    plt.tight_layout()

    # Save figure
    output_filename = f"prediction_{os.path.basename(image_path)}"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved as: {output_filename}")

    plt.show()


def save_prediction_results(image_path, pred_class, confidence, fresh_prob, rotten_prob, is_certain):
    """Save prediction results to a text file"""
    output_file = f"prediction_result_{os.path.basename(image_path)}.txt"

    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("AgriWaste Classifier - Prediction Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Image: {image_path}\n")
        f.write(f"Predicted Class: {pred_class.upper()}\n")
        f.write(f"Confidence: {confidence:.2f}%\n")
        f.write(f"Fresh Probability: {fresh_prob:.2f}%\n")
        f.write(f"Rotten Probability: {rotten_prob:.2f}%\n")
        f.write(f"High Certainty: {'Yes' if is_certain else 'No'}\n")
        f.write("\n" + "=" * 60 + "\n")

    print(f"📄 Results saved to: {output_file}")


def batch_predict(model, image_folder, output_csv=None):
    """Predict on multiple images in a folder"""
    import pandas as pd

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if os.path.splitext(f)[1].lower() in image_extensions
    ]

    if not image_files:
        print(f"No images found in {image_folder}")
        return

    print(f"\n🔍 Found {len(image_files)} images to process\n")

    results = []

    for i, img_path in enumerate(image_files, 1):
        print(f"\nProcessing [{i}/{len(image_files)}]: {os.path.basename(img_path)}")
        try:
            result = predict_image(model, img_path, show_plot=False, save_result=False)
            result['image'] = os.path.basename(img_path)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    # Create DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    df = df[['image', 'class', 'confidence', 'fresh_probability', 'rotten_probability', 'is_certain']]

    # Display summary
    print("\n" + "=" * 60)
    print("📊 BATCH PREDICTION SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Save to CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Results saved to: {output_csv}")

    return df


def main():
    parser = argparse.ArgumentParser(description='AgriWaste Classifier - Predict freshness of produce')
    parser.add_argument('--image', '-i', type=str, help='Path to single image file')
    parser.add_argument('--folder', '-f', type=str, help='Path to folder with multiple images')
    parser.add_argument('--model', '-m', type=str, default=MODEL_PATH, help='Path to model file')
    parser.add_argument('--no-plot', action='store_true', help='Disable visualization')
    parser.add_argument('--save', '-s', action='store_true', help='Save prediction results to file')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file for batch predictions')

    args = parser.parse_args()

    # Load model
    try:
        model = load_model(args.model)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Single image prediction
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Image not found: {args.image}")
            return

        predict_image(
            model,
            args.image,
            show_plot=not args.no_plot,
            save_result=args.save
        )

    # Batch prediction
    elif args.folder:
        if not os.path.exists(args.folder):
            print(f"❌ Folder not found: {args.folder}")
            return

        output_csv = args.output or 'batch_predictions.csv'
        batch_predict(model, args.folder, output_csv)

    else:
        print("❌ Please provide either --image or --folder argument")
        parser.print_help()


if __name__ == "__main__":
    main()

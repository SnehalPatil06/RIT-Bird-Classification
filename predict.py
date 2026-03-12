"""
Bird Species Classification - Prediction Script
Rajarambapu Institute of Technology, Rajaramnagar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os

IMAGE_SIZE = (224, 224)
MODEL_PATH = 'best_model.h5'

# ── Load Model ─────────────────────────────────────────────────────
print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!\n")

# ── Load Class Names ───────────────────────────────────────────────
# These are loaded from the training directory structure
def get_class_names(train_dir='data/train'):
    if os.path.exists(train_dir):
        return sorted(os.listdir(train_dir))
    return [f"Species_{i}" for i in range(260)]

class_names = get_class_names()


def predict_bird(img_path, top_k=5):
    """
    Predict the bird species from an image.

    Args:
        img_path (str): Path to the bird image
        top_k (int): Number of top predictions to return

    Returns:
        list: Top-k predictions with species name and confidence
    """
    # Load and preprocess image
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    predictions = model.predict(img_array, verbose=0)[0]

    # Get top-k predictions
    top_indices = np.argsort(predictions)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            'species': class_names[idx],
            'confidence': float(predictions[idx]) * 100
        })

    return results


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_bird_image>")
        print("Example: python predict.py bird.jpg")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"Error: File '{img_path}' not found.")
        return

    print(f"Classifying: {img_path}\n")
    results = predict_bird(img_path, top_k=5)

    print("=" * 45)
    print(f"  {'RANK':<6} {'SPECIES':<30} {'CONFIDENCE'}")
    print("=" * 45)
    for i, r in enumerate(results, 1):
        bar = '█' * int(r['confidence'] / 5)
        print(f"  #{i:<5} {r['species']:<30} {r['confidence']:.2f}%  {bar}")
    print("=" * 45)
    print(f"\n✅ Top Prediction: {results[0]['species']} ({results[0]['confidence']:.2f}%)")


if __name__ == '__main__':
    main()

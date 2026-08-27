"""
Soil pH prediction from a camera image.

Pipeline:
  1. Load image with OpenCV
  2. Compute mean R, G, B (RGB) and mean H, S, V (HSV) over the whole image
  3. Feed [R, G, B, H, S, V] into the trained sklearn Pipeline
     (StandardScaler -> MLPRegressor) to predict soil pH

Usage:
  python predict_soil_ph.py path/to/soil_photo.jpg
  python predict_soil_ph.py path/to/soil_photo.jpg --model soil_ph_neural_network_model.pkl
"""

import argparse
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import pandas as pd

# Feature order the model was trained on (see scaler.feature_names_in_)
FEATURE_ORDER = ["R", "G", "B", "H", "S", "V"]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = SCRIPT_DIR / "ChatGPT Image Aug 21, 2026, 11_05_32 AM.png"
DEFAULT_MODEL = SCRIPT_DIR / "soil_ph_neural_network_model.pkl"


def extract_color_features(image_path: str) -> np.ndarray:
    """Read an image and compute mean RGB + mean HSV as a 6-element feature vector."""
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Mean RGB (OpenCV loads as BGR, so reorder to R, G, B)
    mean_bgr = img_bgr.reshape(-1, 3).mean(axis=0)
    mean_rgb = mean_bgr[::-1]  # B,G,R -> R,G,B

    # Mean HSV
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_hsv = img_hsv.reshape(-1, 3).mean(axis=0)

    features = np.concatenate([mean_rgb, mean_hsv])  # [R, G, B, H, S, V]
    return features


def classify_ph(ph: float) -> str:
    """Simple agronomic interpretation of predicted pH."""
    if ph < 5.5:
        return "Strongly acidic"
    elif ph < 6.5:
        return "Slightly acidic"
    elif ph <= 7.5:
        return "Neutral"
    elif ph <= 8.5:
        return "Slightly alkaline"
    else:
        return "Strongly alkaline"


def predict_soil_ph(image_path: str, model_path: str) -> dict:
    pipeline = joblib.load(model_path)

    features = extract_color_features(image_path)
    features_df = pd.DataFrame([features], columns=FEATURE_ORDER)

    predicted_ph = float(pipeline.predict(features_df)[0])
    predicted_ph = max(0.0, min(14.0, predicted_ph))  # clamp to valid pH range

    return {
        "features": dict(zip(FEATURE_ORDER, features.tolist())),
        "predicted_ph": round(predicted_ph, 2),
        "classification": classify_ph(predicted_ph),
    }


def main():
    parser = argparse.ArgumentParser(description="Predict soil pH from a photo.")
    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help="Path to the soil photo (jpg/png).",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained model pickle file.",
    )
    args = parser.parse_args()

    try:
        result = predict_soil_ph(args.image, args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nExtracted color features (image average):")
    for name, value in result["features"].items():
        print(f"  {name}: {value:.2f}")

    print(f"\nPredicted Soil pH: {result['predicted_ph']}")
    print(f"Classification:    {result['classification']}\n")


if __name__ == "__main__":
    main()
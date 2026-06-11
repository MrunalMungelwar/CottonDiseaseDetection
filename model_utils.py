"""
Model utilities for loading the CNN and making predictions.
"""

import os

import cv2
import numpy as np

from config import (
    CLASS_NAMES,
    IMG_HEIGHT,
    IMG_WIDTH,
    MODEL_PATH,
)


def model_exists() -> bool:
    """Check if the trained model file exists."""
    return os.path.exists(MODEL_PATH)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image for CNN prediction.

    Args:
        image: BGR numpy array from OpenCV.

    Returns:
        Preprocessed array of shape (1, IMG_HEIGHT, IMG_WIDTH, 3).
    """
    # Resize to model input dimensions
    resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Convert BGR to RGB
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0

    # Add batch dimension
    return np.expand_dims(normalized, axis=0)


def load_model():
    """
    Load the trained Keras model.

    Returns:
        Loaded Keras model or None if file not found.
    """
    if not model_exists():
        return None

    # Import TensorFlow only when needed to speed up Streamlit startup
    import tensorflow as tf

    return tf.keras.models.load_model(MODEL_PATH)


def predict_disease(model, image: np.ndarray) -> dict:
    """
    Predict disease from a leaf image.

    Args:
        model: Loaded Keras model.
        image: BGR numpy array of the leaf image.

    Returns:
        Dictionary with keys:
            - disease (str): Predicted class name.
            - confidence (float): Confidence percentage (0-100).
            - probabilities (dict): Class name to probability mapping.
    """
    processed = preprocess_image(image)
    predictions = model.predict(processed, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    confidence = float(predictions[predicted_idx]) * 100

    probabilities = {
        CLASS_NAMES[i]: float(predictions[i]) * 100
        for i in range(len(CLASS_NAMES))
    }

    return {
        "disease": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 2),
        "probabilities": probabilities,
    }

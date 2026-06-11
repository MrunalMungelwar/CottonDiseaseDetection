"""
Configuration constants for the Cotton Disease Detection project.
Centralizes paths, class labels, and model hyperparameters.
"""

import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset and model paths
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model.keras")
HISTORY_CSV_PATH = os.path.join(BASE_DIR, "prediction_history.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
TRAINING_PLOTS_DIR = os.path.join(BASE_DIR, "training_plots")

# Disease class labels (must match dataset folder names)
CLASS_NAMES = [
    "Healthy",
    "Bacterial Blight",
    "Curl Virus",
    "Fusarium Wilt",
]

# Image dimensions for the CNN
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 30
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
LEARNING_RATE = 0.001
RANDOM_SEED = 42

# Severity thresholds (percentage of infected area)
SEVERITY_MILD_MAX = 20
SEVERITY_MODERATE_MAX = 50

# Supported languages
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
}

# Streamlit page configuration
APP_TITLE = "Cotton Plant Disease Detection"
APP_ICON = "🌿"

# Gemini API Key (Add your key here so users don't need to provide it)
GEMINI_API_KEY = ""

# Cotton Plant Disease Detection and Smart Crop Advisory System using CNN

An end-to-end AI-powered web application that detects cotton plant diseases using a custom Convolutional Neural Network (CNN) and provides smart crop advisory through an intelligent assistant.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Project Overview

Cotton is one of the most important cash crops globally, and disease outbreaks can cause significant yield losses. This system uses deep learning to automatically classify cotton leaf images into four categories and provides farmers with severity analysis, treatment recommendations, and farming advisory — all through an intuitive web interface.

**Supported Disease Classes:**
- Healthy
- Bacterial Blight
- Curl Virus
- Fusarium Wilt

---

## Features

| Feature | Description |
|---------|-------------|
| **Custom CNN Model** | 4-layer convolutional architecture with BatchNorm, Dropout, and Dense layers |
| **Disease Prediction** | Upload or capture leaf images for instant AI analysis |
| **Severity Analysis** | OpenCV-based infected area estimation (Mild / Moderate / Severe) |
| **Disease Information** | Detailed symptoms, causes, prevention, and treatment for each disease |
| **PDF Reports** | Downloadable analysis reports with ReportLab |
| **Analytics Dashboard** | Prediction history, disease distribution, and trend charts |
| **AI Assistant** | Rule-based chatbot with optional Google Gemini API integration |
| **Multilingual** | English, Hindi, and Marathi language support |
| **Webcam Support** | Capture images directly from device camera |

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Webcam (optional, for camera capture)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CottonDiseaseDetection.git
cd CottonDiseaseDetection

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Optional: Gemini AI Assistant

To enable the Gemini-powered chatbot, you can configure your API key so that users don't have to provide one. You can do this in two ways:

**Option 1: In config.py (Recommended)**
Open `config.py` and paste your Gemini API key at the bottom:
```python
GEMINI_API_KEY = "your_api_key_here"
```

**Option 2: Using Environment Variables**
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Linux/Mac
export GEMINI_API_KEY=your_api_key_here
```

---

## Dataset Structure

Organize your training images in the following folder structure:

```
dataset/
├── Healthy/
│   ├── healthy_001.jpg
│   ├── healthy_002.jpg
│   └── ...
├── Bacterial Blight/
│   ├── blight_001.jpg
│   ├── blight_002.jpg
│   └── ...
├── Curl Virus/
│   ├── curl_001.jpg
│   ├── curl_002.jpg
│   └── ...
└── Fusarium Wilt/
    ├── wilt_001.jpg
    ├── wilt_002.jpg
    └── ...
```

**Alternate structure** (also supported automatically):

```
dataset/cotton/
├── healthy/
├── bacterial_blight/
├── curl_virus/
└── fussarium_wilt/
```

**Requirements:**
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Recommended: At least 50-100 images per class
- Images should show clear cotton leaf photographs

To reorganize images into standard folders, run:

```bash
python prepare_dataset.py
```

---

## Training Instructions

### Train the CNN Model

```bash
python train.py
```

**Training pipeline includes:**
1. Image preprocessing and resizing (224×224)
2. Data augmentation (rotation, flip, zoom, brightness)
3. 80/20 train-validation split
4. Custom CNN training with early stopping
5. Accuracy and loss metric display
6. Confusion matrix generation
7. Model saved as `model.keras`

**Training outputs:**
- `model.keras` — Trained model file
- `training_plots/training_history.png` — Accuracy and loss curves
- `training_plots/confusion_matrix.png` — Confusion matrix visualization

### Bootstrap Model (Without Dataset)

If you don't have a dataset yet, the training script will save an untrained model architecture so the app can still load:

```bash
python train.py
# Will save untrained model.keras if dataset is empty
```

---

## Running the Application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501` with five pages:

| Page | Description |
|------|-------------|
| **Home** | Project overview, features, and quick stats |
| **Disease Prediction** | Upload/capture images, get predictions and reports |
| **Analytics Dashboard** | Charts and metrics from prediction history |
| **AI Assistant** | Cotton farming chatbot |
| **About Project** | Technical details and architecture |

---

## Deployment Instructions

### GitHub

1. Create a new GitHub repository
2. Push the project:

```bash
git init
git add .
git commit -m "Initial commit: Cotton Disease Detection System"
git remote add origin https://github.com/yourusername/CottonDiseaseDetection.git
git push -u origin main
```

### Streamlit Community Cloud

1. Push your code to GitHub (including `model.keras` if trained)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"**
4. Select your repository, branch, and set main file to `app.py`
5. Click **"Deploy"**

**Optional secrets** (Settings → Secrets):

```toml
GEMINI_API_KEY = "your_gemini_api_key"
```

**Note:** For Streamlit Cloud, use `opencv-python-headless` (already in requirements.txt) instead of `opencv-python`.

### Local Production

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Screenshots Section

> Add screenshots of your application here after running it.

| Home Page | Disease Prediction |
|-----------|-------------------|
| *Screenshot placeholder* | *Screenshot placeholder* |

| Analytics Dashboard | AI Assistant |
|--------------------|--------------|
| *Screenshot placeholder* | *Screenshot placeholder* |

---

## Project Structure

```
CottonDiseaseDetection/
├── dataset/                  # Training images (4 class folders)
├── training_plots/           # Generated during training
├── app.py                    # Main Streamlit application
├── train.py                  # CNN training script
├── chatbot.py                # AI assistant (rule-based + Gemini)
├── disease_info.py           # Disease knowledge base
├── severity_analysis.py      # OpenCV severity estimation
├── report_generator.py       # PDF report generation
├── dashboard.py              # Analytics dashboard
├── model_utils.py            # Model loading and prediction
├── history_manager.py        # CSV prediction history
├── translations.py           # Multilingual translations
├── config.py                 # Configuration constants
├── model.keras               # Trained CNN model
├── prediction_history.csv    # Prediction log
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore
```

---

## CNN Architecture

```
Input (224×224×3)
    ↓
Conv2D(32) → BatchNorm → ReLU → Conv2D(32) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv2D(64) → BatchNorm → ReLU → Conv2D(64) → BatchNorm → ReLU → MaxPool → Dropout(0.25)
    ↓
Conv2D(128) → BatchNorm → ReLU → Conv2D(128) → BatchNorm → ReLU → MaxPool → Dropout(0.30)
    ↓
Conv2D(256) → BatchNorm → ReLU → Conv2D(256) → BatchNorm → ReLU → MaxPool → Dropout(0.30)
    ↓
Flatten → Dense(256) → BatchNorm → Dropout(0.50) → Dense(128) → Dropout(0.40)
    ↓
Dense(4, Softmax)
```

---

## Contributors

- **Pranjali Deshmukh**
- **Tina Agrawal**
- **Mrunal Mungelwar**

---

## License

This project is licensed under the MIT License. Free for educational and research purposes.

---

## Acknowledgments

- TensorFlow/Keras for deep learning framework
- Streamlit for the web application framework
- OpenCV for image processing
- ReportLab for PDF generation
- Google Gemini for AI assistant capabilities

"""
Cotton Plant Disease Detection and Smart Crop Advisory System
Main Streamlit Web Application
"""

import io

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from chatbot import get_chatbot_response, is_gemini_available
from config import APP_ICON, APP_TITLE, LANGUAGES
from dashboard import render_dashboard
from disease_info import get_disease_info
from history_manager import add_prediction, clear_history, get_history
from model_utils import load_model, model_exists, predict_disease
from report_generator import generate_pdf_report
from severity_analysis import analyze_severity
from translations import get_ui_text, translate_disease_name, translate_severity

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for agriculture theme
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Background and global styles */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    /* Main Header - Glassmorphism */
    .main-header {
        background: rgba(46, 125, 50, 0.85);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 2.5rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .main-header:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.25);
    }

    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 300;
    }

    /* Glass Cards */
    .metric-card, .feature-card, .result-card-healthy, .result-card-disease, .info-section {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        box-shadow: 0 4px 16px 0 rgba(31, 38, 135, 0.07);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }

    .feature-card {
        padding: 2rem 1.5rem;
        text-align: center;
        height: 100%;
        border-top: 4px solid #43A047;
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px 0 rgba(31, 38, 135, 0.12);
        background: rgba(255, 255, 255, 0.9);
    }

    .feature-card h4 {
        color: #1B5E20;
        margin-top: 1rem;
        font-weight: 600;
    }

    /* Result Cards */
    .result-card-healthy {
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.8) 0%, rgba(165, 214, 167, 0.8) 100%);
        padding: 2rem;
        border-left: 6px solid #4CAF50;
        margin: 1.5rem 0;
    }

    .result-card-disease {
        background: linear-gradient(135deg, rgba(255, 235, 238, 0.8) 0%, rgba(255, 205, 210, 0.8) 100%);
        padding: 2rem;
        border-left: 6px solid #F44336;
        margin: 1.5rem 0;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2E7D32, #43A047);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 2.5rem;
        font-weight: 600;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        box-shadow: 0 6px 16px rgba(46, 125, 50, 0.4);
        transform: translateY(-2px);
        border: none;
        color: white;
    }

    /* Sidebar styling */
    div[data-testid="stSidebar"] {
        background: rgba(241, 248, 233, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.5);
    }

    /* Info sections */
    .info-section {
        padding: 1.5rem;
        margin: 0.8rem 0;
        border-left: 5px solid #66BB6A;
        background: rgba(241, 248, 233, 0.7);
    }

    .info-section strong {
        color: #2E7D32;
        font-size: 1.1rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }

    /* Severity Colors */
    .severity-mild { color: #FFA000; font-weight: 700; }
    .severity-moderate { color: #F57C00; font-weight: 700; }
    .severity-severe { color: #D32F2F; font-weight: 700; }
    .severity-none { color: #388E3C; font-weight: 700; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "model" not in st.session_state:
        st.session_state.model = None
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False


def load_model_cached():
    """Load the CNN model once and cache in session state."""
    if not st.session_state.model_loaded:
        if model_exists():
            st.session_state.model = load_model()
            st.session_state.model_loaded = True
        else:
            st.session_state.model = None
            st.session_state.model_loaded = True
    return st.session_state.model


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def is_valid_cotton_leaf(image: np.ndarray) -> bool:
    """
    Check if the image has leaf-like green/yellow/brown colors using HSV.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Define color range for leaf tissue (green, yellow, light brown)
    lower_plant = np.array([18, 25, 25])
    upper_plant = np.array([95, 255, 255])
    mask_plant = cv2.inRange(hsv, lower_plant, upper_plant)

    # Calculate percentage of plant-colored pixels
    plant_pixel_count = np.count_nonzero(mask_plant)
    total_pixels = image.shape[0] * image.shape[1]
    plant_ratio = plant_pixel_count / total_pixels

    return plant_ratio >= 0.10


def create_probability_chart(probabilities: dict, lang: str) -> bytes:
    """Create a horizontal bar chart of prediction probabilities."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [translate_disease_name(k, lang) for k in probabilities.keys()]
    values = list(probabilities.values())
    colors = ["#4CAF50" if v == max(values) else "#81C784" for v in values]

    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.6)
    ax.set_xlabel("Probability (%)")
    ax.set_title(get_ui_text("probability_chart", lang), fontweight="bold")
    ax.set_xlim(0, 100)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
def page_home(lang: str):
    """Render the home page."""
    st.markdown(
        f"""<div class="main-header">
            <h1>🌿 {get_ui_text('welcome_title', lang)}</h1>
            <p>{get_ui_text('welcome_desc', lang)}</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # Feature cards
    st.markdown(f"### ✨ {get_ui_text('features_title', lang)}")
    cols = st.columns(3)
    features = [
        ("🔬", "feature_1"),
        ("📊", "feature_2"),
        ("📄", "feature_3"),
        ("🌐", "feature_4"),
        ("🤖", "feature_5"),
        ("📈", "feature_6"),
    ]
    for i, (icon, key) in enumerate(features):
        with cols[i % 3]:
            st.markdown(
                f"""<div class="feature-card">
                    <div style="font-size:2.5rem;">{icon}</div>
                    <h4>{get_ui_text(key, lang)}</h4>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Quick stats from history
    df = get_history()
    if not df.empty:
        st.markdown("### 📊 Quick Stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(get_ui_text("total_predictions", lang), len(df))
        c2.metric(get_ui_text("healthy_plants", lang), len(df[df["Disease"] == "Healthy"]))
        c3.metric(get_ui_text("diseased_plants", lang), len(df[df["Disease"] != "Healthy"]))
        c4.metric(get_ui_text("avg_confidence", lang), f"{df['Confidence'].mean():.1f}%")

    # How it works
    st.markdown("### 🔄 How It Works")
    steps = st.columns(4)
    step_labels = [
        ("1️⃣", "Upload", "Upload or capture a cotton leaf image"),
        ("2️⃣", "Analyze", "AI model detects disease using CNN"),
        ("3️⃣", "Report", "Get severity analysis and recommendations"),
        ("4️⃣", "Advisory", "Download PDF report and get farming advice"),
    ]
    for col, (emoji, title, desc) in zip(steps, step_labels):
        with col:
            st.markdown(f"**{emoji} {title}**")
            st.caption(desc)


# ---------------------------------------------------------------------------
# Page: Disease Prediction
# ---------------------------------------------------------------------------
def page_prediction(lang: str):
    """Render the disease prediction page."""
    st.markdown(f"## 🔬 {get_ui_text('nav_prediction', lang)}")

    model = load_model_cached()
    if model is None:
        st.error(get_ui_text("model_not_found", lang))
        st.info("Run `python train.py` with your dataset to train the model.")
        return

    # Input method selection
    input_method = st.radio(
        "Input Method",
        [get_ui_text("upload_image", lang), get_ui_text("capture_image", lang)],
        horizontal=True,
        label_visibility="collapsed",
    )

    image = None

    if input_method == get_ui_text("upload_image", lang):
        uploaded_file = st.file_uploader(
            get_ui_text("upload_image", lang),
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input(
            get_ui_text("capture_image", lang),
            label_visibility="collapsed",
        )
        if camera_image is not None:
            image = Image.open(camera_image)

    if image is None:
        st.info(get_ui_text("upload_or_capture", lang))
        return

    # Display uploaded image
    col_img, col_result = st.columns([1, 1])

    with col_img:
        st.image(image, caption="Uploaded Cotton Leaf", use_container_width=True)

    with col_result:
        if st.button(f"🔍 {get_ui_text('predict_btn', lang)}", use_container_width=True):
            with st.spinner("Analyzing leaf image..."):
                cv2_image = pil_to_cv2(image)

                # Validate if image contains a cotton leaf
                if not is_valid_cotton_leaf(cv2_image):
                    st.error(get_ui_text("invalid_image_warning", lang))
                    if "last_prediction" in st.session_state:
                        del st.session_state.last_prediction
                else:
                    # Predict disease
                    result = predict_disease(model, cv2_image)
                    disease = result["disease"]
                    confidence = result["confidence"]
                    probabilities = result["probabilities"]

                    # Severity analysis
                    severity_result = analyze_severity(cv2_image, disease)
                    severity_pct = severity_result["severity_percentage"]
                    severity_level = severity_result["severity_level"]
                    visualization = severity_result["visualization"]

                    # Store in session for report download
                    st.session_state.last_prediction = {
                        "image": cv2_image,
                        "disease": disease,
                        "confidence": confidence,
                        "probabilities": probabilities,
                        "severity_pct": severity_pct,
                        "severity_level": severity_level,
                        "visualization": visualization,
                    }

                    # Save to history
                    add_prediction(disease, confidence, severity_level)

    # Display results if available
    if "last_prediction" in st.session_state:
        pred = st.session_state.last_prediction
        disease = pred["disease"]
        confidence = pred["confidence"]
        is_healthy = disease == "Healthy"

        st.markdown("---")
        st.markdown(f"### 📋 Prediction Results")

        # Result card
        card_class = "result-card-healthy" if is_healthy else "result-card-disease"
        status_text = get_ui_text("healthy_status" if is_healthy else "disease_detected", lang)
        disease_display = translate_disease_name(disease, lang)
        severity_display = translate_severity(pred["severity_level"], lang)

        st.markdown(
            f"""<div class="{card_class}">
                <h3>🌿 {get_ui_text('disease', lang)}: {disease_display}</h3>
                <h3>🎯 {get_ui_text('confidence', lang)}: {confidence:.1f}%</h3>
                <h3>📌 {get_ui_text('status', lang)}: {status_text}</h3>
                <h3>⚠️ {get_ui_text('severity_level', lang)}: {severity_display}
                    ({pred['severity_pct']:.1f}%)</h3>
            </div>""",
            unsafe_allow_html=True,
        )

        # Probability chart and severity visualization
        chart_col, sev_col = st.columns(2)

        with chart_col:
            chart_bytes = create_probability_chart(pred["probabilities"], lang)
            st.image(chart_bytes, caption=get_ui_text("probability_chart", lang))

        with sev_col:
            vis_rgb = cv2.cvtColor(pred["visualization"], cv2.COLOR_BGR2RGB)
            st.image(vis_rgb, caption=get_ui_text("affected_area", lang))

        # Disease information
        st.markdown("---")
        st.markdown(f"### 📖 Disease Information")
        info = get_disease_info(disease, lang)

        info_cols = st.columns(2)
        sections = [
            ("symptoms", "symptoms", "🔍"),
            ("causes", "causes", "🦠"),
            ("prevention", "prevention", "🛡️"),
            ("treatment", "treatment", "💊"),
            ("impact_yield", "impact_on_yield", "📉"),
        ]
        for i, (ui_key, db_key, icon) in enumerate(sections):
            with info_cols[i % 2]:
                st.markdown(
                    f"""<div class="info-section">
                        <strong>{icon} {get_ui_text(ui_key, lang)}</strong><br>
                        {info[db_key]}
                    </div>""",
                    unsafe_allow_html=True,
                )

        # PDF Report download
        st.markdown("---")
        pdf_bytes = generate_pdf_report(
            image=pred["image"],
            disease=disease,
            confidence=confidence,
            severity_level=pred["severity_level"],
            severity_percentage=pred["severity_pct"],
            lang=lang,
        )
        st.download_button(
            label=f"📄 {get_ui_text('download_report', lang)}",
            data=pdf_bytes,
            file_name=f"cotton_disease_report_{disease.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Prediction history
    st.markdown("---")
    st.markdown(f"### 📜 {get_ui_text('prediction_history', lang)}")
    history_df = get_history()
    if not history_df.empty:
        display_df = history_df.copy()
        display_df["Disease"] = display_df["Disease"].apply(
            lambda x: translate_disease_name(x, lang)
        )
        display_df["Severity"] = display_df["Severity"].apply(
            lambda x: translate_severity(x, lang)
        )
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if st.button(f"🗑️ {get_ui_text('clear_history', lang)}"):
            clear_history()
            st.success(get_ui_text("history_cleared", lang))
            st.rerun()
    else:
        st.info(get_ui_text("no_history", lang))


# ---------------------------------------------------------------------------
# Page: AI Assistant
# ---------------------------------------------------------------------------
def page_assistant(lang: str):
    """Render the AI assistant chatbot page."""
    st.markdown(f"## 🤖 {get_ui_text('nav_assistant', lang)}")

    # Show active mode
    mode = "Gemini AI" if is_gemini_available() else "Rule-Based Knowledge"
    mode_label = get_ui_text("gemini_mode", lang) if is_gemini_available() else get_ui_text("rule_based", lang)
    st.caption(f"{get_ui_text('chat_mode', lang)}: **{mode_label}**")

    if not is_gemini_available():
        st.info(
            "💡 Currently using the offline Rule-Based AI knowledge base. "
            "The site administrator has not configured the Gemini AI yet."
        )

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input(get_ui_text("chat_placeholder", lang))

    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get response
        response, mode_used = get_chatbot_response(user_input, lang)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Quick question buttons
    st.markdown("**Quick Questions:**")
    quick_cols = st.columns(4)
    quick_questions = [
        "What is Bacterial Blight?",
        "What causes Curl Virus?",
        "How to prevent Fusarium Wilt?",
        "Best cotton farming practices",
    ]
    for i, q in enumerate(quick_questions):
        with quick_cols[i]:
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                response, _ = get_chatbot_response(q, lang)
                st.session_state.chat_history.append({"role": "user", "content": q})
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------
def page_about(lang: str):
    """Render the about project page."""
    st.markdown(f"## 📖 {get_ui_text('about_title', lang)}")

    st.markdown(
        """
        ### Cotton Plant Disease Detection and Smart Crop Advisory System

        This project is an end-to-end AI-powered system designed to help cotton farmers
        detect plant diseases early and receive actionable crop advisory recommendations.

        ---

        ### 🛠️ Technology Stack

        | Component | Technology |
        |-----------|-----------|
        | Deep Learning | TensorFlow/Keras (Custom CNN) |
        | Web Framework | Streamlit |
        | Image Processing | OpenCV |
        | Data Analysis | Pandas, Matplotlib |
        | Report Generation | ReportLab |
        | AI Assistant | Rule-Based + Google Gemini API |

        ---

        ### 🦠 Supported Diseases

        1. **Healthy** - Normal, disease-free cotton leaves
        2. **Bacterial Blight** - Caused by *Xanthomonas citri* bacteria
        3. **Curl Virus** - Cotton Leaf Curl Virus transmitted by whiteflies
        4. **Fusarium Wilt** - Soil-borne fungal disease

        ---

        ### 🧠 CNN Architecture

        - 4 Convolutional blocks (32 → 64 → 128 → 256 filters)
        - Batch Normalization after each conv layer
        - Max Pooling for spatial reduction
        - Dropout for regularization (0.25 - 0.5)
        - Dense layers (256 → 128 neurons)
        - Softmax output (4 classes)
        - Input size: 224×224×3

        ---

        ### 📁 Project Structure

        ```
        CottonDiseaseDetection/
        ├── dataset/                  # Training images
        │   ├── Healthy/
        │   ├── Bacterial Blight/
        │   ├── Curl Virus/
        │   └── Fusarium Wilt/
        ├── app.py                    # Streamlit web app
        ├── train.py                  # CNN training script
        ├── chatbot.py                # AI assistant
        ├── disease_info.py           # Disease knowledge base
        ├── severity_analysis.py      # OpenCV severity analysis
        ├── report_generator.py       # PDF report generation
        ├── dashboard.py              # Analytics dashboard
        ├── model_utils.py            # Model loading & prediction
        ├── history_manager.py        # Prediction history CSV
        ├── translations.py           # Multilingual support
        ├── config.py                 # Configuration constants
        ├── model.keras               # Trained CNN model
        ├── requirements.txt
        └── README.md
        ```

        ---

        ### 👨‍💻 Developed For

        Cotton farmers, agricultural researchers, and students learning
        about AI applications in agriculture.

        **License:** MIT License - Free for educational and research purposes.

        ---

        ### 👥 Contributors

        - **Mrunal Mungelwar**
        - **Pranjali Deshmukh**
        - **Tina Agrawal**
        """
    )


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    init_session_state()

    lang = st.session_state.language

    # Sidebar navigation
    with st.sidebar:
        st.markdown(
            f"""<div style="text-align:center; padding: 1rem 0;">
                <span style="font-size: 3rem;">🌿</span>
                <h2 style="color: #1B5E20; margin: 0;">CottonGuard AI</h2>
                <p style="color: #666; font-size: 0.85rem;">Smart Crop Advisory</p>
            </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Language selector
        selected_lang_name = st.selectbox(
            get_ui_text("select_language", lang),
            options=list(LANGUAGES.keys()),
            index=list(LANGUAGES.values()).index(lang),
        )
        st.session_state.language = LANGUAGES[selected_lang_name]
        lang = st.session_state.language

        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            options=[
                get_ui_text("nav_home", lang),
                get_ui_text("nav_prediction", lang),
                get_ui_text("nav_dashboard", lang),
                get_ui_text("nav_assistant", lang),
                get_ui_text("nav_about", lang),
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        
        # --- Debug Section ---
        st.markdown("### 🛠️ Debug Info")
        gemini_active = is_gemini_available()
        st.markdown(f"**Gemini API Key Found:** {'✅ Yes' if gemini_active else '❌ No'}")
        st.markdown(f"**Chatbot Mode:** {get_ui_text('gemini_mode', lang) if gemini_active else get_ui_text('rule_based', lang)}")
        
        if not gemini_active:
            st.warning(
                "Gemini is unavailable because the API key is missing. "
                "To enable it, please add your Gemini API key to "
                "the `GEMINI_API_KEY` variable in `config.py`, or set it as "
                "an environment variable (`GEMINI_API_KEY`)."
            )

        st.markdown("---")
        st.caption("CottonGuard AI v1.0")
        st.caption("Powered by Custom CNN")

    # Route to selected page
    home_label = get_ui_text("nav_home", lang)
    prediction_label = get_ui_text("nav_prediction", lang)
    dashboard_label = get_ui_text("nav_dashboard", lang)
    assistant_label = get_ui_text("nav_assistant", lang)
    about_label = get_ui_text("nav_about", lang)

    if page == home_label:
        page_home(lang)
    elif page == prediction_label:
        page_prediction(lang)
    elif page == dashboard_label:
        render_dashboard(lang)
    elif page == assistant_label:
        page_assistant(lang)
    elif page == about_label:
        page_about(lang)


if __name__ == "__main__":
    main()

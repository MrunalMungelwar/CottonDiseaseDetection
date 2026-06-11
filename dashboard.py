"""
Analytics Dashboard module for cotton disease prediction history.
Displays interactive metrics, charts, and trends using Plotly.
"""

from datetime import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from history_manager import get_history
from translations import get_ui_text, translate_disease_name, translate_severity

def render_dashboard(lang: str = "en") -> None:
    """
    Render the interactive analytics dashboard in Streamlit.

    Args:
        lang: Language code for UI translations.
    """
    st.markdown(f"## 📊 {get_ui_text('nav_dashboard', lang)}")

    df = get_history()

    if df.empty:
        st.info(get_ui_text("no_data_dashboard", lang))
        return

    # --- Metrics Row ---
    total = len(df)
    healthy_count = len(df[df["Disease"] == "Healthy"])
    diseased_count = total - healthy_count
    avg_confidence = df["Confidence"].mean()

    # Most common disease
    if diseased_count > 0:
        diseased_df = df[df["Disease"] != "Healthy"]
        most_common = diseased_df["Disease"].mode().iloc[0] if not diseased_df.empty else "N/A"
        most_common_display = translate_disease_name(most_common, lang)
    else:
        most_common_display = "N/A"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric(f"📋 {get_ui_text('total_predictions', lang)}", total)
    col2.metric(f"🌿 {get_ui_text('healthy_plants', lang)}", healthy_count)
    col3.metric(f"🦠 {get_ui_text('diseased_plants', lang)}", diseased_count)
    col4.metric(f"⚠️ {get_ui_text('most_common', lang)}", most_common_display)
    col5.metric(f"🎯 {get_ui_text('avg_confidence', lang)}", f"{avg_confidence:.1f}%")

    st.markdown("---")

    # --- Charts Row ---
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown(f"### {get_ui_text('disease_distribution', lang)}")
        disease_counts = df["Disease"].value_counts().reset_index()
        disease_counts.columns = ["Disease", "Count"]
        disease_counts["Disease"] = disease_counts["Disease"].apply(lambda d: translate_disease_name(d, lang))

        fig_pie = px.pie(
            disease_counts, 
            values="Count", 
            names="Disease", 
            color_discrete_sequence=["#4CAF50", "#F44336", "#FF9800", "#9C27B0"],
            hole=0.4
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown(f"### {get_ui_text('severity_distribution', lang)}")
        severity_counts = df["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        severity_counts["Severity_Translated"] = severity_counts["Severity"].apply(lambda s: translate_severity(s, lang))
        
        severity_colors = {
            "None": "#4CAF50",
            "Mild": "#FFC107",
            "Moderate": "#FF9800",
            "Severe": "#F44336",
        }
        
        fig_bar = px.bar(
            severity_counts, 
            x="Severity_Translated", 
            y="Count",
            color="Severity",
            color_discrete_map=severity_colors,
            labels={"Severity_Translated": "Severity Level"}
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Prediction Trends ---
    st.markdown(f"### {get_ui_text('prediction_trends', lang)}")

    df_copy = df.copy()
    df_copy["Date"] = pd.to_datetime(df_copy["Date"])
    df_copy["DateOnly"] = df_copy["Date"].dt.date
    df_copy = df_copy.sort_values("Date")

    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        st.markdown("**Disease Predictions Over Time**")
        disease_timeline = df_copy.groupby(["DateOnly", "Disease"]).size().reset_index(name="Count")
        fig_area = px.area(
            disease_timeline, 
            x="DateOnly", 
            y="Count", 
            color="Disease",
            color_discrete_sequence=["#4CAF50", "#F44336", "#FF9800", "#9C27B0"]
        )
        fig_area.update_layout(margin=dict(t=20, b=20, l=20, r=20), xaxis_title="Date")
        st.plotly_chart(fig_area, use_container_width=True)

    with trend_col2:
        st.markdown("**Confidence Score Trend**")
        fig_line = px.line(
            df_copy, 
            x="Date", 
            y="Confidence", 
            markers=True,
            line_shape="spline"
        )
        fig_line.update_traces(line_color="#2E7D32", marker=dict(size=8, color="#43A047"))
        fig_line.update_layout(margin=dict(t=20, b=20, l=20, r=20), xaxis_title="Date", yaxis_title="Confidence (%)")
        st.plotly_chart(fig_line, use_container_width=True)


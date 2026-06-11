"""
Prediction history manager.
Stores and retrieves prediction records in a CSV file.
"""

import os
from datetime import datetime

import pandas as pd

from config import HISTORY_CSV_PATH

HISTORY_COLUMNS = ["Date", "Disease", "Confidence", "Severity"]


def _ensure_history_file() -> None:
    """Create the history CSV with headers if it does not exist."""
    if not os.path.exists(HISTORY_CSV_PATH):
        df = pd.DataFrame(columns=HISTORY_COLUMNS)
        df.to_csv(HISTORY_CSV_PATH, index=False)


def add_prediction(
    disease: str,
    confidence: float,
    severity: str,
) -> None:
    """
    Append a new prediction record to the history CSV.

    Args:
        disease: Predicted disease name.
        confidence: Confidence score (0-100).
        severity: Severity level string.
    """
    _ensure_history_file()

    new_record = pd.DataFrame(
        [{
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Disease": disease,
            "Confidence": round(confidence, 2),
            "Severity": severity,
        }]
    )

    new_record.to_csv(
        HISTORY_CSV_PATH,
        mode="a",
        header=False,
        index=False,
    )


def get_history() -> pd.DataFrame:
    """
    Load and return the full prediction history.

    Returns:
        DataFrame with columns: Date, Disease, Confidence, Severity.
    """
    _ensure_history_file()
    try:
        df = pd.read_csv(HISTORY_CSV_PATH)
        if df.empty:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        return df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=HISTORY_COLUMNS)


def clear_history() -> None:
    """Delete all prediction history records."""
    df = pd.DataFrame(columns=HISTORY_COLUMNS)
    df.to_csv(HISTORY_CSV_PATH, index=False)


def get_history_count() -> int:
    """Return the total number of prediction records."""
    df = get_history()
    return len(df)

from pathlib import Path
import re
import numpy as np
import pandas as pd

NO_RESPONSE_VALUES = {
    "", "nr", "n/r", "no responde", "no respondio", "no respondió",
    "no response", "no_response", "sin respuesta", "na", "n/a"
}

def clean_name(value):
    return "" if pd.isna(value) else re.sub(r"\s+", "", str(value).strip())

def clean_token(value):
    return "" if pd.isna(value) else str(value).strip().lower()

def is_no_response(value):
    return clean_token(value) in NO_RESPONSE_VALUES

def normalize_emotion(value):
    if pd.isna(value) or is_no_response(value):
        return np.nan
    token = clean_token(value)
    mapping = {
        "negative": "Negative", "negativo": "Negative", "negativa": "Negative",
        "neutral": "Neutral", "neutro": "Neutral", "neutra": "Neutral",
        "positive": "Positive", "positivo": "Positive", "positiva": "Positive",
    }
    return mapping.get(token, str(value).strip())

def analysis_dir(base_dir: Path) -> Path:
    out = base_dir / "analisis_2"
    out.mkdir(parents=True, exist_ok=True)
    return out
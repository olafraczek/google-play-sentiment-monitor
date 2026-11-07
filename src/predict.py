"""
Predykcja sentymentu.

    predict_sentiment(texts)             -> własny model (TF-IDF + klasyfikator)
"""

from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import LABEL_NAMES, clean_text, score_to_label
from src.train import MODEL_PATH



@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Brak wytrenowanego modelu. Uruchom najpierw: python -m src.train"
        )
    return joblib.load(MODEL_PATH)


def predict_sentiment(texts) -> pd.DataFrame:
    """Zwraca DataFrame z kolumnami: sentiment, label, confidence, prob_positive."""
    model = load_model()
    cleaned = [clean_text(t) for t in texts]
    proba = model.predict_proba(cleaned)          # [P(neg), P(pos)]
    prob_pos = proba[:, 1]
    labels = (prob_pos >= 0.5).astype(int)
    return pd.DataFrame({
        "label": labels,
        "sentiment": [LABEL_NAMES[l] for l in labels],
        "confidence": np.round(np.maximum(prob_pos, 1 - prob_pos), 3),
        "prob_positive": np.round(prob_pos, 3),
    })

def analyze_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje do pobranych recenzji predykcje i etykietę z gwiazdek."""
    df = df.reset_index(drop=True).copy()
    preds = predict_sentiment(df["content"].tolist())
    df = pd.concat([df, preds], axis=1)
    # "prawdziwa" etykieta z gwiazdek (None dla 3 gwiazdek)
    df["label_stars"] = df["score"].apply(score_to_label)
    return df


if __name__ == "__main__":
    examples = [
        "This app is amazing, I use it every day!",
        "Terrible update, it keeps crashing and I lost all my data.",
        "Not bad, but not great either.",
    ]
    print(pd.concat([pd.Series(examples, name="text"), predict_sentiment(examples)], axis=1))

"""
Trening modelu sentymentu.

Porównuje trzy klasyfikatory na tych samych cechach TF-IDF:
    - regresja logistyczna
    - naiwny Bayes (ComplementNB, dobrze radzi sobie z niezbalansowanymi klasami)
    - liniowy SVM (z kalibracją, żeby zwracał prawdopodobieństwa)

Najlepszy model (wg F1 makro na zbiorze testowym) zapisywany jest do
models/sentiment_model.joblib, a metryki do models/metrics.json.

Uruchomienie (z głównego folderu projektu):
    python -m src.train
"""

import json
from datetime import datetime

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # zapis wykresów bez okna
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import LABEL_NAMES, PROJECT_DIR, load_raw_data, prepare_dataset

MODELS_DIR = PROJECT_DIR / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
CONFUSION_PATH = MODELS_DIR / "confusion_matrix.png"

RANDOM_STATE = 42


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),   # pojedyncze słowa + pary słów ("not good")
        min_df=2,
        max_df=0.95,
        max_features=50_000,
        sublinear_tf=True,
    )


def candidate_models() -> dict:
    return {
        "Regresja logistyczna": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=2.0
        ),
        "Naiwny Bayes (Complement)": ComplementNB(alpha=0.5),
        "Liniowy SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", C=0.5), cv=3
        ),
    }


def evaluate(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
        "f1_negative": round(f1_score(y_test, y_pred, pos_label=0), 4),
        "f1_positive": round(f1_score(y_test, y_pred, pos_label=1), 4),
    }


def main():
    print("=== Wczytywanie danych ===")
    data = prepare_dataset(load_raw_data())
    print(f"Recenzji po czyszczeniu: {len(data)}")
    print(data["label"].map(LABEL_NAMES).value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        data["clean_text"],
        data["label"],
        test_size=0.2,
        stratify=data["label"],
        random_state=RANDOM_STATE,
    )

    print("=== Porównanie modeli ===")
    results = {}
    pipelines = {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("tfidf", make_vectorizer()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        pipelines[name] = pipe
        r = results[name]
        print(f"{name:28s} accuracy={r['accuracy']:.3f}  F1 makro={r['f1_macro']:.3f}")

    # przy remisie F1 wybieramy prostszy model (mniejsza liczba = prostszy)
    simplicity = {"Naiwny Bayes (Complement)": 0, "Regresja logistyczna": 1, "Liniowy SVM": 2}
    best_name = max(results, key=lambda n: (results[n]["f1_macro"], -simplicity[n]))

    best = pipelines[best_name]
    print(f"\nNajlepszy model: {best_name}\n")

    y_pred = best.predict(X_test)
    print(classification_report(
        y_test, y_pred, target_names=[LABEL_NAMES[0], LABEL_NAMES[1]], digits=3
    ))

    # --- zapis modelu, metryk i macierzy pomyłek ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best, MODEL_PATH)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=[LABEL_NAMES[0], LABEL_NAMES[1]]).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title(f"Macierz pomyłek – {best_name}")
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa etykieta")
    fig.tight_layout()
    fig.savefig(CONFUSION_PATH, dpi=150)
    plt.close(fig)

    metrics = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "class_counts": {
            LABEL_NAMES[k]: int(v) for k, v in data["label"].value_counts().items()
        },
        "comparison": results,
        "confusion_matrix": cm.tolist(),
        "top_words": top_words(best),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Zapisano model:   {MODEL_PATH}")
    print(f"Zapisano metryki: {METRICS_PATH}")
    print(f"Zapisano wykres:  {CONFUSION_PATH}")


def top_words(pipeline, n: int = 20) -> dict:
    """Słowa najmocniej wskazujące na każdą klasę.

    Każdy z trzech modeli przechowuje "wagi" słów inaczej:
        regresja logistyczna -> coef_ (dodatnia waga = w stronę opinii pozytywnej)
        naiwny Bayes         -> feature_log_prob_ (różnica między klasą pozytywną a negatywną)
        SVM z kalibracją     -> średnia z coef_ wewnętrznych modeli SVM
    """
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        weights = clf.coef_[0]
    elif hasattr(clf, "feature_log_prob_"):
        weights = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]
    elif hasattr(clf, "calibrated_classifiers_"):
        inner = [getattr(c, "estimator", None) or getattr(c, "base_estimator", None)
                 for c in clf.calibrated_classifiers_]
        weights = np.mean([m.coef_[0] for m in inner], axis=0)
    else:
        return {}
    vocab = pipeline.named_steps["tfidf"].get_feature_names_out()
    order = np.argsort(weights)
    return {
        LABEL_NAMES[0]: [vocab[i] for i in order[:n]],
        LABEL_NAMES[1]: [vocab[i] for i in order[::-1][:n]],
    }


if __name__ == "__main__":
    main()
"""
Przygotowanie danych: wczytanie zbioru z Kaggle, czyszczenie tekstu
i zamiana ocen (gwiazdek 1-5) na etykiety sentymentu.

Etykiety:
    1-2 gwiazdki -> 0 (negatywny)
    4-5 gwiazdek -> 1 (pozytywny)
    3 gwiazdki   -> odrzucane (opinie neutralne / niejednoznaczne)
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

LABEL_NAMES = {0: "negatywny", 1: "pozytywny"}

# Możliwe nazwy kolumn w różnych zbiorach z Kaggle
TEXT_COLUMNS = ["content", "review", "text", "Translated_Review", "review_text"]
SCORE_COLUMNS = ["score", "rating", "Rating", "stars"]

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_NON_LETTERS_RE = re.compile(r"[^a-z' ]+")
_SPACES_RE = re.compile(r"\s+")
_REPEATED_RE = re.compile(r"(.)\1{2,}")  


def clean_text(text) -> str:
    """Czyści pojedynczy tekst recenzji.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = text.replace("’", "'")
    text = _REPEATED_RE.sub(r"\1\1", text)
    text = _NON_LETTERS_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text).strip()
    return text


def score_to_label(score):
    """Zamienia liczbę gwiazdek na etykietę (0/1) albo None dla 3 gwiazdek."""
    try:
        score = int(score)
    except (TypeError, ValueError):
        return None
    if score <= 2:
        return 0
    if score >= 4:
        return 1
    return None


def _find_column(df: pd.DataFrame, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Wczytuje wszystkie pliki CSV z data/raw i zwraca kolumny text, score.
    """
    csv_files = sorted(Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Brak plików CSV w {raw_dir}. Pobierz zbiór z Kaggle "
            "i wrzuć plik .csv do folderu data/raw/."
        )

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        text_col = _find_column(df, TEXT_COLUMNS)
        score_col = _find_column(df, SCORE_COLUMNS)
        if text_col is None or score_col is None:
            print(f"[pomijam] {path.name}: brak kolumny z tekstem lub oceną "
                  f"(kolumny: {list(df.columns)})")
            continue
        part = df[[text_col, score_col]].rename(
            columns={text_col: "text", score_col: "score"}
        )
        part["source_file"] = path.name
        frames.append(part)
        print(f"[wczytano] {path.name}: {len(part)} wierszy")

    if not frames:
        raise ValueError("Żaden plik CSV nie zawiera kolumn z tekstem i oceną.")
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Czyści dane i dodaje kolumny clean_text oraz label."""
    df = df.copy()
    df["label"] = df["score"].apply(score_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df["clean_text"] = df["text"].apply(clean_text)
    # bardzo krótkie recenzje ("ok", "good") zostawiamy, ale puste usuwamy
    df = df[df["clean_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["clean_text", "label"])
    return df.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw_data()
    print(f"\nRazem surowych recenzji: {len(raw)}")
    print("Rozkład gwiazdek:")
    print(raw["score"].value_counts().sort_index())

    data = prepare_dataset(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "reviews_clean.csv"
    data.to_csv(out_path, index=False)

    print(f"\nPo czyszczeniu: {len(data)} recenzji")
    print(data["label"].map(LABEL_NAMES).value_counts())
    print(f"Zapisano: {out_path}")

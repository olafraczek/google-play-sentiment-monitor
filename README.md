# Monitor opinii o aplikacjach z Google Play

Aplikacja pobiera recenzje dowolnej aplikacji z Google Play i automatycznie
analizuje ich sentyment (pozytywny / negatywny).

- **Dane treningowe:** zbiór [Google Play Store Reviews](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews) z Kaggle (12 495 recenzji z ocenami 1-5).
- **Dane do predykcji:** recenzje pobierane na żywo biblioteką `google-play-scraper`.
  Zbiór z Kaggle był zebrany tą samą biblioteką, więc dane mają tę samą strukturę (`content`, `score`).
- **Model:** TF-IDF (słowa i pary słów) oraz porównanie trzech klasyfikatorów
  (regresja logistyczna, naiwny Bayes, liniowy SVM). Zapisywany jest najlepszy według F1 makro,
  a przy remisie prostszy.
- **Wyniki:** około 86% accuracy i F1 na zbiorze testowym.
- **GUI:** Streamlit.

## Uruchomienie (Windows, VS Code)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Pobierz zbiór z Kaggle i wrzuć plik `reviews.csv` do folderu `data/raw/`.
2. Wytrenuj model:
   ```
   python -m src.train
   ```
3. Uruchom aplikację:
   ```
   streamlit run app.py
   ```
   Aplikacja otworzy się w przeglądarce pod adresem http://localhost:8501.

## Struktura

| Plik | Zawartość |
|---|---|
| `src/preprocessing.py` | wczytanie CSV, czyszczenie tekstu, gwiazdki → etykiety (1-2 negatywne, 4-5 pozytywne, 3 odrzucane) |
| `src/train.py` | trening i porównanie modeli, zapis modelu, metryk i macierzy pomyłek |
| `src/scraper.py` | wyszukiwanie aplikacji i pobieranie recenzji z Google Play |
| `src/predict.py` | predykcja sentymentu dla pobranych recenzji |
| `app.py` | interfejs graficzny (dashboard) |
| `notebooks/01_eksploracja.ipynb` | eksploracja danych treningowych |
| `models/metrics.json` | wyniki treningu |
| `docs/PREZENTACJA.md` | tekst prezentacji projektu |

## Funkcje aplikacji

- wyszukiwanie aplikacji po nazwie albo podanie ID lub linku z Google Play
  (główna aplikacja, np. TikTok, trafia na pierwsze miejsce listy wyników),
- wybór liczby i kolejności pobieranych recenzji (najnowsze lub najtrafniejsze),
- udział opinii pozytywnych i negatywnych, gwiazdki autora a wynik modelu, recenzje w czasie,
- **weryfikacja modelu na żywych danych**: porównanie predykcji z gwiazdkami autorów,
- chmury słów dla recenzji pozytywnych i negatywnych,
- recenzje, przy których model nie zgadza się z gwiazdkami,
- tabela recenzji z filtrami i eksportem do CSV,
- zakładka z wynikami modelu.

## Ograniczenia

- model działa na recenzjach po angielsku (dane pobierane są ze sklepu w USA),
- model patrzy na słowa i pary słów, więc słabo radzi sobie z sarkazmem i opiniami mieszanymi,
- dane treningowe dotyczą głównie aplikacji do organizacji zadań i pochodzą w większości z 2020 roku.
# Monitor opinii o aplikacjach z Google Play

Aplikacja pobiera recenzje dowolnej aplikacji z Google Play i automatycznie
analizuje ich sentyment (pozytywny / negatywny).

- **Dane treningowe:** zbiór „Google Play Store Reviews” z Kaggle (recenzje z ocenami 1–5).
- **Dane do predykcji:** recenzje pobierane na żywo biblioteką `google-play-scraper`.
  Mają tę samą strukturę (`content`, `score`) co dane treningowe.
- **Model:** TF-IDF (słowa + bigramy) i porównanie trzech klasyfikatorów
  (regresja logistyczna, naiwny Bayes, liniowy SVM). Zapisywany jest najlepszy.
- **GUI:** Streamlit.

## Uruchomienie (Windows, VS Code)

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Pobierz zbiór z Kaggle i wrzuć plik `.csv` do `data/raw/`
   (może być kilka plików – zostaną połączone; potrzebne kolumny: `content` i `score`).
2. Wytrenuj model:
   ```
   python -m src.train
   ```
3. Uruchom aplikację:
   ```
   streamlit run app.py
   ```

Opcjonalnie, do porównania z modelem DistilBERT: `pip install transformers torch`.

## Struktura

| Plik | Zawartość |
|---|---|
| `src/preprocessing.py` | wczytanie CSV, czyszczenie tekstu, gwiazdki → etykiety (1–2 neg, 4–5 poz, 3 odrzucane) |
| `src/train.py` | trening i porównanie modeli, zapis modelu, metryk i macierzy pomyłek |
| `src/scraper.py` | wyszukiwanie aplikacji i pobieranie recenzji z Google Play |
| `src/predict.py` | predykcja (własny model i opcjonalnie transformer) |
| `app.py` | interfejs graficzny |

## Funkcje aplikacji

- wyszukiwanie aplikacji po nazwie albo podanie ID / linku,
- wybór liczby i kolejności pobieranych recenzji,
- udział sentymentu, gwiazdki vs predykcja, sentyment w czasie,
- **weryfikacja modelu na żywych danych** – porównanie predykcji z gwiazdkami autorów,
- chmury słów dla recenzji pozytywnych i negatywnych,
- tabela recenzji z filtrami i eksportem do CSV,
- analiza własnego tekstu,
- zakładka z metrykami modelu.

"""
Pobieranie recenzji aplikacji z Google Play (biblioteka google-play-scraper).

Przykład:
    python -m src.scraper com.spotify.music 100
"""

import re
import sys
from urllib.parse import quote

import pandas as pd
from google_play_scraper import Sort, app, reviews, search
from google_play_scraper.constants.request import Formats
from google_play_scraper.exceptions import NotFoundError
from google_play_scraper.utils.request import get

SORT_OPTIONS = {
    "Najnowsze": Sort.NEWEST,
    "Najtrafniejsze": Sort.MOST_RELEVANT,
}

_APP_ID_RE = re.compile(r"^[A-Za-z][\w]*(\.[\w]+)+$")
_URL_ID_RE = re.compile(r"[?&]id=([\w.]+)")
_DETAILS_LINK_RE = re.compile(r"/store/apps/details\?id=([\w.]+)")


def extract_app_id(user_input: str):
    """Zwraca ID aplikacji z linku do Google Play albo z samego ID.

    'https://play.google.com/store/apps/details?id=com.spotify.music' -> 'com.spotify.music'
    'com.spotify.music' -> 'com.spotify.music'
    'spotify' -> None (to nazwa, trzeba ją wyszukać)
    """
    user_input = user_input.strip()
    match = _URL_ID_RE.search(user_input)
    if match:
        return match.group(1)
    if _APP_ID_RE.match(user_input):
        return user_input
    return None


def _top_result_id(query: str, lang: str, country: str):
    """Zwraca ID aplikacji z wyróżnionej karty na górze wyników wyszukiwania.

    Google Play pokazuje najlepiej pasującą aplikację (np. główny TikTok) jako
    osobną, dużą kartę. Biblioteka google-play-scraper czasem nie potrafi odczytać
    jej ID, więc szukamy go sami: bierzemy pierwszy link do aplikacji
    ("/store/apps/details?id=...") w kodzie strony z wynikami.
    """
    url = Formats.Searchresults.build(query=quote(query), lang=lang, country=country)
    try:
        page = get(url)
    except NotFoundError:
        page = get(Formats.Searchresults.fallback_build(query=quote(query), lang=lang))
    match = _DETAILS_LINK_RE.search(page)
    return match.group(1) if match else None


def search_apps(query: str, lang: str = "en", country: str = "us", n_hits: int = 8) -> list:
    """Wyszukuje aplikacje po nazwie. Zwraca listę słowników (appId, title, ...)."""
    results = search(query, lang=lang, country=country, n_hits=n_hits)
    # bywa, że wyróżniony pierwszy wynik nie ma appId – pomijamy takie
    results = [r for r in results if r.get("appId")]

    # ... ale próbujemy odzyskać tę wyróżnioną aplikację i wstawić ją na początek listy
    try:
        top_id = _top_result_id(query, lang, country)
        if top_id and top_id not in {r["appId"] for r in results}:
            info = app(top_id, lang=lang, country=country)
            results.insert(0, {"appId": top_id, "title": info.get("title") or top_id})
    except Exception:
        pass  # jeśli się nie uda, zostają zwykłe wyniki wyszukiwania
    return results


def find_app(user_input: str, lang: str = "en", country: str = "us") -> list:
    """Zamienia dowolny tekst wpisany przez użytkownika na listę pasujących aplikacji.

    Obsługuje trzy przypadki:
        link do Google Play  -> jedna aplikacja (ID wyciągnięte z linku)
        ID aplikacji         -> jedna aplikacja (jeśli takie ID istnieje)
        nazwa, np. "spotify" -> lista wyników wyszukiwania

    Zwraca listę słowników {"appId": ..., "title": ...}.
    """
    user_input = user_input.strip()
    if not user_input:
        return []

    app_id = extract_app_id(user_input)
    if app_id:
        try:
            info = app(app_id, lang=lang, country=country)
            return [{"appId": app_id, "title": info.get("title") or app_id}]
        except Exception:
            # np. "booking.com" wygląda jak ID, ale nim nie jest -> szukamy po nazwie
            if "play.google.com" in user_input:
                return []

    return [{"appId": r["appId"], "title": r.get("title") or r["appId"]}
            for r in search_apps(user_input, lang=lang, country=country)]


def get_app_info(app_id: str, lang: str = "en", country: str = "us") -> dict:
    """Podstawowe informacje o aplikacji (nazwa, ikona, średnia ocena...)."""
    info = app(app_id, lang=lang, country=country)
    return {
        "appId": app_id,
        "title": info.get("title"),
        "developer": info.get("developer"),
        "icon": info.get("icon"),
        "score": info.get("score"),
        "ratings": info.get("ratings"),
        "installs": info.get("installs"),
        "genre": info.get("genre"),
    }


def fetch_reviews(
    app_id: str,
    count: int = 200,
    lang: str = "en",
    country: str = "us",
    sort: str = "Najnowsze",
) -> pd.DataFrame:
    """Pobiera recenzje aplikacji i zwraca DataFrame.

    Kolumny są takie same jak w zbiorze z Kaggle (content, score, at...),
    więc dane treningowe i dane do predykcji mają tę samą strukturę.
    """
    result, _ = reviews(
        app_id,
        lang=lang,
        country=country,
        sort=SORT_OPTIONS.get(sort, Sort.NEWEST),
        count=count,
    )
    if not result:
        return pd.DataFrame(columns=["reviewId", "userName", "content", "score",
                                     "thumbsUpCount", "at", "appVersion"])

    df = pd.DataFrame(result)
    if "reviewCreatedVersion" in df.columns and "appVersion" not in df.columns:
        df = df.rename(columns={"reviewCreatedVersion": "appVersion"})
    keep = [c for c in ["reviewId", "userName", "content", "score",
                        "thumbsUpCount", "at", "appVersion"] if c in df.columns]
    df = df[keep].copy()
    df["at"] = pd.to_datetime(df["at"])
    df = df.dropna(subset=["content"])
    df = df[df["content"].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    app_id = sys.argv[1] if len(sys.argv) > 1 else "com.spotify.music"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    df = fetch_reviews(app_id, count=n)
    print(f"Pobrano {len(df)} recenzji aplikacji {app_id}")
    print(df[["score", "content"]].head(10).to_string())
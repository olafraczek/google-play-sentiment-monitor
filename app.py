"""
Inteligentny system monitorowania opinii - recenzje aplikacji z Google Play.

Uruchomienie (z głównego folderu projektu, przy aktywnym .venv):
    streamlit run app.py
"""

import html
import json
import re

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from matplotlib import font_manager
from wordcloud import STOPWORDS, WordCloud

from src.predict import analyze_reviews, load_model
from src.preprocessing import clean_text
from src.scraper import fetch_reviews, find_app, get_app_info
from src.train import METRICS_PATH, MODEL_PATH

LANG, COUNTRY = "en", "us"  # model trenowany na angielskich recenzjach

# ---------------------------------------------------------------- kolory i styl
POS, NEG = "#2a78d6", "#eb6834"          # pozytywny / negatywny
COLORS = {"pozytywny": POS, "negatywny": NEG}
INK, MUTED, GRID = "#1d2129", "#5b6270", "#eceef1"
FONT = "Figtree, sans-serif"
BLUES = [[0, "#eef4fc"], [0.5, "#86b6ef"], [1, "#1c5cab"]]

st.set_page_config(page_title="Monitor opinii Google Play",
                   page_icon=":material/reviews:", layout="wide")

st.markdown(f"""
<style>
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
.stMain h3 {{ font-size: 1.3rem; margin-top: 1.2rem; }}
.stSidebar h3 {{ font-size: 1.05rem; }}
.how {{ color: {MUTED}; font-size: 0.95rem; line-height: 1.6; max-width: 460px; padding-top: 64px; }}
.how b {{ color: {INK}; font-weight: 600; }}

.app-head {{ display: flex; gap: 18px; align-items: center; margin: 4px 0 26px; }}
.app-head img {{ width: 64px; height: 64px; border-radius: 14px; }}
.app-head .name {{ font-size: 1.65rem; font-weight: 650; color: {INK}; line-height: 1.2; }}
.app-head .meta {{ color: {MUTED}; font-size: 0.95rem; margin-top: 4px; }}

.stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
          gap: 8px 32px; margin-bottom: 18px; }}
.stat .v {{ font-size: 2rem; font-weight: 650; color: {INK}; line-height: 1.1;
            font-variant-numeric: tabular-nums; }}
.stat .l {{ color: {MUTED}; font-size: 0.9rem; margin-top: 2px; }}

.split {{ display: flex; height: 40px; gap: 3px; margin: 6px 0 6px; }}
.split div {{ display: flex; align-items: center; padding: 0 14px; color: #fff;
              font-weight: 600; font-size: 0.95rem; white-space: nowrap; overflow: hidden; }}
.split .pos {{ background: {POS}; border-radius: 6px 0 0 6px; }}
.split .neg {{ background: {NEG}; border-radius: 0 6px 6px 0; justify-content: flex-end; }}
.split .solo {{ border-radius: 6px; }}
.split-note {{ color: {MUTED}; font-size: 0.85rem; margin-bottom: 30px; }}

.picked {{ border-left: 3px solid {POS}; padding: 6px 0 6px 12px; margin: 4px 0 8px; }}
.picked .t {{ font-weight: 600; color: {INK}; }}
.picked .id {{ color: {MUTED}; font-size: 0.85rem; }}

.brand {{ font-size: 1.25rem; font-weight: 700; line-height: 1.2; color: {INK};
          padding-bottom: 18px; margin-bottom: 8px; border-bottom: 1px solid #e3e5e9; }}

.empty {{ max-width: 560px; padding: 48px 0; color: {MUTED}; font-size: 1.02rem; line-height: 1.6; }}
.empty b {{ color: {INK}; }}

@media (max-width: 760px) {{
  .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .stat .v {{ font-size: 1.6rem; }}
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- pomocnicze
def pct(x):
    return f"{x * 100:.0f}%".replace(".", ",")


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def short_installs(s):
    """'1,000,000,000+' -> '1 mld+' """
    digits = re.sub(r"\D", "", str(s or ""))
    if not digits:
        return None
    n = int(digits)
    for limit, suffix in [(10**9, "mld"), (10**6, "mln"), (10**3, "tys.")]:
        if n >= limit:
            return f"{n // limit} {suffix}+"
    return f"{n}+"


def split_bar(p_pos, n_pos=None, n_neg=None):
    """Pasek proporcji pozytywne / negatywne."""
    def label(name, p, n):
        return f"{name} {pct(p)}" + (f" ({n})" if n is not None else "")
    if p_pos >= 0.999:
        return f'<div class="split"><div class="pos solo" style="flex:1">{label("pozytywne", 1, n_pos)}</div></div>'
    if p_pos <= 0.001:
        return f'<div class="split"><div class="neg solo" style="flex:1">{label("negatywne", 1, n_neg)}</div></div>'
    left = label("pozytywne", p_pos, n_pos) if p_pos >= 0.15 else ""
    right = label("negatywne", 1 - p_pos, n_neg) if p_pos <= 0.85 else ""
    return (f'<div class="split"><div class="pos" style="flex:{p_pos:.4f}">{left}</div>'
            f'<div class="neg" style="flex:{1 - p_pos:.4f}">{right}</div></div>')


def style_fig(fig, title=None, height=340):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=INK), x=0, xanchor="left", y=0.97, yanchor="top") if title else None,
        font=dict(family=FONT, color=MUTED, size=13),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=height, margin=dict(t=84 if title else 40, b=8, l=8, r=8),
        legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", title=None,
                    traceorder="normal", font=dict(color=INK)),
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, color=INK)),
        bargap=0.35,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, ticks="", zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, ticks="", rangemode="tozero")
    return fig


# ---------------------------------------------------------------- cache
@st.cache_data(ttl=600, show_spinner=False)
def cached_find(query):
    return find_app(query, lang=LANG, country=COUNTRY)


@st.cache_data(ttl=600, show_spinner=False)
def cached_reviews(app_id, count, sort):
    return fetch_reviews(app_id, count=count, lang=LANG, country=COUNTRY, sort=sort)


@st.cache_data(ttl=600, show_spinner=False)
def cached_app_info(app_id):
    try:
        return get_app_info(app_id, lang=LANG, country=COUNTRY)
    except Exception:
        return {"appId": app_id, "title": app_id}


# ---------------------------------------------------------------- sprawdzenie modelu
if not MODEL_PATH.exists():
    st.title("Monitor opinii Google Play")
    st.error("Nie znaleziono wytrenowanego modelu. Uruchom w terminalu `python -m src.train`, "
             "a potem odśwież stronę.")
    st.stop()
load_model()

# ---------------------------------------------------------------- panel boczny
with st.sidebar:
    st.markdown('<div class="brand">Monitor opinii<br>Google Play</div>', unsafe_allow_html=True)
    st.subheader("Aplikacja", anchor=False)
    query = st.text_input(
        "Nazwa, ID lub link",
        placeholder="np. spotify",
        help="Wpisz nazwę aplikacji, jej ID (com.spotify.music) albo wklej link ze sklepu Google Play.",
    )

    app_id = None
    if query.strip():
        try:
            hits = cached_find(query)
        except Exception as e:
            hits = []
            st.warning(f"Nie udało się wyszukać aplikacji: {e}")
        if len(hits) == 1:
            app_id = hits[0]["appId"]
            st.markdown(f'<div class="picked"><div class="t">{html.escape(hits[0]["title"])}</div>'
                        f'<div class="id">{app_id}</div></div>', unsafe_allow_html=True)
        elif hits:
            labels = {f"{h['title']} ({h['appId']})": h["appId"] for h in hits}
            choice = st.selectbox("Wybierz z wyników", list(labels))
            app_id = labels[choice]
        else:
            st.info("Nie znaleziono aplikacji o tej nazwie.")

    st.subheader("Recenzje", anchor=False)
    count = st.slider("Liczba recenzji", 50, 1000, 200, step=50)
    sort = st.selectbox("Kolejność", ["Najnowsze", "Najtrafniejsze"])
    run = st.button("Analizuj recenzje", type="primary", width="stretch",
                    disabled=app_id is None)

if run and app_id:
    with st.spinner("Pobieranie recenzji z Google Play..."):
        try:
            raw_df = cached_reviews(app_id, count, sort)
        except Exception as e:
            st.error(f"Nie udało się pobrać recenzji: {e}")
            st.stop()
    if raw_df.empty:
        st.warning("Ta aplikacja nie ma recenzji w języku angielskim albo jej ID jest błędne.")
        st.stop()
    with st.spinner("Analiza sentymentu..."):
        result = analyze_reviews(raw_df)
    st.session_state["result"] = result
    st.session_state["app_info"] = cached_app_info(app_id)


# ---------------------------------------------------------------- wykresy
def stars_vs_sentiment(df):
    fig = go.Figure()
    for sent in ["pozytywny", "negatywny"]:
        counts = df[df["sentiment"] == sent]["score"].value_counts().reindex(range(1, 6), fill_value=0)
        fig.add_bar(x=[f"{s}★" for s in counts.index], y=counts.values, name=sent,
                    marker=dict(color=COLORS[sent], line=dict(color="white", width=2)),
                    hovertemplate="%{x}: %{y} recenzji<extra>" + sent + "</extra>")
    fig.update_layout(barmode="stack", barcornerradius=4)
    fig.update_yaxes(title=None)
    return style_fig(fig, "Ocena autora a wynik modelu")


def sentiment_over_time(df):
    d = df.copy()
    span_days = (d["at"].max() - d["at"].min()).days
    freq, fmt = ("D", "%d.%m") if span_days <= 21 else (("W", "%d.%m") if span_days <= 300 else ("M", "%m.%Y"))
    d["okres"] = d["at"].dt.to_period(freq).dt.start_time
    tab = d.pivot_table(index="okres", columns="sentiment", values="label",
                        aggfunc="size", fill_value=0).reindex(columns=["pozytywny", "negatywny"], fill_value=0)
    share = tab["pozytywny"] / tab.sum(axis=1)
    fig = go.Figure()
    for sent in ["pozytywny", "negatywny"]:
        fig.add_bar(x=tab.index, y=tab[sent], name=sent, customdata=share,
                    marker=dict(color=COLORS[sent], line=dict(color="white", width=1.5)),
                    hovertemplate=f"%{{x|{fmt}}}<br>{sent}: %{{y}}"
                                  "<br>udział pozytywnych: %{customdata:.0%}<extra></extra>")
    fig.update_layout(barmode="stack", barcornerradius=3, bargap=0.25)
    fig.update_xaxes(tickformat=fmt)
    return style_fig(fig, "Recenzje w czasie", height=320)


def confusion_heatmap(matrix, title, x_title, y_title):
    names = ["negatywny", "pozytywny"]
    total = max(sum(sum(r) for r in matrix), 1)
    text = [[f"{v}<br><span style='font-size:12px'>{pct(v / total)}</span>" for v in row] for row in matrix]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=names, y=names, text=text, texttemplate="%{text}",
        textfont=dict(size=17, family=FONT), colorscale=BLUES, showscale=False, xgap=3, ygap=3,
        hovertemplate=f"{y_title}: %{{y}}<br>{x_title}: %{{x}}<br>liczba: %{{z}}<extra></extra>",
    ))
    fig.update_xaxes(title=x_title, side="bottom", showline=False)
    fig.update_yaxes(title=y_title, autorange="reversed", gridcolor="rgba(0,0,0,0)", rangemode="normal")
    return style_fig(fig, title, height=320)


def agreement_matrix(df, col):
    d = df.dropna(subset=["label_stars"])
    return [[int(((d["label_stars"] == t) & (d[col] == p)).sum()) for p in (0, 1)] for t in (0, 1)]


def wordcloud(texts, color):
    text = " ".join(clean_text(t) for t in texts)
    if not text.strip():
        return None
    stop = STOPWORDS | {"app", "apps", "s", "t", "im", "dont", "ve", "ll"}
    wc = WordCloud(font_path=font_manager.findfont("DejaVu Sans"), width=800, height=380, background_color=None, mode="RGBA",
                   stopwords=stop, max_words=70, collocations=False, prefer_horizontal=1.0,
                   color_func=lambda *a, **k: color).generate(text)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0)
    return fig


def accuracy(df, col):
    d = df.dropna(subset=["label_stars"])
    return (d[col] == d["label_stars"]).mean() if len(d) else float("nan")


# ---------------------------------------------------------------- nagłówek aplikacji
def app_header(info, df):
    meta = []
    if info.get("developer"):
        meta.append(html.escape(info["developer"]))
    if info.get("genre"):
        meta.append(html.escape(info["genre"]))
    if info.get("score"):
        meta.append(f"średnia ocena w sklepie {num(info['score'])}")
    inst = short_installs(info.get("installs"))
    if inst:
        meta.append(f"{inst} instalacji")
    icon = f'<img src="{info["icon"]}" alt="">' if info.get("icon") else ""
    st.markdown(f"""
<div class="app-head">{icon}
  <div><div class="name">{html.escape(info.get("title") or info["appId"])}</div>
  <div class="meta">{", ".join(meta)}</div></div>
</div>
<div class="stats">
  <div class="stat"><div class="v">{len(df)}</div><div class="l">przeanalizowanych recenzji</div></div>
  <div class="stat"><div class="v">{pct(df["label"].mean())}</div><div class="l">opinii pozytywnych wg modelu</div></div>
  <div class="stat"><div class="v">{num(df["score"].mean(), 2)}</div><div class="l">średnia gwiazdek w próbie</div></div>
  <div class="stat"><div class="v">{pct(accuracy(df, "label"))}</div><div class="l">zgodności modelu z gwiazdkami</div></div>
</div>
{split_bar(df["label"].mean(), int(df["label"].sum()), int((1 - df["label"]).sum()))}
<div class="split-note">Zgodność liczona dla recenzji z oceną 1-2 (negatywne) i 4-5 gwiazdek (pozytywne). Recenzje z oceną 3 są pomijane.</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- zakładki
tab_an, tab_rev, tab_model = st.tabs(["Przegląd", "Recenzje", "O modelu"])

with tab_an:
    if "result" not in st.session_state:
        st.markdown("""
<div class="empty"><b>Nie wybrano jeszcze aplikacji.</b><br>
Wpisz jej nazwę w panelu po lewej, wybierz ją z listy i kliknij „Analizuj recenzje”.
Aplikacja pobierze najnowsze opinie z Google Play i oceni, które są pozytywne, a które negatywne.</div>
""", unsafe_allow_html=True)
    else:
        df = st.session_state["result"]
        app_header(st.session_state["app_info"], df)

        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(stars_vs_sentiment(df), width="stretch", config={"displayModeBar": False})
        c2.plotly_chart(sentiment_over_time(df), width="stretch", config={"displayModeBar": False})

        st.subheader("Czy model ocenia tak jak autorzy recenzji?", anchor=False)
        c1, c2 = st.columns(2, gap="large")
        c1.plotly_chart(confusion_heatmap(agreement_matrix(df, "label"), None,
                                          "wynik modelu", "wg gwiazdek"),
                        width="stretch", config={"displayModeBar": False})
        m = agreement_matrix(df, "label")
        c2.markdown(f"""<div class="how">
Wiersze to ocena autora zamieniona na etykietę, kolumny to wynik modelu.
Na przekątnej są recenzje, w których model zgadza się z gwiazdkami.<br><br>
<b>{m[1][0]}</b> recenzji z oceną 4-5 model uznał za negatywne, a <b>{m[0][1]}</b>
z oceną 1-2 za pozytywne. Najczęściej są to opinie mieszane albo takie, w których
gwiazdki nie pasują do treści.</div>""", unsafe_allow_html=True)

        st.subheader("Najczęstsze słowa", anchor=False)
        c1, c2 = st.columns(2, gap="large")
        for col, sent in [(c1, "pozytywny"), (c2, "negatywny")]:
            col.markdown(f"<span style='color:{COLORS[sent]};font-weight:600'>"
                         f"Recenzje oznaczone jako {'pozytywne' if sent == 'pozytywny' else 'negatywne'}</span>",
                         unsafe_allow_html=True)
            fig = wordcloud(df.loc[df["sentiment"] == sent, "content"], COLORS[sent])
            if fig:
                col.pyplot(fig)
            else:
                col.caption("Brak recenzji w tej grupie.")

        st.subheader("Recenzje, przy których model nie zgadza się z gwiazdkami", anchor=False)
        st.caption("Nie zawsze myli się model. Autorzy często dają 5 gwiazdek, a w treści opisują błąd, "
                   "albo piszą recenzje mieszane („dobra aplikacja, ale…”).")
        wrong = df.dropna(subset=["label_stars"])
        wrong = wrong[wrong["label"] != wrong["label_stars"]]
        if wrong.empty:
            st.caption("Brak takich recenzji: model zgodził się ze wszystkimi ocenami.")
        else:
            st.dataframe(
                wrong[["score", "sentiment", "confidence", "content"]]
                .sort_values("confidence", ascending=False).head(15),
                hide_index=True, width="stretch",
                column_config={
                    "score": st.column_config.NumberColumn("Ocena", format="%d★", width="small"),
                    "sentiment": st.column_config.TextColumn("Wynik modelu", width="small"),
                    "confidence": st.column_config.NumberColumn("Pewność", format="%.2f", width="small"),
                    "content": st.column_config.TextColumn("Treść", width="large"),
                },
            )

with tab_rev:
    if "result" not in st.session_state:
        st.markdown('<div class="empty">Tu pojawi się lista recenzji po kliknięciu „Analizuj recenzje”.</div>',
                    unsafe_allow_html=True)
    else:
        df = st.session_state["result"]
        f1, f2, f3 = st.columns([1, 1, 1.4])
        sel_sent = f1.multiselect("Wynik modelu", ["pozytywny", "negatywny"],
                                  default=["pozytywny", "negatywny"])
        sel_stars = f2.multiselect("Ocena autora", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
        phrase = f3.text_input("Szukaj w treści", placeholder="np. ads, crash, login")
        view = df[df["sentiment"].isin(sel_sent) & df["score"].isin(sel_stars)]
        if phrase:
            view = view[view["content"].str.contains(phrase, case=False, na=False, regex=False)]

        cols = ["at", "score", "sentiment", "confidence", "content", "thumbsUpCount"]
        cols = [c for c in cols if c in view.columns]
        st.caption(f"Wyświetlane recenzje: {len(view)} z {len(df)}")
        st.dataframe(
            view[cols], hide_index=True, width="stretch", height=540,
            column_config={
                "at": st.column_config.DatetimeColumn("Data", format="YYYY-MM-DD"),
                "score": st.column_config.NumberColumn("Ocena", format="%d★", width="small"),
                "sentiment": st.column_config.TextColumn("Wynik modelu", width="small"),
                "confidence": st.column_config.ProgressColumn(
                    "Pewność", min_value=0.5, max_value=1.0, format="%.2f", width="small"),
                "content": st.column_config.TextColumn("Treść", width="large"),
                "thumbsUpCount": st.column_config.NumberColumn("Polubienia", width="small"),
            },
        )
        st.download_button("Pobierz jako CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name="recenzje_z_sentymentem.csv", mime="text/csv")

with tab_model:
    if not METRICS_PATH.exists():
        st.info("Brak pliku z wynikami treningu. Uruchom `python -m src.train`.")
    else:
        m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.subheader(f"Wybrany model: {m['best_model']}", anchor=False)
        st.caption(f"Wytrenowany {m['trained_at'].replace('T', ', ')} na recenzjach Google Play ze zbioru Kaggle. "
                   f"Zbiór treningowy: {m['n_train']} recenzji, testowy: {m['n_test']}.")

        comp = pd.DataFrame(m["comparison"]).T.reset_index()
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            st.markdown("**Porównanie klasyfikatorów na zbiorze testowym**")
            st.dataframe(comp, hide_index=True, width="stretch", column_config={
                "index": "Model",
                "accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                "f1_macro": st.column_config.NumberColumn("F1 makro", format="%.3f"),
                "f1_negative": st.column_config.NumberColumn("F1 negatywne", format="%.3f"),
                "f1_positive": st.column_config.NumberColumn("F1 pozytywne", format="%.3f"),
            })
            cc = m["class_counts"]
            total = sum(cc.values())
            st.markdown("**Dane treningowe**")
            st.markdown(split_bar(cc.get("pozytywny", 0) / total, cc.get("pozytywny", 0),
                                  cc.get("negatywny", 0)), unsafe_allow_html=True)
        with c2:
            st.plotly_chart(confusion_heatmap(m["confusion_matrix"], "Macierz pomyłek (zbiór testowy)",
                                              "wynik modelu", "prawdziwa etykieta"),
                            width="stretch", config={"displayModeBar": False})

        if m.get("top_words"):
            st.markdown("**Słowa, które najmocniej wpływają na decyzję modelu**")
            c1, c2 = st.columns(2, gap="large")
            for col, sent, label in [(c1, "pozytywny", "W stronę opinii pozytywnej"),
                                     (c2, "negatywny", "W stronę opinii negatywnej")]:
                col.markdown(f"<span style='color:{COLORS[sent]};font-weight:600'>{label}</span><br>"
                             f"<span style='color:{INK}'>{', '.join(m['top_words'][sent])}</span>",
                             unsafe_allow_html=True)
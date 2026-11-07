# Monitor opinii Google Play - tekst do prezentacji

Podział:

| | Osoba 1 | Osoba 2 |
|---|---|---|
| Temat | skąd bierzemy dane i jak uczymy model | jak pobieramy recenzje na żywo i jak działa aplikacja |
| Części | 1. Wstęp i cel projektu<br>2. Dane treningowe z Kaggle<br>3. Czyszczenie tekstu (`preprocessing.py`)<br>4. Eksploracja danych (notatnik)<br>5. Trening i wyniki modelu (`train.py`) | 6. Pobieranie recenzji (`scraper.py`) i przykład z TikTokiem<br>7. Predykcja (`predict.py`)<br>8. Dashboard (`app.py`) - pokaz na żywo<br>9. Ograniczenia i podsumowanie |
| Czas | ok. 6-7 minut | ok. 7-8 minut (z pokazem) |

Fragmenty w nawiasach kwadratowych i kursywą to wskazówki, czego nie czytamy na głos, np. *[pokazujemy plik]*.

---

## OSOBA 1

### 1. Wstęp i cel projektu

Naszym projektem jest inteligentny system monitorowania opinii. W skrócie: to aplikacja, która pobiera z internetu recenzje aplikacji mobilnych z Google Play, a potem sama ocenia, czy każda recenzja jest pozytywna, czy negatywna.

Wybraliśmy Google Play, bo to źródło ma jedną dużą zaletę. Każda recenzja ma tekst i ma gwiazdki, które wystawił sam autor. Dzięki temu możemy nie tylko przewidywać sentyment, ale też na bieżąco sprawdzać, czy nasz model ma rację. Wracamy do tego przy dashboardzie.

Cały projekt działa w dwóch etapach. Najpierw, jednorazowo, uczymy model na gotowym zbiorze recenzji z Kaggle. Potem, już w aplikacji, pobieramy świeże recenzje z Google Play i model je ocenia.

*[pokazujemy strukturę folderów w VS Code]*

Projekt jest w Pythonie i podzieliliśmy go na kilka plików, z których każdy robi jedną rzecz:

- `preprocessing.py` - czyści tekst recenzji,
- `train.py` - trenuje i porównuje modele,
- `scraper.py` - pobiera recenzje z Google Play,
- `predict.py` - ocenia sentyment nowych recenzji,
- `app.py` - to jest nasz dashboard, czyli interfejs graficzny,
- `notebooks/01_eksploracja.ipynb` - notatnik, w którym oglądaliśmy dane przed treningiem.


### 2. Dane treningowe z Kaggle

Do treningu używamy zbioru „Google Play Store Reviews” z Kaggle. To ponad 12 tysięcy recenzji różnych aplikacji, każda z tekstem i liczbą gwiazdek od 1 do 5.

Jeśli chodzi o wymagania dotyczące danych treningowych i danych do predykcji, że mogą one pochodzić z różnych źródeł, ale muszą być spójne pod względem struktury i charakteru. U nas jest to spełnione wręcz idealnie, bo autor zbioru z Kaggle zebrał te dane tą samą biblioteką, której my używamy do pobierania recenzji na żywo, czyli `google-play-scraper`. Kolumny nazywają się tak samo, więc model uczy się dokładnie na takich samych danych, jakie potem dostaje w aplikacji.

W pliku z Kaggle są też inne kolumny, na przykład nazwa użytkownika, zdjęcie profilowe albo odpowiedź twórców aplikacji. Do treningu modelu bierzemy jednak tylko dwie: content (treść recenzji)  i score (liczbę gwiazdek).

### 3. Czyszczenie tekstu - `preprocessing.py`

Model nie może dostać surowego tekstu, bo ludzie piszą recenzje bardzo różnie.Dlatego najpierw zaczynamy od czyszczenia tekstu.

*[pokazujemy funkcję `clean_text`]*

Funkcja `clean_text` robi po kolei takie rzeczy: 
zamienia wszystko na małe litery, usuwa linki, skraca powtórzone litery, (czyli „sooooo” robi się „soo”), usuwa cyfry, emotki i znaki interpunkcyjne, a na koniec usuwa podwójne spacje.

// ( Przykład z prawdziwej recenzji: „Sooooo GOOD!!! 😍” po czyszczeniu to po prostu „soo good”.)

Zostawiłyśmy też tak zwane stop-słowa, czyli krótkie słowa typu „not”, „no”, „never”. W TF-IDF mamy ustawione `ngram_range=(1, 2)`, więc model patrzy nie tylko na pojedyncze słowa, ale też na pary sąsiednich słów, na przykład „not good”. Dzięki temu dobrze rozpoznaje opinie negatywne.

Druga ważna rzecz w tym pliku to zamiana gwiazdek na etykiety. Recenzje z 1 i 2 gwiazdkami traktujemy jako negatywne, z 4 i 5 jako pozytywne, a recenzje z 3 gwiazdkami odrzucamy. Robimy tak, bo trójki to zwykle opinie mieszane i mogły by mylić nasz model.

Na koniec usuwamy jeszcze puste recenzje, czyli takie, z których po czyszczeniu nic nie zostało, na przykład były to same emotki. Usuwamy też duplikaty, czyli recenzje o identycznej treści. Biorą się stąd, że zbiór był pobierany dwa razy, jako najnowsze i jako najtrafniejsze recenzje, a do tego wiele osób pisze dokładnie to samo, na przykład „good” albo „great app”. Gdyby duplikaty zostały, ta sama recenzja mogłaby trafić i do danych treningowych, i do testowych, a wtedy wynik modelu byłby sztucznie zawyżony.

Po tym wszystkim zostaje nam 9659 recenzji: 4930 pozytywnych i 4729 negatywnych. Czyli klasy są dobrze zbalansowane.


### 4. Eksploracja danych – notatnik

Zanim zaczęłyśmy trenować model, obejrzałyśmy dane w notatniku. W surowym zbiorze jest 12 495 recenzji i od razu widać jedną ciekawą rzecz: gwiazdki są rozłożone prawie po równo, każda ocena to około 16–23% recenzji. W prawdziwym sklepie tak nie jest, więc autor zbioru musiał celowo pobierać podobną liczbę recenzji dla każdej oceny. Dla nas to dobrze, bo po odrzuceniu trójek mamy prawie równe klasy: 51% pozytywnych i 49% negatywnych.

Po czyszczeniu zostaje 9659 recenzji. Ciekawe jest to, że duplikatów było dużo więcej wśród opinii pozytywnych. Krótkie pochwały typu „great app” ludzie piszą identycznie, a skargi każdy pisze po swojemu.

Sprawdziłyśmy też długość recenzji. Negatywne są wyraźnie dłuższe: połowa z nich ma ponad 23 słowa, a pozytywnych tylko 16. Czyli jak ktoś narzeka, to się rozpisuje, a jak chwali, to krócej.

Najczęstszym słowem w recenzjach negatywnych okazało się „not”, a na drugim miejscu jest „no”. To potwierdziło, że dobrze zrobiłyśmy, zostawiając te krótkie słowa w tekście. Słowa najbardziej charakterystyczne dla opinii negatywnych to na przykład „uninstalled”, „refund”, „login” i „disappointed”, a dla pozytywnych „helps”, „amazing”, „perfect” i „easy”.

Zauważyłyśmy też, że zbiór dotyczy głównie aplikacji do organizacji zadań i kalendarzy, bo często pojawiają się słowa „tasks”, „calendar” czy „habits”, a recenzje są w większości z 2020 roku. Mimo to model dobrze radzi sobie z innymi aplikacjami, bo słowa typu „crash”, „ads” czy „love” znaczą to samo niezależnie od aplikacji.


### 5. Trening i wyniki modelu - `train.py`

Zaczęłyśmy od  zamiany każdej recenzji na liczby metodą TF-IDF. W dużym skrócie: każde słowo dostaje wagę. Słowo jest ważne, jeśli często pojawia się w danej recenzji, ale rzadko w innych. Słowa, które są wszędzie, na przykład „app”, dostają niską wagę, a słowa charakterystyczne, na przykład „crash” albo „refund”, dostają wysoką.

Patrzymy nie tylko na pojedyncze słowa, ale też na pary sąsiednich słów. Dzięki temu „not good” jest dla modelu osobną cechą i model może się nauczyć, że to jest negatywne.

Dane dzielimy na dwie części: 80% do nauki i 20% do sprawdzenia. 

*[pokazujemy funkcję `candidate_models`]*

Nie wybraliśmy jednego modelu, tylko trenujemy trzy i porównujemy je ze sobą:

- regresję logistyczną,
- naiwny klasyfikator Bayesa,
- liniowy SVM.

Wszystkie trzy to klasyczne, szybkie modele do klasyfikacji tekstu. Do porównania używamy miary F1 makro, czyli średniej skuteczności dla obu klas. Wybraliśmy ją zamiast samego accuracy, bo ona pilnuje, żeby model był dobry zarówno na opiniach pozytywnych, jak i negatywnych.

*[pokazujemy zakładkę „O modelu” w aplikacji albo plik `metrics.json`]*

Wyniki na zbiorze testowym:

| Model | Accuracy | F1 makro |
|---|---|---|
| Regresja logistyczna | 85,9% | 0,859 |
| Naiwny Bayes | 86,3% | 0,863 |
| Liniowy SVM | 86,3% | 0,863 |

Jak widać, wszystkie trzy wypadają bardzo podobnie, w okolicach 86%, a różnice między nimi są tak małe, że w praktyce to remis. Naiwny Bayes i SVM mają dokładnie taki sam wynik, a w takiej sytuacji wybieramy prostszy model, czyli Bayesa.Ten model zapisujemy do pliku i to on jest potem używany w aplikacji.

Z macierzy pomyłek widać jeszcze jedną ciekawą rzecz: model trochę lepiej rozpoznaje opinie negatywne niż pozytywne. Ze 946 negatywnych recenzji w zbiorze testowym dobrze rozpoznał 842, a z 986 pozytywnych dobrze rozpoznał 825.

Na koniec program zapisuje też listę słów, które najmocniej wpływają na decyzję modelu. Po stronie pozytywnej są to na przykład „amazing app”, „helps me”, „very useful” i „excellent”, a po stronie negatywnej „useless”, „uninstalled”, „refund” i „waste of”. Widać tu też, że działają pary słów, bo wśród negatywnych są „not working”, „not worth” i „very bad”. To pokazuje, że model nauczył się sensownych rzeczy.


To tyle jeśli chodzi o powstawanie modelu. Teraz pokażemy, jak używamy go na prawdziwych danych z internetu.

---

## OSOBA 2

### 6. Pobieranie recenzji z Google Play - `scraper.py`

Za pobieranie danych do predykcji z internetu odpowiada plik `scraper.py`.

Google Play nie ma oficjalnego, publicznego API do recenzji. Dlatego używamy biblioteki `google-play-scraper`. Ona robi to samo, co przeglądarka, kiedy przewijamy recenzje w sklepie: wysyła te same zapytania do Google i odczytuje odpowiedź.

W tym pliku są trzy główne funkcje:

- `find_app` zamienia to, co wpisze użytkownik, na konkretną aplikację. Można wpisać nazwę, na przykład „spotify”, można wpisać ID aplikacji, na przykład `com.spotify.music`, a można też po prostu wkleić link ze sklepu. Funkcja rozpoznaje, co to jest, i albo od razu bierze tę aplikację, albo ją wyszukuje.
- `get_app_info` pobiera podstawowe informacje o aplikacji, czyli nazwę, ikonę, średnią ocenę w sklepie i liczbę instalacji.
- `fetch_reviews` pobiera recenzje. Wybieramy, ile ich chcemy i czy najnowsze, czy najtrafniejsze. Pobieramy recenzje po angielsku, bo model był trenowany na angielskich danych.

#### Przykład z TikTokiem - coś, co musieliśmy naprawić

Miałyśmy też pewien problem dotyczący wyszukiwania aplikacji na naszym dashboardzie. Jak wpisywaliśmy „tiktok”, to na liście wyników był TikTok Lite, TikTok Studio i inne poboczne aplikacje, ale nie było głównego TikToka. Z Instagramem było tak samo.

Zaczęłam szukać, czemu tak jest. Okazało się, że Google Play pokazuje wyniki wyszukiwania w dwóch częściach. Na samej górze jest jedna duża, wyróżniona karta z najlepiej pasującą aplikacją, czyli właśnie głównym TikTokiem. A pod nią jest zwykła lista podobnych aplikacji. W kodzie strony te dwie części są zapisane zupełnie inaczej.

Biblioteka odczytuje je dwoma różnymi sposobami. Google co jakiś czas zmienia budowę strony i wtedy biblioteka przestaje znajdować ID aplikacji w tej wyróżnionej karcie. Nazwę jeszcze odczyta, ale ID wychodzi puste.

A nasz kod miał zabezpieczenie, które odrzucało wyniki bez ID. To zabezpieczenie samo w sobie było sensowne, bo bez ID nie da się pobrać recenzji. Tylko że przez to główna aplikacja  wypadała z listy i zostawały same poboczne.

*[pokazujemy funkcję `_top_result_id`]*

Naprawiliśmy to tak, że dopisaliśmy funkcję `_top_result_id`. Ona otwiera tę samą stronę wyników wyszukiwania i szuka w jej kodzie pierwszego linku do aplikacji. Każdy taki link wygląda tak: `/store/apps/details?id=` i potem ID aplikacji. Pierwszy taki link na stronie to właśnie link do tej wyróżnionej karty, czyli do głównego TikToka.

Potem logika jest prosta:

1. Biblioteka zwraca zwykłą listę wyników.
2. Nasza funkcja znajduje ID tej wyróżnionej aplikacji.
3. Jeśli tej aplikacji nie ma jeszcze na liście, pobieramy jej nazwę i wstawiamy ją na pierwsze miejsce.
4. Jeśli już jest na liście, nic nie robimy, żeby się nie dublowała.
5. A jeśli cokolwiek pójdzie nie tak, na przykład nie ma internetu albo Google znowu zmieni stronę, to po prostu zostają zwykłe wyniki i aplikacja się nie wysypuje.

### 7. Predykcja - `predict.py`

Potem przechodzimy do predykcji czyli plik 'predict'
który  łączy wytrenowany model z nowymi recenzjami. Wczytuje zapisany model z dysku, tylko raz, przy pierwszym użyciu, i potem korzysta z niego już z pamięci.

Ważne jest to, że każda nowa recenzja przechodzi przez tę samą funkcję czyszczenia co dane treningowe, bo model musi dostać tekst w takiej postaci, w jakiej się uczył.

Dla każdej recenzji model zwraca prawdopodobieństwo, że jest pozytywna. Jeśli wychodzi co najmniej 50%, uznajemy ją za pozytywną, a jeśli mniej, za negatywną. Z tego liczymy też pewność modelu. Na przykład 0,9 oznacza, że model jest bardzo pewny, a 0,55, że się waha.

Do każdej recenzji dopisujemy też etykietę z gwiazdek autora. Dzięki temu w aplikacji możemy porównać, co powiedział model, a co sam autor.

### 8. Dashboard - `app.py`

Interfejs zrobiliśmy w Streamlicie. 
( To biblioteka, która pozwala zrobić aplikację w przeglądarce w samym Pythonie. Uruchamiamy ją komendą `streamlit run app.py`.)

*[uruchamiamy aplikację i wpisujemy „tiktok”]*

**Panel po lewej.** Tu wpisujemy aplikację, czyli nazwę, ID albo link. Jeśli pasuje kilka aplikacji, pojawia się lista do wyboru, a główna jest na górze. Niżej wybieramy, ile recenzji pobrać, od 50 do 1000, i czy mają być najnowsze, czy najtrafniejsze.Na końcu klikamy „Analizuj recenzje”.

*[klikamy „Analizuj recenzje”]*

**Nagłówek.** Na górze jest ikona i nazwa aplikacji, a obok twórca, kategoria, średnia ocena w sklepie i liczba instalacji. Pod spodem mamy cztery liczby:

- ile recenzji przeanalizowaliśmy,
- jaki procent opinii jest pozytywny według naszego modelu,
- średnia gwiazdek w pobranej próbie,
- zgodność modelu z gwiazdkami, czyli w ilu procentach recenzji model ocenił tak samo jak autor.

Pod tym jest pasek, który pokazuje proporcję: niebieskie to opinie pozytywne, pomarańczowe negatywne.

Przykład: dla TikToka pobraliśmy 700 recenzji. Według modelu 64% z nich jest pozytywnych. Średnia gwiazdek w próbie wyszła 3,79, a w sklepie TikTok ma 4,0. To, że te dwie liczby są tak blisko, pokazuje, że nasza próbka dobrze odzwierciedla całość.

*[jeśli na żywo wyjdą inne liczby, czytamy te z ekranu]*

**Wykres „Ocena autora a wynik modelu”.** Dla każdej liczby gwiazdek widać, ile recenzji model uznał za pozytywne, a ile za negatywne. Dobrze działający model powinien mieć przy 1 i 2 gwiazdkach prawie same pomarańczowe słupki, a przy 4 i 5 prawie same niebieskie. I tak to właśnie wygląda.

**Wykres „Recenzje w czasie”.** Pokazuje, ile było recenzji w kolejnych dniach albo tygodniach i jaka część z nich była pozytywna.

**Macierz „Czy model ocenia tak jak autorzy recenzji?”.** Tu jest prawdziwy sprawdzian modelu na żywych danych z internetu. Wiersze to ocena autora zamieniona na etykietę, a kolumny to wynik modelu. Na przekątnej są recenzje, przy których model i autor się zgadzają. 93 recenzji z oceną 4–5 model uznał za negatywne, a 49 z oceną 1–2 za pozytywne. Najczęściej są to opinie mieszane albo takie, w których gwiazdki nie pasują do treści.

**Chmury słów.** Najczęstsze słowa osobno w recenzjach pozytywnych i negatywnych. Od razu widać, z czego ludzie się cieszą, a na co narzekają. Na przykład w negatywnych często pojawiają się „ads”, „update” .

**Tabela „Recenzje, przy których model nie zgadza się z gwiazdkami”.**  Okazuje się, że często to nie model się myli, tylko gwiazdki nie pasują do treści. 
(POKAZAĆ 2-3 przyklady w tabelce)

*[przechodzimy do zakładki „Recenzje”]*

**Zakładka „Recenzje”.** Pełna tabela wszystkich pobranych recenzji z datą, gwiazdkami, wynikiem modelu, pewnością modelu i treścią. Można filtrować po wyniku modelu, po gwiazdkach i szukać konkretnego słowa, na przykład „ads”. Wyniki można też pobrać jako plik CSV.

*[przechodzimy do zakładki „O modelu”]*

**Zakładka „O modelu”.** Tu są wyniki treningu, o których mówiła pierwsza osoba: porównanie trzech modeli, macierz pomyłek, proporcja klas w danych treningowych i słowa, które najmocniej wpływają na decyzję modelu.

### 9. Ograniczenia i podsumowanie

Na koniec kilka uczciwych ograniczeń naszego rozwiązania.

Po pierwsze, model działa tylko na recenzjach po angielsku, bo na takich był trenowany.

Po drugie, gwiazdki nie zawsze pasują do treści, więc zgodność z gwiazdkami to dobre, ale tylko przybliżone sprawdzenie modelu.

Po trzecie, zależymy od tego, jak Google Play buduje swoją stronę. Jak pokazał przykład z TikTokiem, zmiana po stronie Google może coś zepsuć, więc kod musi być na to przygotowany.

Podsumowując aplikacja pobiera teksty z internetu przez web scraping Google Play i automatycznie analizuje ich sentyment jako pozytywny lub negatywny. Ma graficzny interfejs w Streamlicie. Model jest trenowany na zbiorze z Kaggle, który ma tę samą strukturę co dane pobierane na żywo. Model osiąga około 86% skuteczności na zbiorze testowym, a dodatkowo w aplikacji można na bieżąco sprawdzić, jak radzi sobie na świeżych recenzjach.


"""Frontend Streamlit: wyniki modeli + predykcja CO2 dla jednego auta.

Uruchom z katalogu głównego projektu:
    streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------- ŚCIEŻKI
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "co2.csv"
# modele z notebooka: w katalogu głównym albo jeszcze w notebooks/
MODELS_DIR = next((p for p in [ROOT / "models", ROOT / "notebooks" / "models"] if p.exists()), ROOT / "models")

PLIKI_MODELI = {
    "Ridge (baseline)": "ridge.joblib",
    "Boosting": "boosting.joblib",
    "Las losowy": "las_losowy.joblib",
}

# wyniki walidacji krzyżowej po strojeniu - przepisane z notebooka (komórka 79)
WYNIKI_CV = pd.DataFrame({
    "MAE walidacja": [14.543, 8.670, 8.967],
    "MAE odch. std": [0.521, 0.248, 0.223],
    "RMSE walidacja": [19.644, 12.834, 13.163],
    "R² walidacja": [0.891, 0.953, 0.951],
    "Czas uczenia [s]": [0.018, 2.436, 0.709],
    "Czas strojenia [s]": [1.714, 264.866, 44.693],
}, index=list(PLIKI_MODELI))

# ----------------------------------------------------------------- CECHY (jak w notebooku)
CECHY_KATEGORYCZNE = ["Make", "Vehicle Class", "Fuel Type", "Trans_type", "Gears_kat", "Cylinders_kat"]
CECHY_BINARNE = ["Hybrid_nazwa", "FFV", "Naped_4x4"]
CECHY_NUMERYCZNE = ["Engine Size(L)"]
CECHY = CECHY_NUMERYCZNE + CECHY_KATEGORYCZNE + CECHY_BINARNE
CEL = "CO2 Emissions(g/km)"
CECHY_SPALANIA = [
    "Fuel Consumption City (L/100 km)",
    "Fuel Consumption Hwy (L/100 km)",
    "Fuel Consumption Comb (L/100 km)",
    "Fuel Consumption Comb (mpg)",
]

OPISY_PALIW = {"X": "X – benzyna zwykła", "Z": "Z – benzyna premium", "D": "D – diesel", "E": "E – etanol E85"}
OPISY_SKRZYN = {"A": "automatyczna", "AM": "zautomatyzowana manualna", "AS": "automatyczna sekwencyjna",
                "AV": "bezstopniowa (CVT)", "M": "manualna"}


def przygotuj_cechy(df: pd.DataFrame) -> pd.DataFrame:
    """Inżynieria cech z notebooka (komórki 42 i 50)."""
    df = df.copy()
    df["Make"] = df["Make"].str.upper().str.strip()
    df["Model"] = df["Model"].str.upper().str.strip()

    df["Trans_type"] = df["Transmission"].str.extract(r"^([A-Z]+)")[0]
    biegi = pd.to_numeric(df["Transmission"].str.extract(r"(\d+)")[0], errors="coerce")
    df["Gears_kat"] = biegi.fillna(0).astype("Int64").astype("string")
    df["Cylinders_kat"] = df["Cylinders"].astype("Int64").astype("string")

    nazwa = df["Model"]
    df["Hybrid_nazwa"] = nazwa.str.contains("HYBRID", regex=False).astype(int)
    df["FFV"] = nazwa.str.contains("FFV", regex=False).astype(int)
    df["Naped_4x4"] = nazwa.str.contains(r"4WD|AWD|4X4|4MATIC|QUATTRO|XDRIVE").astype(int)
    return df


# ----------------------------------------------------------------- WCZYTYWANIE
@st.cache_data
def wczytaj_surowe_dane() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Make"] = df["Make"].str.upper().str.strip()
    return df[df["Fuel Type"] != "N"]


@st.cache_data
def zbior_testowy():
    """Odtwarza preprocessing i podział 80/20 (random_state=42) z notebooka."""
    df = pd.read_csv(DATA_PATH)
    df["Make"] = df["Make"].str.upper().str.strip()
    df["Model"] = df["Model"].str.upper().str.strip()
    df = df.drop_duplicates()
    df = df[df["Fuel Type"] != "N"]
    df = przygotuj_cechy(df)
    df = df.drop(columns=CECHY_SPALANIA + ["Transmission", "Cylinders"]).drop_duplicates()
    _, X_test, _, y_test = train_test_split(df[CECHY], df[CEL], test_size=0.2, random_state=42)
    return X_test, y_test


@st.cache_resource
def wczytaj_modele():
    return {nazwa: joblib.load(MODELS_DIR / plik) for nazwa, plik in PLIKI_MODELI.items()}


@st.cache_data
def wyniki_testowe():
    X_test, y_test = zbior_testowy()
    modele = wczytaj_modele()
    predykcje = X_test.copy()
    predykcje["rzeczywiste"] = y_test.values
    metryki = {}
    for nazwa, model in modele.items():
        pred = model.predict(X_test)
        predykcje[nazwa] = pred
        metryki[nazwa] = {
            "MAE test": mean_absolute_error(y_test, pred),
            "RMSE test": np.sqrt(mean_squared_error(y_test, pred)),
            "R² test": r2_score(y_test, pred),
        }
    return pd.DataFrame(metryki).T, predykcje


# ----------------------------------------------------------------- APLIKACJA
st.set_page_config(page_title="Predykcja emisji CO₂", page_icon="🚗", layout="wide")

brakujace = [p for p in PLIKI_MODELI.values() if not (MODELS_DIR / p).exists()]
if brakujace:
    st.error(f"Brak plików modeli w `{MODELS_DIR}`: {', '.join(brakujace)}. "
             "Uruchom ostatnią komórkę notebooka.")
    st.stop()

dane = wczytaj_surowe_dane()
modele = wczytaj_modele()
metryki_test, pred_test = wyniki_testowe()
tabela = WYNIKI_CV.join(metryki_test).sort_values("MAE test")
najlepszy = tabela.index[0]

st.title("🚗 Predykcja emisji CO₂ pojazdów")
st.caption("Emisja CO₂ [g/km] szacowana z danych technicznych auta, bez informacji o spalaniu.")

zakladka_pred, zakladka_wyniki, zakladka_dane = st.tabs(["🔮 Predykcja", "📊 Wyniki modeli", "📁 Dane"])

# ----------------------------------------------------------------- PREDYKCJA
with zakladka_pred:
    st.subheader("Wprowadź parametry pojazdu")

    marki = sorted(dane["Make"].unique())
    klasy = sorted(dane["Vehicle Class"].unique())
    cylindry_lista = sorted(dane["Cylinders"].unique())
    skrzynie = sorted(dane["Transmission"].unique())

    with st.form("formularz"):
        k1, k2, k3 = st.columns(3)
        with k1:
            marka = st.selectbox("Marka", marki, index=marki.index("TOYOTA"))
            model_auta = st.text_input(
                "Nazwa modelu", "RAV4 AWD",
                help="Z nazwy wykrywane są: HYBRID, FFV oraz napęd 4x4 (AWD, 4WD, 4X4, 4MATIC, QUATTRO, XDRIVE).")
            klasa = st.selectbox("Klasa pojazdu", klasy, index=klasy.index("SUV - SMALL"))
        with k2:
            pojemnosc = st.number_input("Pojemność silnika [L]", min_value=0.8, max_value=8.5,
                                        value=2.5, step=0.1, format="%.1f")
            cylindry = st.selectbox("Liczba cylindrów", cylindry_lista, index=cylindry_lista.index(4))
        with k3:
            skrzynia = st.selectbox(
                "Skrzynia biegów", skrzynie, index=skrzynie.index("AS8"),
                help="Litery = typ skrzyni, liczba = liczba biegów. "
                     + ", ".join(f"{k}: {v}" for k, v in OPISY_SKRZYN.items()))
            paliwo = st.selectbox("Rodzaj paliwa", list(OPISY_PALIW), format_func=OPISY_PALIW.get)

        wyslij = st.form_submit_button("Oblicz emisję", type="primary")

    if wyslij:
        wiersz = pd.DataFrame([{
            "Make": marka, "Model": model_auta, "Vehicle Class": klasa,
            "Engine Size(L)": pojemnosc, "Cylinders": cylindry,
            "Transmission": skrzynia, "Fuel Type": paliwo,
        }])
        X = przygotuj_cechy(wiersz)[CECHY]
        wyniki = {nazwa: float(m.predict(X)[0]) for nazwa, m in modele.items()}

        st.divider()
        for kol, (nazwa, wartosc) in zip(st.columns(len(wyniki)), wyniki.items()):
            kol.metric(f"{nazwa} ⭐" if nazwa == najlepszy else nazwa, f"{wartosc:.0f} g/km",
                       help=f"Średni błąd modelu na zbiorze testowym: ±{tabela.loc[nazwa, 'MAE test']:.1f} g/km")

        sredni_co2 = dane[CEL].mean()
        roznica = wyniki[najlepszy] - sredni_co2
        st.info(f"Najlepszy model (**{najlepszy}**) szacuje **{wyniki[najlepszy]:.0f} g/km**, "
                f"czyli {abs(roznica):.0f} g/km {'powyżej' if roznica > 0 else 'poniżej'} średniej w zbiorze "
                f"({sredni_co2:.0f} g/km). Przy 20 000 km rocznie to ok. "
                f"**{wyniki[najlepszy] * 20000 / 1e6:.2f} t CO₂/rok**.")

        if (dane["Make"] == marka).sum() < 10:
            st.warning("Ta marka ma mało obserwacji w danych – predykcja może być mniej pewna.")

        with st.expander("Cechy przekazane do modelu"):
            st.dataframe(X, hide_index=True)

        podobne = dane[(dane["Make"] == marka) & (dane["Vehicle Class"] == klasa)
                       & ((dane["Engine Size(L)"] - pojemnosc).abs() <= 0.3)]
        if not podobne.empty:
            st.markdown("**Podobne auta w danych** (ta sama marka i klasa, pojemność ±0,3 L):")
            st.dataframe(podobne[["Make", "Model", "Engine Size(L)", "Cylinders", "Transmission",
                                  "Fuel Type", CEL]].drop_duplicates().head(10), hide_index=True)

# ----------------------------------------------------------------- WYNIKI
with zakladka_wyniki:
    st.subheader("Porównanie modeli")
    st.dataframe(tabela.round(3))
    st.caption("Walidacja = 5-krotna walidacja krzyżowa na zbiorze treningowym (wyniki z notebooka). "
               "Test = 20% danych odłożonych przed treningiem, liczone na zapisanych modelach. Błędy w g/km.")

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**MAE na zbiorze testowym [g/km]** (niżej = lepiej)")
        st.bar_chart(tabela["MAE test"], horizontal=True)
    with k2:
        st.markdown("**R² na zbiorze testowym** (wyżej = lepiej)")
        st.bar_chart(tabela["R² test"], horizontal=True)

    st.subheader("Predykcja vs wartość rzeczywista (zbiór testowy)")
    wybrany = st.selectbox("Model", list(tabela.index))
    wykres = pred_test[["rzeczywiste", wybrany]].rename(columns={wybrany: "przewidywane"})
    st.scatter_chart(wykres, x="rzeczywiste", y="przewidywane", height=450)

    bledy = pred_test.assign(**{"błąd bezwzględny": (pred_test[wybrany] - pred_test["rzeczywiste"]).abs()})
    st.markdown("**Największe błędy tego modelu**")
    st.dataframe(bledy.sort_values("błąd bezwzględny", ascending=False)
                 [["Make", "Vehicle Class", "Engine Size(L)", "Fuel Type", "rzeczywiste", wybrany, "błąd bezwzględny"]]
                 .head(10).round(1), hide_index=True)

# ----------------------------------------------------------------- DANE
with zakladka_dane:
    st.subheader("Zbiór danych")
    a, b, c = st.columns(3)
    a.metric("Wiersze", f"{len(dane):,}".replace(",", " "))
    b.metric("Marki", dane["Make"].nunique())
    c.metric("Średnia emisja", f"{dane[CEL].mean():.0f} g/km")

    k1, k2 = st.columns(2)
    with k1:
        st.markdown("**Średnia emisja CO₂ wg klasy pojazdu**")
        st.bar_chart(dane.groupby("Vehicle Class")[CEL].mean().sort_values(), horizontal=True)
    with k2:
        st.markdown("**Emisja CO₂ a pojemność silnika**")
        st.scatter_chart(dane, x="Engine Size(L)", y=CEL, color="Fuel Type", height=420)

    st.dataframe(dane, hide_index=True)
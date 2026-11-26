import pandas as pd
import streamlit as st

st.title("Predykcja emisji CO₂")

df = pd.read_csv("data/co2.csv")

predykcja, dane, modele = st.tabs(["Predykcja", "Dane", "Modele"])

with predykcja:
    marka = st.selectbox("Marka", sorted(df["Make"].str.upper().unique()))
    model = st.text_input("Model")
    klasa = st.selectbox("Klasa pojazdu", sorted(df["Vehicle Class"].unique()))
    pojemnosc = st.number_input("Pojemność silnika [L]", min_value=0.5, max_value=9.0, value=2.0, step=0.1)
    cylindry = st.selectbox("Liczba cylindrów", sorted(df["Cylinders"].unique()))
    skrzynia = st.selectbox("Skrzynia biegów", sorted(df["Transmission"].unique()))
    paliwo = st.selectbox("Rodzaj paliwa", ["X", "Z", "D", "E"])

    if st.button("Przewiduj"):
        st.write("Wprowadzone dane:")
        st.write({
            "Make": marka,
            "Model": model,
            "Vehicle Class": klasa,
            "Engine Size(L)": pojemnosc,
            "Cylinders": cylindry,
            "Transmission": skrzynia,
            "Fuel Type": paliwo,
        })

with dane:
    pass

with modele:
    pass
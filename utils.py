import streamlit as st
import pandas as pd
import numpy as np
from auth import login_user, register_user
import os
import matplotlib.pyplot as plt

def analiza_podatkov():
    st.title("Analiza filmov")

    # Naloži podatke
    movies = pd.read_csv("podatki/ml-latest-small/movies.csv")
    ratings = pd.read_csv("podatki/ml-latest-small/ratings.csv")

    # Združi podatke
    merged = ratings.merge(movies, on="movieId")

    # Izlušči leto iz naslova
    merged["year"] = merged["title"].str.extract(r"\((\d{4})\)").astype("Int64")
    merged["genres"] = merged["genres"].str.split("|")

    # Sidebar - Filtri
    st.sidebar.header("Filtri")
    min_ocen = st.sidebar.slider("Minimalno število ocen", min_value=1, max_value=100, value=10)

    # Vsi žanri
    unikatni_žanri = sorted({g for sublist in merged["genres"] for g in sublist if g != '(no genres listed)'})
    izbrani_žanri = st.sidebar.multiselect("Izberi žanre (lahko več):", unikatni_žanri)

    leta = merged["year"].dropna().sort_values().unique()
    izbrano_leto = st.sidebar.selectbox("Izberi leto (neobvezno):", options=[None] + leta.tolist())

    # Razširi žanre
    df = merged.explode("genres")

    # Združi po filmu
    grupirano = df.groupby(["movieId", "title", "year"]).agg(
        povprecje=("rating", "mean"),
        stevilo_ocen=("rating", "count")
    ).reset_index()

    # Filtriranje po žanrih (film mora vsebovati vse izbrane žanre)
    if izbrani_žanri:
        def vsebuje_vse_zanre(zanri_filma):
            return set(izbrani_žanri).issubset(set(zanri_filma))
        merged = merged[merged["genres"].apply(vsebuje_vse_zanre)]

    df = merged.explode("genres")

    # Združi po filmu (na novo!)
    grupirano = df.groupby(["movieId", "title", "year"]).agg(
        povprecje=("rating", "mean"),
        stevilo_ocen=("rating", "count")
    ).reset_index()

    # Filtriranje po letu
    if izbrano_leto:
        grupirano = grupirano[grupirano["year"] == izbrano_leto]

    # Filtriranje po številu ocen
    filtrirano = grupirano[grupirano["stevilo_ocen"] >= min_ocen]

    # Top 10 filmov
    top10 = filtrirano.sort_values(by="povprecje", ascending=False).head(10)

    st.write("Top 10 filmov glede na izbrane filtre:")
    st.dataframe(top10[["title", "year", "povprecje", "stevilo_ocen"]], use_container_width=True)


def primerjava_filmov():
    st.title("Primerjava dveh filmov")

    # Naloži podatke
    movies = pd.read_csv("podatki/ml-latest-small/movies.csv")
    ratings = pd.read_csv("podatki/ml-latest-small/ratings.csv")

    # Izberi filma
    film1 = st.selectbox("Izberi prvi film", movies["title"].unique())
    film2 = st.selectbox("Izberi drugi film", movies["title"].unique())

    if film1 == film2:
        st.warning("Izberi različna filma.")
        return

    # Funkcija za prikaz podatkov o filmu
    def film_stats(title):
        movie_id = movies[movies["title"] == title]["movieId"].values[0]
        film_ratings = ratings[ratings["movieId"] == movie_id]
        povp = film_ratings["rating"].mean()
        stevilo = film_ratings["rating"].count()
        std = film_ratings["rating"].std()
        return film_ratings, povp, stevilo, std

    # Pridobi podatke
    f1_data, f1_avg, f1_cnt, f1_std = film_stats(film1)
    f2_data, f2_avg, f2_cnt, f2_std = film_stats(film2)

    # Statistika
    st.subheader("Statistika")
    df_stats = pd.DataFrame({
        "Film": [film1, film2],
        "Povprečna ocena": [round(f1_avg, 2), round(f2_avg, 2)],
        "Število ocen": [f1_cnt, f2_cnt],
        "Standardni odklon": [round(f1_std, 2), round(f2_std, 2)]
    })
    st.table(df_stats)

    # Histogram ocen
    st.subheader("Histogram ocen")
    fig, ax = plt.subplots()
    ax.hist(f1_data["rating"], bins=10, alpha=0.5, label=film1)
    ax.hist(f2_data["rating"], bins=10, alpha=0.5, label=film2)
    ax.set_xlabel("Ocena")
    ax.set_ylabel("Pogostost")
    ax.legend()
    st.pyplot(fig)

    # Povprečna letna ocena
    st.subheader("Povprečna letna ocena")
    f1_yearly = f1_data.copy()
    f2_yearly = f2_data.copy()
    f1_yearly["year"] = pd.to_datetime(f1_yearly["timestamp"], unit="s").dt.year
    f2_yearly["year"] = pd.to_datetime(f2_yearly["timestamp"], unit="s").dt.year
    f1_avg_year = f1_yearly.groupby("year")["rating"].mean()
    f2_avg_year = f2_yearly.groupby("year")["rating"].mean()

    fig2, ax2 = plt.subplots()
    f1_avg_year.plot(ax=ax2, label=film1)
    f2_avg_year.plot(ax=ax2, label=film2)
    ax2.set_ylabel("Povprečna ocena")
    ax2.set_xlabel("Leto")
    ax2.legend()
    st.pyplot(fig2)

    # Število ocen na leto
    st.subheader("Število ocen na leto")
    f1_cnt_year = f1_yearly.groupby("year")["rating"].count()
    f2_cnt_year = f2_yearly.groupby("year")["rating"].count()

    fig3, ax3 = plt.subplots()
    f1_cnt_year.plot(ax=ax3, label=film1)
    f2_cnt_year.plot(ax=ax3, label=film2)
    ax3.set_ylabel("Število ocen")
    ax3.set_xlabel("Leto")
    ax3.legend()
    st.pyplot(fig3)

def priporocilni_sistem():
    st.title("Priporočilni sistem")

    izbor = st.sidebar.radio("Prijava / Registracija", ["Prijava", "Registracija"])
    uporabnik = st.sidebar.text_input("Uporabniško ime")
    geslo = st.sidebar.text_input("Geslo", type="password")

    if izbor == "Registracija":
        if st.sidebar.button("Registriraj"):
            if register_user(uporabnik, geslo):
                st.success("Registracija uspešna. Prijavi se.")
            else:
                st.error("Uporabnik že obstaja.")
        return

    if izbor == "Prijava":
        if st.sidebar.button("Prijavi"):
            if login_user(uporabnik, geslo):
                st.session_state["user"] = uporabnik
                st.success(f"Prijavljen kot {uporabnik}")
            else:
                st.error("Napačni podatki.")
        if "user" not in st.session_state:
            st.warning("Najprej se registriraj.")
            return

    ratings = pd.read_csv("podatki/ml-latest-small/ratings.csv")
    movies = pd.read_csv("podatki/ml-latest-small/movies.csv")
    ratings["userId"] = ratings["userId"].astype(str)

    current_user = st.session_state["user"]

    st.subheader("Ocenjevanje filmov")
    film = st.selectbox("Izberi film za oceno", movies["title"].unique())
    ocena = st.slider("Tvoja ocena", 0.5, 5.0, 3.0, 0.5)

    if st.button("Shrani oceno"):
        movieId = movies[movies["title"] == film]["movieId"].values[0]
        new_rating = pd.DataFrame({
            "userId": [current_user],
            "movieId": [movieId],
            "rating": [ocena],
            "timestamp": [int(pd.Timestamp.now().timestamp())]
        })
        ratings = pd.concat([ratings, new_rating], ignore_index=True)
        ratings.to_csv("podatki/ml-latest-small/ratings.csv", index=False)
        st.success("Ocena shranjena.")

    user_ratings = ratings[ratings["userId"] == current_user]
    if len(user_ratings) < 10:
        st.info("Vnesel si: {} ocen.".format(len(user_ratings)))
        st.info("Vnesi vsaj 10 ocen za priporočila.")
        return

    st.subheader("Priporočeni filmi zate")

    # Pivot tabela: uporabniki x filmi
    pivot = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    user_vector = pivot.loc[current_user]

    # Pearson korelacija uporabnika z ostalimi
    podobnosti = pivot.corrwith(user_vector, axis=1, method='pearson').dropna()
    podobni_uporabniki = podobnosti.drop(current_user, errors="ignore").sort_values(ascending=False).head(10)

    # Pridobi ocene teh podobnih uporabnikov
    podobni_ocene = pivot.loc[podobni_uporabniki.index]

    # Povprečna ocena vsakega filma med podobnimi uporabniki
    povprecne_ocene = podobni_ocene.mean()

    # Preštej koliko podobnih uporabnikov je ocenilo vsak film
    stevilo_ocen = podobni_ocene.count()

    # Filter: vsaj 5 podobnih uporabnikov mora oceniti film
    prag = 5
    kandidati = povprecne_ocene[stevilo_ocen >= prag]

    # Odstrani filme, ki jih je uporabnik že ocenil
    kandidati = kandidati.drop(user_vector.dropna().index, errors="ignore")

    # Top 10 priporočenih filmov
    top_filmi = kandidati.sort_values(ascending=False).head(10)

    # Prikaz
    priporoceni = movies[movies["movieId"].isin(top_filmi.index.astype(int))].copy()
    priporoceni["ocena"] = priporoceni["movieId"].map(top_filmi)

    st.table(priporoceni[["title", "ocena"]])


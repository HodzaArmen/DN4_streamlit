import streamlit as st
import utils as utils

st.set_page_config(page_title="MovieLens App", layout="wide")

stran = st.sidebar.selectbox("Izberite stran", ["1. Analiza", "2. Primerjava", "3. Priporočila"])

if stran == "1. Analiza":
    utils.analiza_podatkov()
elif stran == "2. Primerjava":
    utils.primerjava_filmov()
else:
    utils.priporocilni_sistem()
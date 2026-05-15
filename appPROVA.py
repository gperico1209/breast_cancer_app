import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Breast_Cancer.csv")

st.sidebar.title("Menu")

pagina = st.sidebar.radio(
    "Seleziona sezione",
    [
        "Home",
        "Overview Dataset",
        "Visualizzazioni",
        "Modelli",
        "What If",
        "Conclusioni"
    ]
)

# HOME
if pagina == "Home":

    st.title("Predittore di Sopravvivenza - Tumore al Seno")

    st.write("""
    Questa web app analizza il dataset SEER Breast Cancer
    utilizzando tecniche di data analysis e machine learning.
    """)

# OVERVIEW
elif pagina == "Overview Dataset":

    st.title("Overview Dataset")

    st.write(df.head())

# VISUALIZZAZIONI
elif pagina == "Visualizzazioni":

    st.title("Visualizzazioni")

# MODELLI
elif pagina == "Modelli":

    st.title("Modelli Predittivi")

# WHAT IF
elif pagina == "What If":

    st.title("Scenario What If")

# CONCLUSIONI
elif pagina == "Conclusioni":

    st.title("Conclusioni")

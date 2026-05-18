import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ======================================================
# CONFIGURAZIONE PAGINA
# ======================================================

st.set_page_config(
    page_title="Breast Cancer Analysis",
    layout="wide"
)


# ======================================================
# CARICAMENTO DATASET
# ======================================================

@st.cache_data
def load_data():
    df = pd.read_csv("Breast_Cancer.csv")
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    return df

df = load_data()


# ======================================================
# VARIABILI USATE
# ======================================================

target = "Status"

features = [
    "Age",
    "T Stage",
    "N Stage",
    "Grade",
    "Estrogen Status",
    "Progesterone Status"
]

df_model = df[features + [target]].copy()
df_model = df_model.dropna()


# ======================================================
# ENCODING ORDINALE
# ======================================================

t_stage_map = {
    "T1": 1,
    "T2": 2,
    "T3": 3,
    "T4": 4
}

n_stage_map = {
    "N1": 1,
    "N2": 2,
    "N3": 3
}

grade_map = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "I": 1,
    "II": 2,
    "III": 3,
    "IV": 4
}

df_model["T Stage"] = df_model["T Stage"].astype(str).map(t_stage_map)
df_model["N Stage"] = df_model["N Stage"].astype(str).map(n_stage_map)
df_model["Grade"] = df_model["Grade"].astype(str).map(grade_map)

df_model = df_model.dropna()


# ======================================================
# TARGET
# ======================================================

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df_model[target].astype(str))


# ======================================================
# FEATURES
# ======================================================

X_raw = df_model[features].copy()

X = pd.get_dummies(
    X_raw,
    columns=["Estrogen Status", "Progesterone Status"],
    drop_first=True
)

feature_names = X.columns


# ======================================================
# TRAIN TEST SPLIT
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ======================================================
# STANDARDIZZAZIONE

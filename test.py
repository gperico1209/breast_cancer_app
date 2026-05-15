import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ======================================================
# CONFIG PAGINA
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
    return df

df = load_data()

# ======================================================
# PREPROCESSING
# ======================================================

df_model = df.copy()

# Rimuove valori mancanti
df_model = df_model.dropna()

encoders = {}

# Trasforma tutte le colonne testuali in numeri
for col in df_model.columns:

    if df_model[col].dtype == "object":

        le = LabelEncoder()

        df_model[col] = le.fit_transform(
            df_model[col].astype(str)
        )

        encoders[col] = le

# ======================================================
# TARGET
# ======================================================

target = "Status"

# ======================================================
# MODELLO LOGISTICO
# ======================================================

if target in df_model.columns:

    X = df_model.drop(columns=[target])
    y = df_model[target]

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Standardizzazione
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Modello
    log_model = LogisticRegression(max_iter=5000)

    log_model.fit(X_train_scaled, y_train)

    # Predizioni
    y_pred = log_model.predict(X_test_scaled)

    y_prob = log_model.predict_proba(X_test_scaled)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Importance
    importance = pd.DataFrame({
        "Variabile": X.columns,
        "Peso": log_model.coef_[0]
    })

    importance["Importanza Assoluta"] = (
        importance["Peso"].abs()
    )

    importance = importance.sort_values(
        by="Importanza Assoluta",
        ascending=False
    )

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("Menu")

pagina = st.sidebar.radio(
    "Vai a:",
    [
        "Home",
        "Overview Dataset",
        "Visualizzazioni",
        "Correlazioni",
        "Regressione Logistica",
        "What If",
        "Conclusioni"
    ]
)

# ======================================================
# HOME
# ======================================================

if pagina == "Home":

    st.title("Predittore di Sopravvivenza - Breast Cancer")

    st.write("""
    Questa web app analizza un dataset clinico relativo al tumore al seno.
    
    L’obiettivo del progetto è:
    
    - comprendere la struttura del dataset;
    - analizzare le principali variabili cliniche;
    - studiare le correlazioni tra variabili;
    - costruire un modello di regressione logistica;
    - prevedere la sopravvivenza del paziente;
    - creare scenari what-if.
    """)

# ======================================================
# OVERVIEW DATASET
# ======================================================

elif pagina == "Overview Dataset":

    st.title("Overview Dataset")

    st.subheader("Dimensioni dataset")

    col1, col2 = st.columns(2)

    col1.metric("Numero righe", df.shape[0])
    col2.metric("Numero colonne", df.shape[1])

    st.subheader("Prime righe")

    st.dataframe(df.head())

    st.subheader("Tipologia variabili")

    tipi = pd.DataFrame({
        "Variabile": df.columns,
        "Tipo": df.dtypes.astype(str)
    })

    st.dataframe(tipi)

    st.subheader("Statistiche descrittive")

    st.dataframe(df.describe())

    st.subheader("Valori mancanti")

    missing = pd.DataFrame({
        "Variabile": df.columns,
        "Missing": df.isnull().sum()
    })

    st.dataframe(missing)

    st.write("""
    Questa sezione permette di comprendere la struttura generale del dataset.
    """)

# ======================================================
# VISUALIZZAZIONI
# ======================================================

elif pagina == "Visualizzazioni":

    st.title("Visualizzazioni")

    # STATUS
    if "Status" in df.columns:

        st.subheader("Distribuzione Status")

        counts = df["Status"].value_counts()

        fig, ax = plt.subplots()

        ax.bar(
            counts.index.astype(str),
            counts.values
        )

        ax.set_xlabel("Status")
        ax.set_ylabel("Frequenza")

        st.pyplot(fig)

    # AGE
    if "Age" in df.columns:

        st.subheader("Distribuzione Età")

        fig, ax = plt.subplots()

        ax.hist(df["Age"], bins=20)

        ax.set_xlabel("Età")
        ax.set_ylabel("Frequenza")

        st.pyplot(fig)

    # TUMOR SIZE
    if "Tumor Size" in df.columns:

        st.subheader("Distribuzione Tumor Size")

        fig, ax = plt.subplots()

        ax.hist(df["Tumor Size"], bins=20)

        ax.set_xlabel("Dimensione tumore")
        ax.set_ylabel("Frequenza")

        st.pyplot(fig)

# ======================================================
# CORRELAZIONI
# ======================================================

elif pagina == "Correlazioni":

    st.title("Matrice di Correlazione")

    corr = df_model.corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(corr)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    fig.colorbar(im)

    st.pyplot(fig)

    st.write("""
    La matrice di correlazione permette di osservare
    le relazioni tra le variabili del dataset.
    """)

# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

elif pagina == "Regressione Logistica":

    st.title("Regressione Logistica")

    st.subheader("Accuracy")

    st.metric(
        "Accuracy modello",
        round(accuracy, 4)
    )

    st.write("""
    L’accuracy misura la percentuale di osservazioni
    classificate correttamente dal modello.
    """)

    # MATRICE DI CONFUSIONE

    st.subheader("Matrice di Confusione")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")

    st.pyplot(fig)

    # REPORT

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    # IMPORTANZA VARIABILI

    st.subheader("Importanza Variabili")

    st.dataframe(importance)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        importance["Variabile"],
        importance["Peso"]
    )

    ax.invert_yaxis()

    ax.set_xlabel("Peso")

    st.pyplot(fig)

    st.write("""
    Le variabili con coefficiente più alto hanno
    maggiore impatto sulla previsione finale.
    """)

    # RISULTATI

    st.subheader("Predizioni del modello")

    risultati = X_test.copy()

    risultati["Valore Reale"] = y_test.values
    risultati["Valore Predetto"] = y_pred

    risultati["Probabilità Classe 0"] = y_prob[:, 0]
    risultati["Probabilità Classe 1"] = y_prob[:, 1]

    st.dataframe(risultati.head(20))

# ======================================================
# WHAT IF
# ======================================================

elif pagina == "What If":

    st.title("Scenario What If")

    input_data = {}

    for col in X.columns:

        # Variabili categoriche
        if col in encoders:

            valori = sorted(
                df[col].astype(str).unique()
            )

            scelta = st.selectbox(
                col,
                valori
            )

            input_data[col] = encoders[col].transform(
                [scelta]
            )[0]

        # Variabili numeriche
        else:

            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            input_data[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    # INPUT DATAFRAME

    input_df = pd.DataFrame([input_data])

    # SCALE

    input_scaled = scaler.transform(input_df)

    # PREDIZIONE

    pred = log_model.predict(input_scaled)[0]

    prob = log_model.predict_proba(input_scaled)[0]

    st.subheader("Risultato")

    if target in encoders:

        pred_label = encoders[target].inverse_transform(
            [pred]
        )[0]

    else:

        pred_label = pred

    st.success(f"Classe predetta: {pred_label}")

    st.write("Probabilità:")

    prob_df = pd.DataFrame({
        "Classe": log_model.classes_,
        "Probabilità": prob
    })

    if target in encoders:

        prob_df["Classe"] = encoders[target].inverse_transform(
            prob_df["Classe"].astype(int)
        )

    st.dataframe(prob_df)

# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    L’analisi del dataset ha permesso di:
    
    - comprendere la distribuzione delle variabili;
    - analizzare le correlazioni;
    - costruire un modello di regressione logistica;
    - prevedere lo stato del paziente;
    - identificare le variabili cliniche più importanti.
    
    La regressione logistica risulta particolarmente utile
    perché il problema analizzato è un problema di classificazione.
    """)

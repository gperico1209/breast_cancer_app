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
    return df

df = load_data()


# ======================================================
# CONTROLLO TARGET
# ======================================================

target = "Status"

if target not in df.columns:
    st.error("Errore: nel dataset non esiste la colonna Status.")
    st.stop()


# ======================================================
# PREPROCESSING SICURO
# ======================================================

df_model = df.copy()
df_model = df_model.dropna()

# Separazione target
y_raw = df_model[target]

# Encoding target Alive/Dead
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw.astype(str))

# Features
X_raw = df_model.drop(columns=[target])

# One-hot encoding per tutte le variabili categoriche
X = pd.get_dummies(X_raw, drop_first=True)

# Salvo nomi colonne finali
feature_names = X.columns

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Modello regressione logistica
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train_scaled, y_train)

# Predizioni
y_pred = log_model.predict(X_test_scaled)
y_prob = log_model.predict_proba(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

# Importanza variabili
importance = pd.DataFrame({
    "Variabile": feature_names,
    "Peso": log_model.coef_[0]
})

importance["Importanza assoluta"] = importance["Peso"].abs()

importance = importance.sort_values(
    by="Importanza assoluta",
    ascending=False
)


# ======================================================
# MENU LATERALE
# ======================================================

st.sidebar.title("Menu progetto")

pagina = st.sidebar.radio(
    "Seleziona sezione:",
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

    st.title("Predittore di Sopravvivenza - Tumore al Seno")

    st.write("""
    Questa web app analizza un dataset clinico relativo al tumore al seno.

    L’obiettivo è studiare le caratteristiche dei pazienti e costruire
    un modello di regressione logistica capace di prevedere lo stato finale
    del paziente: **Alive** oppure **Dead**.
    """)

    st.subheader("Struttura del progetto")

    st.write("""
    La web app è divisa in sezioni:

    1. overview del dataset;
    2. visualizzazione delle variabili;
    3. analisi delle correlazioni;
    4. regressione logistica;
    5. matrice di confusione;
    6. importanza delle variabili;
    7. scenario what-if;
    8. conclusioni.
    """)


# ======================================================
# OVERVIEW DATASET
# ======================================================

elif pagina == "Overview Dataset":

    st.title("Overview del Dataset")

    col1, col2 = st.columns(2)

    col1.metric("Numero righe", df.shape[0])
    col2.metric("Numero colonne", df.shape[1])

    st.subheader("Prime righe del dataset")
    st.dataframe(df.head())

    st.subheader("Variabili presenti")

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
        "Valori mancanti": df.isnull().sum()
    })

    st.dataframe(missing)

    st.write("""
    Questa sezione permette di comprendere la struttura generale del dataset:
    numero di osservazioni, numero di variabili, tipologia delle colonne
    e presenza di eventuali valori mancanti.
    """)


# ======================================================
# VISUALIZZAZIONI
# ======================================================

elif pagina == "Visualizzazioni":

    st.title("Visualizzazioni dei dati")

    if "Status" in df.columns:

        st.subheader("Distribuzione Status")

        counts = df["Status"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Status")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione Alive / Dead")
        st.pyplot(fig)

        st.write("""
        Questo grafico mostra la distribuzione dei pazienti tra le classi
        Alive e Dead. È utile per capire se il dataset è bilanciato.
        """)

    if "Age" in df.columns:

        st.subheader("Distribuzione Età")

        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=20)
        ax.set_xlabel("Età")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione dell'età")
        st.pyplot(fig)

    if "Tumor Size" in df.columns:

        st.subheader("Distribuzione Tumor Size")

        fig, ax = plt.subplots()
        ax.hist(df["Tumor Size"], bins=20)
        ax.set_xlabel("Dimensione tumore")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione dimensione tumore")
        st.pyplot(fig)

    stage_cols = [col for col in df.columns if "Stage" in col]

    for col in stage_cols:

        st.subheader(f"Distribuzione {col}")

        counts = df[col].value_counts()

        fig, ax = plt.subplots()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequenza")
        ax.set_title(f"Distribuzione {col}")
        st.pyplot(fig)


# ======================================================
# CORRELAZIONI
# ======================================================

elif pagina == "Correlazioni":

    st.title("Matrice di correlazione")

    st.write("""
    La matrice di correlazione mostra come le variabili numeriche
    sono legate tra loro.
    """)

    corr = X.corr()

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(corr)

    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_yticks(np.arange(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)

    fig.colorbar(im)

    ax.set_title("Matrice di correlazione")

    st.pyplot(fig)

    st.write("""
    Una correlazione positiva indica che due variabili tendono ad aumentare insieme.
    Una correlazione negativa indica che una variabile tende a diminuire quando
    l’altra aumenta.
    """)


# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

elif pagina == "Regressione Logistica":

    st.title("Regressione Logistica")

    st.write("""
    La regressione logistica è stata utilizzata per prevedere lo stato finale
    del paziente: Alive oppure Dead.
    """)

    st.subheader("Accuracy del modello")

    st.metric("Accuracy", round(accuracy, 4))

    st.write("""
    L’accuracy misura la percentuale di osservazioni classificate correttamente.
    """)

    st.subheader("Matrice di confusione")

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
    ax.set_title("Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(target_encoder.classes_)
    ax.set_yticklabels(target_encoder.classes_)

    st.pyplot(fig)

    st.write("""
    La matrice di confusione confronta i valori reali con quelli predetti.
    Le celle sulla diagonale rappresentano le classificazioni corrette.
    Le celle fuori diagonale rappresentano gli errori del modello.
    """)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    st.write("""
    Il classification report mostra precision, recall e f1-score.
    Queste metriche aiutano a valutare meglio il modello rispetto alla sola accuracy.
    """)

    st.subheader("Importanza delle variabili")

    st.dataframe(importance.head(20))

    fig, ax = plt.subplots(figsize=(10, 6))

    top_importance = importance.head(15)

    ax.barh(
        top_importance["Variabile"],
        top_importance["Peso"]
    )

    ax.invert_yaxis()
    ax.set_xlabel("Peso coefficiente")
    ax.set_ylabel("Variabile")
    ax.set_title("Variabili più importanti")

    st.pyplot(fig)

    st.write("""
    I coefficienti della regressione logistica indicano il peso delle variabili.
    Più il valore assoluto del coefficiente è alto, maggiore è l’importanza
    della variabile nel modello.

    Il segno positivo o negativo indica la direzione dell’effetto sulla classe
    positiva codificata dal modello.
    """)

    st.subheader("Predizioni dettagliate")

    risultati = X_test.copy()

    risultati["Valore reale"] = target_encoder.inverse_transform(y_test)
    risultati["Valore predetto"] = target_encoder.inverse_transform(y_pred)

    for i, classe in enumerate(target_encoder.classes_):
        risultati[f"Probabilità {classe}"] = y_prob[:, i]

    st.dataframe(risultati.head(30))


# ======================================================
# WHAT IF
# ======================================================

elif pagina == "What If":

    st.title("Scenario What If")

    st.write("""
    Modifica le variabili del paziente per simulare diversi scenari.
    """)

    input_raw = {}

    for col in X_raw.columns:

        if df[col].dtype == "object":

            valori = sorted(df[col].astype(str).unique())

            input_raw[col] = st.selectbox(
                col,
                valori
            )

        else:

            min_val = float(df[col].min())
            max_val = float(df[col].max())
            mean_val = float(df[col].mean())

            input_raw[col] = st.slider(
                col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )

    input_df_raw = pd.DataFrame([input_raw])

    input_encoded = pd.get_dummies(input_df_raw)

    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    pred = log_model.predict(input_scaled)[0]

    prob = log_model.predict_proba(input_scaled)[0]

    pred_label = target_encoder.inverse_transform([pred])[0]

    st.subheader("Risultato previsione")

    st.success(f"Classe predetta: {pred_label}")

    prob_df = pd.DataFrame({
        "Classe": target_encoder.classes_,
        "Probabilità": prob
    })

    st.dataframe(prob_df)

    st.write("""
    Lo scenario what-if permette di osservare come cambia la previsione
    modificando le caratteristiche cliniche del paziente.
    """)


# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    L’analisi ha permesso di studiare il dataset Breast Cancer attraverso
    visualizzazioni, correlazioni e regressione logistica.

    La regressione logistica è coerente con l’obiettivo del progetto perché
    il problema è di classificazione: prevedere se il paziente risulta Alive
    oppure Dead.

    La matrice di confusione e il classification report permettono di valutare
    le performance del modello, mentre l’importanza delle variabili consente
    di capire quali caratteristiche cliniche incidono maggiormente sulla previsione.

    La sezione What If rende la web app interattiva e permette di simulare
    diversi scenari modificando i dati del paziente.
    """)

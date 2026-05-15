import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# CONFIGURAZIONE PAGINA
# =========================

st.set_page_config(
    page_title="Breast Cancer Analysis",
    layout="wide"
)


# =========================
# CARICAMENTO DATI
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("Breast_Cancer.csv")
    df.columns = df.columns.str.strip()
    return df

df = load_data()


# =========================
# PREPROCESSING
# =========================

def prepare_model_data(df):
    df_model = df.copy()
    encoders = {}

    for col in df_model.columns:
        if df_model[col].dtype == "object":
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            encoders[col] = le

    return df_model, encoders

df_model, encoders = prepare_model_data(df)

target = "Status"

if target in df_model.columns:

    # Rimuove eventuali valori mancanti
    df_model = df_model.dropna()

    # Separazione features e target
    X = df_model.drop(columns=[target])
    y = df_model[target]

    # Converte tutto in numerico
    X = X.apply(pd.to_numeric)

    # Train test split
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


# =========================
# MODELLO LOGISTICO
# =========================



    log_model = LogisticRegression(max_iter=5000)
    log_model.fit(X_train_scaled, y_train)

    y_pred = log_model.predict(X_test_scaled)
    y_prob = log_model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)

    importance = pd.DataFrame({
        "Variabile": X.columns,
        "Peso": log_model.coef_[0]
    })

    importance["Importanza assoluta"] = importance["Peso"].abs()
    importance = importance.sort_values(
        by="Importanza assoluta",
        ascending=False
    )


# =========================
# SIDEBAR
# =========================

st.sidebar.title("Menu progetto")

pagina = st.sidebar.radio(
    "Seleziona una sezione:",
    [
        "Home",
        "Overview Dataset",
        "Visualizzazioni",
        "Correlazioni",
        "Regressione Logistica",
        "What-if Scenario",
        "Conclusioni"
    ]
)


# =========================
# HOME
# =========================

if pagina == "Home":

    st.title("Predittore di Sopravvivenza - Breast Cancer Dataset")

    st.write("""
    Questa web app analizza un dataset clinico relativo al tumore al seno.
    
    L’obiettivo è studiare le caratteristiche dei pazienti e costruire un modello
    di regressione logistica capace di prevedere lo stato finale del paziente,
    cioè se il paziente risulta **Alive** oppure **Dead**.
    """)

    st.subheader("Obiettivi del progetto")

    st.write("""
    Il progetto è strutturato in diverse parti:

    1. presentazione generale del dataset;
    2. analisi delle variabili;
    3. visualizzazione dei dati;
    4. analisi delle correlazioni;
    5. costruzione del modello di regressione logistica;
    6. valutazione del modello tramite accuracy e matrice di confusione;
    7. interpretazione del peso delle variabili;
    8. simulazione what-if.
    """)


# =========================
# OVERVIEW DATASET
# =========================

elif pagina == "Overview Dataset":

    st.title("Overview del Dataset")

    st.write("""
    In questa sezione viene presentata una visione generale del dataset.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Numero righe", df.shape[0])

    with col2:
        st.metric("Numero colonne", df.shape[1])

    st.subheader("Prime righe del dataset")
    st.dataframe(df.head())

    st.subheader("Variabili presenti")
    variabili = pd.DataFrame({
        "Variabile": df.columns,
        "Tipo": df.dtypes.astype(str)
    })
    st.dataframe(variabili)

    st.subheader("Statistiche descrittive")
    st.dataframe(df.describe())

    st.subheader("Valori mancanti")
    missing = df.isnull().sum().reset_index()
    missing.columns = ["Variabile", "Valori mancanti"]
    st.dataframe(missing)

    st.write("""
    Le statistiche descrittive permettono di osservare media, deviazione standard,
    valori minimi e massimi delle variabili numeriche.
    L’analisi dei valori mancanti serve invece per valutare la qualità del dataset.
    """)


# =========================
# VISUALIZZAZIONI
# =========================

elif pagina == "Visualizzazioni":

    st.title("Visualizzazioni dei dati")

    st.write("""
    In questa sezione vengono mostrati alcuni grafici utili per capire
    come sono distribuite le principali variabili del dataset.
    """)

    if "Status" in df.columns:
        st.subheader("Distribuzione dello stato del paziente")

        status_counts = df["Status"].value_counts()

        fig, ax = plt.subplots()
        ax.bar(status_counts.index.astype(str), status_counts.values)
        ax.set_xlabel("Status")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione Alive / Dead")
        st.pyplot(fig)

        st.write("""
        Questo grafico mostra quanti pazienti appartengono alla classe Alive
        e quanti alla classe Dead. È utile per capire se il dataset è bilanciato
        oppure se una classe è molto più rappresentata dell’altra.
        """)

    if "Age" in df.columns:
        st.subheader("Distribuzione dell’età")

        fig, ax = plt.subplots()
        ax.hist(df["Age"], bins=20)
        ax.set_xlabel("Età")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione dell’età")
        st.pyplot(fig)

        st.write("""
        L’istogramma dell’età permette di osservare in quali fasce di età
        si concentra maggiormente il campione.
        """)

    if "Tumor Size" in df.columns:
        st.subheader("Distribuzione della dimensione del tumore")

        fig, ax = plt.subplots()
        ax.hist(df["Tumor Size"], bins=20)
        ax.set_xlabel("Tumor Size")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione Tumor Size")
        st.pyplot(fig)

        st.write("""
        Questo grafico mostra come è distribuita la dimensione del tumore.
        Valori più elevati possono essere associati a situazioni cliniche più gravi.
        """)

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


# =========================
# CORRELAZIONI
# =========================

elif pagina == "Correlazioni":

    st.title("Analisi delle correlazioni")

    st.write("""
    La matrice di correlazione mostra quanto le variabili numeriche
    sono legate tra loro.
    """)

    corr = df_model.corr()

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
    Una correlazione negativa indica che quando una variabile aumenta, l’altra tende
    a diminuire.

    Nel contesto clinico, questa analisi è utile per capire se alcune variabili
    descrivono aspetti simili del tumore, ad esempio dimensione, stadio e gravità.
    """)


# =========================
# REGRESSIONE LOGISTICA
# =========================

elif pagina == "Regressione Logistica":

    st.title("Modello di Regressione Logistica")

    st.write("""
    La regressione logistica è stata utilizzata per prevedere lo stato finale
    del paziente, cioè se il paziente risulta Alive oppure Dead.
    """)

    if target not in df_model.columns:
        st.error("La colonna Status non è presente nel dataset.")

    else:
        st.subheader("Accuracy del modello")

        st.metric("Accuracy", round(accuracy, 4))

        st.write("""
        L’accuracy indica la percentuale di osservazioni classificate correttamente
        dal modello. Tuttavia, da sola non basta: per questo si analizza anche
        la matrice di confusione.
        """)

        st.subheader("Matrice di confusione")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        ax.imshow(cm)

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_xlabel("Valore predetto")
        ax.set_ylabel("Valore reale")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.write("""
        La matrice di confusione confronta i valori reali con quelli predetti.
        Le celle sulla diagonale rappresentano le predizioni corrette, mentre
        le celle fuori diagonale rappresentano gli errori del modello.
        """)

        st.subheader("Classification report")

        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

        st.write("""
        Il classification report mostra precision, recall e f1-score.
        Queste metriche sono utili per valutare meglio le performance del modello,
        soprattutto quando le classi non sono perfettamente bilanciate.
        """)

        st.subheader("Importanza delle variabili")

        st.dataframe(importance)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importance["Variabile"], importance["Peso"])
        ax.set_xlabel("Peso coefficiente")
        ax.set_ylabel("Variabile")
        ax.set_title("Peso delle variabili nella regressione logistica")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.write("""
        Il peso delle variabili indica quanto ciascuna caratteristica influenza
        la previsione del modello.

        - Un coefficiente positivo aumenta la probabilità della classe positiva.
        - Un coefficiente negativo riduce la probabilità della classe positiva.
        - Più il valore assoluto del coefficiente è alto, più la variabile è importante.

        È importante ricordare che l’interpretazione dipende da come il modello
        ha codificato la variabile target.
        """)

        st.subheader("Risultati dettagliati del modello")

        risultati = X_test.copy()
        risultati["Valore reale"] = y_test.values
        risultati["Valore predetto"] = y_pred
        risultati["Probabilità classe 0"] = y_prob[:, 0]
        risultati["Probabilità classe 1"] = y_prob[:, 1]

        st.dataframe(risultati.head(20))

        st.write("""
        Questa tabella permette di osservare, per ogni paziente del test set,
        il valore reale, il valore predetto e le probabilità associate alle classi.
        """)


# =========================
# WHAT IF SCENARIO
# =========================

elif pagina == "What-if Scenario":

    st.title("Scenario What-if")

    st.write("""
    In questa sezione è possibile modificare alcune variabili del paziente
    e osservare come cambia la previsione del modello.
    """)

    if target not in df_model.columns:
        st.error("La colonna Status non è presente nel dataset.")

    else:
        input_data = {}

        for col in X.columns:

            if df[col].dtype == "object":
                valori = sorted(df[col].astype(str).unique())
                scelta = st.selectbox(col, valori)

                le = encoders[col]
                input_data[col] = le.transform([scelta])[0]

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

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)

        predizione = log_model.predict(input_scaled)[0]
        probabilita = log_model.predict_proba(input_scaled)[0]

        st.subheader("Risultato previsione")

        if target in encoders:
            classe_predetta = encoders[target].inverse_transform([predizione])[0]
        else:
            classe_predetta = predizione

        st.write("Classe predetta:")
        st.success(classe_predetta)

        st.write("Probabilità associate:")
        prob_df = pd.DataFrame({
            "Classe": log_model.classes_,
            "Probabilità": probabilita
        })

        if target in encoders:
            prob_df["Classe"] = encoders[target].inverse_transform(
                prob_df["Classe"].astype(int)
            )

        st.dataframe(prob_df)

        st.write("""
        Lo scenario what-if permette di simulare diversi profili clinici
        e osservare come cambiano le probabilità previste dal modello.
        """)


# =========================
# CONCLUSIONI
# =========================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    L’analisi del dataset ha permesso di studiare le principali caratteristiche
    cliniche dei pazienti e di costruire un modello predittivo basato sulla
    regressione logistica.

    La regressione logistica è coerente con il problema analizzato perché
    l’obiettivo è classificare il paziente in due possibili stati finali:
    Alive oppure Dead.

    Attraverso la matrice di confusione e il classification report è possibile
    valutare la qualità predittiva del modello.

    L’analisi del peso delle variabili consente invece di comprendere quali
    caratteristiche cliniche incidono maggiormente sulla previsione finale.

    Infine, la sezione what-if rende la web app interattiva e permette di simulare
    diversi scenari modificando i valori delle variabili.
    """)

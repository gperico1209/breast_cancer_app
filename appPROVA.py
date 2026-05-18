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
    "Progesterone Status",
    "Reginol Node Positive"
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

# Controllo classi:
# Di solito LabelEncoder assegna:
# Alive = 0
# Dead = 1
classi_modello = list(target_encoder.classes_)


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
# ======================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

log_model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

log_model.fit(X_train_scaled, y_train)

y_pred = log_model.predict(X_test_scaled)
y_prob = log_model.predict_proba(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)


# ======================================================
# IMPORTANZA VARIABILI
# ======================================================

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
        "Variabili utilizzate",
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

    L’obiettivo è costruire un modello di regressione logistica capace di prevedere
    lo stato finale del paziente: **Alive** oppure **Dead**.
    """)

    st.subheader("Scelta metodologica")

    st.write("""
    Le variabili **T Stage**, **N Stage** e **Grade** sono state trasformate
    tramite encoding ordinale, perché rappresentano livelli progressivi
    di gravità clinica.

    In questo modo il modello interpreta correttamente che:

    - T4 è più grave di T1;
    - N3 è più grave di N1;
    - Grade 3 è più grave di Grade 1.
    """)

    st.subheader("Classi del modello")

    st.write("""
    Il modello lavora internamente con classi numeriche.  
    La tabella seguente mostra come sono state codificate le classi:
    """)

    classi_df = pd.DataFrame({
        "Codice numerico": range(len(target_encoder.classes_)),
        "Classe": target_encoder.classes_
    })

    st.dataframe(classi_df)


# ======================================================
# OVERVIEW DATASET
# ======================================================

elif pagina == "Overview Dataset":

    st.title("Overview del Dataset")

    col1, col2 = st.columns(2)

    col1.metric("Numero righe dataset originale", df.shape[0])
    col2.metric("Numero colonne dataset originale", df.shape[1])

    st.subheader("Prime righe del dataset originale")
    st.dataframe(df.head())

    st.subheader("Variabili presenti nel dataset originale")

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


# ======================================================
# VARIABILI UTILIZZATE
# ======================================================

elif pagina == "Variabili utilizzate":

    st.title("Variabili utilizzate nel modello")

    variabili_usate = pd.DataFrame({
        "Variabile": features,
        "Motivazione": [
            "Età del paziente.",
            "Classificazione clinica dell’estensione del tumore primario.",
            "Coinvolgimento dei linfonodi.",
            "Aggressività biologica del tumore.",
            "Presenza di recettori estrogeni.",
            "Presenza di recettori progesterone.",
            "Numero di linfonodi positivi."
        ]
    })

    st.dataframe(variabili_usate)

    st.subheader("Encoding applicato")

    st.write("""
    - T1 = 1, T2 = 2, T3 = 3, T4 = 4;
    - N1 = 1, N2 = 2, N3 = 3;
    - Grade 1 = 1, Grade 2 = 2, Grade 3 = 3, Grade 4 = 4.

    Le variabili ormonali sono invece trasformate in variabili dummy.
    """)


# ======================================================
# VISUALIZZAZIONI
# ======================================================

elif pagina == "Visualizzazioni":

    st.title("Visualizzazioni dei dati")

    st.subheader("Distribuzione Status")

    counts = df_model[target].value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_xlabel("Status")
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione Alive / Dead")
    st.pyplot(fig)

    st.subheader("Distribuzione Età")

    fig, ax = plt.subplots()
    ax.hist(df_model["Age"], bins=20)
    ax.set_xlabel("Età")
    ax.set_ylabel("Frequenza")
    ax.set_title("Distribuzione dell'età")
    st.pyplot(fig)

    for col in ["T Stage", "N Stage", "Grade", "Estrogen Status", "Progesterone Status"]:

        st.subheader(f"Distribuzione {col}")

        counts = df_model[col].value_counts().sort_index()

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
    La matrice di correlazione permette di osservare se alcune variabili sono
    fortemente legate tra loro.
    """)


# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

elif pagina == "Regressione Logistica":

    st.title("Regressione Logistica")

    st.subheader("Accuracy del modello")

    st.metric("Accuracy", round(accuracy, 4))

    st.subheader("Matrice di confusione")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()

    ax.imshow(cm)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Confusion Matrix")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(target_encoder.classes_)
    ax.set_yticklabels(target_encoder.classes_)

    st.pyplot(fig)

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)

    st.subheader("Importanza delle variabili")

    st.dataframe(importance)

    top_importance = importance.head(15)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        top_importance["Variabile"],
        top_importance["Peso"]
    )

    ax.invert_yaxis()
    ax.set_xlabel("Peso coefficiente")
    ax.set_ylabel("Variabile")
    ax.set_title("Peso delle variabili nella regressione logistica")

    st.pyplot(fig)

    st.write("""
    Attenzione: il segno del coefficiente va interpretato rispetto alla classe positiva
    codificata dal modello. Se la classe 1 è **Dead**, allora un coefficiente positivo
    aumenta la probabilità di morte, non di sopravvivenza.
    """)

    st.subheader("Codifica classi")

    classi_df = pd.DataFrame({
        "Codice numerico": range(len(target_encoder.classes_)),
        "Classe": target_encoder.classes_
    })

    st.dataframe(classi_df)

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
    Modifica le variabili cliniche selezionate per simulare diversi scenari
    e osservare come cambia la probabilità stimata di sopravvivenza.
    """)

    st.sidebar.header("Inserisci dati paziente")

    age = st.sidebar.slider(
        "Età",
        min_value=18,
        max_value=100,
        value=int(df_model["Age"].median())
    )

    t_stage_label = st.sidebar.selectbox(
        "T Stage",
        ["T1", "T2", "T3", "T4"]
    )

    n_stage_label = st.sidebar.selectbox(
        "N Stage",
        ["N1", "N2", "N3"]
    )

    grade_label = st.sidebar.selectbox(
        "Grade",
        ["1", "2", "3", "4"]
    )

    estrogen = st.sidebar.selectbox(
        "Estrogen Status",
        sorted(df["Estrogen Status"].astype(str).unique())
    )

    progesterone = st.sidebar.selectbox(
        "Progesterone Status",
        sorted(df["Progesterone Status"].astype(str).unique())
    )

    nodes_pos = st.sidebar.slider(
        "Reginol Node Positive",
        min_value=0,
        max_value=40,
        value=int(df_model["Reginol Node Positive"].median())
    )

    # Input numerico di base
    input_encoded = pd.DataFrame([{
        "Age": age,
        "T Stage": t_stage_map[t_stage_label],
        "N Stage": n_stage_map[n_stage_label],
        "Grade": grade_map[grade_label],
        "Reginol Node Positive": nodes_pos
    }])

    # Crea tutte le colonne mancanti
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Encoding manuale Estrogen Status
    for col in feature_names:
        if col.startswith("Estrogen Status_"):
            categoria = col.replace("Estrogen Status_", "")
            input_encoded[col] = 1 if estrogen == categoria else 0

    # Encoding manuale Progesterone Status
    for col in feature_names:
        if col.startswith("Progesterone Status_"):
            categoria = col.replace("Progesterone Status_", "")
            input_encoded[col] = 1 if progesterone == categoria else 0

    # Riordina colonne come nel training
    input_encoded = input_encoded[feature_names]

    # Standardizzazione identica al training
    input_scaled = scaler.transform(input_encoded)

    # Predizione
    pred = log_model.predict(input_scaled)[0]
    prob = log_model.predict_proba(input_scaled)[0]

    pred_label = target_encoder.inverse_transform([pred])[0]

    # Recupero sicuro degli indici delle classi
    classi = list(target_encoder.classes_)

    indice_alive = classi.index("Alive")
    indice_dead = classi.index("Dead")

    prob_alive = prob[indice_alive] * 100
    prob_dead = prob[indice_dead] * 100

    st.subheader("Risultato previsione")

    if pred_label == "Alive":
        st.success(f"Classe predetta: {pred_label}")
    else:
        st.error(f"Classe predetta: {pred_label}")

    st.metric(
        "Probabilità stimata di sopravvivenza",
        f"{prob_alive:.1f}%"
    )

    st.metric(
        "Probabilità stimata di morte",
        f"{prob_dead:.1f}%"
    )

    st.progress(int(prob_alive))

    st.subheader("Probabilità per classe")

    prob_df = pd.DataFrame({
        "Classe": target_encoder.classes_,
        "Probabilità": prob * 100
    })

    st.dataframe(prob_df)

    st.subheader("Controllo tecnico classi")

    st.write("Ordine classi usato dal modello:")
    st.write(target_encoder.classes_)

    st.write("Probabilità raw del modello:")
    st.write(prob)

    st.info("""
    Questa versione calcola la probabilità di sopravvivenza prendendo esplicitamente
    la probabilità associata alla classe **Alive**, quindi non confonde la classe 0
    con la classe 1.
    """)


# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    Il modello finale utilizza una selezione di variabili cliniche rilevanti
    per prevedere lo stato finale del paziente.

    T Stage, N Stage e Grade sono state codificate come variabili ordinali,
    perché rappresentano livelli progressivi di gravità.

    Le variabili ormonali sono state mantenute perché rappresentano informazioni
    cliniche rilevanti nella valutazione della prognosi.

    La regressione logistica risulta adatta perché il problema è di classificazione:
    prevedere se il paziente appartiene alla classe Alive oppure Dead.

    Nello scenario What If, la probabilità di sopravvivenza viene calcolata
    prendendo direttamente la probabilità associata alla classe **Alive**.
    """)

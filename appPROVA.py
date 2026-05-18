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
# VARIABILI MODELLO
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


# ======================================================
# DATAFRAME MODELLO
# ======================================================

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


df_model["T Stage"] = (
    df_model["T Stage"]
    .astype(str)
    .map(t_stage_map)
)

df_model["N Stage"] = (
    df_model["N Stage"]
    .astype(str)
    .map(n_stage_map)
)

df_model["Grade"] = (
    df_model["Grade"]
    .astype(str)
    .map(grade_map)
)

df_model = df_model.dropna()


# ======================================================
# TARGET
# ======================================================

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(
    df_model[target].astype(str)
)

classi_modello = list(target_encoder.classes_)


# ======================================================
# FEATURES
# ======================================================

X_raw = df_model[features].copy()

X = pd.get_dummies(
    X_raw,
    columns=[
        "Estrogen Status",
        "Progesterone Status"
    ],
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
# MODELLO REGRESSIONE LOGISTICA
# ======================================================

log_model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced"
)

log_model.fit(
    X_train_scaled,
    y_train
)


# ======================================================
# PREDIZIONI
# ======================================================

y_pred = log_model.predict(X_test_scaled)

y_prob = log_model.predict_proba(X_test_scaled)

accuracy = accuracy_score(
    y_test,
    y_pred
)


# ======================================================
# IMPORTANZA VARIABILI
# ======================================================

importance = pd.DataFrame({
    "Variabile": feature_names,
    "Peso": log_model.coef_[0]
})

importance["Importanza assoluta"] = (
    importance["Peso"].abs()
)

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

    st.title(
        "Predittore di Sopravvivenza - Tumore al Seno"
    )

    st.write("""
    Questa web app analizza un dataset clinico
    relativo al tumore al seno.

    L’obiettivo è costruire un modello di
    regressione logistica capace di prevedere
    lo stato finale del paziente:
    Alive oppure Dead.
    """)

    st.subheader("Scelta metodologica")

    st.write("""
    Le variabili T Stage, N Stage e Grade
    sono state trasformate tramite encoding
    ordinale, perché rappresentano livelli
    progressivi di gravità clinica.

    In questo modo il modello interpreta
    correttamente che:

    - T4 è più grave di T1
    - N3 è più grave di N1
    - Grade 3 è più grave di Grade 1
    """)

    st.subheader("Codifica classi")

    classi_df = pd.DataFrame({
        "Codice numerico":
            range(len(target_encoder.classes_)),
        "Classe":
            target_encoder.classes_
    })

    st.dataframe(classi_df)


# ======================================================
# OVERVIEW DATASET
# ======================================================

elif pagina == "Overview Dataset":

    st.title("Overview Dataset")

    col1, col2 = st.columns(2)

    col1.metric(
        "Numero righe",
        df.shape[0]
    )

    col2.metric(
        "Numero colonne",
        df.shape[1]
    )

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


# ======================================================
# VARIABILI UTILIZZATE
# ======================================================

elif pagina == "Variabili utilizzate":

    st.title("Variabili utilizzate")

    variabili_usate = pd.DataFrame({

        "Variabile": features,

        "Motivazione": [

            "Età del paziente",

            "Estensione del tumore primario",

            "Coinvolgimento dei linfonodi",

            "Aggressività biologica",

            "Presenza recettori estrogeni",

            "Presenza recettori progesterone"
        ]
    })

    st.dataframe(variabili_usate)

    st.subheader("Encoding applicato")

    st.write("""
    T Stage:
    - T1 = 1
    - T2 = 2
    - T3 = 3
    - T4 = 4

    N Stage:
    - N1 = 1
    - N2 = 2
    - N3 = 3

    Grade:
    - 1 = basso grado
    - 4 = grado elevato
    """)


# ======================================================
# VISUALIZZAZIONI
# ======================================================

elif pagina == "Visualizzazioni":

    st.title("Visualizzazioni")

    st.subheader("Distribuzione Status")

    counts = (
        df_model[target]
        .value_counts()
    )

    fig, ax = plt.subplots()

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_xlabel("Status")
    ax.set_ylabel("Frequenza")

    st.pyplot(fig)

    st.subheader("Distribuzione Età")

    fig, ax = plt.subplots()

    ax.hist(
        df_model["Age"],
        bins=20
    )

    ax.set_xlabel("Età")
    ax.set_ylabel("Frequenza")

    st.pyplot(fig)

    for col in [
        "T Stage",
        "N Stage",
        "Grade",
        "Estrogen Status",
        "Progesterone Status"
    ]:

        st.subheader(
            f"Distribuzione {col}"
        )

        counts = (
            df_model[col]
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots()

        ax.bar(
            counts.index.astype(str),
            counts.values
        )

        ax.set_xlabel(col)
        ax.set_ylabel("Frequenza")

        st.pyplot(fig)


# ======================================================
# CORRELAZIONI
# ======================================================

elif pagina == "Correlazioni":

    st.title("Matrice di correlazione")

    corr = X.corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    im = ax.imshow(
        corr,
        cmap="coolwarm"
    )

    ax.set_xticks(
        np.arange(len(corr.columns))
    )

    ax.set_yticks(
        np.arange(len(corr.columns))
    )

    ax.set_xticklabels(
        corr.columns,
        rotation=90
    )

    ax.set_yticklabels(
        corr.columns
    )

    # Numeri dentro celle
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):

            valore = round(
                corr.iloc[i, j],
                2
            )

            ax.text(
                j,
                i,
                valore,
                ha="center",
                va="center",
                color="black",
                fontsize=8
            )

    fig.colorbar(im)

    ax.set_title(
        "Matrice di correlazione"
    )

    st.pyplot(fig)

    st.write("""
    Valori vicini a:
    - +1 → forte correlazione positiva
    - -1 → forte correlazione negativa
    - 0 → correlazione assente
    """)


# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

elif pagina == "Regressione Logistica":

    st.title(
        "Regressione Logistica"
    )

    st.subheader(
        "Accuracy del modello"
    )

    st.metric(
        "Accuracy",
        round(accuracy, 4)
    )

    st.subheader(
        "Matrice di confusione"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

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

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        target_encoder.classes_
    )

    ax.set_yticklabels(
        target_encoder.classes_
    )

    st.pyplot(fig)

    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        target_names=
            target_encoder.classes_,
        output_dict=True
    )

    report_df = (
        pd.DataFrame(report)
        .transpose()
    )

    st.dataframe(report_df)

    st.subheader(
        "Importanza Variabili"
    )

    st.dataframe(importance)

    top_importance = importance.head(15)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        top_importance["Variabile"],
        top_importance["Peso"]
    )

    ax.invert_yaxis()

    ax.set_xlabel("Peso")

    st.pyplot(fig)

    st.write("""
    Se la classe positiva del modello
    è Dead:

    - coefficiente positivo
      → aumenta rischio morte

    - coefficiente negativo
      → aumenta probabilità sopravvivenza
    """)


# ======================================================
# WHAT IF
# ======================================================

elif pagina == "What If":

    st.title("Scenario What If")

    st.write("""
    Modifica le variabili cliniche
    per simulare diversi scenari.
    """)

    st.sidebar.header(
        "Inserisci dati paziente"
    )

    age = st.sidebar.slider(
        "Età",
        min_value=18,
        max_value=100,
        value=int(
            df_model["Age"].median()
        )
    )

    t_stage_label = (
        st.sidebar.selectbox(
            "T Stage",
            ["T1", "T2", "T3", "T4"]
        )
    )

    n_stage_label = (
        st.sidebar.selectbox(
            "N Stage",
            ["N1", "N2", "N3"]
        )
    )

    grade_label = (
        st.sidebar.selectbox(
            "Grade",
            ["1", "2", "3", "4"]
        )
    )

    estrogen = (
        st.sidebar.selectbox(
            "Estrogen Status",
            sorted(
                df["Estrogen Status"]
                .astype(str)
                .unique()
            )
        )
    )

    progesterone = (
        st.sidebar.selectbox(
            "Progesterone Status",
            sorted(
                df["Progesterone Status"]
                .astype(str)
                .unique()
            )
        )
    )

    # INPUT BASE
    input_encoded = pd.DataFrame([{

        "Age": age,

        "T Stage":
            t_stage_map[t_stage_label],

        "N Stage":
            n_stage_map[n_stage_label],

        "Grade":
            grade_map[grade_label]

    }])

    # CREA COLONNE MANCANTI
    for col in feature_names:

        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # ENCODING ESTROGEN
    for col in feature_names:

        if col.startswith(
            "Estrogen Status_"
        ):

            categoria = col.replace(
                "Estrogen Status_",
                ""
            )

            input_encoded[col] = (
                1 if estrogen == categoria
                else 0
            )

    # ENCODING PROGESTERONE
    for col in feature_names:

        if col.startswith(
            "Progesterone Status_"
        ):

            categoria = col.replace(
                "Progesterone Status_",
                ""
            )

            input_encoded[col] = (
                1 if progesterone
                == categoria
                else 0
            )

    input_encoded = (
        input_encoded[feature_names]
    )

    # STANDARDIZZAZIONE
    input_scaled = scaler.transform(
        input_encoded
    )

    # PREDIZIONE
    pred = log_model.predict(
        input_scaled
    )[0]

    prob = log_model.predict_proba(
        input_scaled
    )[0]

    pred_label = (
        target_encoder
        .inverse_transform([pred])[0]
    )

    # RECUPERO CLASSI
    classi = list(
        target_encoder.classes_
    )

    indice_alive = (
        classi.index("Alive")
    )

    indice_dead = (
        classi.index("Dead")
    )

    prob_alive = (
        prob[indice_alive] * 100
    )

    prob_dead = (
        prob[indice_dead] * 100
    )

    st.subheader(
        "Risultato previsione"
    )

    if pred_label == "Alive":

        st.success(
            f"Classe predetta: "
            f"{pred_label}"
        )

    else:

        st.error(
            f"Classe predetta: "
            f"{pred_label}"
        )

    st.metric(
        "Probabilità sopravvivenza",
        f"{prob_alive:.1f}%"
    )

    st.metric(
        "Probabilità morte",
        f"{prob_dead:.1f}%"
    )

    st.progress(int(prob_alive))

    st.subheader(
        "Probabilità per classe"
    )

    prob_df = pd.DataFrame({

        "Classe":
            target_encoder.classes_,

        "Probabilità":
            prob * 100
    })

    st.dataframe(prob_df)

    st.info("""
    Lo scenario What If utilizza:
    - encoding ordinale
    - stesso scaler del training
    - stesso modello logistico
    """)


# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    Il modello utilizza una regressione
    logistica per prevedere lo stato
    finale del paziente.

    Le variabili ordinali sono state
    codificate manualmente per rispettare
    la progressione clinica della malattia.

    Le probabilità visualizzate nel
    What If sono direttamente collegate
    al modello addestrato.
    """)

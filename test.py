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
# ENCODING ORDINALE PER VARIABILI CLINICHE
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
# PREPROCESSING
# ======================================================

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(df_model[target].astype(str))

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
# MODELLO REGRESSIONE LOGISTICA
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
    In questa versione del modello sono state mantenute le variabili cliniche principali:

    - Age;
    - T Stage;
    - N Stage;
    - Grade;
    - Estrogen Status;
    - Progesterone Status;
    - Reginol Node Positive.

    Le variabili **T Stage**, **N Stage** e **Grade** sono state trasformate con
    encoding ordinale, perché rappresentano livelli progressivi di gravità clinica.

    In questo modo il modello interpreta correttamente che:

    - T4 è più grave di T1;
    - N3 è più grave di N1;
    - Grade 3 è più grave di Grade 1.
    """)


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
    Per evitare che il modello tratti variabili ordinate come semplici categorie casuali,
    sono state codificate manualmente:

    - T1 = 1, T2 = 2, T3 = 3, T4 = 4;
    - N1 = 1, N2 = 2, N3 = 3;
    - Grade 1 = 1, Grade 2 = 2, Grade 3 = 3, Grade 4 = 4.

    Le variabili ormonali, invece, sono state trasformate tramite one-hot encoding.
    """)

    st.subheader("Variabili escluse")

    variabili_escluse = pd.DataFrame({
        "Variabile esclusa": [
            "Tumor Size",
            "6th Stage",
            "A Stage",
            "differentiate"
        ],
        "Motivo esclusione": [
            "Ridondante con T Stage.",
            "Non inserita per ora, perché riassume informazioni già contenute in T Stage e N Stage.",
            "Troppo generale rispetto alle altre variabili di stadio.",
            "Ridondante con Grade."
        ]
    })

    st.dataframe(variabili_escluse)


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
    fortemente legate tra loro. Una correlazione elevata può indicare ridondanza
    informativa o multicollinearità.
    """)


# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

elif pagina == "Regressione Logistica":

    st.title("Regressione Logistica")

    st.write("""
    La regressione logistica è stata utilizzata per prevedere lo stato finale
    del paziente: **Alive** oppure **Dead**.
    """)

    st.subheader("Accuracy del modello")

    st.metric("Accuracy", round(accuracy, 4))

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
    Il peso delle variabili indica quanto ogni caratteristica influenza
    la previsione del modello.

    L’importanza assoluta indica quali variabili pesano di più,
    indipendentemente dalla direzione dell’effetto.
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
    Modifica le variabili cliniche selezionate per simulare diversi scenari
    e osservare come cambia la probabilità stimata di sopravvivenza.
    """)

    st.sidebar.header("Inserisci dati paziente")

    age = st.sidebar.slider(
        "Età",
        min_value=int(df_model["Age"].min()),
        max_value=int(df_model["Age"].max()),
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
        min_value=int(df_model["Reginol Node Positive"].min()),
        max_value=int(df_model["Reginol Node Positive"].max()),
        value=int(df_model["Reginol Node Positive"].median())
    )

    user_data = {
        "Age": age,
        "T Stage": t_stage_map[t_stage_label],
        "N Stage": n_stage_map[n_stage_label],
        "Grade": grade_map[grade_label],
        "Estrogen Status": estrogen,
        "Progesterone Status": progesterone,
        "Reginol Node Positive": nodes_pos
    }

    input_df_raw = pd.DataFrame([user_data])

    input_encoded = pd.get_dummies(
        input_df_raw,
        columns=["Estrogen Status", "Progesterone Status"],
        drop_first=True
    )

    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    pred = log_model.predict(input_scaled)[0]
    prob = log_model.predict_proba(input_scaled)[0]

    pred_label = target_encoder.inverse_transform([pred])[0]

    st.subheader("Risultato previsione")

    if pred_label == "Alive":
        st.success(f"Classe predetta: {pred_label}")
    else:
        st.error(f"Classe predetta: {pred_label}")

    prob_df = pd.DataFrame({
        "Classe": target_encoder.classes_,
        "Probabilità": prob * 100
    })

    st.dataframe(prob_df)

    if "Alive" in target_encoder.classes_:
        prob_alive = prob_df.loc[
            prob_df["Classe"] == "Alive",
            "Probabilità"
        ].values[0]

        st.metric(
            "Probabilità stimata di sopravvivenza",
            f"{prob_alive:.1f}%"
        )

        st.progress(int(prob_alive))

    st.info("""
    Grazie all’encoding ordinale, il modello interpreta correttamente il peggioramento
    progressivo di T Stage, N Stage e Grade.
    """)


# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    Il modello finale utilizza una selezione di variabili cliniche rilevanti
    per prevedere lo stato finale del paziente.

    In questa versione, T Stage, N Stage e Grade sono state codificate come variabili
    ordinali. Questa scelta è più coerente dal punto di vista clinico, perché tali
    variabili rappresentano livelli progressivi di gravità.

    Sono state escluse variabili potenzialmente ridondanti come Tumor Size,
    6th Stage, A Stage e differentiate, in modo da rendere il modello più semplice
    e interpretabile.

    La regressione logistica risulta adatta perché il problema è di classificazione:
    prevedere se il paziente appartiene alla classe Alive oppure Dead.
    """)

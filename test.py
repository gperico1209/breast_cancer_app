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
# SELEZIONE VARIABILI NON RIDONDANTI
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
# PREPROCESSING
# ======================================================

y_raw = df_model[target]

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y_raw.astype(str))

X_raw = df_model[features]

X = pd.get_dummies(X_raw, drop_first=True)

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
    Per evitare ridondanza informativa e problemi di multicollinearità,
    sono state escluse alcune variabili che descrivono concetti clinici molto simili.

    In particolare sono state rimosse:

    - **Tumor Size**, perché molto legata a T Stage;
    - **6th Stage**, perché sintetizza già T Stage e N Stage;
    - **A Stage**, perché rappresenta una classificazione generale della diffusione;
    - **differentiate**, perché molto simile a Grade.

    Il modello finale utilizza quindi un numero più limitato di variabili,
    ma più interpretabili dal punto di vista clinico.
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

    st.write("""
    Il modello non utilizza tutte le variabili del dataset originale.
    Sono state selezionate solo le variabili considerate più informative
    e meno ridondanti.
    """)

    variabili_usate = pd.DataFrame({
        "Variabile": features,
        "Motivazione": [
            "Età del paziente, utile per valutare il rischio clinico.",
            "Classificazione clinica dell’estensione del tumore primario.",
            "Indica il coinvolgimento dei linfonodi.",
            "Misura l’aggressività biologica del tumore.",
            "Indica la presenza di recettori estrogeni.",
            "Indica la presenza di recettori progesterone.",
            "Numero di linfonodi risultati positivi."
        ]
    })

    st.dataframe(variabili_usate)

    st.subheader("Variabili escluse")

    variabili_escluse = pd.DataFrame({
        "Variabile esclusa": [
            "Tumor Size",
            "6th Stage",
            "A Stage",
            "differentiate"
        ],
        "Motivo esclusione": [
            "Ridondante con T Stage, che sintetizza meglio l’estensione tumorale.",
            "Ridondante perché deriva dalla combinazione di T Stage e N Stage.",
            "Ridondante perché descrive in modo generale la diffusione della malattia.",
            "Ridondante con Grade, perché entrambe misurano aggressività/differenziazione cellulare."
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

    if "Age" in df_model.columns:

        st.subheader("Distribuzione Età")

        fig, ax = plt.subplots()
        ax.hist(df_model["Age"], bins=20)
        ax.set_xlabel("Età")
        ax.set_ylabel("Frequenza")
        ax.set_title("Distribuzione dell'età")
        st.pyplot(fig)

    for col in ["T Stage", "N Stage", "Grade", "Estrogen Status", "Progesterone Status"]:

        st.subheader(f"Distribuzione {col}")

        counts = df_model[col].value_counts()

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
    La matrice di correlazione viene calcolata sulle variabili effettivamente usate
    dal modello dopo la trasformazione delle variabili categoriche.
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
    Questa analisi consente di osservare se alcune variabili sono ancora molto legate
    tra loro. Una correlazione elevata può indicare possibile ridondanza informativa.
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

    st.write("""
    La matrice di confusione confronta i valori reali con quelli predetti.
    Le celle sulla diagonale rappresentano le classificazioni corrette,
    mentre le celle fuori diagonale rappresentano gli errori del modello.
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

    Un coefficiente positivo aumenta la probabilità della classe positiva
    codificata dal modello, mentre un coefficiente negativo la riduce.

    L’importanza assoluta permette invece di capire quali variabili hanno
    maggiore impatto, indipendentemente dalla direzione dell’effetto.
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

    t_stage = st.sidebar.selectbox(
        "T Stage",
        sorted(df_model["T Stage"].astype(str).unique())
    )

    n_stage = st.sidebar.selectbox(
        "N Stage",
        sorted(df_model["N Stage"].astype(str).unique())
    )

    grade = st.sidebar.selectbox(
        "Grade",
        sorted(df_model["Grade"].astype(str).unique())
    )

    estrogen = st.sidebar.selectbox(
        "Estrogen Status",
        sorted(df_model["Estrogen Status"].astype(str).unique())
    )

    progesterone = st.sidebar.selectbox(
        "Progesterone Status",
        sorted(df_model["Progesterone Status"].astype(str).unique())
    )

    nodes_pos = st.sidebar.slider(
        "Reginol Node Positive",
        min_value=int(df_model["Reginol Node Positive"].min()),
        max_value=int(df_model["Reginol Node Positive"].max()),
        value=int(df_model["Reginol Node Positive"].median())
    )

    user_data = {
        "Age": age,
        "T Stage": t_stage,
        "N Stage": n_stage,
        "Grade": grade,
        "Estrogen Status": estrogen,
        "Progesterone Status": progesterone,
        "Reginol Node Positive": nodes_pos
    }

    input_df_raw = pd.DataFrame([user_data])

    input_encoded = pd.get_dummies(input_df_raw, drop_first=True)

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
    Lo scenario What If permette di modificare le variabili cliniche principali
    e osservare come cambia la previsione del modello.
    """)


# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    Il modello finale utilizza una selezione di variabili cliniche non ridondanti
    per prevedere lo stato finale del paziente.

    Sono state escluse variabili che rappresentavano informazioni già presenti
    in altre colonne, come Tumor Size, 6th Stage, A Stage e differentiate.

    Questa scelta permette di:

    - ridurre la multicollinearità;
    - rendere il modello più interpretabile;
    - evitare coefficienti instabili;
    - migliorare la coerenza clinica degli scenari What If.

    La regressione logistica risulta adatta perché il problema è di classificazione:
    prevedere se il paziente appartiene alla classe Alive oppure Dead.
    """)# One-hot encoding per tutte le variabili categoriche
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

# ======================================================
# WHAT IF
# ======================================================

elif pagina == "What If":

    st.title("🩺 Predittore di Sopravvivenza - Tumore al Seno (SEER)")

    st.markdown("""
    Modifica i parametri clinici nella barra laterale per simulare diversi scenari.
    """)

    st.sidebar.header("📊 Inserisci Dati Paziente")

    age = st.sidebar.slider("Età", min_value=30, max_value=90, value=50)

    t_stage = st.sidebar.selectbox(
        "Stadio Tumore (T Stage)",
        options=["T1", "T2", "T3", "T4"]
    )

    n_stage = st.sidebar.selectbox(
        "Stadio Linfonodi (N Stage)",
        options=["N1", "N2", "N3"]
    )

    estrogen = st.sidebar.selectbox(
        "Recettori Estrogeni",
        options=["Positive", "Negative"]
    )

    tumor_size = st.sidebar.slider(
        "Dimensione Tumore (mm)",
        min_value=1,
        max_value=100,
        value=20
    )

    nodes_pos = st.sidebar.slider(
        "Linfonodi Positivi",
        min_value=0,
        max_value=40,
        value=1
    )

    user_data = {
        "Age": age,
        "T Stage": t_stage,
        "N Stage": n_stage,
        "6th Stage": "IIA",
        "differentiate": "Moderately differentiated",
        "Grade": "2",
        "A Stage": "Regional",
        "Tumor Size": tumor_size,
        "Estrogen Status": estrogen,
        "Progesterone Status": "Positive",
        "Regional Node Examined": 10,
        "Reginol Node Positive": nodes_pos
    }

    df_user_raw = pd.DataFrame([user_data])

    input_encoded = pd.get_dummies(df_user_raw)

    input_encoded = input_encoded.reindex(
        columns=feature_names,
        fill_value=0
    )

    input_scaled = scaler.transform(input_encoded)

    probabilita = log_model.predict_proba(input_scaled)[0]

    classi = target_encoder.classes_

    prob_df = pd.DataFrame({
        "Classe": classi,
        "Probabilità": probabilita
    })

    st.divider()

    st.subheader("Risultato Stimato")

    if "Alive" in classi:
        prob_vivo = prob_df.loc[
            prob_df["Classe"] == "Alive",
            "Probabilità"
        ].values[0] * 100
    else:
        prob_vivo = probabilita.max() * 100

    if prob_vivo > 50:
        st.success("### 🎉 Prognosi Favorevole: Alive")
        st.metric(
            label="Probabilità di Sopravvivenza Stimata",
            value=f"{prob_vivo:.1f}%"
        )
    else:
        st.error("### ⚠️ Rischio Elevato: Dead")
        st.metric(
            label="Probabilità di Sopravvivenza Stimata",
            value=f"{prob_vivo:.1f}%",
            delta="- Rischio Critico",
            delta_color="inverse"
        )

    st.progress(int(prob_vivo))

    st.subheader("Probabilità per classe")

    prob_df["Probabilità"] = prob_df["Probabilità"] * 100

    st.dataframe(prob_df)

    st.info("""
    💡 Prova a cambiare lo Stadio dei Linfonodi, i Recettori Estrogeni,
    la Dimensione del Tumore o il numero di Linfonodi Positivi per osservare
    come cambia la probabilità di sopravvivenza in tempo reale.
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

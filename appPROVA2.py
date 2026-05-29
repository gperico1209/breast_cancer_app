import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    f1_score
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

y_pred = log_model.predict(X_test_scaled)
y_prob = log_model.predict_proba(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)


# ======================================================
# MODELLO RANDOM FOREST
# ======================================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_train
)

y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)
accuracy_rf = accuracy_score(y_test, y_pred_rf)


# ======================================================
# IMPORTANZA VARIABILI (LOGISTICA)
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
        "Random Forest",
        "Confronto Modelli",
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

    L’obiettivo è costruire modelli predittivi 
    capaci di prevedere lo stato finale del paziente:
    Alive oppure Dead.
    """)

    st.subheader("Scelta metodologica")

    st.write("""
    Le variabili T Stage, N Stage e Grade
    sono state trasformate tramite encoding
    ordinale, perché rappresentano livelli
    progressivi di gravità clinica.

    In questo modo i modelli interpretano
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

    st.title("📊 Visualizzazioni dei Dati")
    
    sns.set_theme(style="whitegrid", context="talk")

    st.subheader("Distribuzione Status")

    counts = df_model[target].value_counts()
    total_status = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(counts.index.astype(str), counts.values, color=['#4C72B0', '#DD8452'], width=0.6)
    
    for bar in bars:
        height = bar.get_height()
        percentage = (height / total_status) * 100
        ax.annotate(f'{percentage:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel("Status", fontsize=12)
    ax.set_ylabel("Frequenza", fontsize=12)
    ax.set_title("Distribuzione Alive / Dead", fontweight='bold', fontsize=14)
    sns.despine()
    st.pyplot(fig)

    st.write("---") 

    st.subheader("Analisi delle Variabili Numeriche Principali")

    col1, col2, col3 = st.columns(3)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(5, 4.5))
        sns.histplot(df['Age'], bins=20, kde=True, ax=ax1, color='skyblue', stat='percent')
        ax1.set_title('Distribuzione Età', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Età', fontsize=10)
        ax1.set_ylabel('Percentuale (%)', fontsize=10)
        ax1.tick_params(labelsize=9)
        sns.despine()
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        sns.histplot(df['Tumor Size'], bins=20, kde=True, ax=ax2, color='salmon', stat='percent')
        ax2.set_title('Dimensione Tumore (mm)', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Dimensione (mm)', fontsize=10)
        ax2.set_ylabel('Percentuale (%)', fontsize=10)
        ax2.tick_params(labelsize=9)
        sns.despine()
        st.pyplot(fig2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(5, 4.5))
        sns.histplot(df['Survival Months'], bins=20, kde=True, ax=ax3, color='lightgreen', stat='percent')
        ax3.set_title('Mesi di Sopravvivenza', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Mesi', fontsize=10)
        ax3.set_ylabel('Percentuale (%)', fontsize=10)
        ax3.tick_params(labelsize=9)
        sns.despine()
        st.pyplot(fig3)

    st.write("---")

    st.subheader("Distribuzione delle Variabili Categoriche")
    
    cat_cols = ["T Stage", "N Stage", "Grade", "Estrogen Status", "Progesterone Status"]
    total_records = len(df_model)
    
    for i in range(0, len(cat_cols), 2):
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            col = cat_cols[i]
            counts = df_model[col].value_counts().sort_index()
            
            fig, ax = plt.subplots(figsize=(6, 4.5))
            bars = ax.bar(counts.index.astype(str), counts.values, color='#64B5CD', width=0.5)
            
            for bar in bars:
                height = bar.get_height()
                percentage = (height / total_records) * 100
                ax.annotate(f'{percentage:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                            
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Frequenza", fontsize=11)
            ax.set_title(f"Distribuzione {col}", fontweight='bold', fontsize=13)
            ax.tick_params(labelsize=10)
            sns.despine()
            st.pyplot(fig)
            
        with grid_col2:
            if i + 1 < len(cat_cols):
                col = cat_cols[i+1]
                counts = df_model[col].value_counts().sort_index()
                
                fig, ax = plt.subplots(figsize=(6, 4.5))
                bars = ax.bar(counts.index.astype(str), counts.values, color='#8C564B', width=0.5)
                
                for bar in bars:
                    height = bar.get_height()
                    percentage = (height / total_records) * 100
                    ax.annotate(f'{percentage:.1f}%',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=10, fontweight='bold')
                                
                ax.set_xlabel(col, fontsize=11)
                ax.set_ylabel("Frequenza", fontsize=11)
                ax.set_title(f"Distribuzione {col}", fontweight='bold', fontsize=13)
                ax.tick_params(labelsize=10)
                sns.despine()
                st.pyplot(fig)

    st.write("---")
    st.subheader("📊 Analisi Comparativa rispetto allo Status (Alive / Dead)")
    
    for i in range(0, len(cat_cols), 2):
        grid_col1, grid_col2 = st.columns(2)
        
        with grid_col1:
            col = cat_cols[i]
            df_prop = df_model.groupby(col)[target].value_counts(normalize=True).rename("Proporzione").reset_index()
            
            fig, ax = plt.subplots(figsize=(6.5, 4.8))
            bars = sns.barplot(data=df_prop, x=col, y="Proporzione", hue=target, ax=ax, palette=["#FF6B6B", "#4ECDC4"])
            
            for p in ax.patches:
                if p.get_height() > 0: 
                    ax.annotate(f'{p.get_height()*100:.1f}%',
                                xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax.set_xlabel(col, fontsize=11)
            ax.set_ylabel("Proporzione", fontsize=11)
            ax.set_title(f"Proporzione di Status per {col}", fontweight='bold', fontsize=12)
            ax.set_ylim(0, 1.1) 
            ax.legend(title="Stato", fontsize=9, title_fontsize=10)
            sns.despine()
            st.pyplot(fig)
            
        with grid_col2:
            if i + 1 < len(cat_cols):
                col = cat_cols[i+1]
                df_prop = df_model.groupby(col)[target].value_counts(normalize=True).rename("Proporzione").reset_index()
                
                fig, ax = plt.subplots(figsize=(6.5, 4.8))
                bars = sns.barplot(data=df_prop, x=col, y="Proporzione", hue=target, ax=ax, palette=["#FF6B6B", "#4ECDC4"])
                
                for p in ax.patches:
                    if p.get_height() > 0:
                        ax.annotate(f'{p.get_height()*100:.1f}%',
                                    xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=9, fontweight='bold')
                                    
                ax.set_xlabel(col, fontsize=11)
                ax.set_ylabel("Proporzione", fontsize=11)
                ax.set_title(f"Proporzione di Status per {col}", fontweight='bold', fontsize=12)
                ax.set_ylim(0, 1.1)
                ax.legend(title="Stato", fontsize=9, title_fontsize=10)
                sns.despine()
                st.pyplot(fig)

    st.write("---")
    st.subheader("📈 Distribuzione delle Variabili Numeriche per i Pazienti Vivi/Deceduti")
    
    col_num1, col_num2, col_num3 = st.columns(3)
    
    with col_num1:
        fig1, ax1 = plt.subplots(figsize=(5.5, 4.5))
        sns.boxplot(data=df, x=target, y='Age', ax=ax1, palette=["#FF6B6B", "#4ECDC4"])
        ax1.set_title('Età vs Status', fontweight='bold', fontsize=11)
        ax1.set_xlabel('Status')
        ax1.set_ylabel('Età')
        sns.despine()
        st.pyplot(fig1)
        
    with col_num2:
        fig2, ax2 = plt.subplots(figsize=(5.5, 4.5))
        sns.boxplot(data=df, x=target, y='Tumor Size', ax=ax2, palette=["#FF6B6B", "#4ECDC4"])
        ax2.set_title('Dimensione Tumore vs Status', fontweight='bold', fontsize=11)
        ax2.set_xlabel('Status')
        ax2.set_ylabel('Dimensione (mm)')
        sns.despine()
        st.pyplot(fig2)
        
    with col_num3:
        fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
        nome_colonna_nodi = 'Reginol Node Positive' 
        
        sns.boxplot(data=df, x=target, y=nome_colonna_nodi, ax=ax3, palette=["#FF6B6B", "#4ECDC4"])
        ax3.set_title('Distribuzione di Nodi Positivi per Status', fontweight='bold', fontsize=11)
        ax3.set_xlabel('Status')
        ax3.set_ylabel('Regional Nodes Positive')
        sns.despine()
        st.pyplot(fig3)

# ======================================================
# CORRELAZIONI
# ======================================================

elif pagina == "Correlazioni":

    st.title("📊 Matrice di Correlazione")

    corr = X.corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr, 
        annot=True,            
        fmt=".2f",             
        cmap="coolwarm",       
        vmin=-1, vmax=1,       
        center=0,              
        linewidths=0.5,        
        linecolor='white',
        cbar_kws={"shrink": 0.8}, 
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"} 
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    ax.set_title("Matrice di Correlazione delle Feature", fontsize=14, fontweight='bold', pad=20)
    ax.grid(False)

    st.pyplot(fig)

    st.markdown("""
    ### 💡 Guida alla lettura:
    * **+1** $\rightarrow$ **Forte correlazione positiva**: al crescere di una variabile, cresce anche l'altra.
    * **-1** $\rightarrow$ **Forte correlazione negativa**: al crescere di una variabile, l'altra decresce.
    * **0** $\rightarrow$ **Assenza di correlazione**: le due variabili sono linearmente indipendenti.
    """)

# ======================================================
# REGRESSIONE LOGISTICA
# ======================================================

elif pagina == "Regressione Logistica":

    st.title("Regressione Logistica")

    st.subheader("Accuracy del modello")

    st.metric(
        "Accuracy",
        round(accuracy, 4)
    )

    st.subheader("Analisi degli Errori")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        
        labels = np.asarray([
            f"{tn}\n\nTN", 
            f"{fp}\n\nFP\n(Falsi Allarmi)",
            f"{fn}\n\nFN\n(Mancati)", 
            f"{tp}\n\nTP"
        ]).reshape(2, 2)

        sns.heatmap(
            cm, 
            annot=labels,          
            fmt="",             
            cmap="Blues",        
            cbar=True,          
            linewidths=0.5,      
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            ax=ax1
        )

        ax1.set_title("Confusion Matrix", fontweight='bold', pad=15)
        ax1.set_xlabel("Valore Predetto")
        ax1.set_ylabel("Valore Reale")
        
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        
        categorie = ['Veri Negativi\n(TN)', 'Falsi Positivi\n(FP)', 'Falsi Negativi\n(FN)', 'Veri Positivi\n(TP)']
        valori = [tn, fp, fn, tp]
        colori = ['#55a868', '#f1a340', '#c44e52', '#4c72b0'] 
        
        bars = ax2.bar(categorie, valori, color=colori)
        
        ax2.set_title("Breakdown degli Errori e Successi", fontweight='bold', pad=15)
        ax2.set_ylabel("N° Pazienti")
        sns.despine(left=False, bottom=False)
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(valori)*0.02), 
                     int(yval), ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig2)

    st.divider()

    st.subheader("Performance e Distribuzione Probabilità")

    classi = list(target_encoder.classes_)
    indice_alive = classi.index("Alive")
    indice_dead = classi.index("Dead")
    
    y_prob_alive = log_model.predict_proba(X_test_scaled)[:, indice_alive]
    
    y_test_np = np.array(y_test)
    mask_alive = (y_test_np == indice_alive)
    mask_dead = (y_test_np == indice_dead)
    y_test_bin = mask_alive.astype(int)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob_alive)
        roc_auc = auc(fpr, tpr)
        
        ax3.plot(fpr, tpr, color='#55a868', lw=2, label=f'Modello Logistico (AUC = {roc_auc:.3f})')
        ax3.fill_between(fpr, tpr, alpha=0.1, color='#55a868')
        ax3.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Classificatore Casuale (AUC = 0.5)')
        
        ax3.set_title("Curva ROC - Previsione SOPRAVVIVENZA", fontweight='bold', pad=15)
        ax3.set_xlabel("Falsi Positivi")
        ax3.set_ylabel("Sensibilità")
        ax3.legend(loc="lower right")
        
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        
        prob_alive_data = y_prob_alive[mask_alive]
        prob_dead_data = y_prob_alive[mask_dead]
        
        ax4.hist(prob_alive_data, bins=25, alpha=0.6, color='#55a868', label='Realmente Sopravvissute (Alive)')
        ax4.hist(prob_dead_data, bins=25, alpha=0.6, color='#c44e52', label='Realmente Decedute (Dead)')
        ax4.axvline(0.5, color='black', linestyle='--', lw=2, label='Soglia di Default (0.5)')
        
        ax4.set_title("Distribuzione Probabilità (Istogramma)", fontweight='bold', pad=15)
        ax4.set_xlabel("Probabilità di SOPRAVVIVENZA calcolata dal Modello")
        ax4.set_ylabel("Numero di Pazienti")
        ax4.legend(loc="upper right")
        
        st.pyplot(fig4)

    st.divider()

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=target_encoder.classes_,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Importanza Variabili")

    top_importance = importance.head(15)

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x="Peso",
        y="Variabile",
        data=top_importance,
        palette="coolwarm",  
        ax=ax5
    )

    sns.despine(left=True, bottom=True)          
    ax5.grid(axis='x', linestyle='--', alpha=0.5) 
    ax5.axvline(0, color='black', linewidth=1)    

    ax5.set_xlabel("Peso (Coefficiente)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("") 

    st.pyplot(fig5)

    st.info("""
    **Interpretazione dei coefficienti** (Se la classe positiva del modello è Dead):
    - **Coefficiente positivo**: aumenta il rischio di morte.
    - **Coefficiente negativo**: aumenta la probabilità di sopravvivenza.
    """)


# ======================================================
# RANDOM FOREST
# ======================================================

elif pagina == "Random Forest":

    st.title("Random Forest")

    st.subheader("Accuracy del modello")

    st.metric(
        "Accuracy",
        round(accuracy_rf, 4)
    )

    st.subheader("Analisi degli Errori")

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    tn, fp, fn, tp = cm_rf.ravel()

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        
        labels = np.asarray([
            f"{tn}\n\nTN", 
            f"{fp}\n\nFP\n(Falsi Allarmi)",
            f"{fn}\n\nFN\n(Mancati)", 
            f"{tp}\n\nTP"
        ]).reshape(2, 2)

        sns.heatmap(
            cm_rf, 
            annot=labels,          
            fmt="",             
            cmap="Greens",     
            cbar=True,          
            linewidths=0.5,      
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
            ax=ax1
        )

        ax1.set_title("Confusion Matrix (RF)", fontweight='bold', pad=15)
        ax1.set_xlabel("Valore Predetto")
        ax1.set_ylabel("Valore Reale")
        
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        
        categorie = ['Veri Negativi\n(TN)', 'Falsi Positivi\n(FP)', 'Falsi Negativi\n(FN)', 'Veri Positivi\n(TP)']
        valori = [tn, fp, fn, tp]
        colori = ['#55a868', '#f1a340', '#c44e52', '#4c72b0']
        
        bars = ax2.bar(categorie, valori, color=colori)
        
        ax2.set_title("Breakdown degli Errori e Successi", fontweight='bold', pad=15)
        ax2.set_ylabel("N° Pazienti")
        sns.despine(left=False, bottom=False)
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval + (max(valori)*0.02), 
                     int(yval), ha='center', va='bottom', fontweight='bold')

        st.pyplot(fig2)

    st.divider()

    st.subheader("Performance e Distribuzione Probabilità")

    classi = list(target_encoder.classes_)
    indice_alive = classi.index("Alive")
    indice_dead = classi.index("Dead")
    
    y_prob_alive_rf = y_prob_rf[:, indice_alive]
    
    y_test_np = np.array(y_test)
    mask_alive = (y_test_np == indice_alive)
    mask_dead = (y_test_np == indice_dead)
    y_test_bin = mask_alive.astype(int)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob_alive_rf)
        roc_auc = auc(fpr, tpr)
        
        ax3.plot(fpr, tpr, color='#55a868', lw=2, label=f'Random Forest (AUC = {roc_auc:.3f})')
        ax3.fill_between(fpr, tpr, alpha=0.1, color='#55a868')
        ax3.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Classificatore Casuale (AUC = 0.5)')
        
        ax3.set_title("Curva ROC - Previsione SOPRAVVIVENZA", fontweight='bold', pad=15)
        ax3.set_xlabel("Falsi Positivi")
        ax3.set_ylabel("Sensibilità")
        ax3.legend(loc="lower right")
        
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        
        prob_alive_rf_data = y_prob_alive_rf[mask_alive]
        prob_dead_rf_data = y_prob_alive_rf[mask_dead]
        
        ax4.hist(prob_alive_rf_data, bins=25, alpha=0.6, color='#55a868', label='Realmente Sopravvissute (Alive)')
        ax4.hist(prob_dead_rf_data, bins=25, alpha=0.6, color='#c44e52', label='Realmente Decedute (Dead)')
        ax4.axvline(0.5, color='black', linestyle='--', lw=2, label='Soglia di Default (0.5)')
        
        ax4.set_title("Distribuzione Probabilità (Istogramma)", fontweight='bold', pad=15)
        ax4.set_xlabel("Probabilità di SOPRAVVIVENZA calcolata (RF)")
        ax4.set_ylabel("Numero di Pazienti")
        ax4.legend(loc="upper left")
        
        st.pyplot(fig4)

    st.divider()

    st.subheader("Classification Report")

    report_rf = classification_report(
        y_test,
        y_pred_rf,
        target_names=target_encoder.classes_,
        output_dict=True
    )

    report_df_rf = pd.DataFrame(report_rf).transpose()
    st.dataframe(report_df_rf, use_container_width=True)

    st.subheader("Importanza Variabili (Feature Importance)")

    importance_rf = pd.DataFrame({
        "Variabile": feature_names,
        "Importanza": rf_model.feature_importances_
    })

    importance_rf = importance_rf.sort_values(by="Importanza", ascending=False)
    top_importance_rf = importance_rf.head(15)

    fig5, ax5 = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x="Importanza",
        y="Variabile",
        data=top_importance_rf,
        palette="viridis",  
        ax=ax5
    )

    sns.despine(left=True, bottom=True)          
    ax5.grid(axis='x', linestyle='--', alpha=0.5) 

    ax5.set_xlabel("Importanza (Gini / Diminuzione Impurità)", fontsize=11, fontweight="bold")
    ax5.set_ylabel("") 

    st.pyplot(fig5)

    st.info("""
    **Interpretazione dell'importanza (Random Forest)**:
    A differenza della Regressione Logistica, la Random Forest non assegna coefficienti direzionali (positivi o negativi). 
    Il valore mostrato indica puramente la **rilevanza predittiva** della variabile: più la barra è lunga, più quel dato clinico è stato cruciale per le decisioni dell'algoritmo nel separare i pazienti sopravvissuti da quelli deceduti, a prescindere dal fatto che ne aumenti o ne diminuisca il rischio.
    """)


# ======================================================
# CONFRONTO MODELLI
# ======================================================

elif pagina == "Confronto Modelli":

    st.title("Confronto Modelli")
    
    st.write("""
    In questa sezione confrontiamo direttamente le performance della
    **Regressione Logistica** e della **Random Forest**.
    """)

    # Preparazione dati reali
    classi = list(target_encoder.classes_)
    indice_alive = classi.index("Alive")
    y_test_bin = (np.array(y_test) == indice_alive).astype(int)

    # --- Calcoli Regressione Logistica ---
    prob_log = log_model.predict_proba(X_test_scaled)[:, indice_alive]
    pred_log = log_model.predict(X_test_scaled)
    pred_bin_log = (np.array(pred_log) == indice_alive).astype(int)
    fpr_log, tpr_log, _ = roc_curve(y_test_bin, prob_log)
    auc_log = auc(fpr_log, tpr_log)
    f1_log = f1_score(y_test_bin, pred_bin_log)

    # --- Calcoli Random Forest ---
    prob_rf = rf_model.predict_proba(X_test_scaled)[:, indice_alive]
    pred_rf = rf_model.predict(X_test_scaled)
    pred_bin_rf = (np.array(pred_rf) == indice_alive).astype(int)
    fpr_rf, tpr_rf, _ = roc_curve(y_test_bin, prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)
    f1_rf = f1_score(y_test_bin, pred_bin_rf)

    # ==========================================
    # UI: METRICHE AFFIANCATE
    # ==========================================
    st.subheader("Metriche a confronto")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔵 Regressione Logistica")
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")
        st.metric("AUC Score", f"{auc_log:.4f}")
        st.metric("F1 Score (Alive)", f"{f1_log:.4f}")

    with col2:
        st.markdown("### 🟢 Random Forest")
        st.metric("Accuracy", f"{accuracy_rf * 100:.2f}%")
        st.metric("AUC Score", f"{auc_rf:.4f}")
        st.metric("F1 Score (Alive)", f"{f1_rf:.4f}")

    st.divider()

    # ==========================================
    # UI: GRAFICO ROC COMBINATO
    # ==========================================
    st.subheader("Curva ROC Combinata")

    fig, ax = plt.subplots(figsize=(9, 6))

    # Curva Logistica (Blu)
    ax.plot(fpr_log, tpr_log, color='#3498db', lw=2.5, label=f'Logistica (AUC = {auc_log:.3f})')
    
    # Curva Random Forest (Verde)
    ax.plot(fpr_rf, tpr_rf, color='#2ecc71', lw=2.5, label=f'Random Forest (AUC = {auc_rf:.3f})')
    
    # Linea casuale di base
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Classificatore Casuale (0.5)')

    ax.set_title("Confronto Capacità Predittiva (ROC)", fontweight='bold', pad=15)
    ax.set_xlabel("Falsi Positivi", fontweight='bold')
    ax.set_ylabel("Sensibilità (Veri Positivi)", fontweight='bold')
    
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    sns.despine()

    st.pyplot(fig)
    
    st.info("""
    💡 **Come leggere il grafico:**
    La curva che si "gonfia" di più verso l'angolo in alto a sinistra rappresenta il modello migliore. 
    Mettendole sullo stesso grafico, puoi vedere immediatamente quale dei due algoritmi riesce a separare con più precisione i pazienti sopravvissuti da quelli deceduti!
    """)


# ======================================================
# WHAT IF (AGGIORNATO CON SELEZIONE MODELLO)
# ======================================================

elif pagina == "What If":

    st.title("Scenario What If")

    st.write("""
    Modifica le variabili cliniche nel menu laterale e seleziona il modello 
    di intelligenza artificiale desiderato per osservare la previsione.
    """)

    # Menu di selezione del modello richiesto
    modello_scelto = st.selectbox(
        "Seleziona il modello da utilizzare per la simulazione:",
        ["Regressione Logistica", "Random Forest"]
    )

    st.sidebar.header(
        "Inserisci dati paziente"
    )

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

    # INPUT BASE
    input_encoded = pd.DataFrame([{
        "Age": age,
        "T Stage": t_stage_map[t_stage_label],
        "N Stage": n_stage_map[n_stage_label],
        "Grade": grade_map[grade_label]
    }])

    # CREA COLONNE MANCANTI
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # ENCODING ESTROGEN
    for col in feature_names:
        if col.startswith("Estrogen Status_"):
            categoria = col.replace("Estrogen Status_", "")
            input_encoded[col] = 1 if estrogen == categoria else 0

    # ENCODING PROGESTERONE
    for col in feature_names:
        if col.startswith("Progesterone Status_"):
            categoria = col.replace("Progesterone Status_", "")
            input_encoded[col] = 1 if progesterone == categoria else 0

    input_encoded = input_encoded[feature_names]

    # STANDARDIZZAZIONE
    input_scaled = scaler.transform(input_encoded)

    # Elenco indici classi
    classi = list(target_encoder.classes_)
    indice_alive = classi.index("Alive")
    indice_dead = classi.index("Dead")

    # Calcolo predizione basato sul modello scelto
    if modello_scelto == "Regressione Logistica":
        pred = log_model.predict(input_scaled)[0]
        prob = log_model.predict_proba(input_scaled)[0]
    else:
        pred = rf_model.predict(input_scaled)[0]
        prob = rf_model.predict_proba(input_scaled)[0]

    pred_label = target_encoder.inverse_transform([pred])[0]
    prob_alive = prob[indice_alive] * 100
    prob_dead = prob[indice_dead] * 100

    # Interfaccia grafica risultati pulita (Grafici e dataframe superflui eliminati)
    st.subheader(f"Risultato previsione tramite {modello_scelto}")

    if pred_label == "Alive":
        st.success(f"Classe predetta dal modello: **{pred_label}**")
    else:
        st.error(f"Classe predetta dal modello: **{pred_label}**")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Probabilità sopravvivenza", f"{prob_alive:.1f}%")
    with col_m2:
        st.metric("Probabilità morte", f"{prob_dead:.1f}%")

    st.progress(int(prob_alive))

    st.info(f"""
    Lo scenario What If corrente sta impiegando il modello **{modello_scelto}** applicando:
    - Pipeline di encoding ordinale delle variabili cliniche astratte
    - Normalizzazione StandardScaler fittata sul set di allenamento globale
    """)


# ======================================================
# CONCLUSIONI
# ======================================================

elif pagina == "Conclusioni":

    st.title("Conclusioni")

    st.write("""
    Il progetto utilizza modelli di Machine Learning (Regressione Logistica e Random Forest) 
    per prevedere lo stato finale del paziente.

    Le variabili ordinali sono state codificate manualmente per rispettare
    la progressione clinica della malattia, mentre i modelli sono stati addestrati 
    risolvendo lo squilibrio delle classi.

    Confrontando le metriche e la Curva ROC, possiamo valutare quale dei due approcci 
    risulta più robusto e generalizzabile sui dati analizzati.
    """)

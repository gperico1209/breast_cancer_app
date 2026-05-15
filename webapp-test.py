import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 1. CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="SEER Cancer Predictor", layout="wide")
st.title("🩺 Predittore di Sopravvivenza - Tumore al Seno (SEER)")
st.markdown("Modifica i parametri clinici nella barra laterale per simulare diversi scenari (What-If).")

# --- 2. ADDESTRAMENTO IN BACKGROUND (nascosto all'utente) ---
@st.cache_resource # Questo fa in modo che il modello si addestri una volta sola!
def train_model():
    df = pd.read_csv('Breast_Cancer.csv')
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()
    
    # Target
    df['Status_num'] = (df['Status'] == 'Alive').astype(int)
    
    # Feature complete (serve che il modello veda tutte quelle originali)
    features = ['Age', 'T Stage', 'N Stage', '6th Stage', 'differentiate', 'Grade', 
                'A Stage', 'Tumor Size', 'Estrogen Status', 'Progesterone Status', 
                'Regional Node Examined', 'Reginol Node Positive']
    
    X = df[features].copy()
    y = df['Status_num']
    
    # Salviamo i LabelEncoder per riusarli sui dati inseriti dall'utente
    encoders = {}
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, encoders, features, df

model, scaler, encoders, feature_cols, df_original = train_model()

# --- 3. BARRA LATERALE (INPUT UTENTE) ---
st.sidebar.header("📊 Inserisci Dati Paziente")

# Creiamo slider e menu a tendina per le feature più importanti
age = st.sidebar.slider("Età", min_value=30, max_value=90, value=50)
t_stage = st.sidebar.selectbox("Stadio Tumore (T Stage)", options=['T1', 'T2', 'T3', 'T4'])
n_stage = st.sidebar.selectbox("Stadio Linfonodi (N Stage)", options=['N1', 'N2', 'N3'])
estrogen = st.sidebar.selectbox("Recettori Estrogeni", options=['Positive', 'Negative'])
tumor_size = st.sidebar.slider("Dimensione Tumore (mm)", 1, 100, value=20)
nodes_pos = st.sidebar.slider("Linfonodi Positivi", 0, 40, value=1)

# Per non riempire lo schermo, impostiamo i valori medi per le feature meno importanti
user_data = {
    'Age': age,
    'T Stage': t_stage,
    'N Stage': n_stage,
    '6th Stage': 'IIA', 
    'differentiate': 'Moderately differentiated',
    'Grade': '2', # <--- ECCO LA CORREZIONE! Prima era 'II'
    'A Stage': 'Regional',
    'Tumor Size': tumor_size,
    'Estrogen Status': estrogen,
    'Progesterone Status': 'Positive', 
    'Regional Node Examined': 10, 
    'Reginol Node Positive': nodes_pos
}
# --- 4. PREDIZIONE ---
# Trasformiamo il dizionario in un DataFrame (1 riga)
df_user = pd.DataFrame([user_data])

# Applichiamo gli stessi Encoder usati nell'addestramento
for col in df_user.select_dtypes(include=['object']).columns:
    df_user[col] = encoders[col].transform(df_user[col].astype(str))

# Scaliamo i dati
X_user_scaled = scaler.transform(df_user)

# OTTENIAMO LE PROBABILITÀ (Questa è la magia del What-If)
probabilita = model.predict_proba(X_user_scaled)[0]
prob_morto = probabilita[0] * 100
prob_vivo = probabilita[1] * 100

# --- 5. VISUALIZZAZIONE RISULTATI ---
st.divider()
st.subheader("Risultato Stimato")

# Logica visiva
if prob_vivo > 50:
    st.success(f"### 🎉 Prognosi Favorevole: Alive")
    st.metric(label="Probabilità di Sopravvivenza Stimata", value=f"{prob_vivo:.1f}%")
else:
    st.error(f"### ⚠️ Rischio Elevato: Dead")
    st.metric(label="Probabilità di Sopravvivenza Stimata", value=f"{prob_vivo:.1f}%", delta="- Rischio Critico", delta_color="inverse")

# Barra visiva della probabilità
st.progress(int(prob_vivo))

st.info("💡 **Nota:** Prova a cambiare lo *Stadio dei Linfonodi* o i *Recettori Estrogeni* nella barra laterale per vedere come cambia la percentuale di sopravvivenza in tempo reale.")
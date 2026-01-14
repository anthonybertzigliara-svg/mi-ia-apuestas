import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración Estética Pro
st.set_page_config(page_title="PRO AI Betting Terminal", layout="wide")

# Estilo visual avanzado (Dark Mode y tarjetas)
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 20px; border-radius: 12px; border: 1px solid #3b82f6; }
    .winner-box { background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%); padding: 25px; border-radius: 15px; text-align: center; border: 2px solid #60a5fa; margin: 20px 0; }
    h1, h2, h3 { color: #f1f5f9 !important; font-family: 'Inter', sans-serif; }
    .stButton>button { background: #2563eb; color: white; border-radius: 8px; font-weight: bold; height: 3.5em; border: none; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ PRO AI Betting Terminal")
st.sidebar.header("Panel de Inteligencia")

ligas = {
    "🇪🇸 La Liga": "SP1.csv",
    "🇬🇧 Premier League": "E0.csv",
    "🇮🇹 Serie A": "I1.csv",
    "🇩🇪 Bundesliga": "D1.csv"
}

seleccion_liga = st.sidebar.selectbox("Selecciona Competición", list(ligas.keys()))

@st.cache_data
def load_data(archivo):
    try:
        df = pd.read_csv(archivo)
        cols = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG', 'HC', 'AC', 'HY', 'AY']
        return df[cols].dropna()
    except: return None

df = load_data(ligas[seleccion_liga])

if df is not None:
    # --- PROCESAMIENTO IA ---
    le = LabelEncoder()
    equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(equipos)
    
    df['H_ID'], df['A_ID'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_ID', 'A_ID', 'B365H', 'B365D', 'B365A']]
    
    with st.spinner('Calibrando algoritmos de precisión...'):
        m_res = RandomForestClassifier(n_estimators=200).fit(X.values, df['Target'])
        m_gol = RandomForestRegressor(n_estimators=200).fit(X.values, df['FTHG'] + df['FTAG'])
        m_cor = RandomForestRegressor(n_estimators=200).fit(X.values, df['HC'] + df['AC'])
        m_tar = RandomForestRegressor(n_estimators=200).fit(X.values, df['HY'] + df['AY'])

    # --- INTERFAZ ---
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🏟️ Selección de Encuentro")
        l_col, v_col = st.columns(2)
        local = l_col.selectbox("Local", equipos)
        visita = v_col.selectbox("Visitante", equipos, index=1)
        
        st.markdown("### 🏦 Mercado de Cuotas")
        q1, q2, q3 = st.columns(3)
        ch = q1.number_input("Cuota Local", value=2.0)
        cd = q2.number_input("Cuota Empate", value=3.2)
        ca = q3.number_input("Cuota Visita", value=3.5)

        if st.button("🔍 EJECUTAR ANÁLISIS DE ALTO NIVEL"):
            id_l, id_v = le.transform([local])[0], le.transform([visita])[0]
            input_data = [[id_l, id_v, ch, cd, ca]]
            
            # Predicciones
            probs = m_res.predict_proba(input_data)[0]
            goles = m_gol.predict(input_data)[0]
            corns = m_cor.predict(input_data)[0]
            tarjs = m_tar.predict(input_data)[0]

            # 🏆

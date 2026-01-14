import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de página estilo Profesional
st.set_page_config(page_title="PRO AI Betting Terminal", layout="wide", initial_sidebar_state="collapsed")

# Estilo CSS personalizado para que sea "bonita" y profesional
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border: 1px solid #374151; }
    .stButton>button { width: 100%; background-color: #2563eb; color: white; border-radius: 8px; height: 3em; font-weight: bold; }
    .prediction-card { padding: 20px; border-radius: 15px; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); border: 1px solid #3b82f6; margin-bottom: 20px; }
    h1, h2, h3 { color: #f8fafc !important; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 PRO AI Betting Terminal")
st.markdown("---")

# Diccionario de ligas
ligas = {
    "🇪🇸 La Liga": "SP1.csv",
    "🇬🇧 Premier League": "E0.csv",
    "🇮🇹 Serie A": "I1.csv",
    "🇩🇪 Bundesliga": "D1.csv"
}

seleccion_liga = st.sidebar.selectbox("Selecciona Competición", list(ligas.keys()))

@st.cache_data
def load_pro_data(archivo):
    try:
        df = pd.read_csv(archivo)
        cols = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG', 'HC', 'AC', 'HY', 'AY']
        return df[cols].dropna()
    except: return None

df = load_pro_data(ligas[seleccion_liga])

if df is not None:
    # --- PROCESAMIENTO ---
    le = LabelEncoder()
    equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(equipos)
    
    df['H_ID'], df['A_ID'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_ID', 'A_ID', 'B365H', 'B365D', 'B365A']]
    
    # Entrenar modelos (Resultado, Goles, Corners, Tarjetas)
    with st.spinner('Actualizando algoritmos...'):
        m_res = RandomForestClassifier(n_estimators=150).fit(X.values, df['Target'])
        m_gol = RandomForestRegressor(n_estimators=150).fit(X.values, df['FTHG'] + df['FTAG'])
        m_cor = RandomForestRegressor(n_estimators=150).fit(X.values, df['HC'] + df['AC'])
        m_tar = RandomForestRegressor(n_estimators=150).fit(X.values, df['HY'] + df['AY'])

    # --- PANEL LATERAL DE ENTRADA ---
    st.sidebar.subheader("Análisis de Partido")
    local = st.sidebar.selectbox("Local", equipos)
    visita = st.sidebar.selectbox("Visitante", equipos, index=1)
    
    st.sidebar.markdown("### 📈 Cuotas Live")
    ch = st.sidebar.number_input("Cuota 1", value=2.0)
    cd = st.sidebar.number_input("Cuota X", value=3.2)
    ca = st.sidebar.number_input("Cuota 2", value=3.5)

    # --- MAIN UI ---
    col_main_1, col_main_2 = st.columns([2, 1])

    with col_main_1:
        st.markdown(f"""

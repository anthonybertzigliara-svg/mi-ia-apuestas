import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time

# 1. CONFIGURACIÓN DE PÁGINA IMPACTANTE
st.set_page_config(page_title="IA ULTRA BET", layout="wide")

# 2. DISEÑO DE ALTO NIVEL (CSS)
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 30px !important; }
    .stSelectbox, .stNumberInput { background-color: #1e293b !important; }
    .stButton>button { 
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%); 
        color: white; border: none; border-radius: 10px; height: 50px; font-size: 20px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }
    .result-card {
        background: rgba(30, 41, 59, 0.7);
        padding: 30px; border-radius: 20px; border: 1px solid #3b82f6;
        text-align: center; margin-top: 20px;
    }
    .winner-text { color: #4ade80; font-size: 40px; font-weight: 800; text-transform: uppercase; }
    </style>
    """, unsafe_allow_html=True)

# 3. LÓGICA DE DATOS
ligas = {
    "🇪🇸 LA LIGA": "SP1.csv",
    "🇬🇧 PREMIER LEAGUE": "E0.csv",
    "🇮🇹 SERIE A": "I1.csv",
    "🇩🇪 BUNDESLIGA": "D1.csv"
}

st.sidebar.title("💎 IA PREMIUM")
seleccion = st.sidebar.selectbox("LIGA SELECCIONADA", list(ligas.keys()))

@st.cache_data
def cargar_pro(archivo):
    try:
        df = pd.read_csv(archivo)
        columnas = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG', 'HC', 'AC']
        return df[columnas].dropna()
    except: return None

df = cargar_pro(lig

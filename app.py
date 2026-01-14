import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN Y ESTILO PREVIUM
st.set_page_config(page_title="AI ELITE BETTING", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stMetric { background-color: #1e293b !important; border: 1px solid #3b82f6 !important; border-radius: 12px; padding: 15px !important; }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: bold; }
    .winner-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        padding: 25px; border-radius: 20px; text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4); margin: 20px 0; border: 1px solid #60a5fa;
    }
    .market-card {
        background: #1e293b; padding: 20px; border-radius: 15px;
        border-left: 5px solid #3b82f6; margin-bottom: 10px;
    }
    h1, h2, h3 { color: white !important; }
    .stButton>button { 
        background: #2563eb; color: white; border-radius: 10px; 
        font-weight: bold; height: 3.5em; width: 100%; border: none; font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE LIGAS
ligas = {"🇪🇸 La Liga": "SP1.csv", "🇬🇧 Premier League": "E0.csv", "🇮🇹 Serie A": "I1.csv", "🇩🇪 Bundesliga": "D1.csv"}
sel_liga = st.sidebar.selectbox("📂 SELECCIONAR COMPETICIÓN", list(ligas.keys()))

@st.cache_

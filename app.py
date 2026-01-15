import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE ALTA COMPETICIÓN
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(5,5,10,0.98) 0%, rgba(10,25,50,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    .status-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #00f2ff33;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }
    .bet-header {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        color: #00f2ff !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 800;
        margin: 25px 0 15px 0;
        border-left: 5px solid #00f2ff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .racha-v { color: #00ff88; font-weight: bold; margin: 0 3px; }
    .racha-e { color: #ffcc00; font-weight: bold; margin: 0 3px; }
    .racha-d { color: #ff4b4b; font-weight: bold; margin: 0 3px; }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00f2ff 0%, #0077ff 100%) !important;
        color: white !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        padding: 1rem !important;
        border-radius: 15px !important;
        border: none !important;
        box-shadow: 0 0 25px rgba(0, 242, 255, 0.4);
    }
    .metric-value { font-size: 2.5rem; font-weight: 900; color: white; }
    .tag-plus { background: #00ff88; color: black; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    .tag-minus { background: #ff4b4b; color: white; padding: 4px 12px; border-radius: 6px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE DATOS EN TIEMPO REAL
ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER LEAGUE": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "🇫🇷 LIGUE 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
}

@st.cache_data(ttl=3600)
def load_pro_data(url):
    try:
        data = pd.read_csv(url)
        cols = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR','B365H','B365D','B365A','HC','AC','HY','AY']
        return data[cols].dropna()
    except: return None

# Función para extraer racha
def get_streak(df, team):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
    streak = []
    for _, row in recent.iterrows():
        if row['FTR'] == 'D': streak.append('<span class="racha-e">E</span>')
        elif (row['HomeTeam'] == team and row['FTR'] == 'H') or (row['AwayTeam'] == team and row['FTR'] == 'A'):
            streak.append('<span class="racha-v">V</span>')
        else:
            streak.append('<span class="racha-d">D</span>')
    return "".join(streak)

if 'quiniela' not in st.session_state:
    st.session_state.quiniela = []

sel_liga = st.sidebar.selectbox("🌍 SELECCIONAR MERCADO", list(ligas.keys()))
df = load_pro_data(ligas[sel_liga])

if df is not None:
    # Entrenamiento de Red Neuronal / Random Forest
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
    
    m_win = RandomForestClassifier(n_estimators=200).fit(X.values, df['Target'])
    m_goals = RandomForestRegressor(n_estimators=200).fit(X.values, df['FTHG'] + df['FTAG'])
    m_corn = RandomForestRegressor(n_estimators=200).fit(X.values, df['HC'] + df['AC'])
    m_cards = RandomForestRegressor(n_estimators=200).fit(X.values, df['HY'] + df['AY'])

    st.markdown("<h1 style='text-align: center; color: #00f2ff !important;'>WORLD ELITE BETTING AI</h1>", unsafe_allow_html=True)
    
    # 3. SELECTORES CON RACHA VISUAL
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("EQUIPO LOCAL", teams)
    c1.markdown(f"Racha: {get_streak(df, t1)}", unsafe_allow_html=True)
    
    t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
    c2.markdown(f"Racha: {get_streak(df, t2)}", unsafe_allow_html=True)
    
    st.markdown("<div class='bet-header'>Análisis de Cuotas Mundiales</div>", unsafe_allow_html=True)
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input(f"Cuota {t1}", value=2.10, step=0.01)
    vx = qx.number_input("Cuota Empate", value=3.40, step=0.01)
    v2 = q2.number_input(f"Cuota {t2}", value=3.80, step=0.01)

    if st.button("🔥 EJECUTAR ALGORITMO MAESTRO"):
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
        probs = m_win.predict_proba(v_in)[0]
        g, c, cards = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0], m_cards.predict(v_in)[0]
        
        idx = m_win.predict(v_in)[0]
        res_text = t1 if idx == 1 else (t2 if idx == 2 else "Empate")

        # Guardar en Quiniela
        st.session_state.quiniela.append({
            "EVENTO": f"{t1} vs {t2}",
            "PICK": res_text,
            "GOLES": f"{'+2.5' if g > 2.5 else '-2.5'}",
            "CORNERS": f"{'+9.5' if c > 9.5 else '-9.5'}",
            "CONF.": f"{max(probs)*100:.0f}%"
        })

        # Resultados Visuales Elite
        st.markdown("<div class='bet-header'>Predicción de Alta Precisión</div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f'<div class="status-card"><p>GOLES EST.</p><div class="metric-value">{g:.1f}</div><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
        with r2:
            st.markdown(f'<div class="status-card"><p>CÓRNERS EST.</p><div class="metric-value">{c:.0f}</div><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
        with r3:
            st.markdown(f'<div class="status-card"><p>PROBABILIDAD</p><div class="metric-value">{max(probs)*100:.0f}%</div><span class="tag-plus" style="background: #00f2ff;">PROB. ORO</span></div>', unsafe_allow_html=True)

    # 4. QUINIELA DE SESIÓN PROFESIONAL
    if st.session_state.quiniela:
        st.markdown("<div class='bet-header'>📋 QUINIELA MAESTRA (REGISTRO)</div>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(st.session_state.quiniela), use_container_width=True, hide_index=True)
        
        if st.button("🗑️ REINICIAR SISTEMA"):
            st.session_state.quiniela = []
            st.rerun()

else:
    st.error("📡 Error de conexión con el satélite de datos deportivos.")

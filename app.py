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
    }
    .racha-v { color: #00ff88; font-weight: bold; margin: 0 3px; }
    .racha-e { color: #ffcc00; font-weight: bold; margin: 0 3px; }
    .racha-d { color: #ff4b4b; font-weight: bold; margin: 0 3px; }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        padding: 1rem !important;
        border-radius: 15px !important;
        width: 100%;
        box-shadow: 0 0 25px rgba(0, 255, 136, 0.4);
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
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_pro_data(url):
    try:
        data = pd.read_csv(url)
        cols = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR','B365H','B365D','B365A','HC','AC','HY','AY']
        return data[cols].dropna()
    except: return None

def get_stats(df, team):
    # Racha
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].tail(5)
    streak = []
    goles_total = 0
    for _, row in recent.iterrows():
        if row['HomeTeam'] == team:
            goles_total += row['FTHG']
            res = 'V' if row['FTR'] == 'H' else ('E' if row['FTR'] == 'D' else 'D')
        else:
            goles_total += row['FTAG']
            res = 'V' if row['FTR'] == 'A' else ('E' if row['FTR'] == 'D' else 'D')
        
        if res == 'V': streak.append('<span class="racha-v">V</span>')
        elif res == 'E': streak.append('<span class="racha-e">E</span>')
        else: streak.append('<span class="racha-d">D</span>')
    
    return "".join(streak), goles_total / 5

if 'quiniela' not in st.session_state:
    st.session_state.quiniela = []

sel_liga = st.sidebar.selectbox("🌍 MERCADO", list(ligas.keys()))
df = load_pro_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    # 3. INTERFAZ
    st.markdown("<h1 style='text-align: center; color: #00f2ff !important;'>WORLD ELITE BETTING AI</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("LOCAL", teams)
    racha1, g_prom1 = get_stats(df, t1)
    c1.markdown(f"Racha: {racha1}", unsafe_allow_html=True)
    
    t2 = c2.selectbox("VISITANTE", teams, index=1)
    racha2, g_prom2 = get_stats(df, t2)
    c2.markdown(f"Racha: {racha2}", unsafe_allow_html=True)

    # GRÁFICO DE BARRAS DE ATAQUE
    st.markdown("<div class='bet-header'>PODER OFENSIVO (Goles/Partido)</div>", unsafe_allow_html=True)
    chart_data = pd.DataFrame({
        'Equipo': [t1, t2],
        'Goles Promedio': [g_prom1, g_prom2]
    })
    st.bar_chart(chart_data.set_index('Equipo'))

    st.markdown("<div class='bet-header'>ANÁLISIS DE CUOTAS</div>", unsafe_allow_html=True)
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input(f"Cuota {t1}", value=2.10)
    vx = qx.number_input("Cuota X", value=3.40)
    v2 = q2.number_input(f"Cuota {t2}", value=3.80)

    # ENTRENAMIENTO IA
    df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
    m_win = RandomForestClassifier(n_estimators=150).fit(X.values, df['Target'])
    m_goals = RandomForestRegressor(n_estimators=150).fit(X.values, df['FTHG'] + df['FTAG'])
    m_corn = RandomForestRegressor(n_estimators=150).fit(X.values, df['HC'] + df['AC'])

    if st.button("🚀 ANALIZAR Y GUARDAR EN QUINIELA"):
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
        probs = m_win.predict_proba(v_in)[0]
        g, c = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0]
        
        idx = m_win.predict(v_in)[0]
        pick = t1 if idx == 1 else (t2 if idx == 2 else "Empate")

        st.session_state.quiniela.append({
            "PARTIDO": f"{t1}-{t2}", "PICK": pick, 
            "GOLES": f"{'+2.5' if g > 2.5 else '-2.5'}",
            "CORNERS": f"{'+9.5' if c > 9.5 else '-9.5'}", "CONF.": f"{max(probs)*100:.0f}%"
        })

        st.markdown("<div class='bet-header'>PRONÓSTICO IA</div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        r1.markdown(f'<div class="status-card"><p>GOLES</p><div class="metric-value">{g:.1f}</div><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="status-card"><p>CÓRNERS</p><div class="metric-value">{c:.0f}</div><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="status-card"><p>CONFIANZA</p><div class="metric-value">{max(probs)*100:.0f}%</div><span class="tag-plus" style="background:#00f2ff">SCORE ELITE</span></div>', unsafe_allow_html=True)

    if st.session_state.quiniela:
        st.markdown("<div class='bet-header'>📋 MI QUINIELA</div>", unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state.quiniela))
        if st.button("🗑️ LIMPIAR"):
            st.session_state.quiniela = []
            st.rerun()

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN E INTERFAZ PROFESIONAL
st.set_page_config(page_title="IA ELITE BETTING PRO", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,10,0.95) 0%, rgba(20,30,50,0.9) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    /* TARJETAS */
    .status-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(0, 242, 255, 0.2);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .bet-header {
        background: #1e293b;
        color: #00f2ff !important;
        padding: 8px 15px;
        border-radius: 5px;
        font-weight: bold;
        margin: 20px 0 10px 0;
        border-left: 4px solid #00f2ff;
    }
    
    /* BOTÓN ANALIZAR (VERDE NEÓN) */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        border: none !important;
        font-weight: bold !important;
        padding: 0.6rem 2rem !important;
        border-radius: 10px !important;
        width: 100%;
        transition: 0.3s;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.6);
        transform: scale(1.02);
    }

    /* BOTÓN LIMPIAR (ROJO) */
    /* Usamos un selector específico para el segundo botón de la página */
    div.stButton > button:last-child {
        background: #ff4b4b !important;
        color: white !important;
        border: none !important;
        font-size: 0.8rem !important;
        padding: 5px 15px !important;
    }

    .tag-plus { background-color: #00ff88; color: black; padding: 3px 10px; border-radius: 4px; font-weight: 900; }
    .tag-minus { background-color: #ff4b4b; color: white; padding: 3px 10px; border-radius: 4px; font-weight: 900; }
    h1, h3, p, label { color: white !important; }
    .metric-value { font-size: 2rem; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS AUTOMÁTICA
ligas = {
    "🇪🇸 ESPAÑA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 INGLATERRA": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITALIA": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 ALEMANIA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        data = pd.read_csv(url)
        cols = ['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC','HY','AY']
        return data[cols].dropna()
    except: return None

if 'quiniela' not in st.session_state:
    st.session_state.quiniela = []

sel_liga = st.sidebar.selectbox("LIGA", list(ligas.keys()))
df = load_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
    m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
    m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_cards = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    st.markdown("<h1 style='text-align: center;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("LOCAL", teams)
    t2 = col2.selectbox("VISITANTE", teams, index=1)
    
    st.markdown("<div class='bet-header'>CUOTAS 1X2</div>", unsafe_allow_html=True)
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input(f"1 ({t1})", value=2.0, step=0.01)
    vx = qx.number_input("X (Empate)", value=3.2, step=0.01)
    v2 = q2.number_input(f"2 ({t2})", value=3.5, step=0.01)

    # BOTÓN ANALIZAR
    if st.button("🚀 ANALIZAR Y GUARDAR PARTIDO"):
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
        probs = m_win.predict_proba(v_in)[0] 
        g, c, cards = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0], m_cards.predict(v_in)[0]
        
        idx = m_win.predict(v_in)[0]
        ganador_pick = t1 if idx == 1 else (t2 if idx == 2 else "Empate")

        st.session_state.quiniela.append({
            "Partido": f"{t1} vs {t2}",
            "GANADOR": ganador_pick,
            "Goles": f"{'+ 2.5' if g > 2.5 else '- 2.5'} ({g:.1f})",
            "Córners": f"{'+ 9.5' if c > 9.5 else '- 9.5'} ({c:.0f})",
            "Tarjetas": f"{'+ 4.5' if cards > 4.5 else '- 4.5'}",
            "Confianza": f"{max(probs)*100:.0f}%"
        })

        st.markdown("<div class='bet-header'>DETALLES DEL PRONÓSTICO</div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1:
            tag = "tag-plus" if g > 2.5 else "tag-minus"
            st.markdown(f'<div class="status-card"><p>GOLES</p><div class="metric-value">{g:.1f}</div><span class="{tag}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
        with r2:
            tag = "tag-plus" if c > 9.5 else "tag-minus"
            st.markdown(f'<div class="status-card"><p>CÓRNERS</p><div class="metric-value">{c:.0f}</div><span class="{tag}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
        with r3:
            tag = "tag-plus" if cards > 4.5 else "tag-minus"
            st.markdown(f'<div class="status-card"><p>TARJETAS</p><div class="metric-value">{cards:.1f}</div><span class="{tag}">{" + 4.5" if cards > 4.5 else " - 4.5"}</span></div>', unsafe_allow_html=True)

    if st.session_state.quiniela:
        st.markdown("<div class='bet-header'>📋 MI QUINIELA DE SESIÓN</div>", unsafe_allow_html=True)
        q_df = pd.DataFrame(st.session_state.quiniela)
        st.dataframe(q_df, use_container_width=True, hide_index=True)
        
        # BOTÓN LIMPIAR
        if st.button("🗑️ Limpiar Historial de Quiniela"):
            st.session_state.quiniela = []
            st.rerun()

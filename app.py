import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE ALTA VISIBILIDAD (CORRECCIÓN TOTAL DE ETIQUETAS)
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,40,80,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    
    /* FORZAR COLOR BLANCO EN ETIQUETAS DE TEXTO (LOCAL, EMPATE, VISITANTE) */
    label {
        color: white !important;
        font-weight: 900 !important;
        font-size: 1.1rem !important;
        text-shadow: 1px 1px 2px black;
    }

    .stat-text {
        color: #FFFFFF !important;
        font-weight: bold;
        background: rgba(0,0,0,0.5);
        padding: 5px 10px;
        border-radius: 8px;
        border: 1px solid #00f2ff;
    }
    .favor-numb { color: #FFFF00 !important; font-weight: 900; font-size: 1.2rem; }

    .status-card {
        background: rgba(15, 23, 42, 0.95);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        min-height: 180px;
    }
    .metric-title { color: #00f2ff; font-size: 1rem; font-weight: 800; text-transform: uppercase; margin-bottom: 10px; }
    .metric-value { font-size: 3rem; font-weight: 900; color: #FFFFFF; margin: 10px 0; }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        height: 4rem !important;
        border-radius: 15px !important;
        border: 2px solid white !important;
    }

    .tag-plus { background: #00ff88; color: black; padding: 6px 18px; border-radius: 6px; font-weight: 900; display: inline-block; }
    .tag-minus { background: #ff4b4b; color: white; padding: 6px 18px; border-radius: 6px; font-weight: 900; display: inline-block; }
    
    .bet-header {
        background: #00f2ff;
        color: black !important;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 900;
        margin: 25px 0;
        font-size: 1.2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE DATOS
ligas = {
    "🇪🇸 ESPAÑA - LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 INGLATERRA - PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITALIA - SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 ALEMANIA - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try: return pd.read_csv(url)
    except: return None

def get_pro_stats(df, team):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].dropna(subset=['FTR']).tail(5)
    streak = []
    g_favor = 0
    for _, row in recent.iterrows():
        is_home = row['HomeTeam'] == team
        g_favor += row['FTHG'] if is_home else row['FTAG']
        res = 'V' if (is_home and row['FTR']=='H') or (not is_home and row['FTR']=='A') else ('E' if row['FTR']=='D' else 'D')
        color = "#00ff88" if res=='V' else ("#ffcc00" if res=='E' else "#ff4b4b")
        streak.append(f'<span style="color:{color}; font-size:1.2rem;">{res}</span>')
    return " ".join(streak), g_favor / 5 if not recent.empty else 0

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

with st.sidebar:
    st.title("⚽ PANEL CONTROL")
    sel_liga = st.selectbox("COMPETICIÓN", list(ligas.keys()))

df_raw = load_data(ligas[sel_liga])

if df_raw is not None:
    tab_analisis, tab_calendario = st.tabs(["🔥 ANALIZADOR PRO", "📅 CALENDARIO"])

    with tab_calendario:
        futuros = df_raw[df_raw['FTR'].isna()][['Date', 'HomeTeam', 'AwayTeam']].head(15)
        for _, row in futuros.iterrows():
            st.markdown(f'<div style="background:white; color:black; padding:10px; border-radius:10px; margin-bottom:5px; font-weight:bold;">📅 {row["Date"]} | {row["HomeTeam"]} vs {row["AwayTeam"]}</div>', unsafe_allow_html=True)

    with tab_analisis:
        df = df_raw.dropna(subset=['FTR', 'B365H'])
        le = LabelEncoder()
        teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
        le.fit(teams)

        st.markdown("<h1 style='text-align: center; color: #00f2ff; text-shadow: 2px 2px 10px black;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("EQUIPO LOCAL", teams)
        racha1, g_fav1 = get_pro_stats(df, t1)
        c1.markdown(f'<div class="stat-text">Racha: {racha1} | ⚽ Goles Favor: <span class="favor-numb">{g_fav1:.1f}</span></div>', unsafe_allow_html=True)
        
        t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
        racha2, g_fav2 = get_pro_stats(df, t2)
        c2.markdown(f'<div class="stat-text">Racha: {racha2} | ⚽ Goles Favor: <span class="favor-numb">{g_fav2:.1f}</span></div>', unsafe_allow_html=True)

        st.markdown("<div class='bet-header'>📉 AJUSTAR CUOTAS REALES</div>", unsafe_allow_html=True)
        q1, qx, q2 = st.columns(3)
        v1 = q1.number_input("Cuota Local (1)", value=2.0, step=0.01)
        vx = qx.number_input("Cuota Empate (X)", value=3.2, step=0.01)
        v2 = q2.number_input("Cuota Visita (2)", value=3.5, step=0.01)

        # ENTRENAMIENTO IA
        df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
        df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
        X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
        
        m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
        m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
        m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
        m_cards = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

        if st.button("🔥 ANALIZAR Y GUARDAR PRONÓSTICO"):
            v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
            probs = m_win.predict_proba(v_in)[0]
            g, c, cards = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0], m_cards.predict(v_in)[0]
            pick = t1 if m_win.predict(v_in)[0] == 1 else (t2 if m_win.predict(v_in)[0] == 2 else "Empate")

            st.session_state.quiniela.append({
                "PARTIDO": f"{t1}-{t2}", "PICK": pick, "GOLES": f"{g:.1f}", "TARJETAS": f"{cards:.1f}", "CONF": f"{max(probs)*100:.0f}%"
            })

            st.markdown("<div class='bet-header'>📊 RESULTADO DEL ESCANEO IA</div>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown(f'<div class="status-card"><div class="metric-title">GANADOR</div><div class="metric-value" style="color:#00ff88">{pick}</div><span class="tag-plus" style="background:#00f2ff">{max(probs)*100:.0f}%</span></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="status-card"><div class="metric-title">GOLES</div><div class="metric-value">{g:.1f}</div><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="status-card"><div class="metric-title">CÓRNERS</div><div class="metric-value">{c:.0f}</div><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
            with r4:
                st.markdown(f'<div class="status-card"><div class="metric-title">TARJETAS</div><div class="metric-value">{cards:.1f}</div><span class="{"tag-plus" if cards > 4.5 else "tag-minus"}">{" + 4.5" if cards > 4.5 else " - 4.5"}</span></div>', unsafe_allow_html=True)

        if st.session_state.quiniela:
            st.markdown("<div class='bet-header'>📋 MI QUINIELA MAESTRA</div>", unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state.quiniela))
            if st.button("🗑️ LIMPIAR QUINIELA"):
                st.session_state.quiniela = []
                st.rerun()

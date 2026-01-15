import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE MÁXIMA VISIBILIDAD (ELITE NEÓN)
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,40,80,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    /* TARJETAS DE RESULTADO (RECUERDOS DEL DISEÑO ANTERIOR) */
    .status-card {
        background: rgba(15, 23, 42, 0.9);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.2);
    }
    .metric-title { color: #aaa; font-size: 0.9rem; text-transform: uppercase; margin-bottom: 10px; }
    .metric-value { font-size: 2.5rem; font-weight: 900; color: white; margin: 5px 0; }
    
    /* CALENDARIO ULTRA LEGIBLE */
    .cal-card {
        background: #f8f9fa;
        color: #111 !important;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 10px solid #00f2ff;
    }
    .cal-team { font-size: 1.3rem; font-weight: 800; color: #1a1a1a; }
    .cal-date { color: #666; font-weight: bold; }

    /* BOTONES POTENTES */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.4rem !important;
        height: 3.5rem !important;
        width: 100% !important;
        border-radius: 15px !important;
        box-shadow: 0 4px 15px rgba(0, 255, 136, 0.4);
    }
    .tag-plus { background: #00ff88; color: black; padding: 5px 15px; border-radius: 5px; font-weight: 900; }
    .tag-minus { background: #ff4b4b; color: white; padding: 5px 15px; border-radius: 5px; font-weight: 900; }
    
    .bet-header {
        background: #00f2ff;
        color: black !important;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 900;
        margin: 20px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE DATOS (Actualización Automática)
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
    # Racha y Goles a favor (Últimos 5)
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].dropna(subset=['FTR']).tail(5)
    streak = []
    g_favor = 0
    for _, row in recent.iterrows():
        is_home = row['HomeTeam'] == team
        goals = row['FTHG'] if is_home else row['FTAG']
        g_favor += goals
        res = 'V' if (is_home and row['FTR']=='H') or (not is_home and row['FTR']=='A') else ('E' if row['FTR']=='D' else 'D')
        color = "#00ff88" if res=='V' else ("#ffcc00" if res=='E' else "#ff4b4b")
        streak.append(f'<span style="color:{color}; font-weight:bold; margin-right:5px;">{res}</span>')
    return "".join(streak), g_favor / 5 if not recent.empty else 0

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

# SIDEBAR
with st.sidebar:
    st.title("⚙️ PANEL CONTROL")
    sel_liga = st.selectbox("COMPETICIÓN ACTIVA", list(ligas.keys()))
    st.success("Base de datos: Sincronizada ✅")

df_raw = load_data(ligas[sel_liga])

if df_raw is not None:
    tab_analisis, tab_calendario = st.tabs(["🔥 ANALIZADOR PRO", "📅 CALENDARIO SEMANAL"])

    with tab_calendario:
        st.markdown("<h2 style='color:white;'>Próximos Encuentros</h2>", unsafe_allow_html=True)
        futuros = df_raw[df_raw['FTR'].isna()][['Date', 'HomeTeam', 'AwayTeam']].head(12)
        if futuros.empty:
            st.info("No hay más partidos registrados para esta jornada.")
        for _, row in futuros.iterrows():
            st.markdown(f"""
                <div class="cal-card">
                    <div class="cal-date">📅 {row['Date']}</div>
                    <div class="cal-team">{row['HomeTeam']} <span style="color:#00f2ff">vs</span> {row['AwayTeam']}</div>
                </div>
            """, unsafe_allow_html=True)

    with tab_analisis:
        df = df_raw.dropna(subset=['FTR', 'B365H'])
        le = LabelEncoder()
        teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
        le.fit(teams)

        st.markdown("<h1 style='text-align: center; color: #00f2ff; text-shadow: 0 0 10px #00f2ff;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("LOCAL", teams)
        racha1, g_fav1 = get_pro_stats(df, t1)
        c1.markdown(f"Racha: {racha1} | ⚽ Favor: **{g_fav1:.1f}**", unsafe_allow_html=True)
        
        t2 = c2.selectbox("VISITANTE", teams, index=1)
        racha2, g_fav2 = get_pro_stats(df, t2)
        c2.markdown(f"Racha: {racha2} | ⚽ Favor: **{g_fav2:.1f}**", unsafe_allow_html=True)

        st.markdown("<div class='bet-header'>CUOTAS 1X2</div>", unsafe_allow_html=True)
        q1, qx, q2 = st.columns(3)
        v1 = q1.number_input(f"1 ({t1})", value=2.0, step=0.01)
        vx = qx.number_input("X (Empate)", value=3.2, step=0.01)
        v2 = q2.number_input(f"2 ({t2})", value=3.5, step=0.01)

        # LÓGICA IA MAESTRA
        df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
        df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
        X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
        
        m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
        m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
        m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])

        if st.button("🚀 ANALIZAR Y GUARDAR PARTIDO"):
            v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
            probs = m_win.predict_proba(v_in)[0]
            g, c = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0]
            pick = t1 if m_win.predict(v_in)[0] == 1 else (t2 if m_win.predict(v_in)[0] == 2 else "Empate")
            conf = max(probs)*100

            st.session_state.quiniela.append({
                "Partido": f"{t1} vs {t2}", "GANADOR": pick, 
                "Goles": f"{'+2.5' if g > 2.5 else '-2.5'} ({g:.1f})",
                "Córners": f"{'+9.5' if c > 9.5 else '-9.5'} ({c:.0f})", "Conf.": f"{conf:.0f}%"
            })

            st.markdown("<div class='bet-header'>DETALLES DEL PRONÓSTICO</div>", unsafe_allow_html=True)
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="status-card"><div class="metric-title">GANADOR PROBABLE</div><div class="metric-value">{pick}</div><span class="tag-plus" style="background:#00f2ff">{conf:.0f}% Confianza</span></div>', unsafe_allow_html=True)
            with r2:
                st.markdown(f'<div class="status-card"><div class="metric-title">GOLES ESTIMADOS</div><div class="metric-value">{g:.1f}</div><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
            with r3:
                st.markdown(f'<div class="status-card"><div class="metric-title">CÓRNERS ESTIMADOS</div><div class="metric-value">{c:.0f}</div><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)

        if st.session_state.quiniela:
            st.markdown("<div class='bet-header'>📋 MI QUINIELA DE SESIÓN</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state.quiniela), use_container_width=True, hide_index=True)
            if st.button("🗑️ VACIAR TODO EL CUADRO"):
                st.session_state.quiniela = []
                st.rerun()

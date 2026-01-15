import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE ALTA VISIBILIDAD
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,40,80,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    /* TARJETAS DE CALENDARIO (LEGIBLES) */
    .cal-card {
        background: rgba(255, 255, 255, 0.95);
        color: #111 !important;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 8px solid #00f2ff;
        font-weight: bold;
    }
    .cal-date { color: #555; font-size: 0.85rem; }
    
    .bet-header {
        background: #00f2ff;
        color: black !important;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: 900;
        margin: 20px 0;
        text-transform: uppercase;
    }
    /* BOTONES */
    div.stButton > button:first-child {
        background: #00ff88 !important;
        color: black !important;
        font-weight: 900 !important;
        height: 3em !important;
        width: 100% !important;
        border-radius: 10px !important;
        border: 2px solid white !important;
    }
    .racha-v { color: #00ff88; font-weight: bold; }
    .racha-e { color: #ffcc00; font-weight: bold; }
    .racha-d { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE DATOS
ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        return pd.read_csv(url)
    except: return None

# Función de Racha
def get_streak(df, team):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].dropna(subset=['FTR']).tail(5)
    streak = []
    for _, row in recent.iterrows():
        res = 'V' if (row['HomeTeam']==team and row['FTR']=='H') or (row['AwayTeam']==team and row['FTR']=='A') else ('E' if row['FTR']=='D' else 'D')
        color = "racha-v" if res=='V' else ("racha-e" if res=='E' else "racha-d")
        streak.append(f'<span class="{color}">{res}</span>')
    return " ".join(streak)

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

# SIDEBAR
with st.sidebar:
    st.header("⚙️ CONFIGURACIÓN")
    sel_liga = st.selectbox("ELEGIR LIGA", list(ligas.keys()))
    st.info("La IA se actualiza cada hora automáticamente.")

df_raw = load_data(ligas[sel_liga])

if df_raw is not None:
    # 3. PESTAÑAS PARA ORGANIZAR
    tab_analisis, tab_calendario = st.tabs(["🔥 ANALIZADOR PRO", "📅 PRÓXIMOS PARTIDOS"])

    with tab_calendario:
        st.markdown("<h2 style='color:white;'>Calendario Semanal</h2>", unsafe_allow_html=True)
        # Filtramos partidos que no tienen marcador (FTR es nulo)
        futuros = df_raw[df_raw['FTR'].isna()][['Date', 'HomeTeam', 'AwayTeam']].head(15)
        
        if futuros.empty:
            st.warning("No se detectan partidos próximos en esta liga hoy. Prueba con otra liga o espera a la actualización de mañana.")
        else:
            for _, row in futuros.iterrows():
                st.markdown(f"""
                <div class="cal-card">
                    <div class="cal-date">{row['Date']}</div>
                    <div style="font-size:1.2rem;">{row['HomeTeam']} <span style="color:#00f2ff">vs</span> {row['AwayTeam']}</div>
                </div>
                """, unsafe_allow_html=True)

    with tab_analisis:
        df = df_raw.dropna(subset=['FTR', 'B365H'])
        le = LabelEncoder()
        teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
        le.fit(teams)

        st.markdown("<h1 style='text-align: center; color: #00f2ff;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("LOCAL", teams)
        c1.markdown(f"Racha: {get_streak(df, t1)}", unsafe_allow_html=True)
        
        t2 = c2.selectbox("VISITANTE", teams, index=1)
        c2.markdown(f"Racha: {get_streak(df, t2)}", unsafe_allow_html=True)

        st.markdown("<div class='bet-header'>CUOTAS 1X2</div>", unsafe_allow_html=True)
        q1, qx, q2 = st.columns(3)
        v1 = q1.number_input(f"1 ({t1})", value=2.0)
        vx = qx.number_input("X (Empate)", value=3.2)
        v2 = q2.number_input(f"2 ({t2})", value=3.5)

        # Lógica IA
        df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
        df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
        X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
        m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
        m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])

        if st.button("🚀 ANALIZAR Y GUARDAR"):
            v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
            probs = m_win.predict_proba(v_in)[0]
            g = m_goals.predict(v_in)[0]
            pick = t1 if m_win.predict(v_in)[0] == 1 else (t2 if m_win.predict(v_in)[0] == 2 else "Empate")

            st.session_state.quiniela.append({
                "PARTIDO": f"{t1}-{t2}", "PICK": pick, "GOLES": f"{'+2.5' if g > 2.5 else '-2.5'}", "CONF.": f"{max(probs)*100:.0f}%"
            })

            st.markdown("<div class='bet-header'>RESULTADO MAESTRO</div>", unsafe_allow_html=True)
            res1, res2 = st.columns(2)
            res1.metric("PICK RECOMENDADO", pick, f"{max(probs)*100:.0f}% Confianza")
            res2.metric("GOLES ESTIMADOS", f"{g:.1f}", "+2.5" if g > 2.5 else "-2.5")

        if st.session_state.quiniela:
            st.markdown("<div class='bet-header'>📋 MI QUINIELA</div>", unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(st.session_state.quiniela), use_container_width=True)

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO DE MÁXIMA VISIBILIDAD (ELITE TERMINAL)
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,40,80,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    label { color: white !important; font-weight: 900 !important; font-size: 1.1rem !important; }

    /* QUINIELA Y TABLAS */
    .stTable td, .stTable th {
        color: white !important;
        font-weight: bold !important;
        background-color: rgba(0,0,0,0.4) !important;
    }
    thead tr th { background-color: #00f2ff !important; color: black !important; }

    /* TARJETAS DE CALENDARIO (MAX VISIBILIDAD) */
    .cal-box {
        background: white;
        color: #111;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 8px solid #00f2ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .cal-date { color: #555; font-size: 0.9rem; font-weight: bold; }
    .cal-match { font-size: 1.2rem; font-weight: 900; }

    .status-card {
        background: rgba(15, 23, 42, 0.95);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .metric-value { font-size: 2.8rem; font-weight: 900; color: #FFFFFF; }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        height: 3.5rem !important;
        border-radius: 15px !important;
    }
    .tag-plus { background: #00ff88; color: black; padding: 5px 12px; border-radius: 5px; font-weight: 900; }
    .tag-minus { background: #ff4b4b; color: white; padding: 5px 12px; border-radius: 5px; font-weight: 900; }
    .bet-header { background: #00f2ff; color: black; padding: 10px; border-radius: 5px; font-weight: 900; margin: 15px 0; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE LIGAS
ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER LEAGUE": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
    "🇫🇷 LIGUE 1": "https://www.football-data.co.uk/mmz4281/2526/F1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        return df
    except: return None

# ESTADÍSTICAS PRO
def get_pro_stats(df, team):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].dropna(subset=['FTR']).tail(5)
    streak = []
    g_favor = 0
    for _, row in recent.iterrows():
        is_home = row['HomeTeam'] == team
        g_favor += row['FTHG'] if is_home else row['FTAG']
        res = 'V' if (is_home and row['FTR']=='H') or (not is_home and row['FTR']=='A') else ('E' if row['FTR']=='D' else 'D')
        color = "#00ff88" if res=='V' else ("#ffcc00" if res=='E' else "#ff4b4b")
        streak.append(f'<span style="color:{color};">{res}</span>')
    return " ".join(streak), g_favor / 5 if not recent.empty else 0

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

with st.sidebar:
    st.title("⚽ CONTROL DE LIGAS")
    sel_liga = st.selectbox("ELEGIR COMPETICIÓN", list(ligas.keys()))
    st.info("Calendario sincronizado con la temporada 2025/26")

df_raw = load_data(ligas[sel_liga])

if df_raw is not None:
    tab_analisis, tab_calendario = st.tabs(["🔥 ANALIZADOR IA", "📅 CALENDARIO SEMANAL"])

    with tab_calendario:
        st.markdown("<h2 style='color:white;'>Partidos de esta Semana</h2>", unsafe_allow_html=True)
        # Mostrar partidos desde hoy hasta 7 días adelante
        hoy = datetime.now()
        semana_vista = hoy + timedelta(days=7)
        
        # Filtrar partidos futuros (donde FTR es nulo o la fecha es mayor a hoy)
        proximos = df_raw[df_raw['Date'] >= hoy.replace(hour=0, minute=0, second=0)].sort_values('Date').head(20)
        
        if proximos.empty:
            st.warning("No se encontraron partidos próximos en el servidor. Prueba con otra liga.")
        else:
            for _, row in proximos.iterrows():
                st.markdown(f"""
                    <div class="cal-box">
                        <div class="cal-date">📅 {row['Date'].strftime('%d/%m/%Y')}</div>
                        <div class="cal-match">{row['HomeTeam']} <span style="color:#00f2ff">VS</span> {row['AwayTeam']}</div>
                    </div>
                """, unsafe_allow_html=True)

    with tab_analisis:
        # Solo usamos datos con resultados para entrenar a la IA
        df_train = df_raw.dropna(subset=['FTR', 'B365H'])
        le = LabelEncoder()
        teams = sorted(pd.concat([df_train['HomeTeam'], df_train['AwayTeam']]).unique())
        le.fit(teams)

        st.markdown("<h1 style='text-align: center; color: #00f2ff;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("EQUIPO LOCAL", teams)
        racha1, g_fav1 = get_pro_stats(df_train, t1)
        c1.markdown(f'<div style="color:white">Racha: {racha1} | ⚽ Favor: <span style="color:#FFFF00">{g_fav1:.1f}</span></div>', unsafe_allow_html=True)
        
        t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
        racha2, g_fav2 = get_pro_stats(df_train, t2)
        c2.markdown(f'<div style="color:white">Racha: {racha2} | ⚽ Favor: <span style="color:#FFFF00">{g_fav2:.1f}</span></div>', unsafe_allow_html=True)

        st.markdown("<div class='bet-header'>📉 CUOTAS Y MERCADO</div>", unsafe_allow_html=True)
        q1, qx, q2 = st.columns(3)
        v1 = q1.number_input(f"1 ({t1})", value=2.0)
        vx = qx.number_input("Empate (X)", value=3.2)
        v2 = q2.number_input(f"2 ({t2})", value=3.5)

        # CÁLCULOS IA
        if st.button("🚀 ANALIZAR PARTIDO"):
            # Entrenar modelos rápidos
            df_train['H_c'], df_train['A_c'] = le.transform(df_train['HomeTeam']), le.transform(df_train['AwayTeam'])
            df_train['Target'] = df_train['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
            X = df_train[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
            
            m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df_train['Target'])
            m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df_train['FTHG'] + df_train['FTAG'])
            m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df_train['HC'] + df_train['AC'])
            m_cards = RandomForestRegressor(n_estimators=100).fit(X.values, df_train['HY'] + df_train['AY'])

            v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
            probs = m_win.predict_proba(v_in)[0]
            g, c, cards = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0], m_cards.predict(v_in)[0]
            pick = t1 if m_win.predict(v_in)[0] == 1 else (t2 if m_win.predict(v_in)[0] == 2 else "Empate")

            st.session_state.quiniela.append({
                "PARTIDO": f"{t1}-{t2}", "PICK": pick, 
                "GOLES": f"{'+2.5' if g > 2.5 else '-2.5'} ({g:.1f})",
                "TARJETAS": f"{'+4.5' if cards > 4.5 else '-4.5'} ({cards:.1f})",
                "CONF": f"{max(probs)*100:.0f}%"
            })

            # Mostrar resultados en tarjetas
            st.markdown("<div class='bet-header'>PRONÓSTICO GENERADO</div>", unsafe_allow_html=True)
            res1, res2, res3, res4 = st.columns(4)
            res1.markdown(f'<div class="status-card">WINNER<br><span style="color:#00ff88">{pick}</span></div>', unsafe_allow_html=True)
            res2.markdown(f'<div class="status-card">GOLES<br>{g:.1f}<br><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
            res3.markdown(f'<div class="status-card">CÓRNERS<br>{c:.0f}<br><span class="tag-plus"> + 8.5</span></div>', unsafe_allow_html=True)
            res4.markdown(f'<div class="status-card">TARJETAS<br>{cards:.1f}<br><span class="{"tag-plus" if cards > 4.5 else "tag-minus"}">{" + 4.5" if cards > 4.5 else " - 4.5"}</span></div>', unsafe_allow_html=True)

        if st.session_state.quiniela:
            st.markdown("<div class='bet-header'>📋 MI QUINIELA</div>", unsafe_allow_html=True)
            st.table(pd.DataFrame(st.session_state.quiniela))
            if st.button("🗑️ VACIAR TABLA"):
                st.session_state.quiniela = []
                st.rerun()

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO ELITE (MÁXIMA VISIBILIDAD)
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,40,80,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    label { color: white !important; font-weight: 900 !important; font-size: 1.1rem !important; }
    
    .stTable td, .stTable th {
        color: white !important;
        font-weight: bold !important;
        background-color: rgba(0,0,0,0.6) !important;
    }
    thead tr th { background-color: #00f2ff !important; color: black !important; }

    .status-card {
        background: rgba(15, 23, 42, 0.95);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
    }
    .metric-value { font-size: 2.8rem; font-weight: 900; color: #FFFFFF; }
    
    /* DISEÑO DE CALENDARIO GARANTIZADO */
    .cal-box {
        background: rgba(255, 255, 255, 0.95);
        color: #111 !important;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        border-left: 8px solid #00f2ff;
        font-family: sans-serif;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.5rem !important;
        height: 3.5rem !important;
        border-radius: 15px !important;
    }
    .tag-plus { background: #00ff88; color: black; padding: 4px 12px; border-radius: 5px; font-weight: 900; }
    .tag-minus { background: #ff4b4b; color: white; padding: 4px 12px; border-radius: 5px; font-weight: 900; }
    .bet-header { background: #00f2ff; color: black; padding: 10px; border-radius: 5px; font-weight: 900; margin: 15px 0; }
    </style>
    """, unsafe_allow_html=True)

# 2. MOTOR DE DATOS
ligas = {
    "🇪🇸 ESPAÑA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 INGLATERRA": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITALIA": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 ALEMANIA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try: return pd.read_csv(url)
    except: return None

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

with st.sidebar:
    st.title("⚽ LIGAS DISPONIBLES")
    sel_liga = st.selectbox("ELEGIR COMPETICIÓN", list(ligas.keys()))

df_raw = load_data(ligas[sel_liga])

if df_raw is not None:
    tab_analisis, tab_calendario = st.tabs(["🔥 ANALIZADOR PRO", "📅 CALENDARIO DE LA TEMPORADA"])

    with tab_calendario:
        st.markdown("<h2 style='color:white;'>Listado de Partidos</h2>", unsafe_allow_html=True)
        # Seleccionamos todos los partidos que aún no tienen resultado (FTR es nulo)
        # Esto incluye los de mañana, la semana que viene y el resto de la liga
        pendientes = df_raw[df_raw['FTR'].isna()][['Date', 'HomeTeam', 'AwayTeam']]
        
        if pendientes.empty:
            st.warning("No hay más partidos pendientes en este archivo de liga.")
        else:
            for _, row in pendientes.head(30).iterrows(): # Mostramos los siguientes 30 para no saturar
                st.markdown(f"""
                <div class="cal-box">
                    <strong>📅 {row['Date']}</strong><br>
                    <span style="font-size:1.1rem;">{row['HomeTeam']} <span style="color:#00f2ff">VS</span> {row['AwayTeam']}</span>
                </div>
                """, unsafe_allow_html=True)

    with tab_analisis:
        df_train = df_raw.dropna(subset=['FTR', 'B365H'])
        le = LabelEncoder()
        teams = sorted(pd.concat([df_train['HomeTeam'], df_train['AwayTeam']]).unique())
        le.fit(teams)

        st.markdown("<h1 style='text-align: center; color: #00f2ff;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        t1 = c1.selectbox("LOCAL", teams)
        t2 = c2.selectbox("VISITANTE", teams, index=1)

        st.markdown("<div class='bet-header'>CUOTAS 1X2</div>", unsafe_allow_html=True)
        q1, qx, q2 = st.columns(3)
        v1 = q1.number_input(f"Cuota {t1}", value=2.0)
        vx = qx.number_input("Cuota X", value=3.2)
        v2 = q2.number_input(f"Cuota {t2}", value=3.5)

        if st.button("🚀 ANALIZAR Y GUARDAR"):
            # Entrenamiento rápido
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
                "CORNERS": f"{'+9.5' if c > 9.5 else '-9.5'} ({c:.0f})",
                "TARJETAS": f"{'+4.5' if cards > 4.5 else '-4.5'} ({cards:.1f})",
                "CONF": f"{max(probs)*100:.0f}%"
            })

            st.markdown("<div class='bet-header'>RESULTADOS IA</div>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1: st.markdown(f'<div class="status-card"><div style="color:#00f2ff">PICK</div><div class="metric-value" style="color:#00ff88">{pick}</div></div>', unsafe_allow_html=True)
            with r2: st.markdown(f'<div class="status-card"><div style="color:#00f2ff">GOLES</div><div class="metric-value">{g:.1f}</div><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
            with r3: st.markdown(f'<div class="status-card"><div style="color:#00f2ff">CÓRNERS</div><div class="metric-value">{c:.0f}</div><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
            with r4: st.markdown(f'<div class="status-card"><div style="color:#00f2ff">TARJETAS</div><div class="metric-value">{cards:.1f}</div><span class="{"tag-plus" if cards > 4.5 else "tag-minus"}">{" + 4.5" if cards > 4.5 else " - 4.5"}</span></div>', unsafe_allow_html=True)

    if st.session_state.quiniela:
        st.markdown("<div class='bet-header'>📋 MI QUINIELA</div>", unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state.quiniela))
        if st.button("🗑️ LIMPIAR TODO"):
            st.session_state.quiniela = []
            st.rerun()

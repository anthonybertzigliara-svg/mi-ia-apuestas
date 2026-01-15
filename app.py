import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO ORIGINAL DE ALTA VISIBILIDAD
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,20,0.98) 0%, rgba(20,40,80,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    /* FORZAR LETRAS BLANCAS EN TODO EL PANEL */
    label, .stMarkdown p { color: white !important; font-weight: 900 !important; font-size: 1.1rem !important; }
    
    /* TABLA DE QUINIELA SIEMPRE LEGIBLE */
    .stTable td, .stTable th {
        color: white !important;
        font-weight: bold !important;
        background-color: rgba(0,0,0,0.7) !important;
    }
    thead tr th { background-color: #00f2ff !important; color: black !important; }

    /* TARJETAS DE RESULTADO NEÓN */
    .status-card {
        background: rgba(15, 23, 42, 0.95);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
    }
    .metric-title { color: #00f2ff; font-weight: 800; text-transform: uppercase; margin-bottom: 5px; }
    .metric-value { font-size: 2.8rem; font-weight: 900; color: #FFFFFF; }
    
    /* BOTÓN ANALIZAR VERDE GIGANTE */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-weight: 900 !important;
        font-size: 1.6rem !important;
        height: 4rem !important;
        border-radius: 15px !important;
        border: 2px solid white !important;
        margin-top: 20px;
    }

    .tag-plus { background: #00ff88; color: black; padding: 5px 12px; border-radius: 5px; font-weight: 900; }
    .tag-minus { background: #ff4b4b; color: white; padding: 5px 12px; border-radius: 5px; font-weight: 900; }
    .bet-header { background: #00f2ff; color: black; padding: 12px; border-radius: 8px; font-weight: 900; margin: 20px 0; font-size: 1.3rem; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS SIMPLE Y SEGURA
ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try: return pd.read_csv(url).dropna(subset=['FTR', 'B365H'])
    except: return None

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

with st.sidebar:
    st.title("⚽ MENU")
    sel_liga = st.selectbox("LIGA SELECCIONADA", list(ligas.keys()))

df = load_data(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)

    st.markdown("<h1 style='text-align: center; color: #00f2ff; text-shadow: 2px 2px 10px black;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
    
    # SELECCIÓN DE EQUIPOS
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("EQUIPO LOCAL", teams)
    t2 = col2.selectbox("EQUIPO VISITANTE", teams, index=1)

    # CUOTAS
    st.markdown("<div class='bet-header'>DATOS DEL MERCADO (CUOTAS)</div>", unsafe_allow_html=True)
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input(f"Cuota {t1}", value=2.0)
    vx = qx.number_input("Cuota Empate", value=3.2)
    v2 = q2.number_input(f"Cuota {t2}", value=3.5)

    if st.button("🚀 REALIZAR ANÁLISIS"):
        # MODELOS IA
        df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
        df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
        X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
        
        m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
        m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
        m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
        m_cards = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

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

        # TARJETAS DE RESULTADO
        st.markdown("<div class='bet-header'>PREDICCIÓN FINAL DE LA IA</div>", unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1: st.markdown(f'<div class="status-card"><div class="metric-title">GANADOR</div><div class="metric-value" style="color:#00ff88">{pick}</div><span class="tag-plus" style="background:#00f2ff">{max(probs)*100:.0f}%</span></div>', unsafe_allow_html=True)
        with r2: st.markdown(f'<div class="status-card"><div class="metric-title">GOLES</div><div class="metric-value">{g:.1f}</div><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
        with r3: st.markdown(f'<div class="status-card"><div class="metric-title">CÓRNERS</div><div class="metric-value">{c:.0f}</div><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
        with r4: st.markdown(f'<div class="status-card"><div class="metric-title">TARJETAS</div><div class="metric-value">{cards:.1f}</div><span class="{"tag-plus" if cards > 4.5 else "tag-minus"}">{" + 4.5" if cards > 4.5 else " - 4.5"}</span></div>', unsafe_allow_html=True)

    # TABLA DE QUINIELA
    if st.session_state.quiniela:
        st.markdown("<div class='bet-header'>📋 MI QUINIELA GUARDADA</div>", unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state.quiniela))
        if st.button("🗑️ LIMPIAR TODO"):
            st.session_state.quiniela = []
            st.rerun()

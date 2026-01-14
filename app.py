import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN DE PÁGINA
st.set_page_config(page_title="AI ELITE BETTING PRO", layout="wide")

# 2. DISEÑO DE ALTO IMPACTO (CSS CUSTOM)
st.markdown("""
    <style>
    /* Fondo General con Overlay */
    .stApp {
        background: linear-gradient(135deg, rgba(0,0,0,0.9) 0%, rgba(15,23,42,0.8) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }

    /* Estilo de Tarjetas Glassmorphism */
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 5px solid #00f2ff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.8);
        transition: 0.3s;
    }
    
    .main-prediction {
        background: linear-gradient(180deg, rgba(0,242,255,0.1) 0%, rgba(0,0,0,0.4) 100%);
        border: 2px solid #00f2ff;
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.4);
    }

    /* Textos Pro */
    h1 { font-size: 3.5rem !important; font-weight: 800 !important; color: #00f2ff !important; letter-spacing: -2px; }
    h2 { color: white !important; text-transform: uppercase; letter-spacing: 2px; }
    .metric-label { color: #8899ac; font-size: 0.9rem; font-weight: bold; text-transform: uppercase; }
    .metric-value { color: white; font-size: 2.5rem; font-weight: 800; }

    /* Botón Animado */
    .stButton>button {
        background: linear-gradient(90deg, #00f2ff, #0077ff) !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 50px !important;
        transition: all 0.4s ease !important;
        text-transform: uppercase;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 30px rgba(0, 242, 255, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# 3. MOTOR DE DATOS (AUTO-UPDATE)
ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER LEAGUE": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def get_data(url):
    try:
        data = pd.read_csv(url)
        cols = ['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC','HY','AY']
        return data[cols].dropna()
    except: return None

# Sidebar elegante
with st.sidebar:
    st.markdown("## ⚙️ CONFIGURACIÓN")
    sel_liga = st.selectbox("COMPETICIÓN ACTIVA", list(ligas.keys()))
    st.info("La IA actualiza los datos cada hora automáticamente.")

df = get_data(ligas[sel_liga])

if df is not None:
    # Entrenamiento rápido de Modelos
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

    # CABECERA
    st.markdown("<h1 style='text-align: center;'>AI ELITE BETTING</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #8899ac;'>TERMINAL DE INTELIGENCIA DEPORTIVA • {sel_liga}</p>", unsafe_allow_html=True)
    
    # SELECCIÓN DE EQUIPOS
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        t1 = st.selectbox("🏠 EQUIPO LOCAL", teams)
    with c2:
        t2 = st.selectbox("🚀 EQUIPO VISITANTE", teams, index=1)
    
    # CUOTAS
    st.markdown("### 🏦 CUOTAS DE MERCADO")
    q1, qx, q2 = st.columns(3)
    val1 = q1.number_input("LOCAL (1)", value=2.10)
    valx = qx.number_input("EMPATE (X)", value=3.30)
    val2 = q2.number_input("VISITA (2)", value=3.80)

    if st.button("⚡ GENERAR PRONÓSTICO MAESTRO"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], val1, valx, val2]]
        p = m_win.predict_proba(v)[0]
        g, c, cards = m_goals.predict(v)[0], m_corn.predict(v)[0], m_cards.predict(v)[0]
        
        # Lógica Ganador
        idx = m_win.predict(v)[0]
        ganador_final = t1 if idx == 1 else (t2 if idx == 2 else "Empate")

        # PANEL DE PREDICCIÓN PRINCIPAL
        st.markdown(f"""
            <div class="main-prediction">
                <h2 style="color: #00f2ff !important;">PICK RECOMENDADO</h2>
                <h1 style="margin: 20px 0;">{ganador_final.upper()}</h1>
                <p style="font-size: 1.5rem; font-weight: bold;">CONFIANZA DEL SISTEMA: {max(p)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # GRID DE MÉTRICAS DETALLADAS
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class="result-card">
                <div class="metric-label">Goles Totales</div>
                <div class="metric-value">{g:.1f}</div>
                <div style="color: #00ff88; font-weight: bold;">{'OVER 2.5' if g > 2.5 else 'UNDER 2.5'}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class="result-card">
                <div class="metric-label">Córners Est.</div>
                <div class="metric-value">{c:.1f}</div>
                <div style="color: #00f2ff; font-weight: bold;">{'+9.5 CORNERS' if c > 9.5 else '-9.5 CORNERS'}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class="result-card">
                <div class="metric-label">Tarjetas Est.</div>
                <div class="metric-value">{cards:.1f}</div>
                <div style="color: #ffcc00; font-weight: bold;">{'PARTIDO TENSO' if cards > 4.5 else 'PARTIDO LIMPIO'}</div>
            </div>""", unsafe_allow_html=True)

else:
    st.error("Error crítico: No se ha podido establecer conexión con el servidor de datos.")

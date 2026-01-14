import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de página con el título de la imagen
st.set_page_config(page_title="AI ELITE BETTING", layout="wide")

# CSS para replicar exactamente el diseño de neón y estadio
st.markdown("""
    <style>
    /* Fondo del Estadio Nocturno */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* Contenedor principal con transparencia elegante */
    .main {
        background-color: rgba(15, 23, 42, 0.8);
        padding: 25px;
        border-radius: 20px;
        margin-top: 20px;
    }

    /* Títulos estilo neón */
    h1, h2, h3 {
        color: white !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* Cuadros de métricas con borde cian (igual que la imagen) */
    .metric-card {
        background-color: rgba(30, 41, 59, 0.9);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 45px;
        font-weight: 800;
        color: white;
        margin: 10px 0;
    }

    .badge-over { background-color: #00ff88; color: black; padding: 2px 10px; border-radius: 5px; font-weight: bold; }
    .badge-under { background-color: #ffcc00; color: black; padding: 2px 10px; border-radius: 5px; font-weight: bold; }

    /* Botón de Analizar (Cian brillante) */
    .stButton>button {
        background-color: #00f2ff !important;
        color: black !important;
        font-weight: bold !important;
        font-size: 20px !important;
        border-radius: 10px !important;
        width: 100%;
        height: 3em;
        border: none;
    }

    /* Inputs de cuotas */
    input {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #00f2ff !important;
        font-size: 20px !important;
    }
    label { color: #8899ac !important; font-size: 14px !important; }
    </style>
    """, unsafe_allow_html=True)

# Lógica de carga de datos
ligas = {"La Liga": "SP1.csv", "Premier": "E0.csv", "Serie A": "I1.csv", "Bundesliga": "D1.csv"}
st.sidebar.title("PANEL CONTROL")
sel = st.sidebar.selectbox("SELECCIONAR LIGA", list(ligas.keys()))

@st.cache_data
def load(file):
    try:
        df = pd.read_csv(file)
        return df[['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC','HY','AY']].dropna()
    except: return None

df = load(ligas[sel])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    # Preparamos la IA
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H', 'A', 'B365H', 'B365D', 'B365A']]
    
    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_t = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    st.markdown("<h1>AI ELITE BETTING</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3>⚽ TERMINAR INTELIGENTE: {sel.upper()}</h3>", unsafe_allow_html=True)
    
    # Selección de equipos
    c1, v1, c2 = st.columns([1, 0.2, 1])
    t1 = c1.selectbox("EQUIPO LOCAL", teams)
    v1.markdown("<h2 style='text-align:center; margin-top:25px;'>VS</h2>", unsafe_allow_html=True)
    t2 = c2.selectbox("AWAY VISITANTE", teams, index=1)
    
    # Cuotas
    st.markdown("### 📊 CUOTAS DEL MERCADO")
    q_col1, q_col2, q_col3 = st.columns(3)
    q1 = q_col1.number_input("Cuota Local", value=2.00)
    qx = q_col2.number_input("Cuota Empate", value=3.20)
    q2 = q_col3.number_input("Cuota Visita", value=3.50)

    if st.button("🔥 ANALIZAR PARTIDO AHORA"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        probs = m_r.predict_proba(v)[0]
        g, cor, tar = m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        
        # Mostrar Pick Ganador
        gan = t1 if probs[1] > probs[2] and probs[1] > probs[0] else (t2 if probs[2] > probs[1] else "Empate")
        st.markdown(f"""
            <div style="background: rgba(0, 242, 255, 0.1); border: 2px solid #00f2ff; padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 30px;">
                <h2 style="margin:0;">GANADOR PROBABLE: {gan.upper()}</h2>
                <h3 style="color: #00f2ff !important; margin:0;">CONFIANZA: {max(probs)*100:.1f}%</h3>
            </div>
        """, unsafe_allow_html=True)

        # Fila de métricas (Igual que en la imagen)
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">GOLES</div><div class="metric-value">{g:.1f}</div><span class="badge-over">OVER</span></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">CÓRNERS</div><div class="metric-value">{cor:.1f}</div><span class="badge-over">OVER</span></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">TARJETAS</div><div class="metric-value">{tar:.1f}</div><span class="badge-under">UNDER</span></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">LOCAL %</div><div class="metric-value">{probs[1]*100:.0f}%</div><span class="badge-over">WIN</span></div>', unsafe_allow_html=True)

else:
    st.error("Error: Archivos de datos no encontrados.")

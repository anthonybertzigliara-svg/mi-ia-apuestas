import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Configuración de página
st.set_page_config(page_title="IA Predictor Elite", layout="wide")

# 2. Estilo Visual (Fondo de Fútbol + Legibilidad)
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    /* Capa oscura para que se lea todo bien */
    .main {
        background-color: rgba(15, 23, 42, 0.85);
        padding: 30px;
        border-radius: 20px;
    }
    /* Cajas de pronóstico */
    .metric-card {
        background-color: #1e293b;
        border: 2px solid #3b82f6;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
    }
    .winner-box {
        background: linear-gradient(90deg, #1d4ed8, #3b82f6);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 20px;
        color: white;
    }
    /* Forzar visibilidad de inputs */
    input { color: white !important; }
    label { color: #cbd5e1 !important; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 3. Lógica de Datos
ligas = {"🇪🇸 La Liga": "SP1.csv", "🇬🇧 Premier": "E0.csv", "🇮🇹 Serie A": "I1.csv", "🇩🇪 Bundesliga": "D1.csv"}
sel = st.sidebar.selectbox("Selecciona tu Liga", list(ligas.keys()))

@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file)
        return df[['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC','HY','AY']].dropna()
    except: return None

df = load_data(ligas[sel])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    # Entrenamiento rápido
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H', 'A', 'B365H', 'B365D', 'B365A']]
    
    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_t = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    # 4. Interfaz de Usuario
    st.title("🏟️ Terminal Inteligente de Fútbol")
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("Local", teams)
    t2 = c2.selectbox("Visitante", teams, index=1)
    
    st.markdown("### 🏦 Cuotas (€)")
    q_cols = st.columns(3)
    q1 = q_cols[0].number_input("Cuota 1", value=2.00)
    qx = q_cols[1].number_input("Cuota X", value=3.20)
    q2 = q_cols[2].number_input("Cuota 2", value=3.50)

    if st.button("🚀 ANALIZAR PARTIDO"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        probs = m_r.predict_proba(v)[0]
        goles, corners, tarjetas = m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        
        ganador = t1 if probs[1] > probs[2] and probs[1] > probs[0] else (t2 if probs[2] > probs[1] else "Empate")

        # Visualización de Resultados
        st.markdown(f"""
            <div class="winner-box">
                <h2 style="color: white; margin:0;">PRONÓSTICO: {ganador.upper()}</h2>
                <p style="font-size: 20px;">Confianza: {max(probs)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            txt = "MÁS DE 2.5 GOLES" if goles > 2.5 else "MENOS DE 2.5 GOLES"
            st.markdown(f'<div class="metric-card"><h3>Goles</h3><h1>{goles:.1f}</h1><p>{txt}</p></div>', unsafe_allow_html=True)
        with r2:
            txt = "+9.5 CÓRNERS" if corners > 9.5 else "-9.5 CÓRNERS"
            st.markdown(f'<div class="metric-card"><h3>Córners</h3><h1>{corners:.1f}</h1><p>{txt}</p></div>', unsafe_allow_html=True)
        with r3:
            txt = "+4.5 TARJETAS" if tarjetas > 4.5 else "-4.5 TARJETAS"
            st.markdown(f'<div class="metric-card"><h3>Tarjetas</h3><h1>{tarjetas:.1f}</h1><p>{txt}</p></div>', unsafe_allow_html=True)

else:
    st.warning("Por favor, asegúrate de que los archivos CSV estén cargados.")

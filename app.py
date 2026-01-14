import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Configuración de la App
st.set_page_config(page_title="AI ELITE BETTING", layout="wide")

# 2. Estilo Visual Neón (Fondo de Estadio Oscuro)
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main { background-color: rgba(15, 23, 42, 0.85); padding: 2rem; border-radius: 20px; }
    h1, h2, h3, p, label { color: white !important; }
    .metric-card {
        background-color: rgba(30, 41, 59, 0.9);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
    }
    .metric-value { font-size: 48px; font-weight: 900; color: white; margin: 10px 0; }
    .badge { padding: 5px 15px; border-radius: 8px; font-weight: bold; color: black; }
    .bg-over { background-color: #00ff88; }
    .bg-under { background-color: #ffcc00; }
    .stButton>button {
        background: #00f2ff !important; color: black !important;
        font-weight: 900 !important; width: 100%; border-radius: 12px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Lógica de DATOS AUTOMÁTICOS (Sin subir CSVs)
# Estas URLs se actualizan solas cada 24 horas
ligas = {
    "España": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "Inglaterra": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "Italia": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "Alemania": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600) # Se actualiza cada hora automáticamente
def cargar_datos(url):
    try:
        df = pd.read_csv(url)
        cols = ['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC']
        return df[cols].dropna()
    except: return None

seleccion = st.sidebar.selectbox("LIGA", list(ligas.keys()))
df = cargar_datos(ligas[seleccion])

if df is not None:
    le = LabelEncoder()
    equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(equipos)
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H', 'A', 'B365H', 'B365D', 'B365A']]
    
    m_r = RandomForestClassifier().fit(X.values, df['T'])
    m_g = RandomForestRegressor().fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor().fit(X.values, df['HC'] + df['AC'])

    st.markdown("<h1>AI ELITE BETTING</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("LOCAL", equipos)
    t2 = c2.selectbox("VISITANTE", equipos, index=1)
    
    st.markdown("---")
    cq1, cqx, cq2 = st.columns(3)
    q1 = cq1.number_input("Cuota 1", 1.0, 50.0, 2.0)
    qx = cqx.number_input("Cuota X", 1.0, 50.0, 3.2)
    q2 = cq2.number_input("Cuota 2", 1.0, 50.0, 3.5)

    if st.button("🔥 ANALIZAR AHORA"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p, g, cor = m_r.predict_proba(v)[0], m_g.predict(v)[0], m_c.predict(v)[0]
        gan = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] else "Empate")

        st.markdown(f'<div style="border:2px solid #00f2ff; padding:20px; border-radius:15px; text-align:center;"><h2>{gan.upper()}</h2><p>Confianza: {max(p)*100:.1f}%</p></div>', unsafe_allow_html=True)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.markdown(f'<div class="metric-card"><div>GOLES</div><div class="metric-value">{g:.1f}</div></div>', unsafe_allow_html=True)
        with col_res2:
            st.markdown(f'<div class="metric-card"><div>CÓRNERS</div><div class="metric-value">{cor:.1f}</div></div>', unsafe_allow_html=True)

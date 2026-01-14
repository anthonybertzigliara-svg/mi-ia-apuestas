import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN E INTERFAZ
st.set_page_config(page_title="AI ELITE BETTING", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .main { background-color: transparent; }
    h1, h2, h3, p, label { color: white !important; font-family: 'sans-serif'; }
    
    /* Cajas de Neón */
    .metric-card {
        background-color: rgba(15, 23, 42, 0.9);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.3);
        margin-bottom: 10px;
    }
    .metric-value { font-size: 40px; font-weight: 900; color: white; }
    
    /* Botón */
    .stButton>button {
        background: #00f2ff !important;
        color: black !important;
        font-weight: bold !important;
        width: 100%;
        border-radius: 10px !important;
        height: 3em;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA AUTOMÁTICA DE DATOS (SIN CSV LOCALES)
ligas = {
    "España": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "Inglaterra": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "Italia": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "Alemania": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def fetch_data(url):
    try:
        data = pd.read_csv(url)
        # Seleccionamos solo columnas esenciales para evitar errores de memoria
        valid_cols = ['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC']
        return data[valid_cols].dropna()
    except:
        return None

# Sidebar
st.sidebar.title("CONTROL IA")
sel_liga = st.sidebar.selectbox("Selecciona Competición", list(ligas.keys()))

df = fetch_data(ligas[sel_liga])

if df is not None:
    # Preparar IA
    le = LabelEncoder()
    all_teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(all_teams)
    
    df['H_code'] = le.transform(df['HomeTeam'])
    df['A_code'] = le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    
    X = df[['H_code', 'A_code', 'B365H', 'B365D', 'B365A']]
    
    # Modelos (Rápidos)
    model_win = RandomForestClassifier(n_estimators=50).fit(X.values, df['Target'])
    model_goals = RandomForestRegressor(n_estimators=50).fit(X.values, df['FTHG'] + df['FTAG'])
    model_corners = RandomForestRegressor(n_estimators=50).fit(X.values, df['HC'] + df['AC'])

    # INTERFAZ PRINCIPAL
    st.markdown("<h1 style='text-align: center;'>AI ELITE BETTING</h1>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    t1 = col_t1.selectbox("EQUIPO LOCAL", all_teams)
    t2 = col_t2.selectbox("EQUIPO VISITANTE", all_teams, index=1)
    
    st.markdown("### 📊 CUOTAS")
    c_q1, c_qx, c_q2 = st.columns(3)
    q1 = c_q1.number_input("Local", 1.0, 50.0, 2.0)
    qx = c_qx.number_input("Empate", 1.0, 50.0, 3.2)
    q2 = c_q2.number_input("Visita", 1.0, 50.0, 3.5)

    if st.button("🚀 REALIZAR PREDICCIÓN MAESTRA"):
        # Predicción
        input_data = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        prob = model_win.predict_proba(input_data)[0]
        goles = model_goals.predict(input_data)[0]
        corners = model_corners.predict(input_data)[0]
        
        # Resultado Ganador
        idx = model_win.predict(input_data)[0]
        res_text = t1 if idx == 1 else (t2 if idx == 2 else "Empate")
        
        st.markdown(f"""
            <div style="border: 2px solid #00f2ff; padding: 25px; border-radius: 20px; text-align: center; margin-bottom: 20px; background: rgba(0,242,255,0.1);">
                <h3 style="margin:0;">GANADOR PROBABLE</h3>
                <h1 style="font-size: 60px; margin:10px 0; color:#00f2ff !important;">{res_text.upper()}</h1>
                <p style="font-size: 20px;">Confianza de la IA: {max(prob)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # Métricas
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><p>GOLES EST.</p><div class="metric-value">{goles:.1f}</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><p>CÓRNERS EST.</p><div class="metric-value">{corners:.1f}</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><p>PROBABILIDAD</p><div class="metric-value">{max(prob)*100:.0f}%</div></div>', unsafe_allow_html=True)

else:
    st.error("No se pudieron cargar los datos. Verifica la conexión a internet de la app.")

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. Configuración de la aplicación
st.set_page_config(page_title="AI ELITE BETTING", layout="wide")

# 2. Diseño Estético (Igual a la imagen que te gustó)
st.markdown("""
    <style>
    /* Fondo de Estadio Oscuro */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Contenedor Principal */
    .main {
        background-color: rgba(15, 23, 42, 0.85);
        padding: 2rem;
        border-radius: 20px;
    }

    /* Títulos en Blanco */
    h1, h2, h3, p, label {
        color: white !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }

    /* Cuadros de Resultados con Borde Cian */
    .metric-card {
        background-color: rgba(30, 41, 59, 0.9);
        border: 2px solid #00f2ff;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 0 15px rgba(0, 242, 255, 0.2);
    }

    .metric-value {
        font-size: 48px;
        font-weight: 900;
        color: white;
        margin: 10px 0;
    }

    /* Etiquetas de Recomendación */
    .badge {
        padding: 5px 15px;
        border-radius: 8px;
        font-weight: bold;
        color: black;
    }
    .bg-over { background-color: #00ff88; }
    .bg-under { background-color: #ffcc00; }
    .bg-win { background-color: #00f2ff; }

    /* Botón Principal */
    .stButton>button {
        background: #00f2ff !important;
        color: black !important;
        font-weight: 900 !important;
        width: 100%;
        border-radius: 12px !important;
        border: none !important;
        height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Lógica de Datos y Entrenamiento
ligas = {"España": "SP1.csv", "Inglaterra": "E0.csv", "Italia": "I1.csv", "Alemania": "D1.csv"}
st.sidebar.title("🎮 PANEL CONTROL")
seleccion = st.sidebar.selectbox("LIGA SELECCIONADA", list(ligas.keys()))

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo)
        return df[['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC','HY','AY']].dropna()
    except: return None

df = cargar_datos(ligas[seleccion])

if df is not None:
    le = LabelEncoder()
    equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(equipos)
    
    # Entrenamiento rápido de la IA
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H', 'A', 'B365H', 'B365D', 'B365A']]
    
    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    
    # 4. Interfaz de Usuario
    st.markdown("<h1>AI ELITE BETTING</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3>📍 MODO: {seleccion.upper()}</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("LOCAL", equipos)
    t2 = col2.selectbox("VISITANTE", equipos, index=1)
    
    st.markdown("---")
    c_q1, c_qx, c_q2 = st.columns(3)
    q1 = c_q1.number_input("Cuota Local", 1.0, 50.0, 2.0)
    qx = c_qx.number_input("Cuota Empate", 1.0, 50.0, 3.2)
    q2 = c_q2.number_input("Cuota Visita", 1.0, 50.0, 3.5)

    if st.button("🔥 EJECUTAR ANÁLISIS"):
        entrada = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p = m_r.predict_proba(entrada)[0]
        g, cor = m_g.predict(entrada)[0], m_c.predict(entrada)[0]
        
        ganador = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] else "Empate")

        # Pantalla de Resultados
        st.markdown(f"""
            <div style="border: 2px solid #00f2ff; padding: 30px; border-radius: 20px; text-align: center; margin-bottom: 30px; background: rgba(0,242,255,0.05);">
                <h3 style="margin:0;">GANADOR PROBABLE</h3>
                <h1 style="font-size: 70px; margin:10px 0;">{ganador.upper()}</h1>
                <p style="font-size: 24px; color: #00f2ff !important;">Confianza IA: {max(p)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            badge = "bg-over" if g > 2.5 else "bg-under"
            txt = "OVER" if g > 2.5 else "UNDER"
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">GOLES</div><div class="metric-value">{g:.1f}</div><span class="badge {badge}">{txt} 2.5</span></div>', unsafe_allow_html=True)
        with m2:
            badge = "bg-over" if cor > 9.5 else "bg-under"
            txt = "OVER" if cor > 9.5 else "UNDER"
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">CÓRNERS</div><div class="metric-value">{cor:.1f}</div><span class="badge {badge}">{txt} 9.5</span></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div style="color:#00f2ff">WIN RATE</div><div class="metric-value">{max(p)*100:.0f}%</div><span class="badge bg-win">SISTEMA</span></div>', unsafe_allow_html=True)

else:
    st.error("Error: Sube los archivos CSV correspondientes.")

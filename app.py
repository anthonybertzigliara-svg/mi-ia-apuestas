import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN E INTERFAZ DE ALTA FIDELIDAD
st.set_page_config(page_title="IA ELITE BETTING - AUTÓNOMA", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(10,10,10,0.9) 0%, rgba(20,30,50,0.85) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .status-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .bet-header {
        background: linear-gradient(90deg, #00f2ff, #0077ff);
        color: black !important;
        padding: 10px;
        border-radius: 10px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 20px;
    }
    .value-tag { padding: 5px 10px; border-radius: 5px; font-weight: bold; }
    .bg-favor { background-color: #00ff88; color: black; }
    .bg-contra { background-color: #ff4b4b; color: white; }
    h1, h3, p, label { color: white !important; font-family: 'Inter', sans-serif; }
    .metric-value { font-size: 2.5rem; font-weight: 900; }
    </style>
    """, unsafe_allow_html=True)

# 2. SISTEMA DE DATOS AUTÓNOMO (ACTUALIZACIÓN DIARIA AUTOMÁTICA)
# Estas URLs se actualizan solas cada 24h con resultados y estadísticas frescas
ligas = {
    "🇪🇸 ESPAÑA - LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 INGLATERRA - PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 ITALIA - SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 ALEMANIA - BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600) # Se refresca solo cada hora para captar cambios
def load_live_data(url):
    try:
        data = pd.read_csv(url)
        # Seleccionamos columnas de goles, córners y tarjetas para el análisis profundo
        cols = ['HomeTeam','AwayTeam','B365H','B365D','B365A','FTR','FTHG','FTAG','HC','AC','HY','AY']
        return data[cols].dropna()
    except: return None

with st.sidebar:
    st.markdown("## 🛰️ ESTADO DEL SISTEMA")
    sel_liga = st.selectbox("LIGA ACTIVA", list(ligas.keys()))
    st.success("Base de datos: Actualizada ✅")

df = load_live_data(ligas[sel_liga])

if df is not None:
    # 3. ENTRENAMIENTO DE IA EN TIEMPO REAL
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
    
    # Modelos predictivos
    m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
    m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_cards = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    # 4. INTERFAZ DE ANÁLISIS
    st.markdown("<h1 style='text-align: center;'>TERMINAL DE INTELIGENCIA DEPORTIVA</h1>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    t1 = col_t1.selectbox("EQUIPO LOCAL (A FAVOR)", teams)
    t2 = col_t2.selectbox("EQUIPO VISITANTE (EN CONTRA)", teams, index=1)
    
    st.markdown("### 🏟️ CUOTAS ACTUALES")
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input(f"Cuota {t1}", value=2.0)
    vx = qx.number_input("Cuota Empate", value=3.2)
    v2 = q2.number_input(f"Cuota {t2}", value=3.5)

    if st.button("🔥 EJECUTAR ANÁLISIS ELITE"):
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
        probs = m_win.predict_proba(v_in)[0]
        g, c, cards = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0], m_cards.predict(v_in)[0]
        
        # Lógica de Recomendación Principal
        rec_idx = m_win.predict(v_in)[0]
        recomendacion = t1 if rec_idx == 1 else (t2 if rec_idx == 2 else "Evitar / Empate")

        st.markdown(f"""
            <div class="main-prediction" style="border: 3px solid #00f2ff; padding: 30px; border-radius: 20px; text-align: center;">
                <h3 style="color: #00f2ff !important; margin:0;">PICK MAESTRO RECOMENDADO</h3>
                <h1 style="font-size: 4rem; margin:10px 0;">{recomendacion.upper()}</h1>
                <p style="font-size: 1.2rem;">Confianza de la IA: {max(probs)*100:.1f}%</p>
            </div><br>""", unsafe_allow_html=True)

        # 5. DETALLES DE MERCADO: A FAVOR / EN CONTRA
        st.markdown("<div class='bet-header'>Análisis de Mercado: Valor 1X2</div>", unsafe_allow_html=True)
        p_col1, p_colx, p_col2 = st.columns(3)
        
        def get_sentiment(cuota, prob):
            valor = cuota * prob
            if valor > 1.10: return "A FAVOR", "bg-favor"
            return "EN CONTRA", "bg-contra"

        for col, p, c, name in zip([p_col1, p_colx, p_col2], [probs[1], probs[0], probs[2]], [v1, vx, v2], [t1, "Empate", t2]):
            sent, style = get_sentiment(c, p)
            col.markdown(f"""
                <div class="status-card">
                    <p style="margin-bottom:5px; font-weight:bold;">{name}</p>
                    <div class="metric-value">{p*100:.0f}%</div>
                    <div class="value-tag {style}">{sent}</div>
                </div>""", unsafe_allow_html=True)

        # 6. DETALLES DE GOLES, CÓRNERS Y TARJETAS
        st.markdown("<div class='bet-header'>Pronósticos de Eventos (Favor/Contra)</div>", unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        
        # Lógica Goles
        r1.markdown(f"""<div class="status-card">
            <p>GOLES ESTIMADOS</p>
            <div class="metric-value">{g:.1f}</div>
            <div class="value-tag {'bg-favor' if g > 2.5 else 'bg-contra'}">{'A FAVOR +2.5' if g > 2.5 else 'EN CONTRA +2.5'}</div>
        </div>""", unsafe_allow_html=True)
        
        # Lógica Córners
        r2.markdown(f"""<div class="status-card">
            <p>CÓRNERS ESTIMADOS</p>
            <div class="metric-value">{c:.1f}</div>
            <div class="value-tag {'bg-favor' if c > 9.5 else 'bg-contra'}">{'+9.5 CORNERS' if c > 9.5 else 'EN CONTRA +9.5'}</div>
        </div>""", unsafe_allow_html=True)
        
        # Lógica Tarjetas
        r3.markdown(f"""<div class="status-card">
            <p>TARJETAS ESTIMADAS</p>
            <div class="metric-value">{cards:.1f}</div>
            <div class="value-tag {'bg-favor' if cards > 4.5 else 'bg-contra'}">{'TENSO (OVER)' if cards > 4.5 else 'LIMPIO (UNDER)'}</div>
        </div>""", unsafe_allow_html=True)
else:
    st.error("No se ha podido conectar con el satélite de datos deportivos.")

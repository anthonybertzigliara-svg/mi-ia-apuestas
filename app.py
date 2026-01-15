import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. CONFIGURACIÓN E INTERFAZ PRO
st.set_page_config(page_title="AI ELITE BETTING PRO", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(0,0,0,0.9) 0%, rgba(15,23,42,0.8) 100%), 
                    url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
        background-attachment: fixed;
    }
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 15px;
    }
    .main-prediction {
        background: linear-gradient(180deg, rgba(0,242,255,0.1) 0%, rgba(0,0,0,0.4) 100%);
        border: 2px solid #00f2ff;
        padding: 30px;
        border-radius: 20px;
        text-align: center;
    }
    h1 { color: #00f2ff !important; font-weight: 800 !important; }
    h3, p, label { color: white !important; }
    .metric-value { font-size: 2.2rem; font-weight: 800; color: white; }
    .stButton>button {
        background: linear-gradient(90deg, #00f2ff, #0077ff) !important;
        color: white !important; font-weight: bold !important;
        width: 100%; border-radius: 50px !important; border: none !important; height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA AUTOMÁTICA
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

with st.sidebar:
    st.title("⚙️ PANEL CONTROL")
    sel_liga = st.selectbox("LIGA", list(ligas.keys()))

df = get_data(ligas[sel_liga])

if df is not None:
    # 3. ENTRENAMIENTO DE IA
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

    # 4. INTERFAZ
    st.markdown("<h1 style='text-align: center;'>AI ELITE BETTING</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("LOCAL", teams)
    t2 = c2.selectbox("VISITANTE", teams, index=1)
    
    st.markdown("### 🏦 CUOTAS MERCADO")
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input("Cuota 1", value=2.0)
    vx = qx.number_input("Cuota X", value=3.2)
    v2 = q2.number_input("Cuota 2", value=3.5)

    if st.button("⚡ GENERAR PRONÓSTICO MAESTRO"):
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
        probs = m_win.predict_proba(v_in)[0]
        g, c, cards = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0], m_cards.predict(v_in)[0]
        
        # Resultado
        idx = m_win.predict(v_in)[0]
        ganador = t1 if idx == 1 else (t2 if idx == 2 else "Empate")

        st.markdown(f"""
            <div class="main-prediction">
                <h3>PICK RECOMENDADO</h3>
                <h1>{ganador.upper()}</h1>
                <p>Confianza: {max(probs)*100:.1f}%</p>
            </div><br>""", unsafe_allow_html=True)

        # Probabilidades estilo Casa de Apuestas
        st.markdown("### 📈 ANÁLISIS DE VALOR (1X2)")
        p_col1, p_colx, p_col2 = st.columns(3)
        
        with p_col1:
            val = "✅ VALOR" if (v1 * probs[1]) > 1.05 else "❌ NO VALOR"
            st.markdown(f'<div class="result-card"><p>{t1}</p><div class="metric-value">{probs[1]*100:.0f}%</div><p>{val}</p></div>', unsafe_allow_html=True)
        with p_colx:
            val = "✅ VALOR" if (vx * probs[0]) > 1.05 else "❌ NO VALOR"
            st.markdown(f'<div class="result-card"><p>Empate</p><div class="metric-value">{probs[0]*100:.0f}%</div><p>{val}</p></div>', unsafe_allow_html=True)
        with p_col2:
            val = "✅ VALOR" if (v2 * probs[2]) > 1.05 else "❌ NO VALOR"
            st.markdown(f'<div class="result-card"><p>{t2}</p><div class="metric-value">{probs[2]*100:.0f}%</div><p>{val}</p></div>', unsafe_allow_html=True)

        # Estadísticas Detalladas
        st.markdown("### 📊 ESTADÍSTICAS ESTIMADAS")
        r1, r2, r3 = st.columns(3)
        r1.markdown(f'<div class="result-card"><p>GOLES</p><div class="metric-value">{g:.1f}</div></div>', unsafe_allow_html=True)
        r2.markdown(f'<div class="result-card"><p>CÓRNERS</p><div class="metric-value">{c:.1f}</div></div>', unsafe_allow_html=True)
        r3.markdown(f'<div class="result-card"><p>TARJETAS</p><div class="metric-value">{cards:.1f}</div></div>', unsafe_allow_html=True)
else:
    st.error("Error al cargar datos.")

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración Pro
st.set_page_config(page_title="AI ELITE BET", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    .stMetric { background-color: #1e293b !important; border: 1px solid #3b82f6 !important; border-radius: 10px; padding: 10px !important; }
    .winner-card {
        background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%);
        padding: 20px; border-radius: 15px; text-align: center; margin: 15px 0; border: 1px solid #60a5fa;
    }
    .stButton>button { background: #2563eb; color: white; font-weight: bold; width: 100%; border-radius: 10px; height: 3em; }
    </style>
    """, unsafe_allow_html=True)

ligas = {"🇪🇸 La Liga": "SP1.csv", "🇬🇧 Premier": "E0.csv", "🇮🇹 Serie A": "I1.csv", "🇩🇪 Bundesliga": "D1.csv"}
sel = st.sidebar.selectbox("LIGA", list(ligas.keys()))

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
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H', 'A', 'B365H', 'B365D', 'B365A']]

    # Modelos
    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_t = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    st.title(f"🚀 TERMINAL IA: {sel}")
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("LOCAL", teams)
    t2 = c2.selectbox("VISITANTE", teams, index=1)
    
    colq1, colq2, colq3 = st.columns(3)
    q1 = colq1.number_input("Cuota 1", value=2.0)
    qx = colq2.number_input("Cuota X", value=3.2)
    q2 = colq3.number_input("Cuota 2", value=3.5)

    if st.button("📊 ANALIZAR PARTIDO"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p = m_r.predict_proba(v)[0]
        g, cor, tar = m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        
        gan = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] and p[2] > p[0] else "Empate")

        st.markdown(f'<div class="winner-card"><h3>GANADOR PROBABLE</h3><h1>{gan.upper()}</h1><h4>CONFIANZA: {max(p)*100:.1f}%</h4></div>', unsafe_allow_html=True)

        st.subheader("📈 Pronósticos Detallados")
        r1, r2, r3 = st.columns(3)
        r1.metric("Goles Totales", f"{g:.1f}")
        r2.metric("Córners Est.", f"{cor:.1f}")
        r3.metric("Tarjetas Est.", f"{tar:.1f}")

        st.subheader("🎯 Probabilidades 1X2")
        p1, px, p2 = st.columns(3)
        p1.metric(f"Gana {t1}", f"{p[1]*100:.1f}%")
        px.metric("Empate", f"{p[0]*100:.1f}%")
        p2.metric(f"Gana {t2}", f"{p[2]*100:.1f}%")
else:
    st.error("Sube los archivos CSV a GitHub")

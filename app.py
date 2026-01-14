import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración 100% Fútbol
st.set_page_config(page_title="IA ELITE BETTING - STADIUM", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1522778119026-d647f0596c20?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .main { 
        background-color: rgba(10, 15, 30, 0.88); 
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    /* Estilo de los números de cuotas para que brillen */
    input[type="number"] {
        color: #00ff00 !important; /* Verde Neón para las cuotas */
        background-color: #0f172a !important;
        font-weight: 800 !important;
        font-size: 22px !important;
        border: 2px solid #38bdf8 !important;
    }

    .winner-card {
        background: linear-gradient(135deg, rgba(29, 78, 216, 0.95) 0%, rgba(30, 64, 175, 0.95) 100%);
        padding: 30px; border-radius: 25px; text-align: center; margin-bottom: 25px;
        border: 2px solid #60a5fa; box-shadow: 0 0 20px rgba(59, 130, 246, 0.5);
    }

    .metric-box {
        background: rgba(30, 41, 59, 0.95);
        padding: 20px; border-radius: 15px; border: 1px solid #38bdf8;
        text-align: center; margin-bottom: 15px;
    }

    .rec-text { font-size: 18px; font-weight: 900; margin-top: 10px; border-radius: 5px; padding: 5px; }
    .rec-over { color: #22c55e; border: 1px solid #22c55e; }
    .rec-under { color: #ef4444; border: 1px solid #ef4444; }

    .stButton>button { 
        background: #2563eb; color: white; font-weight: bold; width: 100%; 
        border-radius: 12px; height: 3.5em; font-size: 22px; border: 2px solid #38bdf8;
    }
    
    label { color: #ffffff !important; font-size: 18px !important; text-shadow: 1px 1px 2px black; }
    h1, h2, h3 { text-shadow: 2px 2px 4px rgba(0,0,0,0.7); }
    </style>
    """, unsafe_allow_html=True)

# Lógica de carga (Igual que antes pero optimizada)
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

    st.markdown("<h1 style='text-align: center; color: white;'>🏟️ TERMINAL DE INTELIGENCIA DEPORTIVA</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("EQUIPO LOCAL", teams)
    t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
    
    st.markdown("### 🏦 CUOTAS (€)")
    cq1, cq2, cq3 = st.columns(3)
    q1 = cq1.number_input("Local", value=2.00, format="%.2f")
    qx = cq2.number_input("Empate", value=3.20, format="%.2f")
    q2 = cq3.number_input("Visita", value=3.50, format="%.2f")

    if st.button("🚀 CALCULAR PRONÓSTICO ELITE"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p, g, cor, tar = m_r.predict_proba(v)[0], m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        gan = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] and p[2] > p[0] else "Empate")

        st.markdown(f'<div class="winner-card"><h1 style="font-size: 65px; margin:0;">{gan.upper()}</h1><p style="font-size:24px;">Confianza del Sistema: {max(p)*100:.1f}%</p></div>', unsafe_allow_html=True)

        # Resultados con texto explicativo
        r1, r2, r3 = st.columns(3)
        with r1:
            txt = "OVER 2.5 GOLES" if g > 2.5 else "UNDER 2.5 GOLES"
            cl = "rec-over" if g > 2.5 else "rec-under"
            st.markdown(f'<div class="metric-box"><div style="color:#bae6fd">Goles Totales</div><div style="font-size:35px; font-weight:800;">{g:.1f}</div><div class="rec-text {cl}">{txt}</div></div>', unsafe_allow_html=True)
        with r2:
            txt = "+9.5 CÓRNERS" if cor > 9.5 else "-9.5 CÓRNERS"
            cl = "rec-over" if cor > 9.5 else "rec-under"
            st.markdown(f'<div class="metric-box"><div style="color:#bae6fd">Córners</div><div style="font-size:35px; font-weight:800;">{cor:.1f}</div><div class="rec-text {cl}">{txt}</div></div>', unsafe_allow_html=True)
        with r3:
            txt = "+4.5 TARJETAS" if tar > 4.5 else "-4.5 TARJETAS"
            cl = "rec-over" if tar > 4.5 else "rec-under"
            st.markdown(f'<div class="metric-box"><div style="color:#bae6fd">Tarjetas</div><div style="font-size:35px; font-weight:800;">{tar:.1f}</div><div class="rec-text {cl}">{txt}</div></div>', unsafe_allow_html=True)

        # Probabilidades Finales
        st.markdown("### 🎯 PROBABILIDADES REALES")
        p1, px, p2 = st.columns(3)
        p1.markdown(f'<div class="metric-box" style="border-color:#22c55e">Gana {t1}<br><span style="font-size:25px; color:#22c55e;">{p[1]*100:.1f}%</span></div>', unsafe_allow_html=True)
        px.markdown(f'<div class="metric-box" style="border-color:#facc15">Empate<br><span style="font-size:25px; color:#facc15;">{p[0]*100:.1f}%</span></div>', unsafe_allow_html=True)
        p2.markdown(f'<div class="metric-box" style="border-color:#ef4444">Gana {t2}<br><span style="font-size:25px; color:#ef4444;">{p[2]*100:.1f}%</span></div>', unsafe_allow_html=True)
else:
    st.error("⚠️ Sube los archivos CSV a tu repositorio de GitHub.")

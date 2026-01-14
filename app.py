import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración Estilo Estadio Elite
st.set_page_config(page_title="IA ELITE BETTING - VIP", layout="wide")

st.markdown("""
    <style>
    /* FONDO DE ESTADIO NOCTURNO OSCURO */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=2070&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* CAPA OSCURA PARA QUE TODO RESALTE */
    .main { 
        background-color: rgba(0, 5, 15, 0.85); 
        border-radius: 25px;
        padding: 40px;
        margin: 10px;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }

    /* TÍTULOS EN BLANCO PURO CON SOMBRA */
    h1, h2, h3, p, label { 
        color: white !important; 
        text-shadow: 2px 2px 8px rgba(0,0,0,1) !important;
    }

    /* CUOTAS EN VERDE ELÉCTRICO */
    input[type="number"] {
        color: #22c55e !important;
        background-color: #0f172a !important;
        font-weight: 900 !important;
        font-size: 24px !important;
        border: 2px solid #3b82f6 !important;
        border-radius: 12px !important;
    }

    /* TARJETA DE GANADOR (ESTILO PANEL VIP) */
    .winner-card {
        background: linear-gradient(135deg, rgba(30, 64, 175, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        padding: 40px; border-radius: 30px; text-align: center; 
        margin-bottom: 30px; border: 2px solid #38bdf8;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.8);
    }

    /* CAJAS DE MÉTRICAS (Goles, Corners, Tarjetas) */
    .metric-box {
        background: rgba(15, 23, 42, 0.9);
        padding: 25px; border-radius: 20px; 
        border: 1px solid #38bdf8;
        text-align: center; margin-bottom: 20px;
    }

    .rec-text { font-size: 20px; font-weight: 900; margin-top: 15px; padding: 10px; border-radius: 10px; }
    .rec-over { color: #22c55e; background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; }
    .rec-under { color: #ef4444; background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; }

    /* BOTÓN DE ACCIÓN */
    .stButton>button { 
        background: linear-gradient(90deg, #1d4ed8, #3b82f6);
        color: white !important; font-weight: bold; width: 100%; 
        border-radius: 15px; height: 4em; font-size: 24px; border: none;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Lógica del Sistema
ligas = {"🇪🇸 La Liga": "SP1.csv", "🇬🇧 Premier": "E0.csv", "🇮🇹 Serie A": "I1.csv", "🇩🇪 Bundesliga": "D1.csv"}
sel = st.sidebar.selectbox("📂 SELECCIONAR LIGA", list(ligas.keys()))

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

    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_t = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    st.markdown("<h1 style='text-align: center;'>🏟️ TERMINAL DE INTELIGENCIA DEPORTIVA</h1>", unsafe_allow_html=True)
    
    col_t1, col_t2 = st.columns(2)
    t1 = col_t1.selectbox("🏠 EQUIPO LOCAL", teams)
    t2 = col_t2.selectbox("✈️ EQUIPO VISITANTE", teams, index=1)
    
    st.markdown("### 🏦 CUOTAS DE MERCADO (€)")
    cq1, cq2, cq3 = st.columns(3)
    q1 = cq1.number_input("Cuota Local", value=2.00, format="%.2f")
    qx = cq2.number_input("Cuota Empate", value=3.20, format="%.2f")
    q2 = cq3.number_input("Cuota Visita", value=3.50, format="%.2f")

    if st.button("🔥 GENERAR PRONÓSTICO MAESTRO"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p, g, cor, tar = m_r.predict_proba(v)[0], m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        gan = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] and p[2] > p[0] else "Empate")

        st.markdown(f"""
            <div class="winner-card">
                <h3 style="margin:0; color: #38bdf8;">PICK RECOMENDADO</h3>
                <h1 style="font-size: 75px; margin:10px 0;">{gan.upper()}</h1>
                <p style="font-size:26px; font-weight:bold;">Probabilidad de Éxito: {max(p)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        with r1:
            txt = "OVER 2.5 GOLES" if g > 2.5 else "UNDER 2.5 GOLES"
            cl = "rec-over" if g > 2.5 else "rec-under"
            st.markdown(f'<div class="metric-box"><div style="color:#bae6fd">Goles Totales</div><div style="font-size:40px; font-weight:900;">{g:.1f}</div><div class="rec-text {cl}">{txt}</div></div>', unsafe_allow_html=True)
        with r2:
            txt = "+9.5 CÓRNERS" if cor > 9.5 else "-9.5 CÓRNERS"
            cl = "rec-over" if cor > 9.5 else "rec-under"
            st.markdown(f'<div class="metric-box"><div style="color:#bae6fd">Córners Est.</div><div style="font-size:40px; font-weight:900;">{cor:.1f}</div><div class="rec-text {cl}">{txt}</div></div>', unsafe_allow_html=True)
        with r3:
            txt = "+4.5 TARJETAS" if tar > 4.5 else "-4.5 TARJETAS"
            cl = "rec-over" if tar > 4.5 else "rec-under"
            st.markdown(f'<div class="metric-box"><div style="color:#bae6fd">Tarjetas Est.</div><div style="font-size:40px; font-weight:900;">{tar:.1f}</div><div class="rec-text {cl}">{txt}</div></div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración Final Pro - Con Fondo Personalizado
st.set_page_config(page_title="AI ELITE BETTING", layout="wide")

# CSS para el fondo de imagen y ajustes de diseño
st.markdown("""
    <style>
    /* ------------------- FONDO DE IMAGEN ------------------- */
    .stApp {
        background-image: url("https://images.pexels.com/photos/3389536/pexels-photo-3389536.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2"); /* <--- ¡PEGA AQUÍ LA URL DE TU IMAGEN! */
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed; /* Mantiene el fondo fijo al hacer scroll */
    }
    .main { 
        background-color: rgba(15, 23, 42, 0.85); /* Fondo oscuro semitransparente para ver la imagen */
        border-radius: 15px; /* Bordes redondeados para el contenido principal */
        padding: 20px;
        margin: 20px;
    }
    .stSidebar {
        background-color: rgba(15, 23, 42, 0.95); /* Sidebar más oscuro y opaco */
        border-radius: 15px;
    }
    .stMetric { 
        background-color: #1e293b !important; 
        border: 1px solid #3b82f6 !important; 
        border-radius: 10px; padding: 10px !important; 
    }
    div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: bold; }
    
    /* Input de cuotas (fondo y texto) */
    input[type="number"] {
        color: white !important;
        background-color: #1e293b !important;
        font-weight: bold !important;
        font-size: 18px !important;
        border: 1px solid #3b82f6 !important; /* Borde para que destaque */
        border-radius: 8px;
    }
    label { color: #bae6fd !important; font-weight: bold !important; }

    .winner-card {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.9) 0%, rgba(6, 182, 212, 0.9) 100%);
        padding: 25px; border-radius: 20px; text-align: center; margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 2px solid #67e8f9;
    }
    .metric-box {
        background: rgba(30, 41, 59, 0.9); /* Fondo semitransparente */
        padding: 20px; border-radius: 15px; border: 1px solid #38bdf8;
        text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.3); margin-bottom: 10px;
    }
    .metric-title { color: #bae6fd; font-size: 16px; font-weight: 600; }
    .metric-value { color: white; font-size: 32px; font-weight: 800; }
    
    .rec-text { font-size: 18px; font-weight: bold; margin-top: 10px; padding: 5px; border-radius: 8px; }
    .rec-over { color: #4ade80; background: rgba(74, 222, 128, 0.2); }
    .rec-under { color: #facc15; background: rgba(250, 204, 21, 0.2); }

    .stButton>button { 
        background: linear-gradient(90deg, #2563eb, #06b6d4); 
        color: white; font-weight: bold; width: 100%; border-radius: 12px; height: 3.5em; border: none; font-size: 20px;
    }
    h1, h2, h3, h4, h5, h6 { color: white !important; } /* Asegura que todos los títulos sean blancos */
    </style>
    """, unsafe_allow_html=True)

ligas = {"🇪🇸 La Liga": "SP1.csv", "🇬🇧 Premier": "E0.csv", "🇮🇹 Serie A": "I1.csv", "🇩🇪 Bundesliga": "D1.csv"}
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2643/2643501.png", width=100)
st.sidebar.title("PANEL DE CONTROL")
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
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H', 'A', 'B365H', 'B365D', 'B365A']]

    # Modelos
    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_t = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    st.markdown(f"<h1 style='text-align: center;'>⚽ TERMINAL INTELIGENTE: {sel.upper()}</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("EQUIPO LOCAL", teams)
    t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
    
    st.markdown("### 🏦 CUOTAS DEL MERCADO (€)")
    cq1, cq2, cq3 = st.columns(3)
    q1 = cq1.number_input("Cuota Local (€)", value=2.00, step=0.01, format="%.2f")
    qx = cq2.number_input("Cuota Empate (€)", value=3.20, step=0.01, format="%.2f")
    q2 = cq3.number_input("Cuota Visita (€)", value=3.50, step=0.01, format="%.2f")

    if st.button("🔥 ANALIZAR PARTIDO AHORA"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p = m_r.predict_proba(v)[0]
        g, cor, tar = m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        
        gan = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] and p[2] > p[0] else "Empate")

        st.markdown(f"""
            <div class="winner-card">
                <h2 style="color: white; font-size: 60px; margin: 0;">{gan.upper()}</h2>
                <p style="font-size: 20px; color: #e0f2fe;">Probabilidad de Éxito: {max(p)*100:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)

        # MÉTRICAS DETALLADAS
        r1, r2, r3 = st.columns(3)
        with r1:
            txt = "🔥 OVER 2.5" if g > 2.5 else "🧊 UNDER 2.5"
            st.markdown(f'<div class="metric-box"><div class="metric-title">Goles Est.</div><div class="metric-value">{g:.1f}</div><div class="rec-text {"rec-over" if g > 2.5 else "rec-under"}">{txt}</div></div>', unsafe_allow_html=True)
        with r2:
            txt = "🚩 +9.5 CÓRNERS" if cor > 9.5 else "🚩 -9.5 CÓRNERS"
            st.markdown(f'<div class="metric-box"><div class="metric-title">Córners Est.</div><div class="metric-value">{cor:.1f}</div><div class="rec-text {"rec-over" if cor > 9.5 else "rec-under"}">{txt}</div></div>', unsafe_allow_html=True)
        with r3:
            txt = "🟨 +4.5 TARJETAS" if tar > 4.5 else "🟩 -4.5 TARJETAS"
            st.markdown(f'<div class="metric-box"><div class="metric-title">Tarjetas Est.</div><div class="metric-value">{tar:.1f}</div><div class="rec-text {"rec-over" if tar > 4.5 else "rec-under"}">{txt}</div></div>', unsafe_allow_html=True)

        # PROBABILIDADES 1X2 CON COLORES
        st.markdown("### 🎯 PROBABILIDADES 1X2")
        p1, px, p2 = st.columns(3)
        p1.markdown(f'<div class="metric-box" style="border-color:#4ade80"><div class="metric-title">Gana {t1}</div><div class="metric-value" style="color:#4ade80">{p[1]*100:.1f}%</div></div>', unsafe_allow_html=True)
        px.markdown(f'<div class="metric-box" style="border-color:#facc15"><div class="metric-title">Empate (X)</div><div class="metric-value" style="color:#facc15">{p[0]*100:.1f}%</div></div>', unsafe_allow_html=True)
        p2.markdown(f'<div class="metric-box" style="border-color:#ef4444"><div class="metric-title">Gana {t2}</div><div class="metric-value" style="color:#ef4444">{p[2]*100:.1f}%</div></div>', unsafe_allow_html=True)

else:
    st.error("Sube los archivos CSV")

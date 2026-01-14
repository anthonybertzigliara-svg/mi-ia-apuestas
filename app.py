import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración Pro V5
st.set_page_config(page_title="AI ELITE BET V5", layout="wide")

# CSS de Alto Contraste y Diseño
st.markdown("""
    <style>
    .main { background-color: #0f172a; color: white; }
    
    /* Estilo de la tarjeta del ganador principal */
    .winner-card {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        padding: 25px; border-radius: 20px; text-align: center; margin: 20px 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5); border: 2px solid #67e8f9;
    }
    
    /* Nuevas cajas de métricas vibrantes */
    .metric-box {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        padding: 20px; border-radius: 15px; border: 1px solid #38bdf8;
        text-align: center; box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .metric-title { color: #bae6fd; font-size: 16px; margin-bottom: 5px; font-weight: 600; }
    .metric-value { color: white; font-size: 32px; font-weight: 800; margin: 0; }
    
    /* Textos de recomendación (Over/Under) */
    .rec-text { font-size: 18px; font-weight: bold; margin-top: 10px; padding: 5px; border-radius: 8px; }
    .rec-over { color: #4ade80; background: rgba(74, 222, 128, 0.1); } /* Verde */
    .rec-under { color: #facc15; background: rgba(250, 204, 21, 0.1); } /* Amarillo/Naranja */

    /* Botón */
    .stButton>button { 
        background: linear-gradient(90deg, #2563eb, #06b6d4); 
        color: white; font-weight: bold; width: 100%; border-radius: 12px; height: 3.5em; border: none; font-size: 18px;
    }
    /* Ajuste para los inputs numéricos en modo oscuro */
    .stNumberInput input { color: white !important; }
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

    # Modelos Rápidos
    m_r = RandomForestClassifier(n_estimators=100).fit(X.values, df['T'])
    m_g = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_c = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])
    m_t = RandomForestRegressor(n_estimators=100).fit(X.values, df['HY'] + df['AY'])

    st.markdown(f"<h1 style='text-align: center;'>⚽ TERMINAL INTELIGENTE: {sel.upper()}</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("EQUIPO LOCAL", teams)
    t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
    
    st.markdown("### 🏦 CUOTAS DEL MERCADO")
    cq1, cq2, cq3 = st.columns(3)
    q1 = cq1.number_input("Cuota Local (1)", value=2.00, step=0.05)
    qx = cq2.number_input("Cuota Empate (X)", value=3.20, step=0.05)
    q2 = cq3.number_input("Cuota Visita (2)", value=3.50, step=0.05)

    if st.button("🔥 ANALIZAR PARTIDO AHORA"):
        v = [[le.transform([t1])[0], le.transform([t2])[0], q1, qx, q2]]
        p = m_r.predict_proba(v)[0]
        g, cor, tar = m_g.predict(v)[0], m_c.predict(v)[0], m_t.predict(v)[0]
        
        gan = t1 if p[1] > p[2] and p[1] > p[0] else (t2 if p[2] > p[1] and p[2] > p[0] else "Empate")
        conf = max(p)*100

        # TARJETA GANADOR
        st.markdown(f"""
            <div class="winner-card">
                <h3 style="color: #bae6fd; margin:0;">PRONÓSTICO PRINCIPAL</h3>
                <h1 style="color: white; font-size: 60px; margin: 10px 0; text-shadow: 0 0 10px rgba(255,255,255,0.3);">{gan.upper()}</h1>
                <h4 style="color: white; background: rgba(0,0,0,0.2); display: inline-block; padding: 5px 15px; border-radius: 10px;">CONFIANZA IA: {conf:.1f}%</h4>
            </div>
        """, unsafe_allow_html=True)

        st.subheader("📊 Pronósticos Detallados y Recomendaciones")
        
        # Lógica de Recomendaciones
        rec_g = "🔥 OVER 2.5 GOLES" if g > 2.5 else "🧊 UNDER 2.5 GOLES"
        css_g = "rec-over" if g > 2.5 else "rec-under"
        
        rec_c = "🚩 +9.5 CÓRNERS" if cor > 9.5 else "🚩 MENOS DE 9.5 CÓRNERS"
        css_c = "rec-over" if cor > 9.5 else "rec-under"
        
        rec_t = "🟨 PARTIDO CALIENTE (+5.5)" if tar > 5.5 else "🟩 PARTIDO LIMPIO (-5.5)"
        css_t = "rec-over" if tar > 5.5 else "rec-under"

        # Cajas de Métricas Personalizadas
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Goles Esperados</div>
                    <div class="metric-value">{g:.1f}</div>
                    <div class="rec-text {css_g}">{rec_g}</div>
                </div>
            """, unsafe_allow_html=True)
        with r2:
             st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Córners Esperados</div>
                    <div class="metric-value">{cor:.1f}</div>
                    <div class="rec-text {css_c}">{rec_c}</div>
                </div>
            """, unsafe_allow_html=True)
        with r3:
             st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-title">Tarjetas Esperadas</div>
                    <div class="metric-value">{tar:.1f}</div>
                    <div class="rec-text {css_t}">{rec_t}</div>
                </div>
            """, unsafe_allow_html=True)

        st.subheader("🎯 Probabilidades Exactas (1X2)")
        p1_col, px_col, p2_col = st.columns(3)
        with p1_col:
             st.markdown(f"""<div class="metric-box" style="border-color: #22c55e;"><div class="metric-title">Victoria {t1}</div><div class="metric-value" style="color:#22c55e;">{p[1]*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with px_col:
             st.markdown(f"""<div class="metric-box" style="border-color: #facc15;"><div class="metric-title">Empate</div><div class="metric-value" style="color:#facc15;">{p[0]*100:.1f}%</div></div>""", unsafe_allow_html=True)
        with p2_col:
             st.markdown(f"""<div class="metric-box" style="border-color: #ef4444;"><div class="metric-title">Victoria {t2}</div><div class="metric-value" style="color:#ef4444;">{p[2]*100:.1f}%</div></div>""", unsafe_allow_html=True)

else:
    st.error("⚠️ Error: Por favor sube los archivos CSV a GitHub.")

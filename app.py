import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# CONFIGURACIÓN DE IMPACTO VISUAL
st.set_page_config(page_title="ULTRA IA BET", layout="wide")

st.markdown("""
    <style>
    .main { background: #0f172a; color: white; }
    .stMetric { background: #1e293b; border: 2px solid #3b82f6; border-radius: 15px; padding: 15px; }
    .winner-card { 
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
        padding: 30px; border-radius: 20px; text-align: center; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.5); margin: 20px 0;
    }
    .stButton>button { 
        background: #3b82f6; color: white; border-radius: 12px; 
        font-weight: bold; height: 3.5em; width: 100%; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

# LÓGICA DE LIGAS
ligas = {"🇪🇸 La Liga": "SP1.csv", "🇬🇧 Premier": "E0.csv", "🇮🇹 Serie A": "I1.csv", "🇩🇪 Bundesliga": "D1.csv"}
sel_liga = st.sidebar.selectbox("🏆 SELECCIONA TU LIGA", list(ligas.keys()))

@st.cache_data
def load(file):
    try:
        df = pd.read_csv(file)
        return df[['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR', 'FTHG', 'FTAG']].dropna()
    except: return None

df = load(ligas[sel_liga])

if df is not None:
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)
    
    # Entrenar IA rápido
    df['H'], df['A'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['T'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    
    model = RandomForestClassifier(n_estimators=100).fit(df[['H', 'A', 'B365H', 'B365D', 'B365A']].values, df['T'])

    st.title(f"⚡ TERMINAL IA: {sel_liga}")
    
    col1, col2 = st.columns(2)
    with col1: t_l = st.selectbox("🏠 LOCAL", teams)
    with col2: t_v = st.selectbox("✈️ VISITANTE", teams, index=1)
    
    st.markdown("### 🏦 CUOTAS ACTUALES")
    c1, c2, c3 = st.columns(3)
    q_l = c1.number_input("Cuota 1", value=2.0)
    q_e = c2.number_input("Cuota X", value=3.2)
    q_v = c3.number_input("Cuota 2", value=3.5)

    if st.button("🚀 REALIZAR PREDICCIÓN MAESTRA"):
        idx_l, idx_v = le.transform([t_l])[0], le.transform([t_v])[0]
        probs = model.predict_proba([[idx_l, idx_v, q_l, q_e, q_v]])[0] # [E, L, V]
        
        # GANADOR
        p_e, p_l, p_v = probs[0]*100, probs[1]*100, probs[2]*100
        win = t_l if p_l > p_v and p_l > p_e else (t_v if p_v > p_l and p_v > p_e else "Empate")
        
        st.markdown(f"""
            <div class="winner-card">
                <h3 style="color: #bfdbfe; margin:0;">GANADOR PROBABLE</h3>
                <h1 style="color: white; font-size: 50px; margin:10px 0;">{win.upper()}</h1>
                <h4 style="color: #60a5fa;">CONFIANZA: {max(probs)*100:.1f}%</h4>
            </div>
        """, unsafe_allow_html=True)
        
        res1, res2, res3 = st.columns(3)
        res1.metric(f"Gana {t_l}", f"{p_l:.1f}%")
        res2.metric("Empate", f"{p_e:.1f}%")
        res3.metric(f"Gana {t_v}", f"{p_v:.1f}%")
else:
    st.error("Archivo no encontrado")

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 1. ESTILO ELITE
st.set_page_config(page_title="WORLD ELITE BETTING AI", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, rgba(5,5,10,0.98) 0%, rgba(10,25,50,0.95) 100%), 
                    url("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?q=80&w=2000&auto=format&fit=crop");
        background-size: cover;
    }
    .status-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid #00f2ff33;
        padding: 15px;
        border-radius: 12px;
        text-align: center;
    }
    .bet-header {
        background: linear-gradient(90deg, #1e293b, #0f172a);
        color: #00f2ff !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 800;
        margin: 20px 0 10px 0;
        border-left: 5px solid #00f2ff;
    }
    .racha-v { color: #00ff88; font-weight: bold; margin: 0 2px; }
    .racha-e { color: #ffcc00; font-weight: bold; margin: 0 2px; }
    .racha-d { color: #ff4b4b; font-weight: bold; margin: 0 2px; }
    
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #00ff88 0%, #00cc6a 100%) !important;
        color: black !important;
        font-size: 1.2rem !important;
        font-weight: 900 !important;
        padding: 1rem !important;
        border-radius: 15px !important;
        width: 100%;
    }
    .tag-plus { background: #00ff88; color: black; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    .tag-minus { background: #ff4b4b; color: white; padding: 4px 10px; border-radius: 5px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# 2. CARGA DE DATOS
ligas = {
    "🇪🇸 LA LIGA": "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
    "🇬🇧 PREMIER": "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
    "🇮🇹 SERIE A": "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
    "🇩🇪 BUNDESLIGA": "https://www.football-data.co.uk/mmz4281/2526/D1.csv"
}

@st.cache_data(ttl=3600)
def load_data(url):
    try:
        return pd.read_csv(url).dropna(subset=['HomeTeam', 'AwayTeam'])
    except: return None

def get_streak(df, team):
    recent = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].dropna(subset=['FTR']).tail(5)
    streak = []
    for _, row in recent.iterrows():
        res = 'V' if (row['HomeTeam']==team and row['FTR']=='H') or (row['AwayTeam']==team and row['FTR']=='A') else ('E' if row['FTR']=='D' else 'D')
        color = "racha-v" if res=='V' else ("racha-e" if res=='E' else "racha-d")
        streak.append(f'<span class="{color}">{res}</span>')
    return "".join(streak)

if 'quiniela' not in st.session_state: st.session_state.quiniela = []

# BARRA LATERAL CON CALENDARIO
with st.sidebar:
    st.title("⚽ MENU")
    sel_liga = st.selectbox("LIGA", list(ligas.keys()))
    df_raw = load_data(ligas[sel_liga])
    
    if df_raw is not None:
        st.markdown("---")
        st.markdown("### 📅 PRÓXIMOS PARTIDOS")
        # Mostramos los partidos que NO tienen resultado aún (están en el CSV pero FTR es NaN)
        proximos = df_raw[df_raw['FTR'].isna()][['Date', 'HomeTeam', 'AwayTeam']].head(10)
        if proximos.empty:
            st.write("No hay partidos programados en el registro actual.")
        for _, row in proximos.iterrows():
            st.write(f"**{row['Date']}**")
            st.caption(f"{row['HomeTeam']} vs {row['AwayTeam']}")
            st.markdown("---")

# 3. INTERFAZ PRINCIPAL
if df_raw is not None:
    # Filtramos solo partidos jugados para entrenar la IA
    df = df_raw.dropna(subset=['FTR', 'B365H'])
    le = LabelEncoder()
    teams = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(teams)

    st.markdown("<h1 style='text-align: center; color: #00f2ff;'>AI ELITE TERMINAL</h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    t1 = c1.selectbox("EQUIPO LOCAL", teams)
    c1.markdown(f"Racha: {get_streak(df, t1)}", unsafe_allow_html=True)
    
    t2 = c2.selectbox("EQUIPO VISITANTE", teams, index=1)
    c2.markdown(f"Racha: {get_streak(df, t2)}", unsafe_allow_html=True)

    st.markdown("<div class='bet-header'>CUOTAS 1X2</div>", unsafe_allow_html=True)
    q1, qx, q2 = st.columns(3)
    v1 = q1.number_input(f"Cuota {t1}", value=2.0)
    vx = qx.number_input("Cuota X", value=3.2)
    v2 = q2.number_input(f"Cuota {t2}", value=3.5)

    # Entrenamiento IA
    df['H_c'], df['A_c'] = le.transform(df['HomeTeam']), le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    X = df[['H_c', 'A_c', 'B365H', 'B365D', 'B365A']]
    m_win = RandomForestClassifier(n_estimators=100).fit(X.values, df['Target'])
    m_goals = RandomForestRegressor(n_estimators=100).fit(X.values, df['FTHG'] + df['FTAG'])
    m_corn = RandomForestRegressor(n_estimators=100).fit(X.values, df['HC'] + df['AC'])

    if st.button("🚀 ANALIZAR Y GUARDAR"):
        v_in = [[le.transform([t1])[0], le.transform([t2])[0], v1, vx, v2]]
        probs = m_win.predict_proba(v_in)[0]
        g, c = m_goals.predict(v_in)[0], m_corn.predict(v_in)[0]
        pick = t1 if m_win.predict(v_in)[0] == 1 else (t2 if m_win.predict(v_in)[0] == 2 else "Empate")

        st.session_state.quiniela.append({
            "PARTIDO": f"{t1}-{t2}", "PICK": pick, 
            "GOLES": f"{'+2.5' if g > 2.5 else '-2.5'}", "CONF.": f"{max(probs)*100:.0f}%"
        })

        st.markdown("<div class='bet-header'>RESULTADOS</div>", unsafe_allow_html=True)
        res1, res2, res3 = st.columns(3)
        res1.markdown(f'<div class="status-card"><p>GOLES</p><h2>{g:.1f}</h2><span class="{"tag-plus" if g > 2.5 else "tag-minus"}">{" + 2.5" if g > 2.5 else " - 2.5"}</span></div>', unsafe_allow_html=True)
        res2.markdown(f'<div class="status-card"><p>CÓRNERS</p><h2>{c:.0f}</h2><span class="{"tag-plus" if c > 9.5 else "tag-minus"}">{" + 9.5" if c > 9.5 else " - 9.5"}</span></div>', unsafe_allow_html=True)
        res3.markdown(f'<div class="status-card"><p>PICK</p><h2>{pick}</h2><span class="tag-plus" style="background:#00f2ff; color:black">{max(probs)*100:.0f}%</span></div>', unsafe_allow_html=True)

    if st.session_state.quiniela:
        st.markdown("<div class='bet-header'>📋 MI QUINIELA</div>", unsafe_allow_html=True)
        st.table(pd.DataFrame(st.session_state.quiniela))
        if st.button("🗑️ LIMPIAR"):
            st.session_state.quiniela = []
            st.rerun()

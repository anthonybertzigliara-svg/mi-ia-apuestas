import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="IA Pronosticador Pro", layout="wide")

st.title("🏆 IA Predictor Multiliga")
st.sidebar.header("Configuración de Liga")

ligas = {
    "España (La Liga)": "SP1.csv",
    "Inglaterra (Premier)": "E0.csv",
    "Italia (Serie A)": "I1.csv",
    "Alemania (Bundesliga)": "D1.csv"
}

seleccion_liga = st.sidebar.selectbox("Selecciona la Liga", list(ligas.keys()))
archivo_liga = ligas[seleccion_liga]

@st.cache_data
def cargar_datos(archivo):
    try:
        df = pd.read_csv(archivo)
        return df
    except:
        return None

df = cargar_datos(archivo_liga)

if df is not None:
    columnas_clave = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR']
    df = df[columnas_clave].dropna()

    le = LabelEncoder()
    todos_equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(todos_equipos)
    
    df['H_ID'] = le.transform(df['HomeTeam'])
    df['A_ID'] = le.transform(df['AwayTeam'])
    df['Target'] = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))

    X = df[['H_ID', 'A_ID', 'B365H', 'B365D', 'B365A']]
    y = df['Target']
    
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X.values, y)

    st.info(f"📍 Analizando: {seleccion_liga}")
    col1, col2 = st.columns(2)
    with col1:
        local = st.selectbox("Equipo Local", todos_equipos)
    with col2:
        visita = st.selectbox("Equipo Visitante", todos_equipos)

    st.subheader("📊 Cuotas Actuales")
    c1, c2, c3 = st.columns(3)
    ch = c1.number_input("Cuota Local", value=2.0, step=0.01)
    cd = c2.number_input("Cuota Empate", value=3.20, step=0.01)
    ca = c3.number_input("Cuota Visita", value=3.50, step=0.01)

    if st.button("🚀 CALCULAR"):
        id_l = le.transform([local])[0]
        id_v = le.transform([visita])[0]
        probabilidades = model.predict_proba([[id_l, id_v, ch, cd, ca]])[0]
        
        p_empate, p_local, p_visita = probabilidades[0]*100, probabilidades[1]*100, probabilidades[2]*100

        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Gana {local}", f"{p_local:.1f}%")
        m2.metric("Empate", f"{p_empate:.1f}%")
        m3.metric(f"Gana {visita}", f"{p_visita:.1f}%")

        if p_local > 60: st.success(f"✅ Sugerencia: Local")
        elif p_visita > 60: st.success(f"✅ Sugerencia: Visitante")
        else: st.info("💡 Partido parejo")
else:
    st.error(f"Falta el archivo {archivo_liga}")

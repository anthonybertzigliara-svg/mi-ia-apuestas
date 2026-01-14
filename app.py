import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="IA Pro Analytics", layout="wide")

st.title("🛡️ Sistema de Inteligencia Deportiva Profesional")
st.sidebar.header("Panel de Control")

ligas = {
    "España (La Liga)": "SP1.csv",
    "Inglaterra (Premier)": "E0.csv",
    "Italia (Serie A)": "I1.csv",
    "Alemania (Bundesliga)": "D1.csv"
}

seleccion_liga = st.sidebar.selectbox("Selecciona la Competición", list(ligas.keys()))

@st.cache_data
def cargar_datos_pro(archivo):
    try:
        df = pd.read_csv(archivo)
        # Columnas: HomeTeam, AwayTeam, B365H, B365D, B365A, FTR (Resultado), 
        # FTHG/FTAG (Goles), HC/AC (Corners), HY/AY (Amarillas), HST/AST (Tiros puerta)
        cols = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR', 
                'FTHG', 'FTAG', 'HC', 'AC', 'HY', 'AY', 'HST', 'AST']
        return df[cols].dropna()
    except:
        return None

df = cargar_datos_pro(ligas[seleccion_liga])

if df is not None:
    # --- PREPARACIÓN DE DATOS ---
    le = LabelEncoder()
    equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(equipos)
    
    df['H_ID'] = le.transform(df['HomeTeam'])
    df['A_ID'] = le.transform(df['AwayTeam'])
    
    # Entradas para la IA
    X = df[['H_ID', 'A_ID', 'B365H', 'B365D', 'B365A']]
    
    # Objetivos (Lo que queremos predecir)
    y_res = df['FTR'].apply(lambda x: 1 if x == 'H' else (2 if x == 'A' else 0))
    y_goles = df['FTHG'] + df['FTAG']
    y_corners = df['HC'] + df['AC']
    y_tarjetas = df['HY'] + df['AY']

    # --- ENTRENAMIENTO MULTI-MODELO ---
    with st.spinner('Entrenando motores estadísticos...'):
        model_res = RandomForestClassifier(n_estimators=200).fit(X.values, y_res)
        model_goles = RandomForestRegressor(n_estimators=200).fit(X.values, y_goles)
        model_corners = RandomForestRegressor(n_estimators=200).fit(X.values, y_corners)
        model_tarjetas = RandomForestRegressor(n_estimators=200).fit(X.values, y_tarjetas)

    # --- INTERFAZ ---
    c1, c2 = st.columns(2)
    with c1: local = st.selectbox("Equipo Local", equipos)
    with c2: visita = st.selectbox("Equipo Visitante", equipos)

    st.divider()
    
    # Cuotas
    st.subheader("🏦 Mercado de Apuestas")
    q1, q2, q3 = st.columns(3)
    ch = q1.number_input("Cuota Local", value=2.0)
    cd = q2.number_input("Cuota Empate", value=3.2)
    ca = q3.number_input("Cuota Visita", value=3.8)

    if st.button("🔍 GENERAR ANÁLISIS PROFESIONAL"):
        id_l, id_v = le.transform([local])[0], le.transform([visita])[0]
        input_data = [[id_l, id_v, ch, cd, ca]]

        # Predicciones
        prob_res = model_res.predict_proba(input_data)[0]
        pred_goles = model_corners.predict(input_data)[0] # Usando modelo regressor
        pred_corn = model_corners.predict(input_data)[0]
        pred_tarj = model_tarjetas.predict(input_data)[0]
        pred_goles_val = model_goles.predict(input_data)[0]

        # DISEÑO DE RESULTADOS
        st.subheader("📊 Pronósticos de Alta Probabilidad")
        
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Goles Totales", f"{pred_goles_val:.1f}")
        r2.metric("Córners Est.", f"{pred_corn:.1f}")
        r3.metric("Tarjetas Est.", f"{pred_tarj:.1f}")
        r4.metric("Prob. Victoria", f"{max(prob_res)*100:.1f}%")

        # Tarjetas de Análisis
        st.divider()
        a1, a2 = st.columns(2)
        
        with a1:
            st.markdown("### ⚽ Mercado de Goles")
            if pred_goles_val > 2.5:
                st.success("Sugerencia: **Over 2.5 Goles** (Partido Abierto)")
            else:
                st.warning("Sugerencia: **Under 2.5 Goles** (Partido Cerrado)")
        
        with a2:
            st.markdown("### 🚩 Mercado de Córners")
            if pred_corn > 9.5:
                st.success("Sugerencia: **Más de 9.5 Córners**")
            else:
                st.info("Sugerencia: **Menos de 9.5 Córners**")

else:
    st.error("Por favor, verifica que todos los archivos .csv estén en tu GitHub.")

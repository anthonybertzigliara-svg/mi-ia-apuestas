import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuración de la página
st.set_page_config(page_title="IA Pronósticos VIP", layout="centered")

st.title("⚽ IA de Predicciones Deportivas")
st.write("Selecciona los equipos y las cuotas actuales para obtener el pronóstico.")

# Función para cargar datos
@st.cache_data
def cargar_datos():
    # Intentamos cargar los datos reales. Si no está el archivo, usamos uno de ejemplo.
    try:
        df = pd.read_csv('SP1.csv')
        return df
    except:
        st.error("⚠️ No se encontró el archivo 'SP1.csv'. Por favor, súbelo a la carpeta del proyecto.")
        return None

df = cargar_datos()

if df is not None:
    # Preparar el modelo
    le = LabelEncoder()
    todos_los_equipos = sorted(pd.concat([df['HomeTeam'], df['AwayTeam']]).unique())
    le.fit(todos_los_equipos)
    
    df['HomeTeam_ID'] = le.transform(df['HomeTeam'])
    df['AwayTeam_ID'] = le.transform(df['AwayTeam'])
    df['Resultado'] = df['FTR'].apply(lambda x: 1 if x == 'H' else 0)
    
    X = df[['HomeTeam_ID', 'AwayTeam_ID', 'B365H', 'B365D', 'B365A']]
    y = df['Resultado']
    
    modelo = RandomForestClassifier(n_estimators=100)
    modelo.fit(X.values, y)

    # --- INTERFAZ DE USUARIO ---
    col1, col2 = st.columns(2)
    
    with col1:
        local = st.selectbox("Equipo Local", todos_los_equipos)
    with col2:
        visitante = st.selectbox("Equipo Visitante", todos_los_equipos)
    
    st.subheader("Cuotas de la Casa de Apuestas")
    c1, c2, c3 = st.columns(3)
    ch = c1.number_input("Cuota Local", value=2.0)
    cd = c2.number_input("Cuota Empate", value=3.0)
    ca = c3.number_input("Cuota Visita", value=4.0)

    if st.button("📊 ANALIZAR PARTIDO"):
        id_l = le.transform([local])[0]
        id_v = le.transform([visitante])[0]
        
        pred = modelo.predict_proba([[id_l, id_v, ch, cd, ca]])
        prob = pred[0][1] * 100
        
        st.divider()
        st.metric(label=f"Probabilidad de que gane {local}", value=f"{prob:.2f}%")
        
        if prob > 65:
            st.success("🔥 Pronóstico: ALTA PROBABILIDAD DE VICTORIA LOCAL")
        elif prob < 35:
            st.warning("🧊 Pronóstico: BAJA PROBABILIDAD (Considera Empate o Visita)")
        else:
            st.info("⚖️ Pronóstico: PARTIDO EQUILIBRADO")

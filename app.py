import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Configuración visual de la página
st.set_page_config(page_title="IA Pronosticador Pro", layout="wide")

st.title("🏆 IA Predictor Multiliga")
st.sidebar.header("Configuración de Liga")

# Diccionario que conecta el nombre de la liga con su archivo en GitHub
ligas = {
    "España (La Liga)": "SP1.csv",
    "Inglaterra (Premier)": "E0.csv",
    "Italia (Serie A)": "I1.csv",
    "Alemania (Bundesliga)": "D1.csv"
}

# Menú desplegable en la barra lateral
seleccion_liga = st.sidebar.selectbox("Selecciona la Liga que quieres analizar", list(ligas.keys()))
archivo_liga = ligas[seleccion_liga]

# Función para cargar los datos de forma eficiente
@st.cache_data
def cargar_datos(archivo):
    try:
        # Cargamos el CSV correspondiente
        df = pd.read_csv(archivo)
        return df
    except:
        return None

df = cargar_datos(archivo_liga)

if df is not None:
    # 1. LIMPIEZA: Seleccionamos solo las columnas que nos sirven para predecir
    # FTR es el resultado (H=Local, D=Empate, A=Visitante)
    columnas_clave = ['HomeTeam', 'AwayTeam', 'B365H', 'B365D', 'B365A', 'FTR']
    df = df[columnas_clave].dropna()

    # 2. PROCESAMIENTO: Convertimos nombres de equipos a números para la IA
    le = LabelEncoder()
    todos_equipos = sorted(pd.concat([df['HomeTeam'], df['Away

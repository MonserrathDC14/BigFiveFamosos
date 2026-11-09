import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración de la página
st.set_page_config(page_title="Predicción de Personalidad Big Five", layout="centered")

st.title("🎯 Clasificador de Personalidad - Big Five")
st.write("Introduce tus puntuaciones para descubrir a qué clúster perteneces, qué famosos comparten tu perfil y visualizar la comparativa en un gráfico de radar.")

# 1. Cargar el modelo, el escalador y los datos de famosos
@st.cache_resource
def cargar_recursos():
    modelo = joblib.load('modelo_kmeans_famosos.joblib')
    escalador = joblib.load('escalador_scaler.joblib')
    # Cargamos el dataset con los nombres de los famosos
    url_famosos = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df_famosos = pd.read_csv(url_famosos)
    # Aseguramos la predicción de clústeres originales en el df
    features = df_famosos[['op', 'co', 'ex', 'ag', 'ne']]
    features_esc = escalador.transform(features)
    df_famosos['cluster'] = modelo.predict(features_esc)
    return modelo, escalador, df_famosos

try:
    kmeans_cargado, escalador_cargado, df_famosos = cargar_recursos()
except Exception as e:
    st.error(f"Error al cargar los modelos o datos: {e}")
    st.stop()

# 2. Formulario de entrada de usuario
st.sidebar.header("Tus Puntuaciones")
op = st.sidebar.slider("Apertura (op)", 0.0, 100.0, 40.0, 0.1)
co = st.sidebar.slider("Conciencia (co)", 0.0, 100.0, 25.0, 0.1)
ex = st.sidebar.slider("Extroversión (ex)", 0.0, 100.0, 40.0, 0.1)
ag = st.sidebar.slider("Amabilidad (ag)", 0.0, 100.0, 25.0, 0.1)
ne = st.sidebar.slider("Neuroticismo (ne)", 0.0, 100.0, 10.0, 0.1)

# 3. Predicción e inferencia
mis_datos = pd.DataFrame([[op, co, ex, ag, ne]], columns=['op', 'co', 'ex', 'ag', 'ne'])
mis_datos_escalados = escalador_cargado.transform(mis_datos)
mi_cluster = kmeans_cargado.predict(mis_datos_escalados)[0]

# Mostrar Resultados Principales
st.subheader(f"✨ ¡Has sido asignado al Clúster {mi_cluster}! ")

# 4. Mostrar el Head 5 de famosos en ese clúster
st.markdown(f"### 👥 Famosos en el Clúster {mi_cluster}")
famosos_del_cluster = df_famosos[df_famosos['cluster'] == mi_cluster]
st.dataframe(famosos_del_cluster[['usuario', 'op', 'co', 'ex', 'ag', 'ne']].head(5), use_container_width=True)

# 5. Gráfico de radar comparativo (con las 5 variables: op, co, ex, ag, ne)
st.markdown("### 📊 Gráfico de Radar Comparativo")

variables = ['op', 'co', 'ex', 'ag', 'ne']

# Promedios de ese clúster específico
valores_cluster = famosos_del_cluster[variables].mean().tolist()
valores_cluster += valores_cluster[:1]  # Cerrar ciclo

# Valores ingresados por el usuario
valores_usuario = [op, co, ex, ag, ne]
valores_usuario += valores_usuario[:1]  # Cerrar ciclo

# Calcular los ángulos
angulos = np.linspace(0, 2 * np.pi, len(variables), endpoint=False).tolist()
angulos += angulos[:1]

# Crear gráfico polar
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

# Dibujar Clúster
ax.plot(angulos, valores_cluster, color='#1f77b4', linewidth=2, label=f'Promedio Clúster {mi_cluster}')
ax.fill(angulos, valores_cluster, color='#1f77b4', alpha=0.15)

# Dibujar Usuario
ax.plot(angulos, valores_usuario, color='#e377c2', linewidth=2.5, linestyle='--', marker='o', label='Mis Resultados')
ax.fill(angulos, valores_usuario, color='#e377c2', alpha=0.25)

# Estética del radar
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angulos[:-1], ['Apertura (op)', 'Conciencia (co)', 'Extroversión (ex)', 'Amabilidad (ag)', 'Neuroticismo (ne)'], fontsize=10)

plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig)

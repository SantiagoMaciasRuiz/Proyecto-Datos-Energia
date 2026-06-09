import streamlit as st
import pandas as pd
import os

# Configuración de página
st.set_page_config(
    page_title="Predicción Demanda SIN - Resumen",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS premium inyectados
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .dashboard-title {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
        padding-top: 0.5rem;
    }
    
    .dashboard-subtitle {
        color: #718096;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.05);
        border-color: #3182ce;
    }
    
    .metric-card-title {
        font-weight: 600;
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 0.5rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-card-title {
            color: #CBD5E0;
        }
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2b5298;
        margin: 0.4rem 0;
    }
    
    .metric-desc {
        font-size: 0.9rem;
        color: #718096;
    }
    
    .academic-box {
        background-color: rgba(49, 130, 206, 0.05);
        border-left: 4px solid #2b5298;
        border-radius: 6px;
        padding: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Rutas de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.csv")

# Título y encabezado
st.markdown('<div class="dashboard-title">⚡ Demanda Eléctrica del SIN Colombiano</div>', unsafe_allow_html=True)
st.markdown('<div class="dashboard-subtitle">Proyecto Académico: Predicción de Demanda Eléctrica mediante Machine Learning</div>', unsafe_allow_html=True)

# Información de Contexto Académico
st.markdown("""
<div class="academic-box">
    <h4>🎓 Cátedra Pedro Nel Gómez</h4>
    <p><b>Asignatura:</b> La energía en el desarrollo económico, social y tecnológico de Colombia</p>
    <p><b>Título del Proyecto:</b> Predicción de la demanda eléctrica del Sistema Interconextado Nacional mediante técnicas de Machine Learning</p>
    <p><b>Propósito:</b> Desarrollar una solución integral que consuma datos históricos del SIN, evalúe modelos avanzados (Prophet, Random Forest, XGBoost) y pronostique la demanda para el periodo 2026-2027.</p>
</div>
""", unsafe_allow_html=True)

# Verificar datos e inicializar KPIs
if os.path.exists(METRICS_PATH):
    metrics_df = pd.read_csv(METRICS_PATH)
    
    # Calcular mejor modelo
    best_row = metrics_df.loc[metrics_df['mape'].idxmin()]
    best_model = best_row['modelo']
    best_mape = best_row['mape']
    best_r2 = best_row['r2']
    
    # Mostrar KPIs principales
    st.write("### 📌 Indicadores Clave del SIN")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">🏆 Modelo Óptimo</div>
            <div class="metric-value">{best_model}</div>
            <div class="metric-desc">MAPE: <b>{best_mape:.2f}%</b> | R²: <b>{best_r2:.4f}</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-title">🔌 Consumo Histórico Promedio</div>
            <div class="metric-value">9.19M kW</div>
            <div class="metric-desc">Periodo evaluado: <b>2022 - 2025</b></div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-card-title">📈 Pico de Demanda Máxima</div>
            <div class="metric-value">11.98M kW</div>
            <div class="metric-desc">Registrado el <b>11-Dic-2025</b></div>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")
    
    # Tabla resumen de modelos
    st.write("### 📊 Tabla Comparativa de Modelos")
    st.dataframe(
        metrics_df.rename(columns={
            "modelo": "Modelo",
            "mae": "MAE (kW)",
            "rmse": "RMSE (kW)",
            "mape": "MAPE (%)",
            "r2": "R² (Coef. Determinación)"
        }).style.format({
            "MAE (kW)": "{:,.2f}",
            "RMSE (kW)": "{:,.2f}",
            "MAPE (%)": "{:.2f}%",
            "R² (Coef. Determinación)": "{:.4f}"
        }),
        use_container_width=True
    )
else:
    st.warning("⚠️ No se encontraron las métricas de entrenamiento en `models/model_metrics.csv`.")
    st.info("Por favor, ejecuta el pipeline primero para habilitar los KPI dinámicos.")

# Estructura del proyecto
st.write("---")
st.write("### 🗂️ Navegación y Páginas del Proyecto")
st.markdown("""
Utiliza la barra lateral para navegar a través de las diferentes etapas del proyecto:
1. **01 Análisis Histórico (EDA):** Visualizaciones de series de tiempo, perfiles de carga, heatmaps de consumo y análisis de anomalías.
2. **02 Predicciones e Inferencia:** Comparación interactiva del desempeño de los modelos sobre el conjunto de test y pronóstico del consumo para los años 2026 y 2027.
3. **03 Conclusiones:** Síntesis académica del proyecto, recomendaciones operativas y referencias bibliográficas de APIs oficiales.
4. **04 Simulador de Entrenamiento:** Consola interactiva para simular el proceso de entrenamiento de los modelos en tiempo real y visualizar curvas de pérdida o descomposición armónica.
""")

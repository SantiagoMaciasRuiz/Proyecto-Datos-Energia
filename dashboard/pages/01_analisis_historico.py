import streamlit as st
import os

# Configuración de página
st.set_page_config(
    page_title="EDA - Demanda SIN",
    page_icon="📈",
    layout="wide"
)

# Estilos CSS inyectados para diseño adaptativo
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .page-title {
        background: linear-gradient(90deg, #319795 0%, #2b6cb0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        color: #718096;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .chart-container {
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 0.5rem;
        background-color: rgba(128, 128, 128, 0.02);
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Títulos
st.markdown('<div class="page-title">📈 Análisis Exploratorio de Datos (EDA)</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Exploración estadística del consumo eléctrico del SIN colombiano (2022 - 2025)</div>', unsafe_allow_html=True)

# Definir rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
FINDINGS_PATH = os.path.join(PROJECT_ROOT, "reports", "eda_findings.md")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Columnas principales
col_left, col_right = st.columns([1, 1.2])

with col_left:
    st.write("### 📝 Informe de Hallazgos Cuantitativos")
    if os.path.exists(FINDINGS_PATH):
        with open(FINDINGS_PATH, "r", encoding="utf-8") as f:
            findings_content = f.read()
        # Limpiar o renderizar el contenido quitando el título principal duplicado
        lines = findings_content.split("\n")
        filtered_lines = [line for line in lines if not line.startswith("# Informe de Hallazgos")]
        st.markdown("\n".join(filtered_lines))
    else:
        st.warning("⚠️ No se encontró el informe `reports/eda_findings.md`.")
        st.info("Ejecuta el script de EDA (`python scripts/run_eda.py`) para generar las estadísticas.")

with col_right:
    st.write("### 🖼️ Galería de Gráficos Estadísticos")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📅 Vista General", 
        "🗓️ Tendencias", 
        "⏰ Patrones Horarios", 
        "🔥 Mapa de Calor", 
        "📊 Distribución"
    ])
    
    with tab1:
        st.write("#### Serie Temporal de Demanda Horaria Completa")
        fig1_path = os.path.join(FIGURES_DIR, "01_serie_temporal.png")
        if os.path.exists(fig1_path):
            st.image(fig1_path, caption="Demanda del SIN en kW para todo el periodo 2022-2025", use_container_width=True)
        else:
            st.error("Archivo no encontrado: 01_serie_temporal.png")
            
    with tab2:
        st.write("#### Tendencias y Evolución Macroeconómica")
        fig2_path = os.path.join(FIGURES_DIR, "02_tendencia_anual.png")
        fig3_path = os.path.join(FIGURES_DIR, "03_tendencia_mensual.png")
        
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if os.path.exists(fig2_path):
                st.image(fig2_path, caption="Crecimiento Interanual del Consumo", use_container_width=True)
        with col_t2:
            if os.path.exists(fig3_path):
                st.image(fig3_path, caption="Variación Estacional por Meses", use_container_width=True)
                
    with tab3:
        st.write("#### Ciclos y Comportamientos Horarios y Diarios")
        fig4_path = os.path.join(FIGURES_DIR, "04_demanda_por_hora.png")
        fig5_path = os.path.join(FIGURES_DIR, "05_demanda_por_dia_semana.png")
        
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if os.path.exists(fig4_path):
                st.image(fig4_path, caption="Curva de Demanda Diaria Promedio (Hora Pico: 20:00)", use_container_width=True)
        with col_c2:
            if os.path.exists(fig5_path):
                st.image(fig5_path, caption="Perfil de Consumo Semanal (Hábiles vs Festivos)", use_container_width=True)
                
    with tab4:
        st.write("#### Matriz de Calor (Día de la Semana vs Hora de la Jornada)")
        fig6_path = os.path.join(FIGURES_DIR, "06_heatmap_hora_dia.png")
        if os.path.exists(fig6_path):
            st.image(fig6_path, caption="Mapa de Intensidad Horaria de Carga", use_container_width=True)
        else:
            st.error("Archivo no encontrado: 06_heatmap_hora_dia.png")
            
    with tab5:
        st.write("#### Distribuciones Estadísticas y Valores Atípicos")
        fig7_path = os.path.join(FIGURES_DIR, "07_distribucion_demanda.png")
        fig8_path = os.path.join(FIGURES_DIR, "08_outliers.png")
        
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if os.path.exists(fig7_path):
                st.image(fig7_path, caption="Densidad Empírica e Histograma del Consumo", use_container_width=True)
        with col_d2:
            if os.path.exists(fig8_path):
                st.image(fig8_path, caption="Identificación de Anomalías por Boxplots", use_container_width=True)

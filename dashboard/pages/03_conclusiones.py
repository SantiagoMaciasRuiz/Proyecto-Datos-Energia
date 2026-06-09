import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="Conclusiones - Demanda SIN",
    page_icon="📝",
    layout="wide"
)

# Estilos CSS inyectados
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .page-title {
        background: linear-gradient(90deg, #2b6cb0 0%, #1e3c72 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        color: #718096;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }
    
    .conclusion-card {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .conclusion-card h4 {
        color: #2b5298;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Títulos
st.markdown('<div class="page-title">📝 Conclusiones y Referencias</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Resultados del análisis y cierre del proyecto académico</div>', unsafe_allow_html=True)

# Columnas principales
col1, col2 = st.columns(2)

with col1:
    st.write("### 🎓 Conclusiones Académicas")
    
    st.markdown("""
    <div class="conclusion-card">
        <h4>1. Superioridad de los Modelos Tabulares Autoregresivos</h4>
        <p>Los modelos <b>XGBoost</b> (R² = 99.52%) y <b>Random Forest</b> (R² = 99.32%) superaron significativamente al modelo estructural de Prophet. Esto se debe a la alta dependencia de la demanda eléctrica con respecto a la hora del día y la demanda de la hora anterior (lags autorregresivos de corto plazo), los cuales no son explotados directamente por Prophet.</p>
    </div>
    
    <div class="conclusion-card">
        <h4>2. Impacto Crítico de las Variables del Calendario</h4>
        <p>El comportamiento de los días hábiles en Colombia versus fines de semana y festivos oficiales muestra diferencias estadísticamente significativas. Los días de fin de semana registran un <b>8.16% menos de consumo</b>, mientras que los festivos presentan una caída del <b>13.19%</b> debido al cese de actividades industriales y comerciales.</p>
    </div>
    
    <div class="conclusion-card">
        <h4>3. Limitación de Inferencia de Largo Plazo (Efecto de Cascada)</h4>
        <p>Aunque los modelos tabulares son altamente precisos en ventanas cortas (test holdout), su aplicación recursiva para un horizonte de dos años completos (2026-2027) tiende a propagar errores acumulativos (la predicción en <i>t</i> alimenta los rezagos para <i>t+1</i>). Prophet resulta más estable en la proyección macro de tendencias a largo plazo debido a que su formulación matemática aditiva no es recurrente.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.write("### ⚙️ Recomendaciones Operativas para el SIN")
    st.markdown("""
    1. **Planificación de Despacho Diario:** Se recomienda el uso de **XGBoost** para pronósticos operativos de demanda diaria (próximas 24 horas), permitiendo optimizar el despacho de centrales de generación térmicas e hidroeléctricas con un margen de error menor al 1%.
    2. **Recalibración Continua:** Para evitar la acumulación de error en la proyección semanal y mensual, el modelo de aprendizaje supervisado debe ser reentrenado o realimentado con los valores reales registrados en el SIN cada 24 horas.
    3. **Integración Meteorológica:** Futuras iteraciones académicas del modelo deben integrar variables del fenómeno de El Niño/La Niña y temperatura promedio de las principales ciudades colombianas (Bogotá, Medellín, Cali, Barranquilla) para refinar los picos de demanda estacionales.
    """)
    
    st.write("---")
    st.write("### 📚 Fuentes Oficiales y Referencias")
    st.markdown("""
    *   **XM S.A. E.S.P. (Operador del SIN y Administrador del Mercado):** Los datos históricos crudos fueron obtenidos a través de la API oficial de XM (Portal BI de Demanda Horaria). [Visitar XM S.A. E.S.P.](https://www.xm.com.co)
    *   **SIMEM (Sistema de Información de la Carga de Energía y del SIN):** Referencias del mercado energético colombiano y regulaciones de la CREG. [Visitar SIMEM](https://www.simem.co)
    *   **Cátedra Pedro Nel Gómez (Universidad Nacional de Colombia):** Marco analítico sobre "La energía en el desarrollo económico, social y tecnológico de Colombia".
    *   **Prophet Documentation:** Taylor, S. J., & Letham, B. (2018). Forecasting at Scale. *The American Statistician*, 72(1), 37-45.
    *   **Chen, T., & Guestrin, C. (2016).** XGBoost: A Scalable Tree Boosting System. *ACM SIGKDD*.
    """)

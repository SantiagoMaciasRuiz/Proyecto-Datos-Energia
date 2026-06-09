import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Configuración de página
st.set_page_config(
    page_title="Modelamiento y Predicciones - Demanda SIN",
    page_icon="🔮",
    layout="wide"
)

# Estilos CSS inyectados para diseño premium adaptivo
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .page-title {
        background: linear-gradient(90deg, #dd6b20 0%, #2b6cb0 100%);
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
    
    .metric-card {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.06);
        border-color: #3182ce;
    }
    
    .metric-card-title {
        font-weight: 600;
        font-size: 1.3rem;
        color: #2D3748;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-card-title {
            color: #E2E8F0;
        }
    }
    
    .metric-value-primary {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0.4rem 0;
    }
    
    .metric-subvalue {
        font-size: 0.95rem;
        color: #4A5568;
        margin: 0.2rem 0;
    }
    
    @media (prefers-color-scheme: dark) {
        .metric-subvalue {
            color: #A0AEC0;
        }
    }
    
    .recommendation-box {
        background: linear-gradient(90deg, rgba(49, 130, 206, 0.1) 0%, rgba(49, 151, 149, 0.1) 100%);
        border-left: 5px solid #3182ce;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .recommendation-text {
        font-size: 1.1rem;
        font-weight: 500;
    }
    
    .explanation-item {
        margin-bottom: 1rem;
        border-left: 3px solid #718096;
        padding-left: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Títulos
st.markdown('<div class="page-title">🔮 Predicciones e Inferencia de Modelos</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Evaluación de precisión sobre datos históricos y predicción a futuro (2026 - 2027)</div>', unsafe_allow_html=True)

# Rutas de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "..", "models")
METRICS_PATH = os.path.join(MODELS_DIR, "model_metrics.csv")
PREDICTIONS_PATH = os.path.join(MODELS_DIR, "predicciones.csv")
FORECAST_PATH = os.path.join(MODELS_DIR, "forecast.csv")

# Barra lateral para esta página
st.sidebar.markdown("### ⚙️ Agregación Temporal")
freq_option = st.sidebar.selectbox(
    "Agregación:",
    ["Horaria", "Diaria (Promedio)", "Semanal (Promedio)", "Mensual (Promedio)"],
    index=1,
    help="Elige la agregación temporal de las curvas para facilitar el análisis a largo plazo."
)

# Verificar existencia de archivos
files_exist = os.path.exists(METRICS_PATH) and os.path.exists(PREDICTIONS_PATH) and os.path.exists(FORECAST_PATH)

if not files_exist:
    st.warning("⚠️ No se encontraron los archivos de resultados en el directorio `models/`.")
    st.info("Por favor, ejecuta el pipeline de modelamiento para habilitar la visualización.")
else:
    # Cargar datos con invalidación de caché basada en mtime
    @st.cache_data
    def load_data(metrics_mtime, predictions_mtime, forecast_mtime):
        metrics = pd.read_csv(METRICS_PATH)
        predictions = pd.read_csv(PREDICTIONS_PATH)
        forecast = pd.read_csv(FORECAST_PATH)
        
        # Parsear fechas
        predictions['fecha_hora'] = pd.to_datetime(predictions['fecha_hora'])
        forecast['fecha_hora'] = pd.to_datetime(forecast['fecha_hora'])
        
        return metrics, predictions, forecast

    try:
        metrics_mtime = os.path.getmtime(METRICS_PATH)
        predictions_mtime = os.path.getmtime(PREDICTIONS_PATH)
        forecast_mtime = os.path.getmtime(FORECAST_PATH)
        
        metrics_df, predictions_df, forecast_df = load_data(
            metrics_mtime, predictions_mtime, forecast_mtime
        )
        
        # Función para resamplear datos dinámicamente según la frecuencia elegida
        def resample_data(df, option, is_forecast=False):
            if option == "Horaria":
                return df
            
            freq_map = {
                "Diaria (Promedio)": "D",
                "Semanal (Promedio)": "W",
                "Mensual (Promedio)": "MS"
            }
            freq = freq_map.get(option, "D")
            
            resampled_frames = []
            for model_name in df['modelo'].unique():
                sub_df = df[df['modelo'] == model_name].copy()
                sub_df = sub_df.sort_values('fecha_hora')
                sub_df.set_index('fecha_hora', inplace=True)
                
                if is_forecast:
                    resampled = sub_df['prediccion'].resample(freq).mean().reset_index()
                    resampled['modelo'] = model_name
                    resampled['escenario'] = 'future'
                else:
                    resampled = sub_df[['real', 'prediccion']].resample(freq).mean().reset_index()
                    resampled['modelo'] = model_name
                    resampled['escenario'] = 'historical'
                resampled_frames.append(resampled)
                
            return pd.concat(resampled_frames, ignore_index=True)
        
        # Calcular el mejor modelo
        best_model_row = metrics_df.loc[metrics_df['mape'].idxmin()]
        best_model = best_model_row['modelo']
        best_mape = best_model_row['mape']
        best_r2 = best_model_row['r2']
        
        # Recomendación destacada
        st.markdown(f"""
        <div class="recommendation-box">
            <span style="font-size: 2rem;">🏆</span>
            <div class="recommendation-text">
                <b>Modelo Recomendado:</b> El modelo <b>{best_model}</b> presenta el mejor desempeño en la fase de validación temporal, con un 
                <b>MAPE de {best_mape:.2f}%</b> y un coeficiente de determinación <b>R² de {best_r2:.4f}</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparativa de Desempeño
        st.write("### 📊 Comparativa de Desempeño")
        col1, col2, col3 = st.columns(3)
        
        model_colors = {
            "Prophet": "#9B5DE5",
            "Random Forest": "#319795",
            "XGBoost": "#DD6B20"
        }
        
        columns_list = [col1, col2, col3]
        for idx, row in metrics_df.iterrows():
            model_name = row['modelo']
            r2_val = row['r2']
            mape_val = row['mape']
            mae_val = row['mae']
            rmse_val = row['rmse']
            color = model_colors.get(model_name, "#718096")
            
            with columns_list[idx % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-title">
                        <span style="color: {color};">●</span> {model_name}
                    </div>
                    <div class="metric-value-primary" style="color: {color};">R²: {r2_val:.4f}</div>
                    <div class="metric-subvalue"><b>MAPE:</b> {mape_val:.2f}%</div>
                    <div class="metric-subvalue"><b>MAE:</b> {mae_val:,.1f} kW</div>
                    <div class="metric-subvalue"><b>RMSE:</b> {rmse_val:,.1f} kW</div>
                </div>
                """, unsafe_allow_html=True)

        st.write("---")

        # Acordeón Explicativo (UX)
        with st.expander("📖 Guía de Visualización: ¿Qué representa cada línea y modelo?", expanded=True):
            st.markdown("""
            ### 📌 Leyenda de Líneas en los Gráficos
            
            *   **⚫ Demanda Real (Línea Negra continua):** 
                Representa el consumo histórico real medido en kilovatios (kW) por XM S.A. E.S.P. en el Sistema Interconectado Nacional. Es la fuente de verdad del proyecto.
            *   **🟣 Pronóstico Prophet (Línea Morada):** 
                Predicción del modelo Prophet. Este modelo aditivo de Facebook predice basándose en tendencias globales y estacionalidades continuas (diaria, semanal, anual), ideal para capturar crecimientos interanuales macro sin ruidos locales.
            *   **🟢 Pronóstico Random Forest (Línea Verde):** 
                Predicción de Random Forest. Este modelo tabular utiliza rezagos directos (`lag_1`, `lag_24`, `lag_168`), por lo que sus estimaciones en el corto plazo son sumamente sensibles a cambios rápidos.
            *   **🟠 Pronóstico XGBoost (Línea Naranja):** 
                Predicción del algoritmo XGBoost. Mediante árboles de gradiente impulsados de forma secuencial, predice la demanda corrigiendo desviaciones previas, logrando el error más bajo de validación.
            
            ### 🔌 Metodología de Pronóstico e Inferencia
            
            - **Evaluación Histórica (Test):** Compara los valores reales contra las predicciones sobre datos de validación ya registrados en el pasado. Sirve para calcular el error exacto (MAPE, MAE) de cada algoritmo.
            - **Inferencia Futura (Forecast):** Proyecta la demanda hacia el futuro (años 2026 y 2027) donde no existen lecturas reales. Los modelos tabulares operan de forma **recursiva** (una predicción alimenta a la siguiente como rezago `lag_1`), permitiendo observar la inercia estacional pura del sistema eléctrico colombiano.
            """)

        # Pestañas principales
        tab1, tab2 = st.tabs(["📉 Evaluación Histórica (Conjunto de Test)", "🔮 Pronóstico Futuro (Forecast 2026-2027)"])

        with tab1:
            st.write("#### Comparación de Predicciones sobre el Conjunto de Validación (Holdout)")
            st.write(f"Agregación actual: **{freq_option}**.")

            # Filtros temporales
            min_date = predictions_df['fecha_hora'].min()
            max_date = predictions_df['fecha_hora'].max()
            
            col_date_1, col_date_2 = st.columns(2)
            with col_date_1:
                start_date = st.date_input(
                    "Fecha inicial",
                    value=min_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="start_date_pred"
                )
            with col_date_2:
                end_date = st.date_input(
                    "Fecha final",
                    value=max_date.date(),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="end_date_pred"
                )
            
            filtered_preds = predictions_df[
                (predictions_df['fecha_hora'].dt.date >= start_date) &
                (predictions_df['fecha_hora'].dt.date <= end_date)
            ]

            if filtered_preds.empty:
                st.warning("No hay predicciones en el rango de fechas seleccionado.")
            else:
                resampled_preds = resample_data(filtered_preds, freq_option, is_forecast=False)
                
                # Gráfico Plotly
                fig = go.Figure()
                
                # Línea de Demanda Real
                real_demand_unique = resampled_preds[resampled_preds['modelo'] == resampled_preds['modelo'].unique()[0]]
                fig.add_trace(go.Scatter(
                    x=real_demand_unique['fecha_hora'],
                    y=real_demand_unique['real'],
                    name="Demanda Real",
                    line=dict(color="#1A202C", width=3, dash="solid"),
                    mode="lines"
                ))
                
                # Líneas de Predicciones de Modelos
                for m_name in resampled_preds['modelo'].unique():
                    m_data = resampled_preds[resampled_preds['modelo'] == m_name]
                    fig.add_trace(go.Scatter(
                        x=m_data['fecha_hora'],
                        y=m_data['prediccion'],
                        name=f"Predicción {m_name}",
                        line=dict(color=model_colors.get(m_name, "#718096"), width=2),
                        mode="lines"
                    ))
                
                fig.update_layout(
                    xaxis_title="Fecha",
                    yaxis_title="Demanda Real (kW)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_white",
                    margin=dict(l=40, r=40, t=50, b=40),
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.write("#### Pronóstico de Demanda Futura (Horizonte 17,520 Horas - 2026 y 2027)")
            st.write(f"Agregación actual: **{freq_option}**.")
            
            resampled_forecast = resample_data(forecast_df, freq_option, is_forecast=True)
            
            # Gráfico de forecast futuro
            fig_forecast = go.Figure()
            
            for m_name in resampled_forecast['modelo'].unique():
                m_data = resampled_forecast[resampled_forecast['modelo'] == m_name]
                fig_forecast.add_trace(go.Scatter(
                    x=m_data['fecha_hora'],
                    y=m_data['prediccion'],
                    name=f"Pronóstico {m_name}",
                    line=dict(color=model_colors.get(m_name, "#718096"), width=2),
                    mode="lines"
                ))
            
            fig_forecast.update_layout(
                xaxis_title="Fecha",
                yaxis_title="Predicción de Demanda (kW)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white",
                margin=dict(l=40, r=40, t=50, b=40),
                height=500
            )
            
            st.plotly_chart(fig_forecast, use_container_width=True)
            
            st.info("💡 **Nota Académica:** La predicción a largo plazo para modelos autoregresivos (Random Forest y XGBoost) se realiza mediante un bucle de inferencia recursiva. La agregación diaria, semanal o mensual nos permite visualizar el patrón estacional macro y mitigar las desviaciones que se acumulan a nivel de horas.")
            
    except Exception as e:
        st.error(f"Error al procesar predicciones: {str(e)}")
        st.exception(e)

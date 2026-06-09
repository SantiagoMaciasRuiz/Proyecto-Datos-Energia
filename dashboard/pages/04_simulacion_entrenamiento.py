import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Intentar importar XGBoost y Prophet, con fallbacks amigables
try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# Configuración de página
st.set_page_config(
    page_title="Simulador de Entrenamiento - Demanda SIN",
    page_icon="🤖",
    layout="wide"
)

# Estilos CSS premium inyectados
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    .page-title {
        background: linear-gradient(90deg, #4a154b 0%, #2b6cb0 100%);
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
    
    .sim-card {
        background-color: rgba(128, 128, 128, 0.04);
        border: 1px solid rgba(128, 128, 128, 0.12);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.01);
    }
    
    .metric-pill {
        background-color: rgba(49, 130, 206, 0.1);
        border-left: 3px solid #3182ce;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 1.1rem;
        color: #2b6cb0;
        display: inline-block;
        margin-right: 1rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Títulos
st.markdown('<div class="page-title">🤖 Simulador de Entrenamiento de Modelos</div>', unsafe_allow_html=True)
st.markdown('<div class="page-subtitle">Visualiza en tiempo real cómo los modelos de Machine Learning convergen y reconstruyen las curvas de demanda</div>', unsafe_allow_html=True)

# Rutas de datos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, "..", "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "historical_demand_processed.parquet")

# Cargar datos base
@st.cache_data
def load_base_data():
    if not os.path.exists(DATA_PATH):
        return None
    df = pd.read_parquet(DATA_PATH)
    df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
    return df

base_df = load_base_data()

if base_df is None:
    st.error("⚠️ No se encontró el dataset procesado en `data/processed/historical_demand_processed.parquet`.")
    st.info("Por favor, ejecuta el pipeline primero para procesar la serie de tiempo.")
else:
    # Ingeniería de Variables simplificada (causal)
    def prepare_features(df):
        working = df.copy().sort_values('fecha_hora').reset_index(drop=True)
        target = working['demanda'].astype(float)
        
        # Crear rezagos y medias móviles
        working["lag_1"] = target.shift(1)
        working["lag_24"] = target.shift(24)
        working["lag_168"] = target.shift(168)
        working["rolling_mean_24"] = target.shift(1).rolling(24, min_periods=1).mean()
        working["rolling_mean_168"] = target.shift(1).rolling(168, min_periods=1).mean()
        working["rolling_std_24"] = target.shift(1).rolling(24, min_periods=2).std().fillna(0.0)
        
        working = working.dropna(subset=["lag_1", "lag_24", "lag_168"]).reset_index(drop=True)
        return working

    # Preparar el frame
    feature_df = prepare_features(base_df)
    
    # Controles del simulador en la barra lateral
    st.sidebar.markdown("### 🛠️ Configuración de la Simulación")
    
    model_choice = st.sidebar.selectbox(
        "Elige el modelo a entrenar:",
        ["XGBoost", "Random Forest", "Prophet"]
    )
    
    # Parámetros según modelo
    if model_choice == "XGBoost":
        max_estimators = st.sidebar.slider("Rondas de Boosting (Árboles):", 10, 100, 50, step=10)
        learning_rate = st.sidebar.slider("Tasa de Aprendizaje (Learning Rate):", 0.01, 0.20, 0.08, step=0.01)
        max_depth = st.sidebar.slider("Profundidad Máxima del Árbol:", 2, 8, 5)
    elif model_choice == "Random Forest":
        max_estimators = st.sidebar.slider("Número de Árboles (n_estimators):", 10, 100, 50, step=10)
        max_depth = st.sidebar.slider("Profundidad Máxima del Árbol:", 2, 12, 8)
        min_samples_leaf = st.sidebar.slider("Muestras Mínimas por Hoja:", 1, 5, 2)
    else:  # Prophet
        st.sidebar.info("Prophet se entrena analizando y sumando componentes estacionales y de tendencia de forma determinista.")
        sub_sample_days = st.sidebar.slider("Días históricos para ajustar (Ventana):", 30, 180, 90, step=15)

    st.sidebar.write("---")
    st.sidebar.markdown("""
    💡 **Nota Didáctica:** Para que la simulación en la web sea instantánea y fluida, entrenamos los modelos sobre una ventana recortada del dataset histórico (últimos 40 días para modelos tabulares) y validamos sobre la última semana (168 horas).
    """)

    # Preparar datos de entrenamiento y prueba reducidos para velocidad
    # Tomamos los últimos 1000 registros para entrenar rápido y 168 para validar
    val_size = 168
    train_size = 800
    
    sim_data = feature_df.tail(train_size + val_size).reset_index(drop=True)
    train_df = sim_data.iloc[:train_size].reset_index(drop=True)
    test_df = sim_data.iloc[train_size:].reset_index(drop=True)
    
    feature_cols = [
        "hora", "dia", "mes", "trimestre", "anio", "dia_semana", 
        "es_fin_de_semana", "es_festivo", "lag_1", "lag_24", "lag_168",
        "rolling_mean_24", "rolling_mean_168", "rolling_std_24"
    ]
    
    X_train, y_train = train_df[feature_cols], train_df['demanda']
    X_test, y_test = test_df[feature_cols], test_df['demanda']
    
    # Contenedor principal para la simulación
    st.write("### 🎬 Consola de Simulación")
    
    col_play, col_desc = st.columns([1, 3])
    with col_play:
        start_btn = st.button("🚀 Iniciar Simulación de Entrenamiento", use_container_width=True)
    with col_desc:
        st.markdown(f"Presiona el botón para observar el proceso de aprendizaje secuencial de **{model_choice}**.")

    # Marcadores de posición dinámicos
    progress_bar = st.empty()
    status_text = st.empty()
    metrics_placeholder = st.empty()
    
    chart_col1, chart_col2 = st.columns([2, 1.2])
    with chart_col1:
        chart_title = st.empty()
        chart_placeholder = st.empty()
    with chart_col2:
        extra_title = st.empty()
        extra_placeholder = st.empty()

    # Ejecución de la simulación
    if start_btn:
        if model_choice == "XGBoost":
            if not XGB_AVAILABLE:
                st.error("XGBoost no está instalado en este entorno. Por favor, selecciona Random Forest.")
            else:
                st.info("Iniciando Boosting Secuencial...")
                
                # Para guardar pérdidas de entrenamiento y validación
                train_losses = []
                test_losses = []
                steps = list(range(1, max_estimators + 1, 2 if max_estimators > 30 else 1))
                if max_estimators not in steps:
                    steps.append(max_estimators)
                
                for step_idx, n in enumerate(steps):
                    progress_val = (step_idx + 1) / len(steps)
                    progress_bar.progress(progress_val)
                    status_text.markdown(f"🔄 **Ajustando modelo con {n} estimadores...** (Paso {step_idx+1} de {len(steps)})")
                    
                    # Entrenar XGBoost temporal
                    xgb_temp = XGBRegressor(
                        n_estimators=n,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=-1
                    )
                    xgb_temp.fit(X_train, y_train)
                    
                    # Predecir
                    train_pred = xgb_temp.predict(X_train)
                    test_pred = xgb_temp.predict(X_test)
                    
                    # Calcular pérdidas (RMSE)
                    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
                    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                    train_losses.append(train_rmse)
                    test_losses.append(test_rmse)
                    
                    # Métricas de validación
                    mae_val = mean_absolute_error(y_test, test_pred)
                    mape_val = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
                    r2_val = r2_score(y_test, test_pred)
                    
                    # Actualizar KPIs
                    metrics_placeholder.markdown(f"""
                    <div class="sim-card">
                        <div class="metric-pill">R²: {r2_val:.4f}</div>
                        <div class="metric-pill">MAPE: {mape_val:.2f}%</div>
                        <div class="metric-pill">MAE: {mae_val:,.1f} kW</div>
                        <div class="metric-pill">RMSE: {test_rmse:,.1f} kW</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Graficar predicción en tiempo real (Plotly)
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=test_df['fecha_hora'], y=y_test,
                        name="Demanda Real", line=dict(color="#1A202C", width=2.5)
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=test_df['fecha_hora'], y=test_pred,
                        name=f"Predicción (n={n})", line=dict(color="#DD6B20", width=2, dash="dash")
                    ))
                    fig_pred.update_layout(
                        title=f"Ajuste de la curva sobre el conjunto de test (Estimadores: {n})",
                        xaxis_title="Fecha y Hora", yaxis_title="Demanda (kW)",
                        margin=dict(l=20, r=20, t=40, b=20), height=400, template="plotly_white"
                    )
                    chart_placeholder.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Graficar curvas de pérdida
                    fig_loss = go.Figure()
                    fig_loss.add_trace(go.Scatter(
                        x=steps[:len(train_losses)], y=train_losses,
                        name="Entrenamiento (Train)", line=dict(color="#3182ce", width=2)
                    ))
                    fig_loss.add_trace(go.Scatter(
                        x=steps[:len(test_losses)], y=test_losses,
                        name="Validación (Test)", line=dict(color="#e53e3e", width=2)
                    ))
                    fig_loss.update_layout(
                        title="Curva de Aprendizaje (RMSE vs Iteraciones)",
                        xaxis_title="Árboles (Rondas)", yaxis_title="RMSE (kW)",
                        margin=dict(l=20, r=20, t=40, b=20), height=400, template="plotly_white"
                    )
                    extra_placeholder.plotly_chart(fig_loss, use_container_width=True)
                    
                    time.sleep(0.08)
                
                status_text.success(f"🎉 **¡Simulación de XGBoost Finalizada con Éxito!** El modelo convergió en {max_estimators} rondas.")

        elif model_choice == "Random Forest":
            st.info("Iniciando Construcción de Bosque Aleatorio...")
            
            # Guardar OOB / test MSE
            test_losses = []
            steps = list(range(1, max_estimators + 1, 2 if max_estimators > 30 else 1))
            if max_estimators not in steps:
                steps.append(max_estimators)
            
            for step_idx, n in enumerate(steps):
                progress_val = (step_idx + 1) / len(steps)
                progress_bar.progress(progress_val)
                status_text.markdown(f"🌲 **Cultivando {n} árboles de decisión...** (Paso {step_idx+1} de {len(steps)})")
                
                # Entrenar RF
                rf_temp = RandomForestRegressor(
                    n_estimators=n,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=42,
                    n_jobs=-1
                )
                rf_temp.fit(X_train, y_train)
                
                # Predecir
                test_pred = rf_temp.predict(X_test)
                
                # Calcular pérdidas
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                test_losses.append(test_rmse)
                
                # Métricas
                mae_val = mean_absolute_error(y_test, test_pred)
                mape_val = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
                r2_val = r2_score(y_test, test_pred)
                
                # KPIs
                metrics_placeholder.markdown(f"""
                <div class="sim-card">
                    <div class="metric-pill">R²: {r2_val:.4f}</div>
                    <div class="metric-pill">MAPE: {mape_val:.2f}%</div>
                    <div class="metric-pill">MAE: {mae_val:,.1f} kW</div>
                    <div class="metric-pill">RMSE: {test_rmse:,.1f} kW</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico interactivo
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=test_df['fecha_hora'], y=y_test,
                    name="Demanda Real", line=dict(color="#1A202C", width=2.5)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=test_df['fecha_hora'], y=test_pred,
                    name=f"Predicción (Árboles={n})", line=dict(color="#319795", width=2, dash="dash")
                ))
                fig_pred.update_layout(
                    title=f"Ajuste del ensamble sobre el test (Árboles: {n})",
                    xaxis_title="Fecha y Hora", yaxis_title="Demanda (kW)",
                    margin=dict(l=20, r=20, t=40, b=20), height=400, template="plotly_white"
                )
                chart_placeholder.plotly_chart(fig_pred, use_container_width=True)
                
                # Importancia de Variables en el paso actual
                importances = rf_temp.feature_importances_
                imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances})
                imp_df = imp_df.sort_values('importance', ascending=True)
                
                fig_imp = go.Figure()
                fig_imp.add_trace(go.Bar(
                    x=imp_df['importance'], y=imp_df['feature'],
                    orientation='h', marker=dict(color="#319795")
                ))
                fig_imp.update_layout(
                    title="Importancia Dinámica de Variables",
                    xaxis_title="Importancia Gini Relativa", yaxis_title="Variable",
                    margin=dict(l=20, r=20, t=40, b=20), height=400, template="plotly_white"
                )
                extra_placeholder.plotly_chart(fig_imp, use_container_width=True)
                
                time.sleep(0.08)
                
            status_text.success(f"🎉 **¡Bosque Aleatorio finalizado!** Se cultivaron {max_estimators} árboles de forma exitosa.")

        elif model_choice == "Prophet":
            if not PROPHET_AVAILABLE:
                st.error("Prophet no está instalado en este entorno.")
            else:
                st.info("Iniciando ajuste de componentes estacionales Prophet...")
                
                # Preparar datos en el formato de Prophet
                # Tomamos la ventana de días seleccionada
                prophet_full = base_df.tail(sub_sample_days * 24).copy()
                prophet_train = prophet_full.iloc[:-val_size][['fecha_hora', 'demanda']].rename(
                    columns={'fecha_hora': 'ds', 'demanda': 'y'}
                )
                prophet_test_df = prophet_full.iloc[-val_size:][['fecha_hora', 'demanda']].rename(
                    columns={'fecha_hora': 'ds', 'demanda': 'y'}
                )
                
                # Fases del ajuste
                phases = [
                    ("Ajustando Tendencia Lineal Segmentada...", True, False, False, False),
                    ("Añadiendo Estacionalidad Semanal...", True, True, False, False),
                    ("Añadiendo Estacionalidad Anual...", True, True, True, False),
                    ("Integrando Estacionalidad Diaria (Modelo Completo)...", True, True, True, True)
                ]
                
                for idx, (label, trend_flag, weekly, yearly, daily) in enumerate(phases):
                    progress_val = (idx + 1) / len(phases)
                    progress_bar.progress(progress_val)
                    status_text.markdown(f"🔮 **Componente {idx+1}: {label}**")
                    
                    # Configurar modelo con los componentes activos en este paso
                    m = Prophet(
                        growth='linear',
                        yearly_seasonality=yearly,
                        weekly_seasonality=weekly,
                        daily_seasonality=daily
                    )
                    m.fit(prophet_train)
                    
                    # Predecir sobre test
                    forecast = m.predict(prophet_test_df[['ds']])
                    test_pred = forecast['yhat'].values
                    y_test_prophet = prophet_test_df['y'].values
                    
                    # Métricas
                    mae_val = mean_absolute_error(y_test_prophet, test_pred)
                    mape_val = np.mean(np.abs((y_test_prophet - test_pred) / y_test_prophet)) * 100
                    r2_val = r2_score(y_test_prophet, test_pred)
                    
                    # KPIs
                    metrics_placeholder.markdown(f"""
                    <div class="sim-card">
                        <div class="metric-pill">R²: {r2_val:.4f}</div>
                        <div class="metric-pill">MAPE: {mape_val:.2f}%</div>
                        <div class="metric-pill">MAE: {mae_val:,.1f} kW</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gráfico
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=prophet_test_df['ds'], y=y_test_prophet,
                        name="Demanda Real", line=dict(color="#1A202C", width=2.5)
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=prophet_test_df['ds'], y=test_pred,
                        name=f"Suma de Componentes (Fase {idx+1})", line=dict(color="#9B5DE5", width=2)
                    ))
                    fig_pred.update_layout(
                        title=f"Reconstrucción del Pronóstico: {label}",
                        xaxis_title="Fecha y Hora", yaxis_title="Demanda (kW)",
                        margin=dict(l=20, r=20, t=40, b=20), height=400, template="plotly_white"
                    )
                    chart_placeholder.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Graficar el componente individual que se añadió
                    fig_comp = go.Figure()
                    if idx == 0:
                        fig_comp.add_trace(go.Scatter(
                            x=forecast['ds'], y=forecast['trend'],
                            name="Tendencia de Fondo", line=dict(color="#3182ce")
                        ))
                    elif idx == 1:
                        # Extraer componente semanal
                        fig_comp.add_trace(go.Scatter(
                            x=forecast['ds'], y=forecast['weekly'],
                            name="Ciclo Semanal (Efecto Festivos/Fines de Semana)", line=dict(color="#319795")
                        ))
                    elif idx == 2:
                        fig_comp.add_trace(go.Scatter(
                            x=forecast['ds'], y=forecast['yearly'],
                            name="Estacionalidad Anual", line=dict(color="#dd6b20")
                        ))
                    else:
                        fig_comp.add_trace(go.Scatter(
                            x=forecast['ds'], y=forecast['daily'],
                            name="Curva Horaria de Demanda Diaria", line=dict(color="#e53e3e")
                        ))
                        
                    fig_comp.update_layout(
                        title=f"Señal extraída en Fase {idx+1}",
                        xaxis_title="Fecha", yaxis_title="Efecto sobre la demanda (kW)",
                        margin=dict(l=20, r=20, t=40, b=20), height=400, template="plotly_white"
                    )
                    extra_placeholder.plotly_chart(fig_comp, use_container_width=True)
                    
                    time.sleep(1.2)
                    
                status_text.success("🎉 **¡Descomposición de Prophet completa!** Observa cómo la suma secuencial de ondas armónicas da lugar a la curva compleja de consumo real.")

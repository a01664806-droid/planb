# modules/ui_analysis.py
import streamlit as st
import plotly.graph_objects as go
import database
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# ✅ Reutilizamos lógica de EDA / tendencias desde ui_info
from .ui_info import load_info_df, compute_eda_aggregates, line_chart, _axis, THEME_PALETTE


# --- Función para Cargar y Usar el Modelo de Forecast ---
@st.cache_resource
def load_forecast_model():
    """Carga el modelo SARIMA guardado."""
    try:
        model = SARIMAXResults.load("crime_forecaster.pkl")
        return model
    except FileNotFoundError:
        st.error(
            "Error: Archivo 'crime_forecaster.pkl' no encontrado. "
            "Por favor, ejecuta 'train_models.py' primero."
        )
        return None
    except Exception as e:
        st.error(f"Error al cargar el modelo de forecast: {e}")
        return None


@st.cache_data
def get_forecast(_model, steps=7):
    """Genera una predicción de 'steps' días."""
    if _model:
        try:
            forecast = _model.get_forecast(steps=steps)
            forecast_df = forecast.summary_frame(alpha=0.05)
            forecast_df = forecast_df.rename(
                columns={
                    "mean": "Predicción Promedio",
                    "mean_ci_lower": "Límite Inferior (95%)",
                    "mean_ci_upper": "Límite Superior (95%)",
                }
            )
            return forecast_df
        except Exception as e:
            st.warning(f"Error al generar predicción: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# --- Render principal para la página de Analysis ---
def render():
    # Título y descripción
    st.title("🕓 Análisis de Tendencia y Pronóstico")
    st.markdown("Análisis histórico y pronóstico a corto y mediano plazo.")

    # --- Carga de Datos y Modelos ---
    df_tendencia = database.get_historical_tendency()
    model_sarima = load_forecast_model()

    # --- Gráfico Principal: Tendencia y Pronóstico a 5 Meses ---
    st.header("Pronóstico de Tendencia a 5 Meses")
    st.markdown(
        "Esta gráfica muestra la tendencia histórica (últimos 6 meses) "
        "y una predicción del número total de crímenes para los próximos 5 meses."
    )

    if not df_tendencia.empty:
        fig = go.Figure()

        # 1. Línea histórica (solo los últimos 180 días)
        df_tendencia_reciente = df_tendencia.tail(180)
        fig.add_trace(
            go.Scatter(
                x=df_tendencia_reciente["fecha"],
                y=df_tendencia_reciente["total_delitos"],
                mode="lines",
                name="Tendencia Histórica",
            )
        )

        # 2. Obtener y mostrar el pronóstico
        if model_sarima:
            # Pronóstico a 150 días (5 meses)
            df_forecast = get_forecast(model_sarima, steps=150)

            if not df_forecast.empty:
                # 🔒 Corrección para evitar números negativos en pronóstico y bandas
                df_forecast["Predicción Promedio"] = df_forecast[
                    "Predicción Promedio"
                ].clip(lower=0)
                df_forecast["Límite Inferior (95%)"] = df_forecast[
                    "Límite Inferior (95%)"
                ].clip(lower=0)
                df_forecast["Límite Superior (95%)"] = df_forecast[
                    "Límite Superior (95%)"
                ].clip(lower=0)

                # Línea de pronóstico (media)
                fig.add_trace(
                    go.Scatter(
                        x=df_forecast.index,
                        y=df_forecast["Predicción Promedio"],
                        mode="lines",
                        name="Pronóstico",
                        line=dict(dash="dot", color="red"),
                    )
                )
                # Banda de confianza
                fig.add_trace(
                    go.Scatter(
                        x=df_forecast.index,
                        y=df_forecast["Límite Superior (95%)"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        name="Límite Superior",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=df_forecast.index,
                        y=df_forecast["Límite Inferior (95%)"],
                        mode="lines",
                        line=dict(width=0),
                        fillcolor="rgba(255, 0, 0, 0.2)",
                        fill="tonexty",
                        showlegend=False,
                        name="Límite Inferior",
                    )
                )

                fig.update_layout(
                    template="plotly_dark",
                    xaxis_title="Fecha",
                    yaxis_title="Total de Delitos",
                )
                st.plotly_chart(fig, use_container_width=True)

                with st.expander(
                    "Ver tabla con los datos del pronóstico a 5 meses"
                ):
                    st.dataframe(
                        df_forecast[
                            [
                                "Predicción Promedio",
                                "Límite Inferior (95%)",
                                "Límite Superior (95%)",
                            ]
                        ].style.format("{:.0f}")
                    )
            else:
                st.error("No se pudo generar el pronóstico.")
        else:
            st.warning("No se pudo cargar el modelo de pronóstico.")
    else:
        st.warning("No se pudieron cargar los datos de tendencia.")

    # =========================
    # TENDENCIAS EN EL TIEMPO
    # (antes estaban en ui_info)
    # =========================
    st.header("Tendencias en el Tiempo")

    # Cargamos el dataset completo para tendencias globales
    base_df = load_info_df(year_min=2015, limit=None)
    if base_df.empty:
        st.warning("No hay datos suficientes para mostrar tendencias históricas.")
        return

    agg = compute_eda_aggregates(base_df)

    # Tendencia Anual por Alcaldía (Top 8)
    if not agg["yearly_mun_top"].empty:
        st.subheader("Tendencia Anual por Alcaldía (Top 8)")
        st.altair_chart(
            line_chart(
                agg["yearly_mun_top"],
                "anio_inicio",
                "count",
                color_field="alcaldia_std",
                title="Tendencia Anual por Alcaldía (Top 8)",
                width=950,
                height=420,
                x_axis=_axis("Año"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )

    # Tendencia Anual por Clasificación
    if not agg["year_class"].empty:
        st.subheader("Tendencia Anual por Clasificación")
        st.altair_chart(
            line_chart(
                agg["year_class"],
                "anio_inicio",
                "count",
                color_field="crime_classification",
                title="Tendencia Anual por Clasificación",
                width=950,
                height=420,
                x_axis=_axis("Año"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )

    # Tendencia Anual: Violento vs No Violento
    if not agg["yearly_viol"].empty:
        st.subheader("Tendencia Anual: Violento vs No Violento")
        st.altair_chart(
            line_chart(
                agg["yearly_viol"],
                "anio_inicio",
                "count",
                color_field="violence_type",
                title="Tendencia Anual: Violento vs No Violento",
                width=950,
                height=420,
                x_axis=_axis("Año"),
                y_axis=_axis("Número de casos"),
            ),
            use_container_width=True,
        )

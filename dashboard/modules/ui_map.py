# modules/ui_map_predictivo.py

import streamlit as st
import pandas as pd
import json
from pathlib import Path

import folium
from folium.plugins import HeatMap, MarkerCluster # <-- IMPORTANTE: Agregar MarkerCluster
from streamlit_folium import st_folium
import duckdb

# =========================
# RUTAS / CONSTANTES
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "cdmx_insights.db"
CLUSTER_PATH = BASE_DIR / "cluster_info.csv"
COLONIAS_GEOJSON = BASE_DIR / "coloniascdmx.geojson"
ALCALDIAS_GEOJSON = BASE_DIR / "alcaldias.geojson"

CDMX_CENTER = [19.4326, -99.1332]


# =========================
# LOADERS CACHEADOS
# =========================
@st.cache_data
def get_filter_options():
    """
    Obtiene los valores únicos para los filtros desde la base de datos
    y los cachea en la sesión del usuario para una carga más rápida.
    """
    # 🔑 OPTIMIZACIÓN: Cachear en la sesión del usuario
    if 'filter_options' in st.session_state:
        return st.session_state.filter_options

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        years = con.execute("SELECT DISTINCT anio_inicio FROM crimes ORDER BY anio_inicio DESC").df()['anio_inicio'].tolist()
        v_opts = con.execute("SELECT DISTINCT violence_type FROM crimes ORDER BY violence_type").df()['violence_type'].tolist()
        d_opts = con.execute("SELECT DISTINCT delito FROM crimes ORDER BY delito").df()['delito'].tolist()
        a_opts = con.execute("SELECT DISTINCT alcaldia_hecho FROM crimes ORDER BY alcaldia_hecho").df()['alcaldia_hecho'].tolist()
        min_hora, max_hora = con.execute("SELECT MIN(hour(hora_hecho)), MAX(hour(hora_hecho)) FROM crimes").fetchone()
    finally:
        con.close()
    
    filter_options = (years, v_opts, d_opts, a_opts, min_hora, max_hora)
    st.session_state.filter_options = filter_options
    
    return filter_options

@st.cache_data
def load_crime_points(filters):
    """
    Carga puntos de crimen desde DuckDB con filtros dinámicos.
    """
    base_query = """
        SELECT
            latitud,
            longitud,
            delito,
            violence_type AS violence,
            fecha_hecho,
            anio_inicio AS anio,
            mes_inicio AS mes,
            hour(hora_hecho) AS hora,
            alcaldia_hecho AS alcaldia,
            colonia_hecho AS colonia
        FROM crimes
    """
    
    where_clauses = ["latitud IS NOT NULL", "longitud IS NOT NULL"]
    params = []

    if filters["years"]:
        where_clauses.append(f"anio_inicio IN ({','.join(['?']*len(filters['years']))})")
        params.extend(filters["years"])

    if filters["violence"]:
        where_clauses.append(f"violence_type IN ({','.join(['?']*len(filters['violence']))})")
        params.extend(filters["violence"])

    if filters["delitos"]:
        where_clauses.append(f"delito IN ({','.join(['?']*len(filters['delitos']))})")
        params.extend(filters["delitos"])

    if filters["alcaldias"]:
        where_clauses.append(f"alcaldia_hecho IN ({','.join(['?']*len(filters['alcaldias']))})")
        params.extend(filters["alcaldias"])

    if filters["hour_range"]:
        where_clauses.append("hour(hora_hecho) BETWEEN ? AND ?")
        params.extend(filters["hour_range"])

    if where_clauses:
        query = f"{base_query} WHERE {' AND '.join(where_clauses)}"
    else:
        query = base_query

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = con.execute(query, params).df()
    finally:
        con.close()
    
    return df


@st.cache_data
def load_cluster_info():
    if not CLUSTER_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(CLUSTER_PATH)


@st.cache_data
def load_geojson(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        gj = json.load(f)
    return gj


# =========================
# FILTROS
# =========================
def apply_filters(years, v_opts, d_opts, a_opts, min_hora, max_hora):
    st.sidebar.markdown("### 🔎 Filtros del mapa")

    # 🔑 OPTIMIZACIÓN: Cargar el año más reciente por defecto
    default_year = [years[0]] if years else []
    year_sel = st.sidebar.multiselect("Año", years, default=default_year)
    
    v_sel = st.sidebar.multiselect("Tipo de violencia", v_opts, default=[])
    delito_sel = st.sidebar.multiselect("Delito", d_opts, default=[])
    alcaldia_sel = st.sidebar.multiselect("Alcaldía", a_opts, default=[])
    h_range = st.sidebar.slider("Hora del día", min_value=min_hora, max_value=max_hora, value=(min_hora, max_hora))

    return {
        "years": year_sel,
        "violence": v_sel,
        "delitos": delito_sel,
        "alcaldias": alcaldia_sel,
        "hour_range": h_range,
    }


# =========================
# CAPA 1: PUNTOS ("DROPS") - OPTIMIZADA con MarkerCluster
# =========================
def build_points_map(df: pd.DataFrame):
    m = folium.Map(
        location=CDMX_CENTER,
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )
    
    # 🔑 OPTIMIZACIÓN: Inicializar MarkerCluster
    mc = MarkerCluster().add_to(m)

    # Leyenda manual simple
    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 10px 14px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
        font-size: 13px;">
        <b>Leyenda</b><br>
        <span style="color:#d73027;font-size:18px;">●</span> Violento<br>
        <span style="color:#1a759f;font-size:18px;">●</span> No violento
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Usamos iterrows, pero agregamos al cluster
    for _, row in df.iterrows():
        lat = row["latitud"]
        lon = row["longitud"]
        if pd.isna(lat) or pd.isna(lon):
            continue

        v = str(row.get("violence", ""))
        is_violent = "Violent" in v or "VIOLENTO" in v.upper()
        
        # Usamos íconos para MarkerCluster, no CircleMarker
        icon_color = "red" if is_violent else "blue"
        
        delito = row.get("delito", "N/D")
        fecha = row.get("fecha_hecho", "N/D")
        alcaldia = row.get("alcaldia", "N/D")
        colonia = row.get("colonia", "N/D")
        hora = row.get("hora", "N/D")

        popup_html = f"""
        <b>Delito:</b> {delito}<br>
        <b>Violencia:</b> {v}<br>
        <b>Fecha:</b> {fecha} {hora}:00<br>
        <b>Alcaldía:</b> {alcaldia}<br>
        <b>Colonia:</b> {colonia}
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=icon_color, icon="info-sign"), # Usamos un ícono simple
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(mc) # <-- Agregar al MarkerCluster, no al mapa directamente

    return m


# =========================
# CAPA 2: HEATMAP
# =========================
def build_heatmap(df: pd.DataFrame):
    m = folium.Map(
        location=CDMX_CENTER,
        zoom_start=11,
        tiles="CartoDB dark_matter",
        control_scale=True,
    )

    heat_data = (
        df[["latitud", "longitud"]]
        .dropna()
        .to_numpy()
        .tolist()
    )

    HeatMap(
        heat_data,
        radius=13,
        blur=18,
        max_zoom=13,
    ).add_to(m)

    title_html = """
        <h4 style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
                   z-index: 9999; background-color: rgba(0,0,0,0.6); color: white;
                   padding: 6px 10px; border-radius: 8px; font-size: 14px;">
            Heatmap de densidad de incidentes
        </h4>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    return m


# =========================
# CAPA 3: COLONIAS (CHOROPLETH)
# =========================
def build_colonias_map(df: pd.DataFrame):
    gj_colonias = load_geojson(COLONIAS_GEOJSON)
    if gj_colonias is None:
        st.warning("No se encontró el geojson de colonias. Revisa la ruta, ca.")
        return None

    # AJUSTA: "cve_col" por la llave que tengas en tu DB y en el geojson
    df_tmp = df.copy()
    if "colonia" not in df_tmp.columns:
        st.warning("El dataframe no tiene columna 'colonia'. Ajusta el código.")
        return None

    # Simple: conteos por nombre de colonia
    colonia_counts = (
        df_tmp.groupby("colonia")
        .size()
        .reset_index(name="crime_count")
    )

    m = folium.Map(
        location=CDMX_CENTER,
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Para hacerlo robusto, creo un dict de {nombre_colonia_normalizado: crime_count}
    mapping = {
        str(row["colonia"]).upper(): row["crime_count"]
        for _, row in colonia_counts.iterrows()
    }

    def style_function(feature):
        # 🔁 AJUSTA: Revisa qué campo usa tu GeoJSON para el nombre de la colonia (e.g., "NOM_COL")
        nom_col = str(feature["properties"].get("NOM_COL", "")).upper()
        val = mapping.get(nom_col, 0)

        # Escala simple de color
        if val == 0:
            fill = "#f5f5f5"
        elif val < 10:
            fill = "#fee8c8"
        elif val < 30:
            fill = "#fdbb84"
        else:
            fill = "#e34a33"

        return {
            "fillColor": fill,
            "color": "#555555",
            "weight": 0.5,
            "fillOpacity": 0.7,
        }

    def highlight_function(feature):
        return {
            "fillColor": "#ffffbf",
            "color": "#000000",
            "weight": 1.5,
            "fillOpacity": 0.9,
        }

    folium.GeoJson(
        gj_colonias,
        name="Colonias",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=["NOM_COL"],  # 🔁 AJUSTA nombre campo
            aliases=["Colonia:"],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        z-index: 9999;
        background-color: white;
        padding: 10px 14px;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
        font-size: 13px;">
        <b>Intensidad de delitos</b><br>
        <span style="background:#f5f5f5;border-radius:4px;padding:2px 8px;">0</span><br>
        <span style="background:#fee8c8;border-radius:4px;padding:2px 8px;">1 - 9</span><br>
        <span style="background:#fdbb84;border-radius:4px;padding:2px 8px;">10 - 29</span><br>
        <span style="background:#e34a33;border-radius:4px;padding:2px 8px;">30+</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# =========================
# RENDER PRINCIPAL
# =========================
def render():
    st.title("🗺️ Mapa predictivo de crimen CDMX")

    # Carga de opciones para los filtros
    try:
        years, v_opts, d_opts, a_opts, min_hora, max_hora = get_filter_options()
    except Exception as e:
        st.error(f"Error cargando opciones de filtros: {e}")
        return

    # Aplicar filtros y obtener selecciones
    filters = apply_filters(years, v_opts, d_opts, a_opts, min_hora, max_hora)

    # 🔑 OPTIMIZACIÓN: Se elimina la comprobación para cargar datos por defecto
    # if not any([filters["years"], filters["violence"], filters["delitos"], filters["alcaldias"]]):
    #     st.info("Por favor, selecciona al menos un filtro para cargar los datos.")
    #     return

    # Carga de datos filtrados
    try:
        # Usamos un spinner mientras se cargan los datos de DuckDB
        with st.spinner(f"Cargando {', '.join(filters['delitos'] or filters['alcaldias'] or ['delitos...'])}"):
            df_filt = load_crime_points(filters)
    except Exception as e:
        st.error(f"Error cargando datos de delitos: {e}")
        return

    if df_filt.empty:
        st.info("No hay datos de crimen para mostrar con los filtros seleccionados.")
        return
    
    st.success(f"¡Datos cargados! Total de incidentes: {len(df_filt):,}")

    st.markdown(
        """
        Usa las pestañas para cambiar de vista:

        - **📍 Puntos**: Cada incidente agrupado por *cluster* (optimizado para carga rápida).
        - **🔥 Heatmap**: densidad de incidentes.
        - **🧩 Colonias**: intensidad por colonia (choropleth).
        """
    )

    tabs = st.tabs(["📍 Puntos", "🔥 Heatmap", "🧩 Colonias"])

    # ----- TAB 1: PUNTOS -----
    with tabs[0]:
        st.subheader("📍 Incidentes puntuales (MarkerCluster)")
        # El mapa de puntos es ahora muy rápido
        m_points = build_points_map(df_filt)
        st_folium(m_points, width="100%", height=600)

    # ----- TAB 2: HEATMAP -----
    with tabs[1]:
        st.subheader("🔥 Heatmap de densidad")
        m_heat = build_heatmap(df_filt)
        st_folium(m_heat, width="100%", height=600)

    # ----- TAB 3: COLONIAS -----
    with tabs[2]:
        st.subheader("🧩 Intensidad por colonia")
        m_colonias = build_colonias_map(df_filt)
        if m_colonias is not None:
            st_folium(m_colonias, width="100%", height=600)
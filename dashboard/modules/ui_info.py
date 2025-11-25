# modules/ui_info.py
import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go  # Para los gauges
from contextlib import contextmanager

# ✅ helpers propios
from .helpers import MONTH_NAMES
# ✅ runner DuckDB centralizado
from .data import run_query, get_violence_time_metrics

# =================== THEME ===================
def _enable_transparent_theme():
    theme_conf = {
        "config": {
            "background": "transparent",
            "view": {"fill": "transparent"},
            "axis": {
                "labelFontSize": 13,
                "titleFontSize": 14,
                "labelColor": "#E5E7EB",
                "titleColor": "#E5E7EB",
                "gridColor": "#33415555",
                "domainColor": "#64748B"
            },
            "title": {"fontSize": 16, "fontWeight": "bold", "color": "#E5E7EB"},
            "legend": {"labelFontSize": 13, "titleFontSize": 14,
                       "labelColor": "#E5E7EB", "titleColor": "#E5E7EB"}
        }
    }
    alt.themes.register("transp_dark", lambda: theme_conf)
    alt.themes.enable("transp_dark")

_enable_transparent_theme()

THEME_PALETTE = [
    "#1E3A8A", "#3B82F6", "#60A5FA", "#93C5FD", "#A3B3C2",
    "#64748B", "#0EA5E9", "#14B8A6", "#F59E0B", "#64748B",
]

CLASSIFICATION_TRANSLATIONS = {
    "Patrimony": "Patrimonio",
    "Freedom and Sexual Segurity": "Libertad y Seguridad Sexual",
    "Personal Freedom": "Libertad Personal",
    "Life and Integrity": "Vida e Integridad",
    "Family": "Familia",
    "Society": "Sociedad",
    "Others": "Otros",
}

VIOLENCE_TRANSLATIONS = {
    "Violent": "Violento",
    "Non-Violent": "No violento",
}

# =================== HELPERS ===================
MONTH_NUM = {v: k for k, v in MONTH_NAMES.items()}  # "Enero"->1, etc.

def _cfg(width=700, height=350):
    return {"width": width, "height": height}

def _axis(title, orient=None):
    base = {"title": title, "grid": True, "gridColor": "#33415555"}
    if orient:
        base["orient"] = orient
    return base

def _auto_height(count, base=260, per_item=38):
    """Calcula altura dinámica para listas categóricas."""
    try:
        n = int(count)
    except (TypeError, ValueError):
        return base
    return max(base, n * per_item)

def _safe_uniques(series: pd.Series) -> list:
    """Regresa valores únicos limpios (sin NaN, '', 'nan', 'delito de bajo impacto')."""
    try:
        vals = []
        for v in series.dropna().unique().tolist():
            s = str(v).strip()
            if not s:
                continue
            low = s.lower()
            if low in {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}:
                continue
            vals.append(s)
        vals = sorted(vals)
    except Exception:
        vals = []
    return vals

def _classification_label(value):
    """Traduce las clasificaciones a español para mostrarlas en la UI."""
    if value is None:
        return "Sin clasificación"
    try:
        if pd.isna(value):
            return "Sin clasificación"
    except TypeError:
        pass
    value_str = str(value)
    return CLASSIFICATION_TRANSLATIONS.get(value_str, value_str)

def _violence_label(value):
    """Traduce el tipo de violencia a español para la interfaz."""
    if value is None:
        return "Sin tipo"
    try:
        if pd.isna(value):
            return "Sin tipo"
    except TypeError:
        pass
    value_str = str(value)
    return VIOLENCE_TRANSLATIONS.get(value_str, value_str)

def _format_hour_24(hour_value) -> str:
    """Convierte una hora (0-23) a formato 24h con sufijo :00."""
    try:
        hour_int = int(hour_value) % 24
    except (TypeError, ValueError):
        return str(hour_value)
    return f"{hour_int:02d}:00"

def _format_period_label(label: str) -> str:
    """Formatea etiquetas tipo 'Noche (19-07)' a 'Noche (19:00 – 07:00)'."""
    if not isinstance(label, str) or "(" not in label or ")" not in label:
        return label
    try:
        name, range_part = label.split("(", 1)
        range_part = range_part.rstrip(")")
        start_raw, end_raw = [part.strip() for part in range_part.split("-", 1)]
        start_fmt = _format_hour_24(start_raw)
        end_fmt = _format_hour_24(end_raw)
        return f"{name.strip()} ({start_fmt} – {end_fmt})"
    except ValueError:
        return label

def _tooltip(field: str, type_code: str, *, title: str | None = None, fmt: str | None = None):
    """
    Construye tooltips con formato de miles por defecto para valores cuantitativos.
    Permite sobreescribir el formato cuando se requiera algo distinto (p.ej. porcentajes).
    """
    kwargs = {}
    if title:
        kwargs["title"] = title
    if fmt:
        kwargs["format"] = fmt
    elif (type_code or "").upper().startswith("Q"):
        kwargs["format"] = ",.0f"
    return alt.Tooltip(f"{field}:{type_code}", **kwargs)

# =================== DATA LOAD (DuckDB) ===================
@st.cache_data(ttl=3600, show_spinner=False)
def load_info_df(year_min: int = 2016, limit: int | None = None) -> pd.DataFrame:
    """
    Trae columnas necesarias desde DuckDB y las normaliza a nombres esperados.
    Por defecto carga TODO (limit=None).
    """
    limit_clause = "LIMIT ?" if limit is not None else ""
    params = [year_min] + ([int(limit)] if limit is not None else [])

    query = f"""
        SELECT
            anio_hecho                                     AS anio_inicio,
            EXTRACT(MONTH FROM CAST(fecha_hecho AS DATE))  AS mes_inicio,
            EXTRACT(HOUR  FROM hora_hecho)                 AS hour_hecho,
            alcaldia_hecho                                 AS alcaldia_std,
            TRY_CAST(colonia_hecho AS VARCHAR)             AS colonia_std,
            categoria_delito                               AS delito_std,
            crime_classification,
            violence_type
        FROM crimes
        WHERE anio_hecho >= ?
        {limit_clause}
    """
    df = run_query(query, params=params)
    if df.empty:
        return df

    # Normalizaciones numéricas
    for col in ["anio_inicio", "mes_inicio", "hour_hecho"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for needed in ["alcaldia_std", "colonia_std", "delito_std", "crime_classification", "violence_type"]:
        if needed not in df.columns:
            df[needed] = pd.Series(dtype="object")

    # Limpieza de strings y mapeo de 'nan' / bajo impacto a NA
    low_impact_labels = {
        "delito de bajo impacto",
        "delitos de bajo impacto",
        "bajo impacto"
    }
    for col in ["delito_std", "crime_classification"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace(["", "nan", "NaN", "None"], pd.NA)
            df[col] = df[col].mask(
                df[col].str.lower().isin(low_impact_labels),
                pd.NA
            )

    return df

# =================== CACHES ===================
@st.cache_data(ttl=3600, show_spinner=False)
def get_violence_time_metrics_cached() -> pd.DataFrame:
    try:
        return get_violence_time_metrics()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, max_entries=2)
def compute_eda_aggregates(df: pd.DataFrame):
    YEAR_RANGE, MONTH_RANGE, HOUR_RANGE = range(2016, 2025), range(1, 13), range(0, 24)

    df = df.copy()
    df["anio_inicio"] = pd.to_numeric(df.get("anio_inicio"), errors="coerce")
    df["mes_inicio"]  = pd.to_numeric(df.get("mes_inicio"),  errors="coerce")
    if "hour_hecho" in df.columns:
        df["hour_hecho"] = pd.to_numeric(df["hour_hecho"], errors="coerce")

    # Drop rows with NaN in 'alcaldia_std' to prevent it from appearing in graphs
    if "alcaldia_std" in df.columns:
        df["alcaldia_std"] = df["alcaldia_std"].replace("", pd.NA)
        df = df.dropna(subset=["alcaldia_std"])

    year_counts = (
        df.dropna(subset=["anio_inicio"])
          .groupby("anio_inicio").size()
          .reindex(YEAR_RANGE, fill_value=0)
          .reset_index(name="count").rename(columns={"anio_inicio": "year"})
    )
    month_counts = (
        df.dropna(subset=["mes_inicio"])
          .groupby("mes_inicio").size()
          .reindex(MONTH_RANGE, fill_value=0)
          .reset_index(name="count").rename(columns={"mes_inicio": "month"})
    )
    month_counts["month_name"] = month_counts["month"].map(
        lambda m: MONTH_NAMES.get(int(m), str(m))
    )

    if "hour_hecho" in df.columns:
        hour_counts = (
            df.dropna(subset=["hour_hecho"]).groupby("hour_hecho").size()
              .reindex(HOUR_RANGE, fill_value=0)
              .reset_index(name="count").rename(columns={"hour_hecho": "hour"})
        )
    else:
        hour_counts = pd.DataFrame({"hour": list(HOUR_RANGE), "count": 0})

    alc_series = df.get("alcaldia_std", pd.Series([], dtype="object"))
    if not alc_series.empty:
        alc_counts = (
            alc_series.dropna()
            .value_counts()
            .rename_axis("alcaldia")
            .reset_index(name="count")
        )
    else:
        alc_counts = pd.DataFrame(columns=["alcaldia", "count"])

    col_series = df.get("colonia_std", pd.Series([], dtype="object"))
    if not col_series.empty:
        col_counts = (
            col_series.dropna()
            .value_counts()
            .rename_axis("colonia")
            .reset_index(name="count")
        )
    else:
        col_counts = pd.DataFrame(columns=["colonia", "count"])

    # Conteo de delitos (sin NaN ni bajo impacto)
    del_series = df.get("delito_std", pd.Series([], dtype="object")).astype("string")
    if not del_series.empty:
        del_series = del_series.dropna().apply(lambda x: x.strip())
        del_series = del_series[
            ~del_series.str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
        del_counts = (
            del_series.value_counts()
            .rename_axis("delito")
            .reset_index(name="count")
        )
        if del_counts.empty:
            del_counts = pd.DataFrame(columns=["delito", "count"])
    else:
        del_counts = pd.DataFrame(columns=["delito", "count"])

    # Conteo de clasificación (sin NaN ni bajo impacto)
    cls_series = df.get("crime_classification", pd.Series([], dtype="object")).astype("string")
    if not cls_series.empty:
        cls_series = cls_series.dropna().apply(lambda x: x.strip())
        cls_series = cls_series[
            ~cls_series.str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
        cls_counts = cls_series.value_counts().reset_index()
        if not cls_counts.empty:
            cls_counts.columns = ["classification", "count"]
        else:
            cls_counts = pd.DataFrame(columns=["classification", "count"])
    else:
        cls_counts = pd.DataFrame(columns=["classification", "count"])

    if "violence_type" in df.columns:
        v_counts_raw = df["violence_type"].astype("string")
        v_counts_raw = v_counts_raw.dropna().apply(lambda x: x.strip())
        v_counts = v_counts_raw.value_counts().reset_index()
        v_counts.columns = ["violence", "count"]
        v_counts["violence_label"] = v_counts["violence"].apply(_violence_label)
    else:
        v_counts = pd.DataFrame(columns=["violence", "count", "violence_label"])

    if {"alcaldia_std", "crime_classification"}.issubset(df.columns):
        stacked = df.groupby(["alcaldia_std", "crime_classification"]).size().reset_index(name="count")
        stacked["crime_classification"] = stacked["crime_classification"].astype("string")
        stacked = stacked[
            ~stacked["crime_classification"].str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
    else:
        stacked = pd.DataFrame(columns=["alcaldia_std", "crime_classification", "count"])

    if {"anio_inicio", "alcaldia_std"}.issubset(df.columns):
        yearly_mun = (
            df.dropna(subset=["anio_inicio"])
              .groupby(["anio_inicio", "alcaldia_std"]).size().reset_index(name="count")
        )
        totals = yearly_mun.groupby("alcaldia_std")["count"].sum().sort_values(ascending=False)
        top8 = totals.head(8).index.tolist()
        yearly_mun_top = yearly_mun[yearly_mun["alcaldia_std"].isin(top8)].copy()
        yearly_mun_top["anio_inicio"] = pd.Categorical(
            yearly_mun_top["anio_inicio"], categories=range(2015, 2026), ordered=True
        )
    else:
        top8, yearly_mun_top = [], pd.DataFrame(columns=["anio_inicio", "alcaldia_std", "count"])

    if {"anio_inicio", "crime_classification"}.issubset(df.columns):
        year_class = (
            df.dropna(subset=["anio_inicio"])
              .groupby(["anio_inicio", "crime_classification"]).size().reset_index(name="count")
        )
        year_class["anio_inicio"] = pd.Categorical(
            year_class["anio_inicio"], categories=range(2015, 2026), ordered=True
        )
        year_class["crime_classification"] = year_class["crime_classification"].astype("string")
        year_class = year_class[
            ~year_class["crime_classification"].str.lower().isin(
                {"nan", "none", "delito de bajo impacto", "delitos de bajo impacto", "bajo impacto"}
            )
        ]
    else:
        year_class = pd.DataFrame(columns=["anio_inicio", "crime_classification", "count"])

    if {"anio_inicio", "violence_type"}.issubset(df.columns):
        yearly_viol = (
            df.dropna(subset=["anio_inicio"])
              .groupby(["anio_inicio", "violence_type"]).size().reset_index(name="count")
        )
        yearly_viol["anio_inicio"] = pd.Categorical(
            yearly_viol["anio_inicio"], categories=range(2015, 2026), ordered=True
        )
    else:
        yearly_viol = pd.DataFrame(columns=["anio_inicio", "violence_type", "count"])

    return {
        "year_counts": year_counts,
        "month_counts": month_counts,
        "hour_counts": hour_counts,
        "alc_counts": alc_counts,
        "col_counts": col_counts,
        "del_counts": del_counts,
        "cls_counts": cls_counts,
        "v_counts": v_counts,
        "stacked": stacked,
        "yearly_mun_top": yearly_mun_top,
        "year_class": year_class,
        "yearly_viol": yearly_viol,
        "top8": top8,
    }

# =================== CHART PRIMS ===================
def bar_chart(df, x, y, *, x_type="O", y_type="Q", sort=None, title=None, color=None, width=700, height=350, x_axis=None, y_axis=None):
    chart = alt.Chart(df, **_cfg(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", sort=sort, axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[_tooltip(x, x_type), _tooltip(y, y_type)],
    )
    return chart.properties(title=title) if title else chart

def barh_chart(df, y, x, *, y_type="N", x_type="Q", sort=None, title=None, color=None, width=700, height=350, x_axis=None, y_axis=None):
    chart = alt.Chart(df, **_cfg(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", sort=sort, axis=y_axis),
        tooltip=[_tooltip(y, y_type), _tooltip(x, x_type)],
    )
    return chart.properties(title=title) if title else chart

def line_chart(df, x, y, color_field=None, *, x_type="O", y_type="Q", title=None, width=700, height=350, x_axis=None, y_axis=None, point=False):
    mk = alt.Chart(df, **_cfg(width, height)).mark_line(point=point, strokeWidth=2).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[_tooltip(x, x_type), _tooltip(y, y_type)],
    )
    if color_field and color_field in df.columns and not df.empty:
        uniq = max(1, df[color_field].nunique())
        mk = mk.encode(
            color=alt.Color(
                f"{color_field}:N",
                scale=alt.Scale(range=THEME_PALETTE[:uniq]),
            )
        )
    return mk.properties(title=title) if title else mk

def donut_chart(
    df,
    field,
    count_col,
    *,
    title="",
    colors=None,
    width=700,
    height=360,
    inner_radius=70,
    legend_title=None,
):
    df_chart = df.copy() if isinstance(df, pd.DataFrame) else df
    percent_col = "__percent"
    percent_label_col = "__percent_label"

    if isinstance(df_chart, pd.DataFrame) and not df_chart.empty:
        total = df_chart[count_col].sum()
        if total and total != 0:
            df_chart[percent_col] = (df_chart[count_col] / total) * 100
            df_chart[percent_label_col] = df_chart[percent_col].map(lambda v: f"{v:.1f}%")

    n = max(3, len(df_chart)) if isinstance(df_chart, pd.DataFrame) else 3
    colors = colors or THEME_PALETTE[:n]

    base = alt.Chart(df_chart, **_cfg(width, height))
    arc = base.mark_arc(innerRadius=inner_radius).encode(
        theta=alt.Theta(f"{count_col}:Q"),
        color=alt.Color(
            f"{field}:N",
            scale=alt.Scale(range=colors),
            legend=alt.Legend(title=legend_title) if legend_title is not None else None,
        ),
        tooltip=[
            _tooltip(field, "N", title="Categoría"),
            _tooltip(count_col, "Q", title="Conteo"),
        ],
    )
    if title:
        arc = arc.properties(title=title)

    if isinstance(df_chart, pd.DataFrame) and percent_label_col in df_chart.columns:
        text = base.mark_text(
            radius=inner_radius + 40,
            color="#F8FAFC",
            fontSize=14,
            fontWeight="bold",
        ).encode(
            theta=alt.Theta(f"{count_col}:Q"),
            text=alt.Text(f"{percent_label_col}:N"),
        )
        return arc + text
    return arc

def heatmap_chart(df, x, y, z, *, title="", width=700, height=350, x_axis=None, y_axis=None):
    chart = alt.Chart(df, **_cfg(width, height)).mark_rect().encode(
        x=alt.X(f"{x}:N", axis=x_axis),
        y=alt.Y(f"{y}:N", axis=y_axis),
        color=alt.Color(f"{z}:Q", scale=alt.Scale(scheme="blues")),
        tooltip=[_tooltip(x, "N"), _tooltip(y, "N"), _tooltip(z, "Q")],
    )
    return chart.properties(title=title) if title else chart

def section_title(text: str):
    st.markdown(
        f"<h2 style='font-size:32px; color:#E5E7EB; margin:0.5rem 0 0.1rem;'>{text}</h2>",
        unsafe_allow_html=True,
    )

@contextmanager
def chart_block(title: str):
    """Helper container that injects a header with an inline filter slot."""
    with st.container():
        header_cols = st.columns([4, 1])
        header_cols[0].markdown(f"#### {title}")
        yield header_cols[1]

# =================== RENDER ===================
def render():
    st.title("Análisis Exploratorio de Datos (AED)")

    # Carga TODO (sin límite por defecto)
    base_df = load_info_df(year_min=2015, limit=None)
    if base_df.empty:
        st.warning("No hay datos para mostrar en Info.")
        return

    # =================== SIDEBAR FILTERS (globales) ===================
    with st.sidebar:
        st.markdown("### Filtros Globales")

        # Rango de años dinámico
        y_min, y_max = int(base_df["anio_inicio"].min()), int(base_df["anio_inicio"].max())
        year_range = st.slider(
            "Años",
            min_value=y_min,
            max_value=y_max,
            value=(max(y_min, 2016), y_max),
            step=1,
        )

        # Meses (multi)
        month_options = [MONTH_NAMES[m] for m in range(1, 13)]
        months_sel = st.multiselect("Meses", options=month_options, default=month_options)

        # Rango de horas
        hour_has_col = "hour_hecho" in base_df.columns
        if hour_has_col:
            hour_range = st.slider("Hora del día", 0, 23, (0, 23), step=1)
        else:
            hour_range = (0, 23)

        # Alcaldías / Clasificación / Violencia / Delitos
        alcs = _safe_uniques(base_df["alcaldia_std"]) if "alcaldia_std" in base_df else []
        cls  = _safe_uniques(base_df["crime_classification"]) if "crime_classification" in base_df else []
        viol = _safe_uniques(base_df["violence_type"]) if "violence_type" in base_df else []
        dels = _safe_uniques(base_df["delito_std"]) if "delito_std" in base_df else []

        alc_sel = st.multiselect("Alcaldías", options=alcs, default=alcs[:])
        cls_sel = st.multiselect(
            "Clasificación",
            options=cls,
            default=cls[:],
            format_func=_classification_label,
        )
        viol_sel = st.multiselect(
            "Violencia",
            options=viol,
            default=viol[:],
            format_func=_violence_label,
        )

        # Para no saturar, top N delitos por frecuencia como opciones por defecto
        if dels:
            top_del = (
                pd.Series(base_df["delito_std"], dtype="object")
                .value_counts()
                .head(30)
                .index
                .tolist()
            )
            top_del = [
                d
                for d in top_del
                if str(d).strip().lower()
                not in {
                    "nan",
                    "none",
                    "delito de bajo impacto",
                    "delitos de bajo impacto",
                    "bajo impacto",
                }
            ]
            delito_default = top_del
        else:
            delito_default = []
        delito_sel = st.multiselect(
            "Delitos (30 principales por defecto)", options=dels, default=delito_default
        )

        if st.button("🔄 Reset filtros"):
            st.experimental_rerun()

    # =================== APPLY FILTERS ===================
    df = base_df.copy()

    # Filtro robusto de delitos de bajo impacto
    if "delito_std" in df.columns:
        df["delito_std"] = df["delito_std"].astype("string")
        mask_bajo = df["delito_std"].str.lower().str.contains("bajo impacto", na=False)
        df = df[~mask_bajo]

    df = df[(df["anio_inicio"] >= year_range[0]) & (df["anio_inicio"] <= year_range[1])]

    months_num = [MONTH_NUM.get(mn, None) for mn in months_sel]
    months_num = [m for m in months_num if m is not None]
    if months_num:
        df = df[df["mes_inicio"].isin(months_num)]

    if hour_has_col and hour_range != (0, 23):
        h0, h1 = hour_range
        df = df[(df["hour_hecho"] >= h0) & (df["hour_hecho"] <= h1)]

    if "alcaldia_std" in df.columns and alc_sel:
        df = df[df["alcaldia_std"].isin(alc_sel)]

    if "crime_classification" in df.columns and cls_sel:
        df = df[df["crime_classification"].isin(cls_sel)]

    if "violence_type" in df.columns and viol_sel:
        df = df[df["violence_type"].isin(viol_sel)]

    if "delito_std" in df.columns and delito_sel:
        df = df[df["delito_std"].isin(delito_sel)]

    # =================== AGG & METRICS ===================
    agg = compute_eda_aggregates(df)
    df_metrics = get_violence_time_metrics_cached()

    # ====== CONTEXTO + KPIs ======
    st.header("Contexto del Problema")
    st.markdown(
        """
    La Ciudad de México enfrenta desafíos significativos en materia de seguridad.
    Este estudio analiza incidentes de robo dividiéndolos en **violentos** y **no violentos**
    para identificar patrones temporales y espaciales.
    """
    )
    hypothesis_card = """
    <div style="
        background: rgba(30, 58, 138, 0.25);
        border: 1px solid rgba(59, 130, 246, 0.5);
        border-radius: 16px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    ">
        <div style="font-size: 1.1rem; font-weight: 700; color: #E0E7FF; margin-bottom: 0.5rem;">
            Nuestra hipótesis
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 1rem;">
            <div style="
                flex: 1;
                min-width: 180px;
                background: rgba(30, 64, 175, 0.5);
                border-radius: 12px;
                padding: 0.85rem;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: #BFDBFE;">80%</div>
                <div style="color: #E5E7EB;">Delitos violentos durante la noche</div>
            </div>
            <div style="
                flex: 1;
                min-width: 180px;
                background: rgba(249, 158, 11, 0.35);
                border-radius: 12px;
                padding: 0.85rem;
            ">
                <div style="font-size: 2rem; font-weight: 700; color: #FDE68A;">80%</div>
                <div style="color: #E5E7EB;">Delitos no violentos durante el día</div>
            </div>
        </div>
        <p style="color: #E2E8F0; margin-top: 0.75rem; margin-bottom: 0;">
            Planteamos que los crímenes violentos concentran hasta el 80% de los incidentes nocturnos,
            mientras que los delitos no violentos alcanzan una proporción similar durante el día.
        </p>
    </div>
    """
    st.markdown(hypothesis_card, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Incidentes filtrados", f"{len(df):,}")
    c2.metric("Años seleccionados", f"{year_range[0]}–{year_range[1]}")
    c3.metric("Alcaldías activas", str(len(alc_sel) if alc_sel else 0))

    st.subheader("Resultados del Análisis de Horarios (Violencia)")
    if not df_metrics.empty:
        df_metrics = df_metrics.rename(
            columns={
                "franja_horaria": "Periodo",
                "porcentaje": "Porcentaje",
                "total": "Total",
            }
        )
        df_metrics_display = df_metrics.copy()

        df_metrics_display["Periodo"] = df_metrics_display["Periodo"].apply(
            _format_period_label
        )
        if "Total" in df_metrics_display.columns:
            df_metrics_display["Total"] = df_metrics_display["Total"].apply(
                lambda v: f"{int(round(v)):,}" if pd.notna(v) else ""
            )
        if "Porcentaje" in df_metrics_display.columns:
            df_metrics_display["Porcentaje"] = df_metrics_display["Porcentaje"].apply(
                lambda v: f"{float(v):.1f}%" if pd.notna(v) else ""
            )

        st.dataframe(df_metrics_display, use_container_width=True, hide_index=True)

        try:
            noche_pct = float(
                df_metrics[df_metrics["Periodo"] == "Noche (19-07)"]["Porcentaje"].iloc[0]
            )
            dia_pct = float(
                df_metrics[df_metrics["Periodo"] == "Día (07-19)"]["Porcentaje"].iloc[0]
            )

            st.markdown("#### Distribución de crímenes violentos por horario")

            col1, col2 = st.columns(2)

            # Gauge Noche
            with col1:
                fig_noche = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=noche_pct,
                        number={"suffix": "%"},
                        title={
                            "text": "Crímenes Violentos de Noche<br><span style='font-size:0.8em'>(19:00–07:00)</span>"
                        },
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#1E3A8A"},
                            "steps": [
                                {"range": [0, 33], "color": "#E5E7EB"},
                                {"range": [33, 66], "color": "#93C5FD"},
                                {"range": [66, 100], "color": "#1D4ED8"},
                            ],
                        },
                    )
                )
                fig_noche.update_layout(
                    margin=dict(l=20, r=20, t=60, b=10),
                    height=280,
                )
                st.plotly_chart(fig_noche, use_container_width=True)

            # Gauge Día
            with col2:
                fig_dia = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=dia_pct,
                        number={"suffix": "%"},
                        title={
                            "text": "Crímenes Violentos de Día<br><span style='font-size:0.8em'>(07:00–19:00)</span>"
                        },
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#F59E0B"},
                            "steps": [
                                {"range": [0, 33], "color": "#FEF3C7"},
                                {"range": [33, 66], "color": "#FCD34D"},
                                {"range": [66, 100], "color": "#D97706"},
                            ],
                        },
                    )
                )
                fig_dia.update_layout(
                    margin=dict(l=20, r=20, t=60, b=10),
                    height=280,
                )
                st.plotly_chart(fig_dia, use_container_width=True)

        except IndexError:
            st.warning("No se pudieron calcular las métricas de día/noche.")
    else:
        st.warning("No se pudieron cargar las métricas de violencia.")

    st.info(
        "Conclusión: la hipótesis no se sostiene totalmente, aunque la noche sigue siendo significativa."
    )

    # === TEMPORAL DISTRIBUTIONS ===
    section_title("Temporal")
    st.caption("Distribuciones temporales")

    year_df = agg["year_counts"]
    if year_df.empty:
        st.info("Sin datos para el gráfico anual.")
    else:
        with chart_block("Crímenes por Año") as filter_col:
            y_min, y_max = int(year_df["year"].min()), int(year_df["year"].max())
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    year_window = st.slider(
                        "Periodo a mostrar",
                        min_value=y_min,
                        max_value=y_max,
                        value=(y_min, y_max),
                        step=1,
                        key="year_chart_slider",
                    )
            yr_filtered = year_df[
                (year_df["year"] >= year_window[0])
                & (year_df["year"] <= year_window[1])
            ]
            st.altair_chart(
                bar_chart(
                    yr_filtered,
                    "year",
                    "count",
                    sort=list(range(2016, 2025)),
                    color=THEME_PALETTE[0],
                    x_axis=_axis("Año"),
                    y_axis=_axis("Número de casos"),
                ),
                use_container_width=True,
            )

    month_df = agg["month_counts"]
    if month_df.empty:
        st.info("Sin datos mensuales para graficar.")
    else:
        with chart_block("Crímenes por Mes") as filter_col:
            month_options = month_df["month_name"].tolist()
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    months_selected = st.multiselect(
                        "Meses a incluir",
                        options=month_options,
                        default=month_options,
                        key="month_chart_filter",
                    )
            selected_months = months_selected or month_options
            month_filtered = month_df[month_df["month_name"].isin(selected_months)]
            month_filtered = month_filtered.sort_values("month")
            st.altair_chart(
                bar_chart(
                    month_filtered,
                    "month_name",
                    "count",
                    sort=[MONTH_NAMES[m] for m in range(1, 13)],
                    color=THEME_PALETTE[2],
                    x_axis=_axis("Mes"),
                    y_axis=_axis("Número de casos"),
                ),
                use_container_width=True,
            )

    hour_df = agg["hour_counts"]
    if hour_df.empty:
        st.info("Sin datos horarios.")
    else:
        with chart_block("Distribución por Hora del Día") as filter_col:
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    hour_window = st.slider(
                        "Horas",
                        min_value=0,
                        max_value=23,
                        value=(0, 23),
                        step=1,
                        key="hour_chart_slider",
                    )
            hr_filtered = hour_df[
                (hour_df["hour"] >= hour_window[0])
                & (hour_df["hour"] <= hour_window[1])
            ]
            st.altair_chart(
                bar_chart(
                    hr_filtered,
                    "hour",
                    "count",
                    sort=list(range(24)),
                    color=THEME_PALETTE[1],
                    x_axis=_axis("Hora del día"),
                    y_axis=_axis("Número de casos"),
                    width=980,
                ),
                use_container_width=True,
            )

    # === SPATIAL & CATEGORY DISTRIBUTIONS ===
    section_title("Espacial & Categorías")
    st.caption("División espacial y categorías")

    alc_df = agg["alc_counts"]
    if alc_df.empty:
        st.info("No hay registros por alcaldía.")
    else:
        with chart_block("Distribución por Alcaldía") as filter_col:
            alc_options = alc_df["alcaldia"].tolist()
            default_alc = alc_options if len(alc_options) <= 8 else alc_options[:8]
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    alc_selected = st.multiselect(
                        "Selecciona alcaldías",
                        options=alc_options,
                        default=default_alc,
                        key="alc_chart_filter",
                    )
            chosen_alc = alc_selected or alc_options
            alc_filtered = alc_df[alc_df["alcaldia"].isin(chosen_alc)]
            alc_filtered = alc_filtered.sort_values("count", ascending=True)
            alc_height = _auto_height(len(alc_filtered), base=280, per_item=40)
            st.altair_chart(
                barh_chart(
                    alc_filtered,
                    "alcaldia",
                    "count",
                    color=THEME_PALETTE[3],
                    x_axis=_axis("Número de casos"),
                    y_axis=_axis("Alcaldía", orient="left"),
                    height=alc_height,
                ),
                use_container_width=True,
            )

    col_df = agg["col_counts"]
    if col_df.empty:
        st.info("Sin datos de colonias disponibles.")
    else:
        with chart_block("Colonias con más incidentes") as filter_col:
            max_top = max(1, min(30, len(col_df)))
            min_top = 1 if max_top < 5 else 5
            default_top = min(10, max_top)
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    top_col = st.slider(
                        "Cantidad de colonias",
                        min_value=min_top,
                        max_value=max_top,
                        value=default_top,
                        step=1,
                        key="colonias_top_slider",
                    )
            cols_filtered = col_df.head(top_col).sort_values("count", ascending=True)
            col_height = _auto_height(len(cols_filtered), base=280, per_item=40)
            st.altair_chart(
                barh_chart(
                    cols_filtered,
                    "colonia",
                    "count",
                    color=THEME_PALETTE[4],
                    x_axis=_axis("Número de casos"),
                    y_axis=_axis("Colonia", orient="left"),
                    height=col_height,
                ),
                use_container_width=True,
            )

    del_df = agg["del_counts"]
    if del_df.empty:
        st.info("Sin datos de delitos para graficar.")
    else:
        with chart_block("Delitos con mayor frecuencia") as filter_col:
            max_top = max(1, min(30, len(del_df)))
            min_top = 1 if max_top < 5 else 5
            default_top = min(10, max_top)
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    top_del = st.slider(
                        "Cantidad de delitos",
                        min_value=min_top,
                        max_value=max_top,
                        value=default_top,
                        step=1,
                        key="delitos_top_slider",
                    )
            delitos_filtered = del_df.head(top_del).sort_values("count", ascending=True)
            del_height = _auto_height(len(delitos_filtered), base=280, per_item=40)
            st.altair_chart(
                barh_chart(
                    delitos_filtered,
                    "delito",
                    "count",
                    color=THEME_PALETTE[5],
                    x_axis=_axis("Número de casos"),
                    y_axis=_axis("Tipo de delito", orient="left"),
                    height=del_height,
                ),
                use_container_width=True,
            )

    # === CLASSIFICATION & VIOLENCE ===
    section_title("Clasificación & Violencia")
    st.caption("Clasificación y violencia")

    cls_df = agg["cls_counts"]
    if cls_df.empty:
        st.info("Sin datos de clasificación para mostrar.")
    else:
        with chart_block("Distribución por Clasificación") as filter_col:
            cls_options = cls_df["classification"].dropna().tolist()
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    cls_selected = st.multiselect(
                        "Clasificaciones",
                        options=cls_options,
                        default=cls_options,
                        format_func=_classification_label,
                        key="classification_chart_filter",
                    )
            chosen_cls = cls_selected or cls_options
            cls_filtered = cls_df[cls_df["classification"].isin(chosen_cls)].copy()
            cls_filtered["classification_label"] = cls_filtered["classification"].apply(_classification_label)
            total_cls = cls_filtered["count"].sum()
            if total_cls > 0:
                cls_filtered["pct"] = (cls_filtered["count"] / total_cls) * 100
                cls_filtered["pct_label"] = cls_filtered["pct"].map(lambda v: f"{v:.1f}%")
            else:
                cls_filtered["pct"] = 0
                cls_filtered["pct_label"] = "0%"
            cls_filtered = cls_filtered.sort_values("pct", ascending=True)
            n_cls = max(1, len(cls_filtered))
            chart_height = max(240, n_cls * 50)
            base = alt.Chart(cls_filtered, **_cfg(height=chart_height))
            bars = base.mark_bar(cornerRadius=6).encode(
                x=alt.X("pct:Q", axis=_axis("Porcentaje (%)")),
                y=alt.Y(
                    "classification_label:N",
                    sort="-x",
                    axis=_axis("Clasificación", orient="left"),
                ),
                color=alt.Color(
                    "classification_label:N",
                    scale=alt.Scale(range=THEME_PALETTE[:n_cls]),
                    legend=None,
                ),
                tooltip=[
                    _tooltip("classification_label", "N", title="Clasificación"),
                    _tooltip("count", "Q", title="Casos"),
                    _tooltip("pct", "Q", title="Porcentaje", fmt=".1f"),
                ],
            )
            labels = base.mark_text(
                align="left",
                baseline="middle",
                dx=5,
                color="#F8FAFC",
                fontSize=13,
                fontWeight="bold",
            ).encode(
                x=alt.X("pct:Q"),
                y=alt.Y("classification_label:N", sort="-x"),
                text=alt.Text("pct_label:N"),
            )
            st.altair_chart(bars + labels, use_container_width=True)

    v_counts = agg["v_counts"]
    if v_counts.empty:
        st.info("No se encontró columna 'violence_type'.")
    else:
        with chart_block("Violento vs No Violento") as filter_col:
            label_field = "violence_label" if "violence_label" in v_counts.columns else "violence"
            viol_options = v_counts[label_field].tolist()
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    viol_selected = st.multiselect(
                        "Tipo de violencia",
                        options=viol_options,
                        default=viol_options,
                        key="violence_chart_filter",
                    )
            chosen_violence = viol_selected or viol_options
            viol_filtered = v_counts[v_counts[label_field].isin(chosen_violence)]
            if viol_filtered.empty:
                st.info("No hay datos para el filtro seleccionado.")
            else:
                viol_filtered = viol_filtered.copy()
                total_viol = viol_filtered["count"].sum()
                if total_viol > 0:
                    viol_filtered["pct"] = (viol_filtered["count"] / total_viol) * 100
                    viol_filtered["pct_label"] = viol_filtered["pct"].map(lambda v: f"{v:.1f}%")
                else:
                    viol_filtered["pct"] = 0
                    viol_filtered["pct_label"] = "0%"
                viol_filtered = viol_filtered.sort_values("pct", ascending=True)
                colors = (
                    ["#1E3A8A", "#F59E0B"]
                    if set(v_counts.get("violence", [])) >= {"Violent", "Non-Violent"}
                    else THEME_PALETTE[: max(1, len(viol_filtered))]
                )
                base = alt.Chart(viol_filtered, **_cfg(height=220))
                bars = base.mark_bar(cornerRadius=10).encode(
                    x=alt.X("pct:Q", axis=_axis("Porcentaje (%)")),
                    y=alt.Y(
                        f"{label_field}:N",
                        sort="-x",
                        axis=_axis("Tipo de violencia", orient="left"),
                    ),
                    color=alt.Color(
                        f"{label_field}:N",
                        scale=alt.Scale(range=colors),
                        legend=None,
                    ),
                    tooltip=[
                        _tooltip(label_field, "N", title="Tipo"),
                        _tooltip("count", "Q", title="Casos"),
                        _tooltip("pct", "Q", title="Porcentaje", fmt=".1f"),
                    ],
                )
                labels = base.mark_text(
                    align="left",
                    baseline="middle",
                    dx=5,
                    color="#F8FAFC",
                    fontSize=14,
                    fontWeight="bold",
                ).encode(
                    x=alt.X("pct:Q"),
                    y=alt.Y(f"{label_field}:N", sort="-x"),
                    text=alt.Text("pct_label:N"),
                )
                st.altair_chart(bars + labels, use_container_width=True)

    stacked = agg["stacked"]
    if stacked.empty:
        st.info("Sin datos para el stacked por alcaldía y clasificación.")
    else:
        with chart_block("Alcaldía vs Clasificación de Delito (apilado)") as filter_col:
            order_mun = (
                stacked.groupby("alcaldia_std")["count"]
                .sum()
                .sort_values(ascending=True)
                .index
                .tolist()
            )
            cls_options = (
                stacked["crime_classification"].dropna().unique().tolist()
                if "crime_classification" in stacked.columns
                else []
            )
            with filter_col:
                with st.expander("Filtros", expanded=False):
                    colA, colB = st.columns(2)
                    alc_selected = colA.multiselect(
                        "Alcaldías",
                        options=order_mun,
                        default=order_mun,
                        key="stacked_alc_filter",
                    )
                    cls_selected = colB.multiselect(
                        "Clasificación",
                        options=cls_options,
                        default=cls_options,
                        format_func=_classification_label,
                        key="stacked_cls_filter",
                    )
            chosen_alc = alc_selected or order_mun
            chosen_cls = cls_selected or cls_options
            stacked_filtered = stacked[
                stacked["alcaldia_std"].isin(chosen_alc)
                & stacked["crime_classification"].isin(chosen_cls)
            ]
            if stacked_filtered.empty:
                st.info("Selecciona al menos una alcaldía y una clasificación.")
            else:
                stacked_filtered = stacked_filtered.copy()
                stacked_filtered["cls_label"] = stacked_filtered["crime_classification"].apply(
                    _classification_label
                )
                alc_order = [
                    alc for alc in order_mun if alc in stacked_filtered["alcaldia_std"].unique()
                ]
                chart_height = _auto_height(len(alc_order), base=360, per_item=32)
                ch = (
                    alt.Chart(stacked_filtered, **_cfg(width=950, height=chart_height))
                    .mark_bar()
                    .encode(
                        x=alt.X("count:Q", axis=_axis("Número de casos")),
                        y=alt.Y(
                            "alcaldia_std:N",
                            sort=alc_order,
                            axis=_axis("Alcaldía", orient="left"),
                        ),
                        color=alt.Color(
                            "cls_label:N",
                            scale=alt.Scale(
                                range=THEME_PALETTE[
                                    : max(3, stacked_filtered["cls_label"].nunique())
                                ]
                            ),
                            legend=alt.Legend(title="Clasificación"),
                        ),
                        tooltip=[
                            _tooltip("alcaldia_std", "N", title="Alcaldía"),
                            _tooltip("cls_label", "N", title="Clasificación"),
                            _tooltip("count", "Q", title="Casos"),
                        ],
                    )
                )
                st.altair_chart(ch, use_container_width=True)

  

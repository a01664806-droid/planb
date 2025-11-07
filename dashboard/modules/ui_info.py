# modules/ui_info.py
import streamlit as st
import pandas as pd
import altair as alt

from .data import load_info_csv
from .helpers import MONTH_NAMES

# ---- Tema global transparente (top-level) ----
def _enable_transparent_theme():
    theme_conf = {"config": {"background": "transparent", "view": {"fill": "transparent"}}}
    alt.themes.register("transp_dark", lambda: theme_conf)
    alt.themes.enable("transp_dark")
_enable_transparent_theme()

# Paleta
THEME_PALETTE = [
    "#1E3A8A", "#3B82F6", "#60A5FA", "#93C5FD", "#A3B3C2",
    "#64748B", "#0EA5E9", "#14B8A6", "#F59E0B", "#EF4444",
]

# ---- helpers UI ----
def _pad_dict(v=5):
    v = int(v) if isinstance(v, (int, float)) else 5
    return {"left": v, "right": v, "top": v, "bottom": v}

def _cfg(width=700, height=350):
    # Para charts SIN layer
    return {"width": width, "height": height, "padding": _pad_dict(5)}

def _cfg_sub(width=700, height=350):
    # Para sub-charts dentro de un layer (NO padding aquí)
    return {"width": width, "height": height}

def _axis(title, orient=None):
    base = {
        "title": title,
        "grid": True,
        "gridColor": "#33415555",
        "labelColor": "#E5E7EB",
        "titleColor": "#E5E7EB",
        "tickColor": "#94A3B855",
    }
    if orient:
        base["orient"] = orient
    return base

def _text_label_dict(dy=-8):
    return dict(align="center", baseline="bottom", dy=dy, fontWeight="bold", color="#E5E7EB")

# ---- chart builders (sin padding en subcapas) ----
def bar_with_labels(df, x, y, *, x_type="O", y_type="Q", sort=None, title="", color=None, width=700, height=350, x_axis=None, y_axis=None):
    base = alt.Chart(df, **_cfg_sub(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", sort=sort, axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[f"{x}:{x_type}", f"{y}:{y_type}"]
    ).properties(title=title)
    labels = alt.Chart(df, **_cfg_sub(width, height)).mark_text(**_text_label_dict()).encode(
        x=alt.X(f"{x}:{x_type}", sort=sort),
        y=alt.Y(f"{y}:{y_type}"),
        text=alt.Text(f"{y}:{y_type}")
    )
    return (base + labels).properties(padding=_pad_dict(5))

def barh_with_labels(df, y, x, *, y_type="N", x_type="Q", sort=None, title="", color=None, width=700, height=350, x_axis=None, y_axis=None):
    base = alt.Chart(df, **_cfg_sub(width, height)).mark_bar(color=color).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", sort=sort, axis=y_axis),
        tooltip=[f"{y}:{y_type}", f"{x}:{x_type}"]
    ).properties(title=title)
    labels = alt.Chart(df, **_cfg_sub(width, height)).mark_text(align="left", dx=5, color="#E5E7EB", fontWeight="bold").encode(
        x=alt.X(f"{x}:{x_type}"),
        y=alt.Y(f"{y}:{y_type}", sort=sort),
        text=alt.Text(f"{x}:{x_type}")
    )
    return (base + labels).properties(padding=_pad_dict(5))

def line_with_labels(df, x, y, color_field=None, *, x_type="O", y_type="Q", title="", width=700, height=350, x_axis=None, y_axis=None, point=True, label_dy=-10):
    mk = alt.Chart(df, **_cfg_sub(width, height)).mark_line(point=point).encode(
        x=alt.X(f"{x}:{x_type}", axis=x_axis),
        y=alt.Y(f"{y}:{y_type}", axis=y_axis),
        tooltip=[f"{x}:{x_type}", f"{y}:{y_type}"]
    )
    if color_field:
        mk = mk.encode(color=alt.Color(f"{color_field}:N", scale=alt.Scale(range=THEME_PALETTE[:df[color_field].nunique()])))
    mk = mk.properties(title=title)
    labels = alt.Chart(df, **_cfg_sub(width, height)).mark_text(**_text_label_dict(dy=label_dy)).encode(
        x=alt.X(f"{x}:{x_type}"),
        y=alt.Y(f"{y}:{y_type}"),
        text=alt.Text(f"{y}:{y_type}")
    )
    if color_field:
        labels = labels.encode(color=alt.Color(f"{color_field}:N", legend=None))
    return (mk + labels).properties(padding=_pad_dict(5))

def donut_chart(df, field, count_col, *, title="", colors=None, width=700, height=360, inner_radius=70):
    colors = colors or THEME_PALETTE[:max(3, len(df))]
    donut = alt.Chart(df, **_cfg_sub(width, height)).mark_arc(innerRadius=inner_radius).encode(
        theta=alt.Theta(f"{count_col}:Q"),
        color=alt.Color(f"{field}:N", scale=alt.Scale(range=colors)),
        tooltip=[alt.Tooltip(f"{field}:N", title="Category"),
                 alt.Tooltip(f"{count_col}:Q", title="Count")]
    ).properties(title=title)
    return donut

def center_text_chart(text, *, width=700, height=360):
    return alt.Chart(pd.DataFrame({"text":[text]}), **_cfg_sub(width, height)).mark_text(
        align="center", fontSize=16, fontWeight="bold", color="#E5E7EB"
    ).encode(text="text:N")

# ---------------------- RENDER ----------------------
def render():
    st.title("ℹ️ Exploratory Data Analysis (EDA)")
    st.caption("Aquí se usa **carpetasFGJ_acumulado_2025_01.csv** del folder actual.")

    df = load_info_csv()
    if df.empty:
        st.warning("No hay datos para mostrar en Info.")
        return

    # RANGOS
    YEAR_RANGE  = list(range(2015, 2025))
    MONTH_RANGE = list(range(1, 13))
    HOUR_RANGE  = list(range(0, 24))

    # Coerciones
    df["anio_inicio"] = pd.to_numeric(df["anio_inicio"], errors="coerce")
    df["mes_inicio"]  = pd.to_numeric(df["mes_inicio"],  errors="coerce")
    if "hour_hecho" in df.columns:
        df["hour_hecho"] = pd.to_numeric(df["hour_hecho"], errors="coerce")

    # Agregados
    year_counts = (
        df.dropna(subset=["anio_inicio"])
          .groupby("anio_inicio").size()
          .reindex(YEAR_RANGE, fill_value=0)
          .reset_index(name="count")
          .rename(columns={"anio_inicio":"year"})
    )

    month_counts = (
        df.dropna(subset=["mes_inicio"])
          .groupby("mes_inicio").size()
          .reindex(MONTH_RANGE, fill_value=0)
          .reset_index(name="count")
          .rename(columns={"mes_inicio":"month"})
    )
    month_counts["month_name"] = month_counts["month"].map(lambda m: MONTH_NAMES.get(int(m), str(m)))

    if "hour_hecho" in df.columns:
        hour_counts = (
            df.dropna(subset=["hour_hecho"])
              .groupby("hour_hecho").size()
              .reindex(HOUR_RANGE, fill_value=0)
              .reset_index(name="count")
              .rename(columns={"hour_hecho":"hour"})
        )
    else:
        hour_counts = pd.DataFrame({"hour": HOUR_RANGE, "count": 0})

    # ===================== Univariate =====================
    with st.expander("📈 Univariate: Years, Months, Hours", expanded=True):
        c1, c2 = st.columns(2)

        # Crimes by Year
        with c1:
            ch = bar_with_labels(
                year_counts, x="year", y="count",
                x_type="O", y_type="Q",
                sort=YEAR_RANGE,
                title="Crimes by Year",
                color=THEME_PALETTE[0],
                width=700, height=350,
                x_axis=_axis("Year"),
                y_axis=_axis("Number of Cases"),
            )
            st.altair_chart(ch, use_container_width=True)

        # Crimes by Month
        with c2:
            ch = bar_with_labels(
                month_counts, x="month_name", y="count",
                x_type="N", y_type="Q",
                sort=[MONTH_NAMES[m] for m in MONTH_RANGE],
                title="Crimes by Month",
                color=THEME_PALETTE[2],
                width=700, height=350,
                x_axis=_axis("Month"),
                y_axis=_axis("Number of Cases"),
            )
            st.altair_chart(ch, use_container_width=True)

        # By Hour
        ch = bar_with_labels(
            hour_counts, x="hour", y="count",
            x_type="O", y_type="Q",
            sort=HOUR_RANGE,
            title="Distribution of Crimes by Hour of Day",
            color=THEME_PALETTE[1],
            width=980, height=350,
            x_axis=_axis("Hour of Day"),
            y_axis=_axis("Number of Cases"),
        )
        st.altair_chart(ch, use_container_width=True)

    # ===================== Spatial & Categories =====================
    with st.expander("🗺️ Spatial & Categories", expanded=False):
        c1, c2 = st.columns(2)

        # District (horizontal)
        with c1:
            alc_counts = df["alcaldia_std"].value_counts().sort_values(ascending=True).reset_index()
            alc_counts.columns = ["alcaldia","count"]
            ch = barh_with_labels(
                alc_counts, y="alcaldia", x="count",
                y_type="N", x_type="Q",
                sort=alc_counts["alcaldia"].tolist(),
                title="Distribution of Crimes by District",
                color=THEME_PALETTE[3],
                width=700, height=350,
                x_axis=_axis("Number of Cases"),
                y_axis=_axis("District", orient="left"),
            )
            st.altair_chart(ch, use_container_width=True)

        # Top 10 Neighborhoods
        with c2:
            col_counts = df["colonia_std"].value_counts().nlargest(10).sort_values(ascending=True).reset_index()
            col_counts.columns = ["colonia","count"]
            ch = barh_with_labels(
                col_counts, y="colonia", x="count",
                y_type="N", x_type="Q",
                sort=col_counts["colonia"].tolist(),
                title="Top 10 Neighborhoods by Crime Count",
                color=THEME_PALETTE[4],
                width=700, height=350,
                x_axis=_axis("Number of Cases"),
                y_axis=_axis("Neighborhood", orient="left"),
            )
            st.altair_chart(ch, use_container_width=True)

        # Top 10 Crimes
        del_counts = df["delito_std"].value_counts().nlargest(10).sort_values(ascending=True).reset_index()
        del_counts.columns = ["delito","count"]
        ch = barh_with_labels(
            del_counts, y="delito", x="count",
            y_type="N", x_type="Q",
            sort=del_counts["delito"].tolist(),
            title="Top 10 Crimes by Count",
            color=THEME_PALETTE[5],
            width=700, height=320,
            x_axis=_axis("Number of Cases"),
            y_axis=_axis("Crime Type", orient="left"),
        )
        st.altair_chart(ch, use_container_width=True)

    # ===================== Classification & Violence =====================
    with st.expander("🚨 Classification & Violence", expanded=False):
        c1, c2 = st.columns(2)

        # Classification donut
        with c1:
            cls_counts = df["crime_classification"].value_counts().reset_index()
            cls_counts.columns = ["classification","count"]
            total = int(cls_counts["count"].sum())
            cls_counts["pct"] = (cls_counts["count"] / total * 100).round(1)

            donut = donut_chart(
                cls_counts, field="classification", count_col="count",
                title="Distribution by Crime Classification",
                colors=THEME_PALETTE[:max(3, len(cls_counts))],
                width=700, height=360,
            )
            center = center_text_chart(f"Total\n{total:,}", width=700, height=360)
            st.altair_chart((donut + center).properties(padding=_pad_dict(5)), use_container_width=True)

        # Violent vs Non-Violent donut
        with c2:
            if "violence_type" in df.columns:
                v_counts = df["violence_type"].value_counts().reset_index()
                v_counts.columns = ["violence","count"]
                v_counts["pct"] = (v_counts["count"] / v_counts["count"].sum() * 100).round(1)
                colors = ["#EF4444", "#60A5FA"] if set(v_counts["violence"]) >= {"Violent", "Non-Violent"} else THEME_PALETTE[:len(v_counts)]
                donut = donut_chart(
                    v_counts, field="violence", count_col="count",
                    title="Violent vs Non-Violent",
                    colors=colors,
                    width=700, height=360,
                )
                st.altair_chart(donut.properties(padding=_pad_dict(5)), use_container_width=True)
            else:
                st.info("No se encontró columna 'violence_type'.")

        # Municipality vs Classification (stacked) — SIN layer extra
        stacked = df.groupby(['alcaldia_std','crime_classification']).size().reset_index(name="count")
        order_mun = (stacked.groupby("alcaldia_std")["count"].sum().sort_values(ascending=True).index.tolist())
        ch = alt.Chart(stacked, **_cfg(width=900, height=420)).mark_bar().encode(
            x=alt.X("count:Q", axis=_axis("Number of Cases")),
            y=alt.Y("alcaldia_std:N", sort=order_mun, axis=_axis("Municipality", orient="left")),
            color=alt.Color("crime_classification:N", scale=alt.Scale(range=THEME_PALETTE[:max(3, stacked['crime_classification'].nunique())])),
            tooltip=["alcaldia_std:N","crime_classification:N","count:Q"]
        ).properties(title="Municipality vs Crime Classification (stacked)")
        st.altair_chart(ch, use_container_width=True)

        # Violence by Municipality — SIN layer extra
        if "violence_type" in df.columns:
            viol = df.groupby(['alcaldia_std','violence_type']).size().reset_index(name="count")
            order_mun_v = (viol.groupby("alcaldia_std")["count"].sum().sort_values(ascending=True).index.tolist())
            pal_viol = ["#EF4444", "#60A5FA"] if viol["violence_type"].nunique()==2 else THEME_PALETTE[:viol["violence_type"].nunique()]
            ch = alt.Chart(viol, **_cfg(width=900, height=420)).mark_bar().encode(
                x=alt.X("count:Q", axis=_axis("Number of Cases")),
                y=alt.Y("alcaldia_std:N", sort=order_mun_v, axis=_axis("Municipality", orient="left")),
                color=alt.Color("violence_type:N", scale=alt.Scale(range=pal_viol)),
                tooltip=["alcaldia_std:N","violence_type:N","count:Q"]
            ).properties(title="Distribution of Violence Type by Municipality")
            st.altair_chart(ch, use_container_width=True)

        # Violence within Patrimony (layered con labels)
        pat = df[df.get('crime_classification','') == 'Patrimony']
        if not pat.empty and "violence_type" in pat.columns:
            pat_counts = pat["violence_type"].value_counts().reset_index()
            pat_counts.columns = ["violence","count"]
            ch = bar_with_labels(
                pat_counts, x="violence", y="count",
                x_type="N", y_type="Q",
                title="Violence within Patrimony",
                color="#EF4444",
                width=700, height=280,
                x_axis=_axis("Type"),
                y_axis=_axis("Number of Cases"),
            )
            st.altair_chart(ch, use_container_width=True)

    # ===================== Trends over Time =====================
    with st.expander("📈 Trends over Time", expanded=False):
        # Yearly by Municipality (Top 8)
        yearly_mun = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','alcaldia_std'])
              .size().reset_index(name='count')
        )
        totals = yearly_mun.groupby("alcaldia_std")["count"].sum().sort_values(ascending=False)
        top8 = totals.head(8).index.tolist()
        yearly_mun_top = yearly_mun[yearly_mun["alcaldia_std"].isin(top8)].copy()
        yearly_mun_top["anio_inicio"] = pd.Categorical(yearly_mun_top["anio_inicio"], categories=YEAR_RANGE, ordered=True)

        ch = line_with_labels(
            yearly_mun_top, x="anio_inicio", y="count", color_field="alcaldia_std",
            x_type="O", y_type="Q",
            title="Trend of Crimes per Year by Municipality (Top 8)",
            width=950, height=420,
            x_axis=_axis("Year"),
            y_axis=_axis("Number of Cases"),
            point=True, label_dy=-10
        )
        st.altair_chart(ch, use_container_width=True)

        # Yearly by Classification
        year_class = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','crime_classification'])
              .size().reset_index(name='count')
        )
        year_class["anio_inicio"] = pd.Categorical(year_class["anio_inicio"], categories=YEAR_RANGE, ordered=True)

        ch = line_with_labels(
            year_class, x="anio_inicio", y="count", color_field="crime_classification",
            x_type="O", y_type="Q",
            title="Trend of Crimes per Year by Classification",
            width=950, height=420,
            x_axis=_axis("Year"),
            y_axis=_axis("Number of Cases"),
            point=True, label_dy=-10
        )
        st.altair_chart(ch, use_container_width=True)

        # Yearly Violent vs Non-Violent
        if "violence_type" in df.columns:
            yearly_viol = (
                df.dropna(subset=['anio_inicio'])
                  .groupby(['anio_inicio','violence_type'])
                  .size().reset_index(name='count')
            )
            yearly_viol["anio_inicio"] = pd.Categorical(yearly_viol["anio_inicio"], categories=YEAR_RANGE, ordered=True)

            ch = line_with_labels(
                yearly_viol, x="anio_inicio", y="count", color_field="violence_type",
                x_type="O", y_type="Q",
                title="Trend of Violent and Non-Violent Crimes per Year",
                width=950, height=420,
                x_axis=_axis("Year"),
                y_axis=_axis("Number of Cases"),
                point=True, label_dy=-10
            )
            st.altair_chart(ch, use_container_width=True)

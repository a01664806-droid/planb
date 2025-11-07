# modules/ui_info.py
import streamlit as st
import pandas as pd
import altair as alt

from .data import load_info_csv
from .helpers import MONTH_NAMES

# Paleta azul-gris consistente con tu config.toml (dark)
THEME_PALETTE = [
    "#1E3A8A",  # indigo-800 (primario)
    "#3B82F6",  # blue-500
    "#60A5FA",  # blue-400
    "#93C5FD",  # blue-300
    "#A3B3C2",  # gris azulado claro
    "#64748B",  # slate-500
    "#0EA5E9",  # sky-500
    "#14B8A6",  # teal-500
    "#F59E0B",  # amber-500
    "#EF4444",  # red-500 (violento)
]

def _cfg(width=700, height=350):
    return {"width": width, "height": height, "background": "transparent", "padding": 5}

def _axis(title, orient=None):
    base = {
        "title": title,
        "grid": True,
        "gridColor": "#33415555",  # grid sutil
        "labelColor": "#E5E7EB",
        "titleColor": "#E5E7EB",
        "tickColor": "#94A3B855"
    }
    if orient:
        base["orient"] = orient
    return base

# --- FIX Altair v5: helpers de labels (dict, no MarkConfig) ---
def _text_label_dict(dy=-8):
    return dict(align="center", baseline="bottom", dy=dy, fontWeight="bold", color="#E5E7EB")

def _bar_labels(chart, y_field):
    chart_spec = chart.to_dict()
    padding = chart_spec.pop("padding", 5)
    chart_without_padding = alt.Chart.from_dict(chart_spec)
    text = chart_without_padding.mark_text(**_text_label_dict()).encode(text=alt.datum[y_field])
    return (chart_without_padding + text).properties(padding=padding)

def _line_labels(chart, y_field, dy=-10):
    chart_spec = chart.to_dict()
    padding = chart_spec.pop("padding", 5)
    chart_without_padding = alt.Chart.from_dict(chart_spec)
    text = chart_without_padding.mark_text(**_text_label_dict(dy=dy)).encode(text=alt.datum[y_field])
    return (chart_without_padding + text).properties(padding=padding)

def render():
    st.title("ℹ️ Exploratory Data Analysis (EDA)")
    st.caption("Aquí se usa **carpetasFGJ_acumulado_2025_01.csv** del folder actual.")

    df = load_info_csv()
    if df.empty:
        st.warning("No hay datos para mostrar en Info.")
        return

    # ------- Normalizaciones y rangos fijos -------
    YEAR_RANGE  = list(range(2015, 2025))
    MONTH_RANGE = list(range(1, 13))
    HOUR_RANGE  = list(range(0, 24))

    df["anio_inicio"] = pd.to_numeric(df["anio_inicio"], errors="coerce")
    df["mes_inicio"]  = pd.to_numeric(df["mes_inicio"],  errors="coerce")
    if "hour_hecho" in df.columns:
        df["hour_hecho"] = pd.to_numeric(df["hour_hecho"], errors="coerce")

    # Datos agregados
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
            ch = alt.Chart(year_counts, **_cfg()).mark_bar(color=THEME_PALETTE[0]).encode(
                x=alt.X("year:O", sort=YEAR_RANGE, axis=_axis("Year")),
                y=alt.Y("count:Q", axis=_axis("Number of Cases")),
                tooltip=["year:O","count:Q"]
            ).properties(title="Crimes by Year")
            st.altair_chart(_bar_labels(ch, "count"), use_container_width=True)

        # Crimes by Month
        with c2:
            ch = alt.Chart(month_counts, **_cfg()).mark_bar(color=THEME_PALETTE[2]).encode(
                x=alt.X("month_name:N", sort=[MONTH_NAMES[m] for m in MONTH_RANGE], axis=_axis("Month")),
                y=alt.Y("count:Q", axis=_axis("Number of Cases")),
                tooltip=["month_name:N","count:Q"]
            ).properties(title="Crimes by Month")
            st.altair_chart(_bar_labels(ch, "count"), use_container_width=True)

        # Distribution of Crimes by Hour of Day
        ch = alt.Chart(hour_counts, **_cfg(width=980)).mark_bar(color=THEME_PALETTE[1]).encode(
            x=alt.X("hour:O", sort=HOUR_RANGE, axis=_axis("Hour of Day")),
            y=alt.Y("count:Q", axis=_axis("Number of Cases")),
            tooltip=["hour:O","count:Q"]
        ).properties(title="Distribution of Crimes by Hour of Day")
        st.altair_chart(_bar_labels(ch, "count"), use_container_width=True)

    # ===================== Spatial & Categories =====================
    with st.expander("🗺️ Spatial & Categories", expanded=False):
        c1, c2 = st.columns(2)

        # Distribution of Crimes by District (horizontal, ascendente)
        with c1:
            alc_counts = df["alcaldia_std"].value_counts().sort_values(ascending=True).reset_index()
            alc_counts.columns = ["alcaldia","count"]
            ch = alt.Chart(alc_counts, **_cfg()).mark_bar().encode(
                x=alt.X("count:Q", axis=_axis("Number of Cases")),
                y=alt.Y("alcaldia:N", sort=alc_counts["alcaldia"].tolist(), axis=_axis("District", orient="left")),
                color=alt.value(THEME_PALETTE[3]),
                tooltip=["alcaldia:N","count:Q"]
            ).properties(title="Distribution of Crimes by District")
            labels = alt.Chart(alc_counts, **_cfg()).mark_text(
                align="left", dx=5, color="#E5E7EB", fontWeight="bold"
            ).encode(x="count:Q", y=alt.Y("alcaldia:N", sort=alc_counts["alcaldia"].tolist()), text="count:Q")
            st.altair_chart(ch + labels, use_container_width=True)

        # Top 10 Neighborhoods by Crime Count
        with c2:
            col_counts = df["colonia_std"].value_counts().nlargest(10).sort_values(ascending=True).reset_index()
            col_counts.columns = ["colonia","count"]
            ch = alt.Chart(col_counts, **_cfg()).mark_bar(color=THEME_PALETTE[4]).encode(
                x=alt.X("count:Q", axis=_axis("Number of Cases")),
                y=alt.Y("colonia:N", sort=col_counts["colonia"].tolist(), axis=_axis("Neighborhood", orient="left")),
                tooltip=["colonia:N","count:Q"]
            ).properties(title="Top 10 Neighborhoods by Crime Count")
            labels = alt.Chart(col_counts, **_cfg()).mark_text(
                align="left", dx=5, color="#E5E7EB", fontWeight="bold"
            ).encode(x="count:Q", y=alt.Y("colonia:N", sort=col_counts["colonia"].tolist()), text="count:Q")
            st.altair_chart(ch + labels, use_container_width=True)

        # Top 10 Crimes by Count
        del_counts = df["delito_std"].value_counts().nlargest(10).sort_values(ascending=True).reset_index()
        del_counts.columns = ["delito","count"]
        ch = alt.Chart(del_counts, **_cfg(height=320)).mark_bar(color=THEME_PALETTE[5]).encode(
            x=alt.X("count:Q", axis=_axis("Number of Cases")),
            y=alt.Y("delito:N", sort=del_counts["delito"].tolist(), axis=_axis("Crime Type", orient="left")),
            tooltip=["delito:N","count:Q"]
        ).properties(title="Top 10 Crimes by Count")
        labels = alt.Chart(del_counts, **_cfg(height=320)).mark_text(
            align="left", dx=5, color="#E5E7EB", fontWeight="bold"
        ).encode(x="count:Q", y=alt.Y("delito:N", sort=del_counts["delito"].tolist()), text="count:Q")
        st.altair_chart(ch + labels, use_container_width=True)

    # ===================== Classification & Violence =====================
    with st.expander("🚨 Classification & Violence", expanded=False):
        c1, c2 = st.columns(2)

        # Distribution by Crime Classification (donut)
        with c1:
            cls_counts = df["crime_classification"].value_counts().reset_index()
            cls_counts.columns = ["classification","count"]
            total = int(cls_counts["count"].sum())
            cls_counts["pct"] = (cls_counts["count"] / total * 100).round(1)

            donut = alt.Chart(cls_counts, **_cfg(height=360)).mark_arc(innerRadius=70).encode(
                theta=alt.Theta("count:Q"),
                color=alt.Color(
                    "classification:N",
                    scale=alt.Scale(range=THEME_PALETTE[:max(3, len(cls_counts))])
                ),
                tooltip=[alt.Tooltip("classification:N", title="Classification"),
                         alt.Tooltip("count:Q", title="Count"),
                         alt.Tooltip("pct:Q", title="%")]
            ).properties(title="Distribution by Crime Classification")

            center_text = alt.Chart(pd.DataFrame({"text":[f"Total\n{total:,}"]}), **_cfg(height=360)).mark_text(
                align="center", fontSize=16, fontWeight="bold", color="#E5E7EB"
            ).encode(text="text:N")
            st.altair_chart(donut + center_text, use_container_width=True)

        # Violent vs Non-Violent (donut)
        with c2:
            if "violence_type" in df.columns:
                v_counts = df["violence_type"].value_counts().reset_index()
                v_counts.columns = ["violence","count"]
                v_counts["pct"] = (v_counts["count"] / v_counts["count"].sum() * 100).round(1)
                colors = ["#EF4444", "#60A5FA"] if set(v_counts["violence"]) >= {"Violent", "Non-Violent"} else THEME_PALETTE[:len(v_counts)]

                donut = alt.Chart(v_counts, **_cfg(height=360)).mark_arc(innerRadius=70).encode(
                    theta=alt.Theta("count:Q"),
                    color=alt.Color("violence:N", scale=alt.Scale(range=colors)),
                    tooltip=[alt.Tooltip("violence:N", title="Type"),
                             alt.Tooltip("count:Q", title="Count"),
                             alt.Tooltip("pct:Q", title="%")]
                ).properties(title="Violent vs Non-Violent")
                st.altair_chart(donut, use_container_width=True)
            else:
                st.info("No se encontró columna 'violence_type'.")

        # Municipality vs Crime Classification (stacked horizontal)
        stacked = (
            df.groupby(['alcaldia_std','crime_classification'])
              .size().reset_index(name="count")
        )
        order_mun = (stacked.groupby("alcaldia_std")["count"].sum().sort_values(ascending=True).index.tolist())

        ch = alt.Chart(stacked, **_cfg(width=900, height=420)).mark_bar().encode(
            x=alt.X("count:Q", axis=_axis("Number of Cases")),
            y=alt.Y("alcaldia_std:N", sort=order_mun, axis=_axis("Municipality", orient="left")),
            color=alt.Color("crime_classification:N",
                            scale=alt.Scale(range=THEME_PALETTE[:max(3, stacked['crime_classification'].nunique())])),
            tooltip=["alcaldia_std:N","crime_classification:N","count:Q"]
        ).properties(title="Municipality vs Crime Classification (stacked)")
        st.altair_chart(ch, use_container_width=True)

        # Distribution of Violence Type by Municipality (stacked horizontal)
        if "violence_type" in df.columns:
            viol = (
                df.groupby(['alcaldia_std','violence_type'])
                  .size().reset_index(name="count")
            )
            order_mun_v = (viol.groupby("alcaldia_std")["count"].sum().sort_values(ascending=True).index.tolist())
            pal_viol = ["#EF4444", "#60A5FA"] if viol["violence_type"].nunique()==2 else THEME_PALETTE[:viol["violence_type"].nunique()]

            ch = alt.Chart(viol, **_cfg(width=900, height=420)).mark_bar().encode(
                x=alt.X("count:Q", axis=_axis("Number of Cases")),
                y=alt.Y("alcaldia_std:N", sort=order_mun_v, axis=_axis("Municipality", orient="left")),
                color=alt.Color("violence_type:N", scale=alt.Scale(range=pal_viol)),
                tooltip=["alcaldia_std:N","violence_type:N","count:Q"]
            ).properties(title="Distribution of Violence Type by Municipality")
            st.altair_chart(ch, use_container_width=True)

        # Violence within Patrimony (barras para claridad en dark theme)
        pat = df[df.get('crime_classification','') == 'Patrimony']
        if not pat.empty and "violence_type" in pat.columns:
            pat_counts = pat["violence_type"].value_counts().reset_index()
            pat_counts.columns = ["violence","count"]
            ch = alt.Chart(pat_counts, **_cfg(height=280)).mark_bar(color="#EF4444").encode(
                x=alt.X("violence:N", axis=_axis("Type")),
                y=alt.Y("count:Q", axis=_axis("Number of Cases")),
                tooltip=["violence:N","count:Q"]
            ).properties(title="Violence within Patrimony")
            st.altair_chart(_bar_labels(ch, "count"), use_container_width=True)

    # ===================== Trends over Time =====================
    with st.expander("📈 Trends over Time", expanded=False):
        # Trend of Crimes per Year by Municipality (Top 8)
        yearly_mun = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','alcaldia_std'])
              .size().reset_index(name='count')
        )
        totals = yearly_mun.groupby("alcaldia_std")["count"].sum().sort_values(ascending=False)
        top8 = totals.head(8).index.tolist()
        yearly_mun_top = yearly_mun[yearly_mun["alcaldia_std"].isin(top8)].copy()
        yearly_mun_top["anio_inicio"] = pd.Categorical(yearly_mun_top["anio_inicio"], categories=YEAR_RANGE, ordered=True)

        ch = alt.Chart(yearly_mun_top, **_cfg(width=950, height=420)).mark_line(point=True).encode(
            x=alt.X("anio_inicio:O", sort=YEAR_RANGE, axis=_axis("Year")),
            y=alt.Y("count:Q", axis=_axis("Number of Cases")),
            color=alt.Color("alcaldia_std:N", scale=alt.Scale(range=THEME_PALETTE[:len(top8)])),
            tooltip=["alcaldia_std:N","anio_inicio:O","count:Q"]
        ).properties(title="Trend of Crimes per Year by Municipality (Top 8)")
        st.altair_chart(_line_labels(ch, "count"), use_container_width=True)

        # Trend of Crimes per Year by Classification
        year_class = (
            df.dropna(subset=['anio_inicio'])
              .groupby(['anio_inicio','crime_classification'])
              .size().reset_index(name='count')
        )
        year_class["anio_inicio"] = pd.Categorical(year_class["anio_inicio"], categories=YEAR_RANGE, ordered=True)

        ch = alt.Chart(year_class, **_cfg(width=950, height=420)).mark_line(point=True).encode(
            x=alt.X("anio_inicio:O", sort=YEAR_RANGE, axis=_axis("Year")),
            y=alt.Y("count:Q", axis=_axis("Number of Cases")),
            color=alt.Color("crime_classification:N",
                            scale=alt.Scale(range=THEME_PALETTE[:year_class['crime_classification'].nunique()])),
            tooltip=["crime_classification:N","anio_inicio:O","count:Q"]
        ).properties(title="Trend of Crimes per Year by Classification")
        st.altair_chart(_line_labels(ch, "count"), use_container_width=True)

        # Trend of Violent and Non-Violent Crimes per Year
        if "violence_type" in df.columns:
            yearly_viol = (
                df.dropna(subset=['anio_inicio'])
                  .groupby(['anio_inicio','violence_type'])
                  .size().reset_index(name='count')
            )
            yearly_viol["anio_inicio"] = pd.Categorical(yearly_viol["anio_inicio"], categories=YEAR_RANGE, ordered=True)
            pal_viol = ["#EF4444", "#60A5FA"] if yearly_viol["violence_type"].nunique()==2 else THEME_PALETTE[:yearly_viol["violence_type"].nunique()]

            ch = alt.Chart(yearly_viol, **_cfg(width=950, height=420)).mark_line(point=True).encode(
                x=alt.X("anio_inicio:O", sort=YEAR_RANGE, axis=_axis("Year")),
                y=alt.Y("count:Q", axis=_axis("Number of Cases")),
                color=alt.Color("violence_type:N", scale=alt.Scale(range=pal_viol)),
                tooltip=["violence_type:N","anio_inicio:O","count:Q"]
            ).properties(title="Trend of Violent and Non-Violent Crimes per Year")
            st.altair_chart(_line_labels(ch, "count"), use_container_width=True)

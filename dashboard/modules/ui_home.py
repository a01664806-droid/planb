import streamlit as st
from pathlib import Path
import urllib.request

def render():
    st.set_page_config(page_title="CDMX: From Incidents to Insights", layout="wide")

    # ======================
    # Configuración de imagen
    # ======================
    # Cambia a False si quieres cargar local (assets/)
    USE_REMOTE = True

    # (A) Remoto: foto libre (Wikimedia)
    cdmx_img_url = "https://upload.wikimedia.org/wikipedia/commons/8/8b/Mexico_City_Reforma_skyline.jpg"

    # (B) Local: coloca tus archivos en ./assets/
    # - Recomendado: tener PNG/JPG como fallback si tu archivo principal es AVIF.
    local_assets = Path("assets")
    local_assets.mkdir(exist_ok=True)

    # Pon aquí tus nombres reales:
    local_img_avif = local_assets / "dashboard/CDMX IMAGEN INICIAL.jpg"   # tu archivo
    local_img_fallback = local_assets / "dashboard/CDMX IMAGEN INICIAL.jpg"    # fallback recomendado

    # Si seleccionas remoto=False y no existe fallback, intento descargar la imagen de ejemplo
    if not USE_REMOTE and not (local_img_avif.exists() or local_img_fallback.exists()):
        try:
            example_path = local_assets / "cdmx_reforma.jpg"
            if not example_path.exists():
                urllib.request.urlretrieve(cdmx_img_url, example_path)
            # usamos este como fallback
            local_img_fallback = example_path
        except Exception as e:
            st.warning(f"No pude preparar imagen local: {e}. Usaré URL remota.")
            USE_REMOTE = True

    # ======================
    # Estilos globales
    # ======================
    st.markdown(
        """
        <style>
        /* Ajuste del contenedor principal */
        .main > div {
            padding-top: 1.2rem;
        }
        /* Título héroe centrado */
        .hero-title {
            text-align: center;
            color: #FFFFFF;
            font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
            font-weight: 800;
            font-size: clamp(36px, 5vw, 64px);
            letter-spacing: 0.2px;
            text-shadow: 2px 3px 10px rgba(0,0,0,0.35);
            margin: 0.2rem 0 0.4rem 0;
        }
        .hero-sub {
            text-align: center;
            color: #C9D1D9;
            font-size: clamp(14px, 1.6vw, 18px);
            margin-bottom: 1.1rem;
        }
        .soft-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .no-top-margin h3, .no-top-margin h2 {
            margin-top: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ======================
    # Hero (solo título centrado arriba)
    # ======================
    st.markdown(
        """
        <h1 class="hero-title">CDMX: From Incidents to Insights</h1>
        <div class="hero-sub">
            Discover patterns, visualize trends, and explore insights about urban safety across Mexico City.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ======================
    # Cuadrantes (2x2) debajo del título
    # ======================

    # Fila 1 (dos columnas)
    colA, colB = st.columns(2)

    # (1) Arriba-izquierda: "Did you know?"
    with colA:
        st.container()
        st.subheader("💡 Did you know?")
        st.info("En la Ciudad de México se denuncian aproximadamente 26 delitos por hora, lo que equivale a alrededor de 624 delitos al día. Esta cifra incluye más de cien tipos de delitos, desde robos y fraudes hasta violencia familiar y sexual. Aunque los homicidios y robos han disminuido en los últimos años, otros delitos como violencia familiar y amenazas han aumentado, por lo que el ritmo general de denuncias se mantiene alto.**.")

    # (2) Arriba-derecha: Imagen
    with colB:
        st.markdown("### 🌆 Mexico City", help="Skyline — Paseo de la Reforma")
        if USE_REMOTE:
            st.image(
                cdmx_img_url,
                caption="Skyline — Paseo de la Reforma (CC BY-SA 4.0, Jonathan Salvador)",
                use_container_width=True,
            )
        else:
            # Intento: mostrar AVIF si existe; si no, fallback
            if local_img_avif.exists():
                try:
                    # AVIF no siempre es soportado por Streamlit; si falla, mostramos fallback
                    st.image(str(local_img_avif), use_container_width=True, caption="CDMX (AVIF)")
                except Exception:
                    if local_img_fallback.exists():
                        st.image(str(local_img_fallback), use_container_width=True, caption="CDMX (fallback)")
                    else:
                        st.warning("No pude mostrar AVIF y no encontré fallback PNG/JPG.")
            elif local_img_fallback.exists():
                st.image(str(local_img_fallback), use_container_width=True, caption="CDMX (fallback)")
            else:
                st.warning("No hay imagen local disponible.")

    st.divider()

    # Fila 2 (dos columnas)
    colC, colD = st.columns(2)

    # (3) Abajo-izquierda: Público objetivo + Contenido
    with colC:
        st.subheader("🧭 For the public")
        st.markdown(
            """
            Our goal is to **help you feel safer** by turning complex data into clear, actionable insights:

            - Plain-language explanations and friendly visuals.  
            - Borough- and station-level context to orient decisions.  
            - Transparent methods and sources (no black boxes).  
            - Practical tips tied to patterns in time and place.
            """
        )

    # (4) Abajo-derecha: What's inside
    with colD:
        st.subheader("📦 What's inside")
        st.markdown(
            """
            - **🗺️ Map** — Explore density, clusters, and layers.  
            - **📊 Info (EDA)** — Trends by year, month, hour, and borough.  
            - **🤖 Predictive Models** — Station/time risk signals.  
            - **👥 Our Team** — Mission, people, and values.

            **Quick navigation**  
            - Map uses `alcaldias.geojson` or `alcaldias2.geojson`.  
            - Info (EDA) uses `carpetasFGJ_acumulado_2025_01.csv`.
            """
        )

    st.divider()

    # ======================
    # Footer
    # ======================
    st.caption(
        "This platform combines machine learning, geospatial analysis, and open data to support data-driven safety strategies."
    )
    st.caption(
        "Photo: “Mexico City Reforma skyline” — Jonathan Salvador — CC BY‑SA 4.0 — via Wikimedia Commons."
    )

if __name__ == "__main__":
    render()
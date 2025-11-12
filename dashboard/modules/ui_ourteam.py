import streamlit as st
import os
from pathlib import Path

# --- FUNCIÓN AUXILIAR PARA RUTAS RELATIVAS ---
def get_absolute_path(relative_path: str) -> Path:
    """
    Construye una ruta absoluta basándose en la ubicación del archivo app.py
    y la carpeta 'images' en la raíz del proyecto.
    """
    # 1. Obtiene la ruta al directorio actual del módulo (modules)
    MODULE_DIR = Path(__file__).parent
    
    # 2. Sube un nivel para llegar a la raíz del proyecto (donde está app.py)
    ROOT_DIR = MODULE_DIR.parent
    
    # 3. Une la ruta raíz con la ruta relativa específica (ej: images/damian.png)
    return ROOT_DIR / relative_path

def render():
    # Estilos CSS (Se mantienen sin cambios)
    st.markdown("""
        <style>
        .team-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr); /* 3 columnas en la parte superior */
            gap: 18px;
            margin-top: 1rem;
        }
        .team-grid-bottom {
            display: grid;
            grid-template-columns: repeat(2, 1fr); /* 2 columnas en la parte inferior */
            gap: 18px;
        }
        .card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.04);
            text-align: center;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        .quote {
            font-size: .95rem;
            color: #374151;
            line-height: 1.35;
            margin-top: 8px;
        }
        .section {
            background: rgba(0, 0, 0, .02);
            border: 1px solid rgba(0, 0, 0, .06);
            border-radius: 16px;
            padding: 16px;
        }
        .kicker {
            letter-spacing: .08em;
            text-transform: uppercase;
            font-weight: 600;
            font-size: .8rem;
            color: #6b7280;
            margin-bottom: 6px;
        }
        .card img {
            width: 80%; /* Hacemos la imagen más pequeña */
            height: auto; /* Ajustamos la altura automáticamente */
            object-fit: cover;
            border-radius: 999px;
            margin-bottom: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Nuestro Equipo")
    st.write("Conoce a las personas detrás de nuestro equipo y descubre nuestra misión y los objetivos que guían todo lo que hacemos")

    # --- LÓGICA DE RUTAS ADAPTADA ---
    # Ahora las rutas son relativas a la carpeta 'images/' en la raíz del proyecto
    team = [
        {
            "name": "Damian Calderon Capallera",
            "img_path": get_absolute_path("images/damian.png"), # <- Ruta relativa a la carpeta images/
            "quote": "Forecasting crime risk helps allocate resources before spikes happen."
        },
        {
            "name": "Daniela Martínez Xolalpa",
            "img_path": get_absolute_path("images/dani.png"), # <- Ruta relativa a la carpeta images/
            "quote": "Space-time patterns reveal where prevention can be most effective."
        },
        {
            "name": "José de Jesús Rodríguez Rocha",
            "img_path": get_absolute_path("images/chuy.png"), # <- Ruta relativa a la carpeta images/
            "quote": "From noise to narrative: modeling explains the ‘why’, not just the ‘what’."
        },
        {
            "name": "Fernando Vázquez Rivera",
            "img_path": get_absolute_path("images/fercho.png"), # <- Ruta relativa a la carpeta images/
            "quote": "Responsible prediction means insight that informs action—never bias."
        }
    ]

    # Mostrar información del equipo (Parte superior: 3 columnas)
    st.markdown('<div class="kicker">Team</div>', unsafe_allow_html=True)
    cols = st.columns(len(team))
    for idx, m in enumerate(team):
        with cols[idx]:
            # Usamos m["img_path"] y comprobamos si existe
            if m["img_path"].exists():
                # st.image acepta un objeto Path o un string
                st.image(m["img_path"], width=120)
            else:
                # Mensaje de error para facilitar la depuración
                st.warning(f"❌ Imagen no encontrada. Asegúrate de que '{m['img_path'].name}' esté en la carpeta 'images' en la raíz de tu proyecto.")
                st.info(f"Ruta intentada: {m['img_path']}")
                
            st.markdown(f"**{m['name']}**", unsafe_allow_html=True)
            st.markdown(f'<div class="quote">“{m["quote"]}”</div>', unsafe_allow_html=True)

    st.write("")  # Espacio entre secciones

    # Misión (Se mantiene sin cambios)
    st.markdown('<div class="kicker">Nuestra misión</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Nuestra misión es transformar los datos en conocimiento que impulse un cambio significativo.</strong>
        En una ciudad donde se reportan cientos de delitos cada día, comprender el cuándo, dónde y por qué de cada incidente es esencial. Creemos que los datos, cuando se analizan con propósito y precisión, pueden iluminar los patrones que configuran la seguridad urbana y ayudar a guiar decisiones más inteligentes.</p>

        <p>A través del uso de análisis de datos, visualización e información social, nuestro objetivo es descubrir las historias ocultas dentro de los números, revelando cómo las rutinas diarias, el comportamiento social y la estructura de la ciudad influyen en la dinámica del crimen. Al hacerlo, buscamos apoyar estrategias de prevención basadas en evidencia que hagan nuestras comunidades más seguras y resilientes.</p>

        <p>Sostenemos la idea de que la información por sí sola no es suficiente: debe conducir a la comprensión, y la comprensión debe conducir a la acción. Ese es el camino que seguimos: pasar de los incidentes a las ideas.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")  # Espacio entre secciones

    # Objetivos (Se mantiene sin cambios)
    st.markdown('<div class="kicker">Nuestros objetivos</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Nuestro objetivo es transformar los datos complejos sobre delitos en información clara y accionable que ayude a comprender y prevenir la inseguridad urbana.</strong>
         Aspiramos a combinar tecnología, pensamiento analítico y conciencia social para identificar tendencias significativas y apoyar a los responsables de la toma de decisiones, investigadores y comunidades en el desarrollo de estrategias efectivas para la seguridad y la prevención.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
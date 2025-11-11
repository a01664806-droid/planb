import streamlit as st
from pathlib import Path

def render():
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
    st.title("Our team")
    st.write("Get to know the people behind our team, and discover our mission and the goals that guide everything we do")

    # Team data (cambia nombres/fotos si quieres). Usaremos la clase Path para manejar las rutas de las imágenes.
    team = [
        {
            "name": "Damian Calderon Capallera",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/damian.png"),  # Ruta local con Path
            "quote": "Forecasting crime risk helps allocate resources before spikes happen."
        },
        {
            "name": "Daniela Martínez Xolalpa",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/dani.png"),  # Ruta local con Path
            "quote": "Space-time patterns reveal where prevention can be most effective."
        },
        {
            "name": "José de Jesús Rodríguez Rocha",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/chuy.png"),  # Ruta local con Path
            "quote": "From noise to narrative: modeling explains the ‘why’, not just the ‘what’."
        },
        {
            "name": "Fernando Vázquez Rivera",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/fercho.png"),  # Ruta local con Path
            "quote": "Responsible prediction means insight that informs action—never bias."
        },
        {
            "name": "Luis Gallegos Pérez",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/luis.png"),  # Ruta local con Path
            "quote": "Innovative solutions come from the ability to see patterns that others miss."
        },
        {
            "name": "Alejandro Cortés",
            "img": Path("/Users/damcalde/RETO/planb/dashboard/alejandro.png"),  # Ruta local con Path
            "quote": "Transforming data into meaningful insights is key to strategic decision-making."
        },
    ]

    # Mostrar información del equipo (Parte superior: 3 columnas)
    st.markdown('<div class="kicker">Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    for m in team[:3]:  # Mostramos solo los 3 primeros para la fila superior
        st.markdown(f"""
            <div class="card">
                {st.image(m['img'], caption=m['name'], use_container_width=True)}  <!-- Imagen usando st.image -->
                <div style="font-weight:700">{m['name']}</div>
                <div class="quote">“{m['quote']}”</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Mostrar información del equipo (Parte inferior: 2 columnas)
    st.markdown('<div class="team-grid-bottom">', unsafe_allow_html=True)
    for m in team[3:]:  # Mostramos los siguientes 3 (parte inferior)
        st.markdown(f"""
            <div class="card">
                {st.image(m['img'], caption=m['name'], use_container_width=True)}  <!-- Imagen usando st.image -->
                <div style="font-weight:700">{m['name']}</div>
                <div class="quote">“{m['quote']}”</div>
            </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")  # spacing

    # Mission
    st.markdown('<div class="kicker">Our mission</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Our mission is to transform raw data into knowledge that drives meaningful change.</strong>
        In a city where hundreds of crimes are reported every day, understanding the when, where, and why
        behind each incident is essential. We believe that data, when analyzed with purpose and precision,
        can illuminate the patterns that shape urban safety and help guide smarter decisions.</p>

        <p>Through the use of data analytics, visualization, and social insight, our goal is to uncover the stories
        hidden within the numbers — revealing how daily routines, social behavior, and city structure influence crime
        dynamics. By doing so, we aim to support evidence-based prevention strategies that make our communities safer
        and more resilient.</p>

        <p>We stand by the idea that information alone is not enough — it must lead to understanding, and understanding
        must lead to action. That is the path we follow: moving from incidents to insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")  # spacing

    # Goals
    st.markdown('<div class="kicker">Our goals</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section">
        <p><strong>Our goal is to turn complex crime data into clear, actionable insights that help understand and prevent urban insecurity.</strong>
        We aim to combine technology, analytical thinking, and social awareness to identify meaningful trends and support
        decision-makers, researchers, and communities in developing effective strategies for safety and prevention.</p>
        </div>
        """,
        unsafe_allow_html=True
    )


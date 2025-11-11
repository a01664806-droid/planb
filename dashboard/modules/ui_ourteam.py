import streamlit as st

def render():
    st.markdown("""
        <style>
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); 
            gap: 18px;
            margin-top: 1rem;
        }
        .card {
            background: var(--background-color, #ffffff);
            border: 1px solid rgba(0,0,0,.08);
            border-radius: 16px;
            padding: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.04);
            text-align: center;
        }
        .card img {
            width: 100%;
            height: 120px;
            object-fit: cover;
            border-radius: 999px;
            display: block;
            margin: 0 auto 10px;
            border: 3px solid rgba(0, 0, 0, .06);
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
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("Our team")
    st.write("Get to know the people behind our team, and discover our mission and the goals that guide everything we do")

    # Team data (cambia nombres/fotos si quieres). Las imágenes usan placeholders públicos.
    team = [
        {
            "name": "Damian Calderon Capallera",
            "img": "dashboard/damian.png",  # Ruta local
            "quote": "Forecasting crime risk helps allocate resources before spikes happen."
        },
        {
            "name": "Daniela Martínez Xolalpa",
            "img": "dashboard/dani.png",  # Ruta local
            "quote": "Space-time patterns reveal where prevention can be most effective."
        },
        {
            "name": "José de Jesús Rodríguez Rocha",
            "img": "dashboard/chuy.png",  # Ruta local
            "quote": "From noise to narrative: modeling explains the ‘why’, not just the ‘what’."
        },
        {
            "name": "Fernando Vázquez Rivera",
            "img": "dashboard/fercho.png",  # Ruta local
            "quote": "Responsible prediction means insight that informs action—never bias."
        },
    ]

    # Mostrar información del equipo
    st.markdown('<div class="kicker">Team</div>', unsafe_allow_html=True)
    st.markdown('<div class="team-grid">', unsafe_allow_html=True)
    for m in team:
        # Mostrar imagen si existe
        try:
            st.markdown(
                f"""
                <div class="card">
                    <img src="{m['img']}" alt="{m['name']}">
                    <div style="font-weight:700">{m['name']}</div>
                    <div class="quote">“{m['quote']}”</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception as e:
            st.write(f"Error cargando la imagen de {m['name']}: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.write("")  # spacing

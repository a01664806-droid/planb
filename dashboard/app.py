import streamlit as st
<<<<<<< HEAD
from modules import ui_home, ui_analysis, ui_map, ui_info
=======
from modules import ui_home, ui_analysis, ui_map, ui_info, ui_ourteam
>>>>>>> 57695c1260b2c477ebf3d336435ea8dea6a39431

st.set_page_config(
    page_title="CDMX Crime Intelligence Platform",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar nav
st.sidebar.title("🔎 Navegación")
page = st.sidebar.radio(
    "Ir a:",
    ["🏠 Home", "📊 Analysis", "🗺️ Map", "ℹ️ Info", "👥 Our Team"],
    index=2
)

# Router
if page == "🏠 Home":
    ui_home.render()
elif page == "📊 Analysis":
    ui_analysis.render()
elif page == "🗺️ Map":
    ui_map.render()
elif page == "ℹ️ Info":
    ui_info.render()
else:
<<<<<<< HEAD
    st.title("👥 Our Team")
    st.markdown("**Project Leads:** Tú y tu bandita 🔥  \n**Contacto:** agrega tus correos y roles aquí, ca.")
=======
    ui_ourteam.render()
    
>>>>>>> 57695c1260b2c477ebf3d336435ea8dea6a39431

import pandas as pd
import numpy as np

def map_to_time_slot(hour):
    """Convierte una hora (0-23) en una franja horaria categórica."""
    if 0 <= hour <= 5: return 'Madrugada'
    elif 6 <= hour <= 11: return 'Mañana'
    elif 12 <= hour <= 18: return 'Tarde'
    return 'Noche' # 19-23

def preprocess_inputs_mapa_v3(fecha, hora, lat, lon, alcaldia, categoria, kmeans_model):
    """
    Toma los inputs crudos y los transforma para el pipeline V3 (con zona_hora).
    """
    fecha_dt = pd.to_datetime(fecha)
    dia_de_la_semana = fecha_dt.dayofweek
    es_fin_de_semana = int(dia_de_la_semana >= 5)
    mes = fecha_dt.month
    dia_del_mes = fecha_dt.day
    es_quincena = int(dia_del_mes in [14,15,16, 28,29,30,31,1,2])
    
    coords = pd.DataFrame({'latitud': [lat], 'longitud': [lon]})
    
    if kmeans_model is None:
        zona_cluster = 0 
    else:
        try:
            zona_cluster = kmeans_model.predict(coords)[0]
        except Exception:
            zona_cluster = 0 
    
    franja_horaria = map_to_time_slot(hora)
    zona_hora = f"{zona_cluster}_{franja_horaria}" 
    mes_sin = np.sin(2 * np.pi * mes / 12)
    mes_cos = np.cos(2 * np.pi * mes / 12)
    
    input_data = {
        'alcaldia_hecho': [alcaldia], 'categoria_delito': [categoria], 'dia_de_la_semana': [dia_de_la_semana],
        'es_fin_de_semana': [es_fin_de_semana], 'es_quincena': [es_quincena], 'zona_hora': [zona_hora], 
        'mes_sin': [mes_sin], 'mes_cos': [mes_cos],
        'latitud': [lat], 'longitud': [lon], 'hora_hecho': [hora], 'mes_hecho': [mes],
        'zona_cluster': [zona_cluster], 'franja_horaria': [franja_horaria] 
    }
    
    input_df = pd.DataFrame(input_data)
    return input_df

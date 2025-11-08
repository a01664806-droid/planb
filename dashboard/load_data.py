import duckdb
import pandas as pd
import os

# --- Configuración ---
# 1. Escribe el nombre de tu archivo CSV grande aquí
CSV_FILE_NAME = 'cleaned_crime_data.csv' 

# 2. Este es el nombre del archivo de base de datos que se creará
DB_FILE_NAME = 'cdmx_insights.db'
# --- Fin de Configuración ---


def create_database():
    """
    Lee el CSV y crea el archivo de base de datos DuckDB.
    """
    
    # Verifica si el CSV existe
    if not os.path.exists(CSV_FILE_NAME):
        print(f"Error: No se encontró el archivo '{CSV_FILE_NAME}' en esta carpeta.")
        print("Por favor, pon tu archivo CSV en la misma carpeta que este script.")
        return

    # Verifica si la base de datos ya existe para no duplicar
    if os.path.exists(DB_FILE_NAME):
        print(f"El archivo '{DB_FILE_NAME}' ya existe.")
        overwrite = input("¿Quieres sobrescribirlo? (s/n): ").lower()
        if overwrite != 's':
            print("Operación cancelada.")
            return
        else:
            os.remove(DB_FILE_NAME)

    print(f"Iniciando carga de '{CSV_FILE_NAME}'... Esto puede tardar unos minutos.")

    try:
        # Conecta (o crea) el archivo de la base de datos
        con = duckdb.connect(DB_FILE_NAME)

        # Crea la tabla 'crimes' directamente desde el CSV
        # DuckDB es muy rápido para esto e infiere los tipos de datos.
        con.execute(f"CREATE TABLE crimes AS SELECT * FROM read_csv_auto('{CSV_FILE_NAME}')")
        
        print("\n¡Éxito! Tabla 'crimes' creada.")
        
        # Opcional: Verificamos
        count_result = con.execute("SELECT COUNT(*) FROM crimes").fetchdf()
        print(f"Total de filas insertadas: {count_result.iloc[0,0]}")
        
        con.close()
        print(f"\n¡Base de datos '{DB_FILE_NAME}' creada exitosamente!")

    except Exception as e:
        print(f"Ocurrió un error: {e}")

# --- Ejecución Principal ---
if __name__ == "__main__":
    create_database()
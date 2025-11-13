# Importar las bibliotecas necesarias
import geopandas as gpd
import pandas as pd
import os
from pathlib import Path
from shapely.geometry.base import BaseGeometry
from shapely import wkt as shapely_wkt

# Cargar el archivo GPKG


def load_gpkg(gpkg_path):
    try:
        gdf = gpd.read_file(gpkg_path)
        return gdf
    except Exception as e:
        print(f"Error al leer el GPKG '{gpkg_path}': {e}")
        raise

# Función para guardar los datos tabulares en un formato CSV o Excel


def save_to_csv_or_excel(gdf, output_path, file_format='csv'):
    # Asegurar índices simples para evitar desalineos
    gdf = gdf.reset_index(drop=True)

    geom_col = getattr(gdf, "_geometry_column_name", None)
    if geom_col is None and 'geometry' in gdf.columns:
        geom_col = 'geometry'

    # Preparar DataFrame de salida sin la columna geométrica (mantener columnas tabulares)
    if geom_col and geom_col in gdf.columns:
        df_out = gdf.drop(columns=[geom_col]).copy()
    else:
        df_out = gdf.copy()

    # Si existe la columna geométrica, generar columna WKT alineada
    if geom_col and geom_col in gdf.columns:
        sample = gdf[geom_col].dropna().head(10)
        sample_list = sample.tolist()
        from collections import Counter
        type_counts = Counter(type(v) for v in sample_list)
        type_counts_readable = {t.__name__: c for t, c in type_counts.items()}
        print(
            f"Tipo de valores en columna de geometría ('{geom_col}') (muestra):")
        print(type_counts_readable)

        first = sample_list[0] if len(sample_list) > 0 else None

        try:
            if first is None:
                print(
                    "Aviso: columna de geometría vacía o sólo nulos. Se exportarán nulos en WKT.")
                geometry_wkt = pd.Series(
                    [None] * len(gdf), index=gdf.index, name='geometry')
            elif isinstance(first, str):
                # Intentar parsear WKT y normalizar
                parsed = gdf[geom_col].apply(lambda s: shapely_wkt.loads(
                    s) if (isinstance(s, str) and s.strip() != '') else None)
                geometry_wkt = parsed.apply(
                    lambda geom: geom.wkt if geom is not None else None)
                geometry_wkt = pd.Series(
                    geometry_wkt.values, index=gdf.index, name='geometry')
                print("Convierte strings WKT a geometrías y luego a WKT para exportar.")
            else:
                # Asumir objetos geométricos (shapely) u otros; convertir a WKT o a string
                if isinstance(first, BaseGeometry):
                    geometry_wkt = gdf[geom_col].apply(
                        lambda geom: geom.wkt if geom is not None else None)
                    geometry_wkt = pd.Series(
                        geometry_wkt.values, index=gdf.index, name='geometry')
                    print("Convierte objetos geométricos a WKT para exportar.")
                else:
                    # Fallback: convertir a string
                    geometry_wkt = gdf[geom_col].astype(str)
                    geometry_wkt = pd.Series(
                        geometry_wkt.values, index=gdf.index, name='geometry')
                    print(
                        "Aviso: columna de geometría no reconocida; se convertirán los valores a string.")
        except Exception as e:
            print("Error al procesar la columna geométrica:", e)
            geometry_wkt = pd.Series(gdf[geom_col].astype(
                str).values, index=gdf.index, name='geometry')

        # Añadir columna WKT al DataFrame de salida (alineada por índice)
        df_out['geometry'] = geometry_wkt
    else:
        print("Aviso: no se detectó columna de geometría activa en el GeoDataFrame; se exportan las columnas existentes.")

    # Crear carpeta de salida si es necesario
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Guardar archivo
    try:
        if file_format == 'csv':
            df_out.to_csv(output_path, index=False)
        elif file_format == 'excel':
            df_out.to_excel(output_path, index=False)
        else:
            print(
                f"Formato '{file_format}' no soportado. Usa 'csv' o 'excel'.")
            return
        print(f"Archivo guardado en: {output_path}")
    except Exception as e:
        print(f"Error al guardar el archivo '{output_path}': {e}")
        raise

# Función principal


def process_gpkg(gpkg_path, output_path, file_format='csv'):
    # Verificar existencia del archivo GPKG
    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        print(f"Archivo GPKG no encontrado: {gpkg_path}")
        return

    # Cargar el archivo GPKG
    gdf = load_gpkg(gpkg_path)

    # Información para inspección: columnas y tipos
    print("Columnas y tipos del GeoDataFrame:")
    print(gdf.dtypes)
    print("Primeras filas:")
    print(gdf.head())

    # Guardar los datos en el formato deseado
    save_to_csv_or_excel(gdf, output_path, file_format)


# Rutas de los archivos
# Se asume que el GPKG está en la carpeta Data de la raíz del proyecto:
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\Trabajo de Grado
gpkg_path = PROJECT_ROOT / 'Data' / 'precipitacion_acumulada_2024.gpkg'

# Salida en output/area_protegida dentro de la raíz del proyecto
output_dir = PROJECT_ROOT / 'output' / 'area_protegida'
output_path = output_dir / 'precipitacion_acumulada_2024.csv'

# Llamar la función principal para procesar el archivo GPKG
# Cambia 'csv' a 'excel' si prefieres ese formato
if __name__ == '__main__':
    process_gpkg(gpkg_path, output_path, file_format='csv')

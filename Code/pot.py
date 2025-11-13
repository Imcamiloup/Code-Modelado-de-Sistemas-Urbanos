from pathlib import Path
import geopandas as gpd
import pandas as pd


def _detect_year_series(gdf: gpd.GeoDataFrame) -> pd.Series | None:
    # Intentar detectar columna de año común, incluyendo FECHA_CAPT
    candidate_names = [
        "fecha_capt", "periodo", "anio", "año", "year", "fecha", "date",
    ]
    lower_cols = {c.lower(): c for c in gdf.columns}

    col_name = None
    # 1) Coincidencia exacta por nombre común
    for cand in candidate_names:
        if cand in lower_cols:
            col_name = lower_cols[cand]
            break

    # 2) Si no hay coincidencia exacta, buscar por subcadenas relevantes
    if col_name is None:
        keywords = ["fecha", "date", "year", "anio", "año", "period"]
        for lc, orig in lower_cols.items():
            if any(k in lc for k in keywords):
                col_name = orig
                break

    if col_name is None:
        # No se detectó una columna obvia
        return None

    s = gdf[col_name]
    # Si es fecha o texto, intentar parsear
    if pd.api.types.is_datetime64_any_dtype(s):
        years = s.dt.year
    elif pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        years = pd.to_numeric(s, errors="coerce")
    else:
        years = pd.to_datetime(s, errors="coerce", dayfirst=True).dt.year

    # Filtrar a años razonables
    years = years.where((years >= 1900) & (years <= 2100))
    if years.notna().any():
        return years
    return None


def _clean_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Limpiar geometrías antes de operaciones espaciales
    try:
        from shapely import make_valid
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf = gdf.copy()

    gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty]
    gdf["geometry"] = gdf.buffer(0)
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    gdf = gdf[gdf.geometry.area > 0]
    return gdf


def main():
    # Ruta al shapefile relativa a este script: ../Data/estructuraecologicaprincipal/
    script_dir = Path(__file__).resolve().parent
    shp_path = (script_dir.parent / "Data" / "estructuraecologicaprincipal" / "EstructuraEcologicaPrincipal.shp").resolve()

    if not shp_path.exists():
        raise FileNotFoundError(f"No se encontró el shapefile en: {shp_path}")

    # Cargar el shapefile
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        raise ValueError("El shapefile no contiene geometrías.")

    # Diagnóstico previo: distribución de años en FECHA_CAPT (si existe) ANTES de cualquier limpieza/ reproyección
    if "FECHA_CAPT" in gdf.columns:
        fc = pd.to_datetime(gdf["FECHA_CAPT"], errors="coerce", dayfirst=True)
        years_fc = fc.dt.year
        vc = years_fc.value_counts(dropna=False).sort_index()
        print("\nDiagnóstico FECHA_CAPT -> año (antes de limpieza):")
        print(vc.to_string())
        if years_fc.isna().any():
            ejemplos_na = gdf.loc[years_fc.isna(), "FECHA_CAPT"].astype(str).head(10).tolist()
            print("Ejemplos no parseables (máx 10):", ejemplos_na)

    # Asegurar un CRS proyectado en metros para calcular área (Bogotá: EPSG:3116)
    target_crs = "EPSG:3116"
    if gdf.crs is None:
        # Si no tiene CRS, asumimos que viene en geográficas de Colombia; el usuario debería confirmar
        # Aún así, se re-proyecta a 3116 para poder medir en metros
        gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)

    # Limpiar geometrías
    # Guardar diagnóstico de años ANTES de limpieza usando la detección genérica
    years_before = _detect_year_series(gdf)
    if years_before is not None:
        yb = years_before.value_counts(dropna=False).sort_index()
        print("\nAños detectados antes de limpieza (todas las filas):")
        print(yb.to_string())

    gdf = _clean_geometries(gdf)

    if gdf.empty:
        raise ValueError("No quedan geometrías válidas de tipo poligonal para calcular el área.")

    # Disolver todas las geometrías para evitar doble conteo por solapes
    # Fallback a unary_union si fuese necesario
    try:
        dissolved = gdf.dissolve()
        geom = dissolved.geometry.iloc[0]
    except Exception:
        from shapely.ops import unary_union
        geom = unary_union(gdf.geometry)
        if geom.is_empty:
            raise ValueError("La unión geométrica resultó vacía.")
        # Asegurar geometría válida final
        try:
            from shapely import make_valid as _mv
            geom = _mv(geom)
        except Exception:
            geom = geom.buffer(0)

    # Calcular áreas
    area_m2 = geom.area
    area_ha = area_m2 / 10_000
    area_km2 = area_m2 / 1_000_000

    # Mostrar resultados
    print("CRS usado:", target_crs)
    print(f"Área total de la Estructura Ecológica de Bogotá:")
    print(f" - {area_m2:,.2f} m²")
    print(f" - {area_ha:,.2f} ha")
    print(f" - {area_km2:,.4f} km²")

    # Intentar calcular áreas por año
    years = _detect_year_series(gdf)
    if years is None or years.isna().all():
        print("No se detectó una columna de año válida en el shapefile. Columnas disponibles:", list(gdf.columns))
        return

    gdf = gdf.assign(__anio=years)
    gdf_year = gdf.dropna(subset=["__anio"]).copy()

    # Diagnóstico: conteo por año DESPUÉS de limpieza
    print("\nAños detectados después de limpieza (filas con año válido):")
    print(gdf_year["__anio"].value_counts().sort_index().to_string())

    if gdf_year.empty:
        print("No hay registros con año válido para calcular áreas por año.")
        return

    resultados = []
    for y, sub in gdf_year.groupby("__anio"):
        sub = _clean_geometries(sub)
        if sub.empty:
            continue
        try:
            dissolved_y = sub.dissolve()
            geom_y = dissolved_y.geometry.iloc[0]
        except Exception:
            from shapely.ops import unary_union
            geom_y = unary_union(sub.geometry)
            try:
                from shapely import make_valid as _mv
                geom_y = _mv(geom_y)
            except Exception:
                geom_y = geom_y.buffer(0)
        area_m2_y = geom_y.area
        resultados.append({
            "anio": int(y),
            "area_m2": area_m2_y,
            "area_ha": area_m2_y / 10_000,
            "area_km2": area_m2_y / 1_000_000,
        })

    if resultados:
        df_res = pd.DataFrame(resultados).sort_values("anio")
        out_dir = (script_dir.parent / "Output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "areas_estructura_ecologica_por_anio.csv"
        df_res.to_csv(csv_path, index=False)
        print("\nÁreas por año (primeras filas):")
        print(df_res.head())
        print(f"\nArchivo exportado: {csv_path}")


if __name__ == "__main__":
    main()


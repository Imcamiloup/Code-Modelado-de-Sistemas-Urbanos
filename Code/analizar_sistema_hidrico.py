from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit("Se requiere geopandas para ejecutar este script. Instálalo y vuelve a intentar.")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def extract_year_from_text(series: pd.Series) -> pd.Series:
    pat = re.compile(r"(19|20)\d{2}")
    s = series.astype(str).str.extract(pat, expand=False)
    return pd.to_numeric(s, errors="coerce")


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = (script_dir.parent / "Data" / "sistema_hidrico.gpkg").resolve()
    out_dir = (script_dir.parent / "Output" / "sistema_hidrico").resolve()
    ensure_dir(out_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el GPKG: {data_path}")

    # Cargar capa
    layer = "Sistema_hidrico"
    gdf = gpd.read_file(data_path, layer=layer)

    # Asegurar CRS métrico (EPSG:3116 - MAGNA-SIRGAS / Colombia Bogotá)
    if gdf.crs is None:
        # Según inspección, srs_id=4686. Géografico (grados). Asignamos si falta.
        gdf = gdf.set_crs(4686, allow_override=True)
    gdf_m = gdf.to_crs(3116)

    # Calcular área en m²
    gdf_m["area_m2"] = gdf_m.geometry.area

    # Tablas de resumen
    cols_keep = [c for c in ["componente", "categoria", "elemento", "nombre_tot", "acto_admin", "responsabl"] if c in gdf_m.columns]
    df = gdf_m[cols_keep + ["area_m2"]].copy()

    # Totales generales
    total_area = df["area_m2"].sum()
    pd.DataFrame({"indicador": ["area_total_m2"], "valor": [total_area]}).to_csv(out_dir / "area_total.csv", index=False)

    # Por categoría
    if "categoria" in df.columns:
        por_categoria = df.groupby("categoria", dropna=False)["area_m2"].sum().sort_values(ascending=False)
        por_categoria.to_csv(out_dir / "area_por_categoria.csv")

        plt.figure(figsize=(10, 6))
        por_categoria.head(20).plot(kind="bar")
        plt.title("Área por categoría (m²) - Top 20")
        plt.ylabel("Área (m²)")
        plt.xlabel("Categoría")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y", linestyle=":", alpha=0.5)
        plt.savefig(out_dir / "area_por_categoria_top20.png", dpi=150)
        plt.close()

    # Por elemento
    if "elemento" in df.columns:
        por_elemento = df.groupby("elemento", dropna=False)["area_m2"].sum().sort_values(ascending=False)
        por_elemento.to_csv(out_dir / "area_por_elemento.csv")

        plt.figure(figsize=(10, 6))
        por_elemento.head(20).plot(kind="bar")
        plt.title("Área por elemento (m²) - Top 20")
        plt.ylabel("Área (m²)")
        plt.xlabel("Elemento")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.grid(axis="y", linestyle=":", alpha=0.5)
        plt.savefig(out_dir / "area_por_elemento_top20.png", dpi=150)
        plt.close()

    # Intento de histórico: extraer año desde 'acto_admin' si existe
    if "acto_admin" in df.columns:
        anio = extract_year_from_text(df["acto_admin"])  # podría ser NaN si no hay año
        if anio.notna().any():
            df_hist = df.assign(anio=anio).dropna(subset=["anio"])  # solo filas con año detectado
            hist = df_hist.groupby("anio")["area_m2"].sum().sort_index()
            hist.to_csv(out_dir / "area_total_por_anio_desde_acto_admin.csv")

            plt.figure(figsize=(10, 5))
            plt.plot(hist.index, hist.values, marker="o", linewidth=1.8)
            plt.title("Área total del sistema hídrico por año (extraído de acto_admin)")
            plt.xlabel("Año")
            plt.ylabel("Área (m²)")
            plt.grid(True, linestyle=":", alpha=0.5)
            plt.tight_layout()
            plt.savefig(out_dir / "area_total_por_anio_desde_acto_admin.png", dpi=150)
            plt.close()

    print(f"Listo. Resultados en: {out_dir}")


if __name__ == "__main__":
    main()

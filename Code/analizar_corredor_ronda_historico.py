from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    raise SystemExit("Se requiere geopandas para este script.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def main():
    base = Path(__file__).resolve().parent.parent
    gpkg = base / "Data" / "corredor_ecologico_ronda.gpkg"
    out = base / "Output" / "corredor_ronda_historico"
    ensure_dir(out)

    if not gpkg.exists():
        raise FileNotFoundError(gpkg)

    layer = "corredor_ecologico_ronda"
    gdf = gpd.read_file(gpkg, layer=layer)

    # Normalizar nombres de columnas
    cols_lower = {c: c.lower() for c in gdf.columns}
    gdf = gdf.rename(columns=cols_lower)

    # CRS: si no está en un CRS métrico, pasar a EPSG:3116
    if gdf.crs is None:
        # muchos datasets de Bogotá vienen en 4686 (geográfico). Asignar si falta.
        gdf = gdf.set_crs(4686, allow_override=True)
    try:
        gdf_m = gdf.to_crs(3116)
    except Exception:
        # si la definición es inválida, forzar 4686 y luego 3116
        gdf = gdf.set_crs(4686, allow_override=True)
        gdf_m = gdf.to_crs(3116)

    # Calcular área m²
    gdf_m["area_m2"] = gdf_m.geometry.area

    # Detectar año desde 'fecha_capt' o similares
    date_candidates = [c for c in gdf_m.columns if c.lower() in {"fecha_capt", "fecha", "fechacapt"}]
    if date_candidates:
        dcol = date_candidates[0]
        fechas = pd.to_datetime(gdf_m[dcol], errors="coerce")
    else:
        # fallback: intentar extraer año de 'acto_admin' si existe
        if "acto_admin" in gdf_m.columns:
            fechas = pd.to_datetime(gdf_m["acto_admin"].astype(str).str.extract(r"((?:19|20)\d{2})", expand=False), errors="coerce")
        else:
            fechas = pd.Series(pd.NaT, index=gdf_m.index)

    gdf_m["anio"] = fechas.dt.year

    # Filtrar registros con año
    gdfy = gdf_m.dropna(subset=["anio"]).copy()
    if gdfy.empty:
        print("No se detectaron años en la capa; no es posible construir histórico.")
        # Aun así exportar totales sin año
        gdf_m[["area_m2"]].sum().to_frame(name="valor").to_csv(out / "area_total.csv")
        return

    # Tablas por año (total)
    area_por_anio = gdfy.groupby("anio")["area_m2"].sum().sort_index()
    area_por_anio.to_csv(out / "area_total_por_anio.csv")

    plt.figure(figsize=(10, 5))
    plt.plot(area_por_anio.index, area_por_anio.values, marker="o", linewidth=1.8)
    plt.title("Área total del corredor ecológico de ronda por año")
    plt.xlabel("Año")
    plt.ylabel("Área (m²)")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out / "area_total_por_anio.png", dpi=150)
    plt.close()

    # Por año y categoría
    if "categoria" in gdfy.columns:
        tabla_cat = (
            gdfy.groupby(["anio", "categoria"], dropna=False)["area_m2"].sum().reset_index()
        )
        tabla_cat.to_csv(out / "area_por_anio_y_categoria.csv", index=False)

        piv = tabla_cat.pivot(index="anio", columns="categoria", values="area_m2").sort_index()
        # asegurar numérico
        piv = piv.apply(pd.to_numeric, errors="coerce").fillna(0)
        if piv.select_dtypes(include="number").size > 0 and not piv.empty:
            ax = piv.plot(kind="bar", stacked=True, figsize=(12, 6))
            ax.set_title("Área por año y categoría (m²)")
            ax.set_xlabel("Año")
            ax.set_ylabel("Área (m²)")
            plt.tight_layout()
            plt.savefig(out / "area_por_anio_y_categoria.png", dpi=150)
            plt.close()
        else:
            print("[INFO] No hay datos numéricos para graficar por categoría.")

    # Por año y elemento
    if "elemento" in gdfy.columns:
        tabla_ele = (
            gdfy.groupby(["anio", "elemento"], dropna=False)["area_m2"].sum().reset_index()
        )
        tabla_ele.to_csv(out / "area_por_anio_y_elemento.csv", index=False)

        top_elements = (
            tabla_ele.groupby("elemento")["area_m2"].sum().sort_values(ascending=False).head(10).index
        )
        piv_e = tabla_ele[tabla_ele["elemento"].isin(top_elements)].pivot(index="anio", columns="elemento", values="area_m2").sort_index()
        piv_e = piv_e.apply(pd.to_numeric, errors="coerce").fillna(0)
        if piv_e.select_dtypes(include="number").size > 0 and not piv_e.empty:
            ax = piv_e.plot(kind="bar", stacked=True, figsize=(12, 6))
            ax.set_title("Área por año y elemento (m²) - Top 10 elementos")
            ax.set_xlabel("Año")
            ax.set_ylabel("Área (m²)")
            plt.tight_layout()
            plt.savefig(out / "area_por_anio_y_elemento_top10.png", dpi=150)
            plt.close()
        else:
            print("[INFO] No hay datos numéricos para graficar por elemento.")

    print(f"Listo. Resultados en: {out}")


if __name__ == "__main__":
    main()

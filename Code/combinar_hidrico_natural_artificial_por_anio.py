from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    raise SystemExit("Se requiere geopandas para ejecutar este script.")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize(name: str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]", "_", str(name))
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def cargar_sistema_hidrico(base: Path) -> gpd.GeoDataFrame:
    gpkg = base / "Data" / "sistema_hidrico.gpkg"
    gdf = gpd.read_file(gpkg, layer="Sistema_hidrico")
    gdf.columns = [c.lower() for c in gdf.columns]
    if gdf.crs is None:
        gdf = gdf.set_crs(4686, allow_override=True)
    gdf = gdf.to_crs(3116)
    # Clasificar tipo Natural/Artificial
    def clasificar(elem: str) -> str:
        s = (elem or "").strip().lower()
        if "artificial" in s:
            return "Artificial"
        if "natural" in s:
            return "Natural"
        return "Desconocido"
    if "elemento" in gdf.columns:
        gdf["tipo"] = gdf["elemento"].astype(str).map(clasificar)
    else:
        gdf["tipo"] = "Desconocido"
    gdf["area_m2"] = gdf.geometry.area
    return gdf


def cargar_corredor_por_anio(base: Path) -> pd.Series:
    gpkg = base / "Data" / "corredor_ecologico_ronda.gpkg"
    if not gpkg.exists():
        return pd.Series(dtype="int64")
    gdf = gpd.read_file(gpkg, layer="corredor_ecologico_ronda")
    gdf.columns = [c.lower() for c in gdf.columns]
    if gdf.crs is None:
        gdf = gdf.set_crs(4686, allow_override=True)
    gdf = gdf.to_crs(3116)
    anio = pd.to_datetime(gdf.get("fecha_capt"), errors="coerce").dt.year
    gdf["anio"] = anio
    return gdf


def main():
    base = Path(__file__).resolve().parent.parent
    out = base / "Output" / "hidrico_natural_artificial"
    ensure_dir(out)

    sis = cargar_sistema_hidrico(base)

    # Totales globales por tipo
    totales = sis.groupby("tipo")["area_m2"].sum().reset_index()
    totales.to_csv(out / "totales_por_tipo.csv", index=False)

    # Intentar construir histórico mediante intersección con corredor por año
    corr = cargar_corredor_por_anio(base)
    if isinstance(corr, gpd.GeoDataFrame) and corr["anio"].notna().any():
        years = sorted(corr["anio"].dropna().astype(int).unique().tolist())
    else:
        years = []

    registros = []
    if years:
        for y in years:
            poly = corr[corr["anio"] == y]
            if poly.empty:
                continue
            inter = gpd.overlay(sis[["tipo", "geometry"]], poly[["geometry"]], how="intersection")
            if inter.empty:
                continue
            inter["area_m2"] = inter.geometry.area
            agg = inter.groupby("tipo")["area_m2"].sum().reset_index()
            agg.insert(0, "anio", y)
            registros.append(agg)
    # Si no hay años, construimos una fila única 'sin_anio' con totales
    if not registros:
        tmp = totales.copy()
        tmp.insert(0, "anio", pd.NA)
        registros = [tmp]

    resumen = pd.concat(registros, ignore_index=True)
    resumen.to_csv(out / "area_por_anio_y_tipo.csv", index=False)

    # Gráfico
    piv = resumen.pivot(index="anio", columns="tipo", values="area_m2").sort_index()
    # Asegurar numérico
    piv = piv.apply(pd.to_numeric, errors="coerce")
    if piv.notna().any().any():
        piv = piv.fillna(0)
        if piv.index.isna().any():
            piv.index = piv.index.astype("string").fillna("total")
        ax = piv.plot(kind="bar", stacked=True, figsize=(11, 6))
        ax.set_title("Área del sistema hídrico Natural vs Artificial por año")
        ax.set_xlabel("Año")
        ax.set_ylabel("Área (m²)")
        plt.tight_layout()
        plt.grid(axis="y", linestyle=":", alpha=0.5)
        plt.savefig(out / "area_natural_vs_artificial_por_anio.png", dpi=150)
        plt.close()

    print(f"Listo. Resultados en: {out}")


if __name__ == "__main__":
    main()

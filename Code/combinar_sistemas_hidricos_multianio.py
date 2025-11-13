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


def infer_year_from_gdf(gdf: gpd.GeoDataFrame, fallback_name: str) -> int | None:
    # 1) intentar desde columna 'acto_admin'
    # usar grupo CAPTURANTE para compatibilidad con pandas .str.extract
    year_pat = re.compile(r"((?:19|20)\d{2})")
    # mapeo explícito confirmado por el usuario
    fname = fallback_name.lower()
    if fname == "sistema_hidrico.gpkg":
        return 2020
    if fname == "sistema_hidrico2.gpkg":
        return 2004
    col = None
    for c in gdf.columns:
        if str(c).lower() in {"acto_admin", "acto", "acto_adm", "normativa"}:
            col = c
            break
    if col is not None:
        yrs = gdf[col].astype(str).str.extract(year_pat, expand=False)
        yrs = pd.to_numeric(yrs, errors="coerce").dropna().astype(int)
        if not yrs.empty:
            # escoger el año más frecuente
            return int(yrs.mode().iloc[0])
    # 2) intentar desde nombre de archivo
    m = year_pat.search(fallback_name)
    if m:
        return int(m.group(1))
    return None


def classify_type(elem: str) -> str:
    s = (elem or "").strip().lower()
    if "artificial" in s:
        return "Artificial"
    if "natural" in s:
        return "Natural"
    return "Desconocido"


def process_gpkg(path: Path) -> pd.DataFrame:
    layer = "Sistema_hidrico"
    gdf = gpd.read_file(path, layer=layer)
    gdf.columns = [c.lower() for c in gdf.columns]
    if gdf.crs is None:
        gdf = gdf.set_crs(4686, allow_override=True)
    gdf = gdf.to_crs(3116)
    # tipo
    if "elemento" in gdf.columns:
        gdf["tipo"] = gdf["elemento"].astype(str).map(classify_type)
    else:
        gdf["tipo"] = "Desconocido"
    gdf["area_m2"] = gdf.geometry.area
    year = infer_year_from_gdf(gdf, path.name)
    gdf["anio"] = year
    return gdf[["anio", "tipo", "area_m2"]]


def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "Data"
    out = base / "Output" / "sistemas_hidricos_multianio"
    ensure_dir(out)

    files = sorted([p for p in data_dir.glob("sistema_hidrico*.gpkg")])
    if not files:
        raise SystemExit("No se encontraron archivos sistema_hidrico*.gpkg en Data")

    frames = []
    for p in files:
        try:
            df = process_gpkg(p)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] No se pudo procesar {p.name}: {e}")
    if not frames:
        raise SystemExit("No se pudieron procesar archivos para combinar.")
    all_df = pd.concat(frames, ignore_index=True)

    # Si algún archivo no trae año, lo marcamos como 'desconocido' y no entra al gráfico por año
    all_df_known = all_df.dropna(subset=["anio"]).copy()
    if all_df_known.empty:
        print("No se pudo inferir año para los archivos. Exporto solo totales por tipo.")
        tot = all_df.groupby("tipo")["area_m2"].sum().reset_index()
        tot.to_csv(out / "totales_por_tipo.csv", index=False)
        return

    all_df_known["anio"] = all_df_known["anio"].astype(int)
    resumen = (
        all_df_known.groupby(["anio", "tipo"])['area_m2'].sum().reset_index()
    )
    resumen.to_csv(out / "area_por_anio_y_tipo.csv", index=False)

    # Gráficos y tabla resumen
    piv = resumen.pivot(index="anio", columns="tipo", values="area_m2").sort_index()
    piv = piv.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Tabla condensada: Natural, Artificial, Total y porcentajes
    cols = {c: c for c in piv.columns}
    natural = piv.get("Natural", pd.Series(0, index=piv.index))
    artificial = piv.get("Artificial", pd.Series(0, index=piv.index))
    total = natural.add(artificial, fill_value=0)
    resumen_tabla = (
        pd.DataFrame({
            "anio": piv.index.astype(int),
            "natural_m2": natural.values,
            "artificial_m2": artificial.values,
            "total_m2": total.values,
        })
        .assign(
            pct_natural=lambda d: (d["natural_m2"]/d["total_m2"]).where(d["total_m2"]>0).fillna(0),
            pct_artificial=lambda d: (d["artificial_m2"]/d["total_m2"]).where(d["total_m2"]>0).fillna(0),
        )
    )
    resumen_tabla.to_csv(out / "resumen_natural_artificial_por_anio.csv", index=False)

    # Gráfico combinado apilado
    ax = piv.plot(kind="bar", stacked=True, figsize=(11, 6))
    ax.set_title("Área del sistema hídrico por año: Natural vs Artificial")
    ax.set_xlabel("Año")
    ax.set_ylabel("Área (m²)")
    plt.tight_layout(); plt.grid(axis='y', linestyle=':', alpha=0.5)
    plt.savefig(out / "area_natural_vs_artificial_por_anio.png", dpi=150)
    plt.close()

    # Gráficos separados
    for col in [c for c in piv.columns if c in ("Natural", "Artificial")]:
        s = piv[col]
        plt.figure(figsize=(9,5))
        plt.plot(s.index, s.values, marker='o', linewidth=1.8)
        plt.title(f"Año vs {col}")
        plt.xlabel("Año"); plt.ylabel("Área (m²)")
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out / f"anio_vs_{col.lower()}.png", dpi=150)
        plt.close()

    print(f"Listo. Resultados en: {out}")


if __name__ == "__main__":
    main()

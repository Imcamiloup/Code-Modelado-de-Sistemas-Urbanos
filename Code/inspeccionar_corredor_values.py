from pathlib import Path
import pandas as pd
try:
    import geopandas as gpd
except Exception:
    raise SystemExit("Se requiere geopandas.")

base = Path(__file__).resolve().parent.parent
p = base / "Data" / "corredor_ecologico_ronda.gpkg"
layer = "corredor_ecologico_ronda"

gdf = gpd.read_file(p, layer=layer)
gdf.columns = [c.lower() for c in gdf.columns]

if "fecha_capt" in gdf.columns:
    gdf["anio"] = pd.to_datetime(gdf["fecha_capt"], errors="coerce").dt.year
else:
    gdf["anio"] = pd.NaT

print("Columnas:", list(gdf.columns))
print("Años:", sorted([int(x) for x in gdf["anio"].dropna().unique().tolist()]))
for col in ["categoria", "elemento", "eje", "tema"]:
    if col in gdf.columns:
        vals = gdf[col].dropna().astype(str).str.strip().str.upper().unique().tolist()
        print(col.upper(), "(n=", len(vals), "):", sorted(vals)[:50])

from pathlib import Path
try:
    import geopandas as gpd
except Exception:
    raise SystemExit("Se requiere geopandas.")

base = Path(__file__).resolve().parent.parent
p = base / "Data" / "sistema_hidrico.gpkg"
layer = "Sistema_hidrico"

gdf = gpd.read_file(p, layer=layer)
gdf.columns = [c.lower() for c in gdf.columns]
print("Columnas:", list(gdf.columns))
for col in ["componente", "categoria", "elemento", "nombre_tot"]:
    if col in gdf.columns:
        vals = gdf[col].dropna().astype(str).str.strip().unique().tolist()
        vals = sorted(vals)
        print(col.upper(), "(n=", len(vals), "):", vals[:60])

from pathlib import Path
import sqlite3
import json
import sys


def inspect_gpkg(path: Path):
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute(
        "SELECT table_name, data_type, identifier, description FROM gpkg_contents "
        "WHERE data_type IN ('features','tiles','attributes')"
    )
    rows = cur.fetchall()
    out = []
    for table_name, data_type, identifier, description in rows:
        cur.execute(f"PRAGMA table_info('{table_name}')")
        cols = [{"name": r[1], "type": r[2]} for r in cur.fetchall()]
        cur.execute(
            "SELECT column_name, geometry_type_name, srs_id FROM gpkg_geometry_columns WHERE table_name=?",
            (table_name,),
        )
        g = cur.fetchone()
        geom = None
        if g:
            geom = {"geom_column": g[0], "geom_type": g[1], "srs_id": g[2]}
        out.append(
            {
                "layer": table_name,
                "identifier": identifier,
                "type": data_type,
                "description": description,
                "geometry": geom,
                "columns": cols,
            }
        )
    con.close()
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        gpkg = Path(sys.argv[1]).resolve()
    else:
        gpkg = Path(__file__).resolve().parent.parent / "Data" / "sistema_hidrico.gpkg"
    if not gpkg.exists():
        raise SystemExit(f"No se encontró: {gpkg}")
    info = inspect_gpkg(gpkg)
    print(json.dumps(info, ensure_ascii=False, indent=2))

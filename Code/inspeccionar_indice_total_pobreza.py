from pathlib import Path
import pandas as pd

base = Path(__file__).resolve().parent.parent
xls = base / "Data" / "Indice_total_de_pobreza.xlsx"
if not xls.exists():
    raise SystemExit(f"No existe {xls}")

df = pd.read_excel(xls)
print("Columnas:", list(df.columns))
print(df.head(5).to_string(index=False))

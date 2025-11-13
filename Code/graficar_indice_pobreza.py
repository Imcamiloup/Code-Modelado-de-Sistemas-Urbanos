from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def detect_year_column(df: pd.DataFrame) -> pd.Series:
    candidates = ["Año", "Ano", "Anio", "Year", "Periodo", "PERIODO", "periodo", "Fecha", "FECHA", "fecha"]
    for c in candidates:
        if c in df.columns:
            s = df[c]
            if pd.api.types.is_datetime64_any_dtype(s):
                return s.dt.year
            y = pd.to_numeric(s, errors="coerce")
            if y.notna().any():
                return y
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
            if parsed.notna().any():
                return parsed.dt.year
    fecha_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    if fecha_cols:
        return df[fecha_cols[0]].dt.year
    # Fallback: intentar extraer un año de 4 dígitos con regex desde cualquier columna tipo objeto
    year_pattern = re.compile(r"(19|20)\d{2}")
    for c in df.columns:
        if df[c].dtype == object:
            ser = df[c].astype(str).str.extract(year_pattern, expand=False)
            cand = pd.to_numeric(ser, errors="coerce")
            if cand.notna().mean() > 0.5:  # si al menos la mitad parecen años
                return cand
    raise ValueError("No se pudo detectar una columna de año en el archivo.")


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = (script_dir.parent / "Data" / "indice_de_pobreza.xlsx").resolve()
    out_dir = (script_dir.parent / "Output" / "plots_indice_pobreza").resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    ensure_dir(out_dir)

    df = pd.read_excel(data_path)
    df = df.dropna(axis=1, how="all")
    print("Columnas en el archivo:", list(df.columns))

    anio = detect_year_column(df)
    df = df.assign(anio=anio)

    # Posibles nombres de columnas para IPM y Gini
    ipm_candidates = ["IPM", "Indice de Pobreza", "Índice de Pobreza", "Indice_Pobreza", "IPM (%)", "indice"]
    gini_candidates = ["Gini", "GINI", "gini", "Coeficiente de Gini", "Coef. Gini"]

    def pick(colnames, candidates):
        for c in candidates:
            if c in colnames:
                return c
        return None

    ipm_col = pick(df.columns, ipm_candidates)
    gini_col = pick(df.columns, gini_candidates)

    if ipm_col is None and gini_col is None:
        raise ValueError("No se encontraron columnas de IPM ni Gini en el archivo.")

    for c in [ipm_col, gini_col]:
        if c is None:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_cols = [c for c in [ipm_col, gini_col] if c is not None]
    agg = (
        df.groupby("anio")[agg_cols]
        .mean(numeric_only=True)
        .sort_index()
    )

    if ipm_col is not None:
        s = agg[ipm_col].dropna().sort_index()
        if not s.empty:
            plt.figure(figsize=(9, 5))
            plt.plot(s.index, s.values, marker="o", linewidth=1.8, color="#1f77b4")
            plt.title("Año vs IPM")
            plt.xlabel("Año")
            plt.ylabel(ipm_col)
            plt.grid(True, linestyle=":", alpha=0.5)
            f = out_dir / f"Anio_vs_{sanitize_filename(ipm_col)}.png"
            plt.tight_layout(); plt.savefig(f, dpi=150); plt.close()
            print(f"Gráfico IPM guardado en: {f}")
        else:
            print("Serie IPM vacía tras limpieza.")

    if gini_col is not None:
        s = agg[gini_col].dropna().sort_index()
        if not s.empty:
            plt.figure(figsize=(9, 5))
            plt.plot(s.index, s.values, marker="o", linewidth=1.8, color="#ff7f0e")
            plt.title("Año vs Gini")
            plt.xlabel("Año")
            plt.ylabel(gini_col)
            plt.grid(True, linestyle=":", alpha=0.5)
            f = out_dir / f"Anio_vs_{sanitize_filename(gini_col)}.png"
            plt.tight_layout(); plt.savefig(f, dpi=150); plt.close()
            print(f"Gráfico Gini guardado en: {f}")
        else:
            print("Serie Gini vacía tras limpieza.")


if __name__ == "__main__":
    main()

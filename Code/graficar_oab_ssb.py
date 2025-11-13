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
    candidates = [
        "Año", "Ano", "Anio", "Year", "Periodo", "PERIODO", "periodo",
    ]
    for c in candidates:
        if c in df.columns:
            s = df[c]
            # Si es datetime, tomar .dt.year
            if pd.api.types.is_datetime64_any_dtype(s):
                return s.dt.year
            # Intentar convertir directo a numérico (por si el texto es 2020, etc.)
            y = pd.to_numeric(s, errors="coerce")
            if y.notna().any():
                return y
            # Algunos libros traen fechas como texto; intentar parsear
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=True)
            if parsed.notna().any():
                return parsed.dt.year
    # Como fallback, buscar la primera columna datetime
    fecha_cols = [
        c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    if fecha_cols:
        return df[fecha_cols[0]].dt.year
    raise ValueError("No se pudo detectar una columna de año en el archivo.")


def main():
    script_dir = Path(__file__).resolve().parent
    data_path = (script_dir.parent / "Data" / "OAB - SSB - La Ciudad.xlsx").resolve()
    out_dir = (script_dir.parent / "Output" / "plots_oab_ssb").resolve()

    if not data_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {data_path}")

    ensure_dir(out_dir)

    # Leer primera hoja por defecto
    df = pd.read_excel(data_path)
    df = df.dropna(axis=1, how="all")
    print("Columnas en el archivo:", list(df.columns))

    # Detectar columna de año
    anio = detect_year_column(df)
    df = df.assign(anio=anio)

    # Columnas a graficar
    target_cols = ["CAC", "CAL", "CAPL"]
    disponibles = [c for c in target_cols if c in df.columns]
    if not disponibles:
        raise ValueError("No se encontraron las columnas CAC, CAL, CAPL en el archivo.")

    # Asegurar que los objetivos sean numéricos
    for c in disponibles:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Agregar por año (promedio)
    agg = (
        df.groupby("anio")[disponibles]
        .mean(numeric_only=True)
        .sort_index()
    )

    # Graficar en una sola figura
    plt.figure(figsize=(10, 6))
    for c in disponibles:
        s = agg[c].dropna().sort_index()
        if s.empty:
            continue
        plt.plot(s.index, s.values, marker="o", linewidth=1.8, label=c)

    plt.title("Año vs CAC, CAL y CAPL")
    plt.xlabel("Año")
    plt.ylabel("Valor")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(title="Indicador")

    out_file = out_dir / f"Anio_vs_{sanitize_filename('_'.join(disponibles))}.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

    print(f"Gráfico guardado en: {out_file}")


if __name__ == "__main__":
    main()

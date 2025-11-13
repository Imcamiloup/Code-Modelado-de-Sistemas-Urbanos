import os
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def sanitize_filename(name: str) -> str:
    name = str(name)
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name[:100]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_numeric(series: pd.Series, out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(series.index, series.values, marker="o", linewidth=1)
    plt.title(f"{series.name} (línea)")
    plt.xlabel("Índice")
    plt.ylabel("AVPH")
    plt.grid(True, linestyle=":", alpha=0.5)
    fname = out_dir / f"{sanitize_filename(series.name)}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_categorical(series: pd.Series, out_dir: Path, top_n: int = 30) -> None:
    counts = series.astype(str).value_counts().head(top_n)
    plt.figure(figsize=(10, 6))
    counts.plot(kind="bar")
    plt.title(f"{series.name} (barras - top {top_n})")
    plt.xlabel("Categoría")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5)
    fname = out_dir / f"{sanitize_filename(series.name)}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def plot_datetime(series: pd.Series, out_dir: Path, freq: str = "D") -> None:
    # Conteo por periodo (p. ej., por día)
    s = pd.to_datetime(series, errors="coerce").dropna()
    if s.empty:
        return
    counts = s.dt.to_period(freq).value_counts().sort_index()
    idx = counts.index.astype(str)
    plt.figure(figsize=(10, 5))
    plt.plot(idx, counts.values, marker="o", linewidth=1)
    plt.title(f"{series.name} (conteo por {freq})")
    plt.xlabel("Periodo")
    plt.ylabel("Conteo")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, linestyle=":", alpha=0.5)
    fname = out_dir / f"{sanitize_filename(series.name)}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()


def main():
    # Rutas relativas: este script está en ./Code; los datos en ../Data
    script_dir = Path(__file__).resolve().parent
    data_path = (script_dir.parent / "Data" /
                 "estructura_ecologica.xlsx").resolve()
    out_dir = (script_dir.parent / "Output" /
               "plots_estructura_ecologica").resolve()

    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de datos: {data_path}")

    ensure_dir(out_dir)

    # Leer Excel (primera hoja por defecto)
    df = pd.read_excel(data_path)

    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how="all")

    # Intentar inferir fechas
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(
                    df[col], errors="coerce", dayfirst=True)
                # Considerar datetime si una proporción relevante es válida
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
            except Exception:
                pass

    print(f"Columnas detectadas: {list(df.columns)}")
    print(f"Guardando gráficos en: {out_dir}")

    # Determinar año a partir de 'Periodo'
    anio = None
    if 'Periodo' in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df['Periodo']):
            anio = df['Periodo'].dt.year
        else:
            # Si ya es numérico o texto de año
            anio = pd.to_numeric(df['Periodo'], errors='coerce')
    else:
        # Si no existe 'Periodo', intentar con la primera columna tipo fecha
        fecha_cols = [
            c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
        if fecha_cols:
            anio = df[fecha_cols[0]].dt.year

    if anio is None or anio.isna().all():
        raise ValueError(
            "No se pudo determinar el año a partir de la columna 'Periodo'.")

    df = df.assign(anio=anio)

    # Columnas objetivo
    objetivos = ['APH', 'PUT', 'TAVU', 'AVPH']
    disponibles = [c for c in objetivos if c in df.columns]
    if not disponibles:
        raise ValueError(
            "Ninguna de las columnas objetivo ['APH', 'PUT', 'TAVU', 'AVPH'] está presente en los datos.")

    # Agregación por año (promedio)
    agg = (
        df.groupby('anio')[disponibles]
        .mean(numeric_only=True)
        .sort_index()
    )

    # Graficar cada serie Año vs columna
    for col in disponibles:
        s = agg[col].dropna()
        if s.empty:
            continue
        # Asegurar alineación X=índice de la serie filtrada
        s = s.sort_index()
        plt.figure(figsize=(10, 5))
        plt.plot(s.index, s.values, marker="o", linewidth=1)
        plt.title(f"Año vs {col}")
        plt.xlabel("Año")
        plt.ylabel(col)
        plt.grid(True, linestyle=":", alpha=0.5)
        fname = out_dir / f"Anio_vs_{sanitize_filename(col)}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
        plt.close()

    print("Proceso completado.")


if __name__ == "__main__":
    main()

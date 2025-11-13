from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    base = Path(__file__).resolve().parent.parent
    xls = base / "Data" / "Indice_total_de_pobreza.xlsx"
    out = base / "Output" / "indice_total_pobreza"
    ensure_dir(out)

    df = pd.read_excel(xls)
    # Normalizar nombres
    df = df.rename(columns={
        'Año': 'anio', 'Ano': 'anio', 'Anio': 'anio',
        'IPM': 'IPM', 'IPMo': 'IPM',
        'Gini': 'Gini', 'GINI': 'Gini',
        'IPC': 'IPC'
    })
    # Tipos numéricos
    for c in ['anio', 'IPM', 'Gini', 'IPC']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['anio']).sort_values('anio')

    # 1) Año vs IPM
    if 'IPM' in df.columns:
        s = df[['anio', 'IPM']].dropna()
        if not s.empty:
            plt.figure(figsize=(9,5))
            plt.plot(s['anio'], s['IPM'], marker='o', linewidth=1.8, color='#1f77b4')
            plt.title('Año vs IPM (Índice de Pobreza Monetaria)')
            plt.xlabel('Año'); plt.ylabel('IPM')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout(); plt.savefig(out / 'anio_vs_ipm.png', dpi=150); plt.close()

    # 2) Año vs Gini
    if 'Gini' in df.columns:
        s = df[['anio', 'Gini']].dropna()
        if not s.empty:
            plt.figure(figsize=(9,5))
            plt.plot(s['anio'], s['Gini'], marker='o', linewidth=1.8, color='#ff7f0e')
            plt.title('Año vs Gini')
            plt.xlabel('Año'); plt.ylabel('Gini')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout(); plt.savefig(out / 'anio_vs_gini.png', dpi=150); plt.close()

    # 3) Año vs IPC
    if 'IPC' in df.columns:
        s = df[['anio', 'IPC']].dropna()
        if not s.empty:
            plt.figure(figsize=(9,5))
            plt.plot(s['anio'], s['IPC'], marker='o', linewidth=1.8, color='#2ca02c')
            plt.title('Año vs IPC (Índice per cápita monetario)')
            plt.xlabel('Año'); plt.ylabel('IPC')
            plt.grid(True, linestyle=':', alpha=0.5)
            plt.tight_layout(); plt.savefig(out / 'anio_vs_ipc.png', dpi=150); plt.close()

    # Exportar CSV limpio
    df.to_csv(out / 'indice_total_de_pobreza_limpio.csv', index=False)
    print(f"Listo. Resultados en: {out}")


if __name__ == '__main__':
    main()

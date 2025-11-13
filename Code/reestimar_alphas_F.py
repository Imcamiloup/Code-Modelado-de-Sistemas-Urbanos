from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reestimación de alphas para F(t) evitando dominancia por escala.
# Objetivo: aproximar IPM (en proporción) usando una combinación de (IPM/100, Gini, 1/IPC) estandarizadas.
# Salida: alphas (estandarizados), serie F_norm y reporte.

LAMBDA = 1e-3  # ridge suave


def zscore(a: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(a))
    sd = float(np.std(a))
    if sd == 0:
        return np.zeros_like(a, dtype=float), mu, sd
    return (a - mu) / sd, mu, sd


def main():
    base = Path(__file__).resolve().parent.parent
    xls = base / "Data" / "Indice_total_de_pobreza.xlsx"
    out = base / "Output" / "indice_total_pobreza"
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xls)
    df = df.rename(columns={
        'Año': 'anio', 'Ano': 'anio', 'Anio': 'anio',
        'IPM': 'IPM', 'IPMo': 'IPM',
        'Gini': 'Gini', 'GINI': 'Gini',
        'IPC': 'IPC'
    })
    for c in ['anio', 'IPM', 'Gini', 'IPC']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['anio', 'IPM', 'Gini', 'IPC']).sort_values('anio').reset_index(drop=True)

    ipm_prop = (df['IPM'].to_numpy(float)) / 100.0
    gini = df['Gini'].to_numpy(float)
    inv_ipc = 1.0 / df['IPC'].to_numpy(float)

    # Estandarizar columnas de la fórmula
    z_ipm, mu_ipm, sd_ipm = zscore(ipm_prop)
    z_gini, mu_g, sd_g = zscore(gini)
    z_invipc, mu_inv, sd_inv = zscore(inv_ipc)

    Xz = np.column_stack([z_ipm, z_gini, z_invipc])
    yz = z_ipm.copy()  # objetivo: reproducir IPM (proporción) en escala estandarizada

    # Ridge: beta = (X'X + λI)^(-1) X'y
    I = np.eye(Xz.shape[1])
    beta = np.linalg.solve(Xz.T @ Xz + LAMBDA * I, Xz.T @ yz)
    alpha1, alpha2, alpha3 = beta.tolist()

    # Serie F en escala normalizada (z)
    Fz = Xz @ beta
    # Llevar Fz a [0,1] para uso práctico
    Fmin, Fmax = float(Fz.min()), float(Fz.max())
    Fr = (Fz - Fmin) / (Fmax - Fmin) if Fmax > Fmin else np.zeros_like(Fz)

    # Guardar resultados
    pd.DataFrame({
        'alpha1_z': [alpha1], 'alpha2_z': [alpha2], 'alpha3_z': [alpha3],
        'lambda_ridge': [LAMBDA]
    }).to_csv(out / 'alphas_F_estandarizadas.csv', index=False)

    df_out = df[['anio', 'IPM', 'Gini', 'IPC']].copy()
    df_out['F_norm'] = Fr
    df_out['Fz'] = Fz
    df_out.to_csv(out / 'F_estandarizado_por_anio.csv', index=False)

    # Gráfico Año vs F_norm
    plt.figure(figsize=(9,5))
    plt.plot(df['anio'], Fr, marker='o', linewidth=1.8)
    plt.title('Año vs F (normalizado)')
    plt.xlabel('Año'); plt.ylabel('F normalizado [0,1]')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(); plt.savefig(out / 'anio_vs_F_normalizado.png', dpi=150); plt.close()

    print('Listo: alphas y F normalizado generados en', out)


if __name__ == '__main__':
    main()

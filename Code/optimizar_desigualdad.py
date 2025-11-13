import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def cargar_datos() -> pd.DataFrame:
    base = Path(__file__).resolve().parent.parent
    xls = base / "Data" / "Indice_total_de_pobreza.xlsx"
    df = pd.read_excel(xls)
    # Normalizar nombres
    df = df.rename(columns={
        'Año': 'anio', 'Ano': 'anio', 'Anio': 'anio',
        'IPM': 'IPM', 'IPMo': 'IPM',
        'Gini': 'Gini', 'GINI': 'Gini',
        'IPC': 'IPC'
    })
    for c in ['anio', 'IPM', 'Gini', 'IPC']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['anio', 'IPM', 'Gini', 'IPC']).sort_values('anio').reset_index(drop=True)
    return df


def ols_with_intercept(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    # X no incluye intercepto; lo añadimos
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return beta  # [b0, b1, b2, ...]


def main():
    df = cargar_datos()

    # Modelar IPM como función de Gini y log(IPC)
    y = df['IPM'].to_numpy(float)
    x1 = df['Gini'].to_numpy(float)
    x2 = np.log(df['IPC'].to_numpy(float))

    X = np.column_stack([x1, x2])

    # Ajuste OLS con intercepto (todos los años)
    beta = ols_with_intercept(y, X)
    b0, b1, b2 = beta

    # Predicciones y métricas
    y_pred = b0 + X @ np.array([b1, b2])
    resid = y - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))

    # Coeficientes estandarizados (beta*)
    y_std = y.std(ddof=0)
    x_std = X.std(axis=0, ddof=0)
    beta_std = np.array([b1 * x_std[0] / y_std, b2 * x_std[1] / y_std])

    # Salidas
    base = Path(__file__).resolve().parent.parent
    out = base / "Output" / "indice_total_pobreza"
    out.mkdir(parents=True, exist_ok=True)

    # Coeficientes
    coef_df = pd.DataFrame({
        'intercepto': [b0], 'beta_gini': [b1], 'beta_log_ipc': [b2],
        'beta_gini_estandarizado': [beta_std[0]], 'beta_log_ipc_estandarizado': [beta_std[1]],
        'R2': [r2], 'RMSE': [rmse], 'MAE': [mae]
    })
    coef_df.to_csv(out / 'modelo_ipm_gini_logipc.csv', index=False)

    # Serie real vs predicha
    df_out = df[['anio']].copy()
    df_out['IPM_real'] = y
    df_out['IPM_pred'] = y_pred
    df_out.to_csv(out / 'ipm_real_vs_pred.csv', index=False)

    # Gráfica comparativa
    plt.figure(figsize=(10,5))
    plt.plot(df['anio'], y, marker='o', linewidth=1.6, label='IPM real')
    plt.plot(df['anio'], y_pred, marker='s', linewidth=1.6, label='IPM predicho')
    plt.title('IPM: real vs predicho (OLS con Gini y log(IPC))')
    plt.xlabel('Año'); plt.ylabel('IPM')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout(); plt.savefig(out / 'ipm_real_vs_pred.png', dpi=150); plt.close()

    # Validación temporal (train primeros años, test últimos)
    n = len(df)
    test_n = max(2, int(round(n * 0.25)))
    split = n - test_n
    Xtr, ytr = X[:split, :], y[:split]
    Xte, yte = X[split:, :], y[split:]
    beta_tv = ols_with_intercept(ytr, Xtr)
    b0_t, b1_t, b2_t = beta_tv
    ytr_pred = b0_t + Xtr @ np.array([b1_t, b2_t])
    yte_pred = b0_t + Xte @ np.array([b1_t, b2_t])
    # métricas
    def metrics(y_true, y_hat):
        res = y_true - y_hat
        ssr = float(np.sum(res**2)); sst = float(np.sum((y_true - y_true.mean())**2))
        return {
            'R2': 1 - ssr/sst if sst > 0 else np.nan,
            'RMSE': float(np.sqrt(np.mean(res**2))),
            'MAE': float(np.mean(np.abs(res)))
        }
    m_train = metrics(ytr, ytr_pred)
    m_test = metrics(yte, yte_pred)

    # Exportar reporte a Excel
    xlsx = out / 'reporte_modelo_ipm.xlsx'
    with pd.ExcelWriter(xlsx, engine='xlsxwriter') as writer:
        coef_df.to_excel(writer, sheet_name='coeficientes', index=False)
        df_out.to_excel(writer, sheet_name='serie_real_vs_pred', index=False)
        pd.DataFrame([{'conjunto':'train', **m_train}, {'conjunto':'test', **m_test}]).to_excel(writer, sheet_name='metricas', index=False)

    # Gráfico con split
    years = df['anio'].to_numpy()
    plt.figure(figsize=(10,5))
    plt.plot(years[:split], ytr, 'o-', label='Train real', alpha=0.8)
    plt.plot(years[:split], ytr_pred, 's--', label='Train pred', alpha=0.8)
    plt.plot(years[split:], yte, 'o-', label='Test real', alpha=0.8)
    plt.plot(years[split:], yte_pred, 's--', label='Test pred', alpha=0.8)
    plt.title('IPM real vs predicho (validación temporal)')
    plt.xlabel('Año'); plt.ylabel('IPM')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout(); plt.savefig(out / 'ipm_validacion_temporal.png', dpi=150); plt.close()

    print('Coeficientes OLS:')
    print(f'intercepto: {b0}\nbeta_gini: {b1}\nbeta_log_ipc: {b2}')
    print(f'R2: {r2}\nRMSE: {rmse}\nMAE: {mae}')


if __name__ == '__main__':
    main()

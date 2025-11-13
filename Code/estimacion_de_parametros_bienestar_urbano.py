import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ------------------------- Cargar datos -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / 'Data' / 'Consolidado_Bienestar_Urbano.xlsx'
df = pd.read_excel(file_path)


def pick_col(d, candidates):
    for c in candidates:
        if c in d.columns:
            return d[c].values.astype(float)
    return None


t_years = pick_col(df, ['Año', 'Anio', 'Year'])
if t_years is None:
    t_years = np.arange(len(df), dtype=int) + 2000
else:
    t_years = t_years.astype(int)

# Series requeridas
P_data = pick_col(df, ['Poblacion', 'Población', 'P'])
U_data = pick_col(df, ['Huella_Urbana', 'Huella Urbana',
                  'HuellaUrbana', 'Area_Urbanizada', 'Área_Urbanizada', 'U'])
E_data = pick_col(df, ['Infraestructura_Ecologica',
                  'Estructura_Ecologica', 'Area_Protegida', 'Áreas_Protegidas', 'E'])
W_data = pick_col(df, ['Bienestar', 'W', 'Bienestar_Urbano'])
F_data = pick_col(df, ['Desigualdad', 'Gini', 'Inequidad', 'F'])

if any(x is None for x in [P_data, U_data, E_data, W_data, F_data]):
    raise ValueError("Faltan columnas: se requieren P, U, E, W y F.")

# Asegurar longitudes iguales
n = min(map(len, [P_data, U_data, E_data, W_data, F_data, t_years]))
P_data, U_data, E_data, W_data, F_data, t_years = \
    P_data[:n], U_data[:n], E_data[:n], W_data[:n], F_data[:n], t_years[:n]

print(f"Filas usadas: {n}")

# ------------------ Normalización (z-score) ------------------
eps = 1e-9


def zfit(x):
    x = np.asarray(x, dtype=float)
    mu, sd = np.nanmean(x), np.nanstd(x) + eps
    return (x - mu)/sd, mu, sd


def zunscale(z, mu, sd):
    return z*sd + mu


Pz, P_mu, P_sd = zfit(P_data)
Uz, U_mu, U_sd = zfit(U_data)
Ez, E_mu, E_sd = zfit(E_data)
Wz_obs, W_mu, W_sd = zfit(W_data)
Fz, F_mu, F_sd = zfit(F_data)

# ------------------ Modelo: dWz/dt = a*Pz + bU*Uz + bE*Ez + c*Wz - d*Fz + c0 ------------------


def simulate_W(params):
    a, bU, bE, c, d, c0 = params
    Wz = np.empty(n, dtype=float)
    Wz[0] = Wz_obs[0]
    for t in range(n-1):
        dW = a*Pz[t] + bU*Uz[t] + bE*Ez[t] + c*Wz[t] - d*Fz[t] + c0
        Wz[t+1] = Wz[t] + dW  # dt = 1 año
    return Wz


def objective(params):
    Wz_sim = simulate_W(params)
    W_sim = zunscale(Wz_sim, W_mu, W_sd)
    return np.mean((W_sim - W_data)**2)

# ------------------ Estimación por OLS + refinamiento ------------------


def fit_ols(lambda_ridge=1e-6):
    # y = Wz[t+1] - Wz[t] = a Pz + bU Uz + bE Ez + c Wz - d Fz + c0
    y = Wz_obs[1:] - Wz_obs[:-1]
    X = np.column_stack([
        Pz[:-1],
        Uz[:-1],
        Ez[:-1],
        Wz_obs[:-1],
        -Fz[:-1],             # signo negativo incorporado => coef = d
        np.ones(len(y))       # intercepto c0
    ])
    XtX = X.T @ X
    beta = np.linalg.solve(XtX + lambda_ridge*np.eye(X.shape[1]), X.T @ y)
    a, bU, bE, c, d, c0 = beta
    return float(a), float(bU), float(bE), float(c), float(d), float(c0)


# 1) OLS
a_ols, bU_ols, bE_ols, c_ols, d_ols, c0_ols = fit_ols()
mse_ols = objective([a_ols, bU_ols, bE_ols, c_ols, d_ols, c0_ols])
print("Parámetros OLS:", {"a": a_ols, "bU": bU_ols,
      "bE": bE_ols, "c": c_ols, "d": d_ols, "c0": c0_ols})
print("MSE OLS:", mse_ols)

# 2) Refinamiento con límites
initial = [a_ols, bU_ols, bE_ols, c_ols, max(0.0, d_ols), c0_ols]
bounds = [(-3, 3), (-3, 3), (-3, 3), (-2, 2), (0, 3), (-1, 1)]
res = minimize(objective, initial, method='L-BFGS-B', bounds=bounds)

params_best = res.x
mse_opt = res.fun
if mse_ols < mse_opt:
    params_best = np.array(
        [a_ols, bU_ols, bE_ols, c_ols, d_ols, c0_ols], dtype=float)
    mse_opt = mse_ols
    print("Se mantiene OLS (mejor MSE).")
else:
    print("Se usa refinamiento L-BFGS-B (mejor MSE).")

a, bU, bE, c, d, c0 = params_best
print("Parámetros finales:", {"a": a, "bU": bU,
      "bE": bE, "c": c, "d": d, "c0": c0})
print("MSE final:", mse_opt)

# Exportar parámetros y estadísticos de normalización a JSON
params_output = {
    "parametros": {
        "a": float(a),
        "bU": float(bU),
        "bE": float(bE),
        "c": float(c),
        "d": float(d),
        "c0": float(c0)
    },
    "normalizacion": {
        "P": {"mu": float(P_mu), "sd": float(P_sd)},
        "U": {"mu": float(U_mu), "sd": float(U_sd)},
        "E": {"mu": float(E_mu), "sd": float(E_sd)},
        "W": {"mu": float(W_mu), "sd": float(W_sd)},
        "F": {"mu": float(F_mu), "sd": float(F_sd)}
    },
    "mse": float(mse_opt),
    "anio_inicio": int(t_years[0]),
    "anio_fin": int(t_years[-1])
}

params_file = PROJECT_ROOT / 'Output' / 'parametros_estimados.json'
with open(params_file, 'w', encoding='utf-8') as f:
    json.dump(params_output, f, indent=2, ensure_ascii=False)
print(f"Parámetros exportados a: {params_file}")

# ------------------ Simulación final y gráfico ------------------
Wz_sim = simulate_W(params_best)
W_sim = zunscale(Wz_sim, W_mu, W_sd)

# Crear carpeta Output y ruta del archivo
output_dir = PROJECT_ROOT / 'Output'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'ajuste_W_modelo.png'

plt.figure(figsize=(10, 6))
plt.plot(t_years, W_data, 'o-', label='W real', color='blue')
plt.plot(t_years, W_sim,  '--', label='W simulado', color='red')
plt.xlabel('Año')
plt.ylabel('Bienestar urbano (W)')
plt.title('dW/dt = a·P + bU·U + bE·E + c·W − d·F + c0 (z-score, dt=1)')
plt.legend()
plt.tight_layout()

# Guardar y mostrar
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Gráfica guardada en: {output_path}')

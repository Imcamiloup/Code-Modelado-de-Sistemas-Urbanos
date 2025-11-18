import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.integrate import odeint

# ------------------------- Cargar parámetros estimados -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
params_file = PROJECT_ROOT / 'Output' / 'parametros_estimados_ext.json'

if not params_file.exists():
    raise FileNotFoundError(f"No se encontró {params_file}")

with open(params_file, 'r', encoding='utf-8') as f:
    params_data = json.load(f)

params = params_data['parametros_ext']
alpha_W = params['alpha_W']
beta_W = params['beta_W']
gamma_W = params['gamma_W']
delta_W = params['delta_W']
lambda_U = params['lambda_U']
lambda_E = params['lambda_E']
alpha_F = params['alpha_F']
beta_F = params['beta_F']
r = params['r']
K_est = params['K_est']
gamma_F = params['gamma_F']

print("="*70)
print("SIMULACIÓN CORREGIDA DEL MODELO CONTINUO")
print("="*70)

# ------------------------- Cargar datos -------------------------
file_path = PROJECT_ROOT / 'Data' / 'Consolidado_Bienestar_Urbano.xlsx'
df = pd.read_excel(file_path)


def pick_col(d, candidates):
    for col in candidates:
        if col in d.columns:
            return d[col].values.astype(float)
    return None


t_years_data = pick_col(df, ['Año', 'Anio', 'Year'])
if t_years_data is None:
    t_years_data = np.arange(len(df), dtype=int) + 2012
else:
    t_years_data = t_years_data.astype(int)

P_data = pick_col(df, ['Poblacion', 'Población', 'P'])
U_data = pick_col(df, ['Huella_Urbana', 'Huella Urbana', 'HuellaUrbana', 'U'])
E_data = pick_col(df, ['Infraestructura_Ecologica',
                  'Estructura_Ecologica', 'E'])
W_data = pick_col(df, ['Bienestar', 'W'])
F_data = pick_col(df, ['Desigualdad', 'Gini', 'F'])

n_hist = len(t_years_data)
print(
    f"\nDatos históricos: {t_years_data[0]} a {t_years_data[-1]} ({n_hist} observaciones)")

# ------------------------- Normalización -------------------------
P_mu, P_sd = P_data.mean(), P_data.std()
U_mu, U_sd = U_data.mean(), U_data.std()
E_mu, E_sd = E_data.mean(), E_data.std()
W_mu, W_sd = W_data.mean(), W_data.std()
F_mu, F_sd = F_data.mean(), F_data.std()


def znorm(x, mu, sd):
    return (x - mu) / sd


def zunscale(z, mu, sd):
    return z * sd + mu


P_norm = znorm(P_data, P_mu, P_sd)
U_norm = znorm(U_data, U_mu, U_sd)
E_norm = znorm(E_data, E_mu, E_sd)
W_norm = znorm(W_data, W_mu, W_sd)
F_norm = znorm(F_data, F_mu, F_sd)

# ------------------------- Ajuste polinómico CONSERVADOR -------------------------
deg_U = 2
deg_E = 3

t_norm = np.linspace(0, 1, n_hist)

coeffs_U = np.polyfit(t_norm, U_norm, deg_U)
coeffs_E = np.polyfit(t_norm, E_norm, deg_E)

poly_U = np.poly1d(coeffs_U)
poly_E = np.poly1d(coeffs_E)

print(f"\nPolinomios ajustados (grados reducidos para estabilidad):")
print(f"  U(t): grado {deg_U}")
print(f"  E(t): grado {deg_E}")

# ------------------------- Sistema de EDOs -------------------------


def sistema_EDO(y, t_val, t_interp, U_series, E_series):
    """
    Sistema de EDOs con interpolación de U y E
    """
    P, F, W = y

    # Interpolar U y E en el tiempo actual
    U = np.interp(t_val, t_interp, U_series)
    E = np.interp(t_val, t_interp, E_series)

    # Limitar valores para estabilidad numérica
    P = np.clip(P, -5, 5)
    F = np.clip(F, -5, 5)
    W = np.clip(W, -5, 5)

    dP_dt = r * P * (1 - P / K_est) + gamma_F * F
    dF_dt = -alpha_F * F + beta_F * W
    dW_dt = alpha_W * P + beta_W * \
        (U + E) + gamma_W * W - delta_W * F + lambda_U * U + lambda_E * E

    return [dP_dt, dF_dt, dW_dt]


# ------------------------- SIMULACIÓN HISTÓRICA -------------------------
U_norm_fit = poly_U(t_norm)
E_norm_fit = poly_E(t_norm)

y0 = [P_norm[0], F_norm[0], W_norm[0]]

# Usar tiempo continuo normalizado
t_sim = np.linspace(0, n_hist-1, n_hist)
sol = odeint(sistema_EDO, y0, t_sim, args=(t_sim, U_norm_fit, E_norm_fit))

P_sim_norm = sol[:, 0]
F_sim_norm = sol[:, 1]
W_sim_norm = sol[:, 2]

P_sim = zunscale(P_sim_norm, P_mu, P_sd)
F_sim = zunscale(F_sim_norm, F_mu, F_sd)
W_sim = zunscale(W_sim_norm, W_mu, W_sd)
U_sim = zunscale(U_norm_fit, U_mu, U_sd)
E_sim = zunscale(E_norm_fit, E_mu, E_sd)

# ------------------------- PROYECCIÓN -------------------------
n_future = 5
t_years_future = np.arange(
    t_years_data[-1] + 1, t_years_data[-1] + n_future + 1)

# Extrapolar U y E linealmente
U_trend = (U_norm[-1] - U_norm[-3]) / 2
E_trend = (E_norm[-1] - E_norm[-3]) / 2

U_future_norm = np.array([U_norm[-1] + U_trend * (i+1)
                         for i in range(n_future)])
E_future_norm = np.array([E_norm[-1] + E_trend * (i+1)
                         for i in range(n_future)])

# Series completas para proyección
U_proj_norm = np.concatenate([U_norm_fit, U_future_norm])
E_proj_norm = np.concatenate([E_norm_fit, E_future_norm])
t_proj_full = np.arange(n_hist + n_future)

# Integrar proyección
y0_proj = [P_sim_norm[-1], F_sim_norm[-1], W_sim_norm[-1]]
t_proj_sim = np.linspace(n_hist-1, n_hist + n_future - 1, n_future + 1)

sol_proj = odeint(sistema_EDO, y0_proj, t_proj_sim,
                  args=(t_proj_full, U_proj_norm, E_proj_norm))

P_proj_norm = sol_proj[1:, 0]  # ✅ Excluir primer punto (duplicado)
F_proj_norm = sol_proj[1:, 1]
W_proj_norm = sol_proj[1:, 2]

P_proj = zunscale(P_proj_norm, P_mu, P_sd)
F_proj = zunscale(F_proj_norm, F_mu, F_sd)
W_proj = zunscale(W_proj_norm, W_mu, W_sd)
U_proj = zunscale(U_future_norm, U_mu, U_sd)
E_proj = zunscale(E_future_norm, E_mu, E_sd)

print(f"\n✅ Proyección completada: {t_years_future[0]} a {t_years_future[-1]}")
print(f"   Dimensiones verificadas:")
print(f"   - t_years_future: {len(t_years_future)}")
print(f"   - P_proj: {len(P_proj)}")

# ------------------------- GRÁFICOS -------------------------
output_dir = PROJECT_ROOT / 'Output' / 'Simulaciones_Continuo'
output_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
axes = axes.flatten()

variables = [
    ('Población', 'P', P_data, P_sim, P_proj, 'blue'),
    ('Huella Urbana', 'U', U_data, U_sim, U_proj, 'green'),
    ('Infraestructura Ecológica', 'E', E_data, E_sim, E_proj, 'forestgreen'),
    ('Desigualdad', 'F', F_data, F_sim, F_proj, 'orange'),
    ('Bienestar Urbano', 'W', W_data, W_sim, W_proj, 'red')
]

for idx, (nombre, simbolo, data_real, sim_hist, sim_proj, color) in enumerate(variables):
    ax = axes[idx]

    # Datos reales
    ax.plot(t_years_data, data_real, 'o', label=f'{simbolo} real',
            color=color, markersize=6, alpha=0.7, zorder=3)

    # Simulación histórica
    ax.plot(t_years_data, sim_hist, '-', label=f'{simbolo} simulado',
            color=color, linewidth=2.5, alpha=0.9, zorder=2)

    # Proyección (✅ Ahora ambos tienen longitud 5)
    ax.plot(t_years_future, sim_proj, '--', label=f'{simbolo} proyectado',
            color=color, linewidth=2.5, alpha=0.7, zorder=2)

    # Línea de separación
    ax.axvline(x=t_years_data[-1], color='gray',
               linestyle=':', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Año', fontsize=11)
    ax.set_ylabel(f'{nombre}', fontsize=11)
    ax.set_title(nombre, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3)

axes[5].axis('off')

plt.suptitle('Simulación Corregida del Sistema: Modelo Continuo de Bienestar Urbano\nBogotá 2012-2029',
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

output_path = output_dir / 'simulacion_completa_corregida.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {output_path}")

# ------------------------- MÉTRICAS DE AJUSTE -------------------------
print("\n" + "="*70)
print("MÉTRICAS DE AJUSTE (Período histórico)")
print("="*70)


def calcular_metricas(real, simulado):
    residuos = real - simulado
    rmse = np.sqrt(np.mean(residuos**2))
    mae = np.mean(np.abs(residuos))
    mape = np.mean(np.abs(residuos / (real + 1e-10))) * 100
    r2 = 1 - (np.sum(residuos**2) / np.sum((real - real.mean())**2))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R²': r2}


for nombre, simbolo, data_real, sim_hist, _, _ in variables:
    metricas = calcular_metricas(data_real, sim_hist)
    print(f"\n{nombre} ({simbolo}):")
    print(f"  RMSE = {metricas['RMSE']:.4f}")
    print(f"  MAE  = {metricas['MAE']:.4f}")
    print(f"  MAPE = {metricas['MAPE']:.2f}%")
    print(f"  R²   = {metricas['R²']:.4f}")

# ------------------------- TABLA DE RESULTADOS -------------------------
df_resultados = pd.DataFrame({
    'Año': np.concatenate([t_years_data, t_years_future]),
    'P_real': np.concatenate([P_data, [np.nan]*n_future]),
    'P_sim': np.concatenate([P_sim, P_proj]),
    'U_real': np.concatenate([U_data, [np.nan]*n_future]),
    'U_sim': np.concatenate([U_sim, U_proj]),
    'E_real': np.concatenate([E_data, [np.nan]*n_future]),
    'E_sim': np.concatenate([E_sim, E_proj]),
    'F_real': np.concatenate([F_data, [np.nan]*n_future]),
    'F_sim': np.concatenate([F_sim, F_proj]),
    'W_real': np.concatenate([W_data, [np.nan]*n_future]),
    'W_sim': np.concatenate([W_sim, W_proj]),
    'Segmento': ['Histórico']*n_hist + ['Proyección']*n_future
})

csv_path = output_dir / 'resultados_simulacion_continuo.csv'
xlsx_path = output_dir / 'resultados_simulacion_continuo.xlsx'

df_resultados.to_csv(csv_path, index=False, encoding='utf-8-sig')
df_resultados.to_excel(xlsx_path, index=False)

print(f"\n✅ Tabla de resultados guardada:")
print(f"   - {csv_path}")
print(f"   - {xlsx_path}")

# ------------------------- VALORES PROYECTADOS -------------------------
print("\n" + "="*70)
print("VALORES PROYECTADOS (2025-2029)")
print("="*70)

for i, year in enumerate(t_years_future):
    print(f"\nAño {year}:")
    print(f"  P = {P_proj[i]:.2f}")
    print(f"  U = {U_proj[i]:.2f}")
    print(f"  E = {E_proj[i]:.2f}")
    print(f"  F = {F_proj[i]:.4f}")
    print(f"  W = {W_proj[i]:.2f}")

print("\n" + "="*70)
print("✅ SIMULACIÓN COMPLETADA EXITOSAMENTE")
print("="*70)

plt.show()

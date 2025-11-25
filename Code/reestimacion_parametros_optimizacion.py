import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.integrate import odeint
from scipy.optimize import differential_evolution

# ------------------------- Cargar datos -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

print("="*70)
print("RE-ESTIMACIÓN DE PARÁMETROS CON OPTIMIZACIÓN GLOBAL")
print("="*70)
print(
    f"\nDatos: {t_years_data[0]}-{t_years_data[-1]} ({n_hist} observaciones)")

# ------------------------- Normalización -------------------------
P_mu, P_sd = P_data.mean(), P_data.std()
U_mu, U_sd = U_data.mean(), U_data.std()
E_mu, E_sd = E_data.mean(), E_data.std()
W_mu, W_sd = W_data.mean(), W_data.std()
F_mu, F_sd = F_data.mean(), F_data.std()


def znorm(x, mu, sd):
    return (x - mu) / sd


P_norm = znorm(P_data, P_mu, P_sd)
U_norm = znorm(U_data, U_mu, U_sd)
E_norm = znorm(E_data, E_mu, E_sd)
W_norm = znorm(W_data, W_mu, W_sd)
F_norm = znorm(F_data, F_mu, F_sd)

# ------------------------- Ajuste polinómico de U y E -------------------------
t_norm = np.linspace(0, 1, n_hist)

# Usar grados bajos para estabilidad
coeffs_U = np.polyfit(t_norm, U_norm, 2)
coeffs_E = np.polyfit(t_norm, E_norm, 3)

poly_U = np.poly1d(coeffs_U)
poly_E = np.poly1d(coeffs_E)

U_interp = poly_U(t_norm)
E_interp = poly_E(t_norm)

# ------------------------- Sistema de EDOs Simplificado -------------------------


def sistema_EDO(y, t, params, t_interp, U_series, E_series):
    """
    Sistema simplificado con menos parámetros para mejor identificabilidad
    """
    P, F, W = y

    # Desempaquetar parámetros
    r, gamma_F, alpha_F, beta_F, alpha_W, beta_W, gamma_W, delta_W = params

    # Interpolar U y E
    U = np.interp(t, t_interp, U_series)
    E = np.interp(t, t_interp, E_series)

    # EDOs simplificadas (sin lambda_U y lambda_E para reducir parámetros)
    dP_dt = r * P + gamma_F * F
    dF_dt = -alpha_F * F + beta_F * W
    dW_dt = alpha_W * P + beta_W * (U + E) + gamma_W * W - delta_W * F

    return [dP_dt, dF_dt, dW_dt]


# ------------------------- Función objetivo -------------------------


def objetivo(params):
    """
    Minimiza el error cuadrático medio entre datos y simulación
    """
    try:
        # Condiciones iniciales
        y0 = [P_norm[0], F_norm[0], W_norm[0]]

        # Integrar
        t_sim = np.linspace(0, n_hist-1, n_hist)
        sol = odeint(sistema_EDO, y0, t_sim, args=(
            params, t_sim, U_interp, E_interp))

        P_sim = sol[:, 0]
        F_sim = sol[:, 1]
        W_sim = sol[:, 2]

        # Error normalizado (dar igual peso a cada variable)
        error_P = np.mean((P_norm - P_sim)**2)
        error_F = np.mean((F_norm - F_sim)**2)
        error_W = np.mean((W_norm - W_sim)**2)

        # Error total (suma ponderada)
        # Dar más peso a F (más variable)
        error_total = error_P + 2*error_F + error_W

        return error_total

    except:
        return 1e10  # Penalizar parámetros que causan errores numéricos


# ------------------------- Optimización -------------------------
print("\n🔄 Iniciando optimización global (esto puede tomar varios minutos)...")

# Límites de los parámetros [r, gamma_F, alpha_F, beta_F, alpha_W, beta_W, gamma_W, delta_W]
bounds = [
    (-0.5, 0.5),      # r (tasa de crecimiento poblacional)
    (-5, 15),         # gamma_F (efecto de desigualdad en población)
    (0, 1),           # alpha_F (decaimiento de desigualdad)
    (-0.01, 0.01),    # beta_F (efecto de bienestar en desigualdad)
    (-0.01, 0.01),    # alpha_W (efecto de población en bienestar)
    (-10, 10),        # beta_W (efecto de infraestructura en bienestar)
    (-1, 0),          # gamma_W (auto-regulación de bienestar, debe ser negativo)
    (-1, 1)           # delta_W (efecto de desigualdad en bienestar)
]

# Optimización con algoritmo evolutivo
result = differential_evolution(
    objetivo,
    bounds,
    maxiter=300,
    popsize=30,
    atol=1e-6,
    seed=42,
    workers=1,
    updating='deferred',
    disp=True
)

params_opt = result.x
error_final = result.fun

print(f"\n✅ Optimización completada!")
print(f"   Error final: {error_final:.6f}")

# ------------------------- Parámetros optimizados -------------------------
r_opt, gamma_F_opt, alpha_F_opt, beta_F_opt, alpha_W_opt, beta_W_opt, gamma_W_opt, delta_W_opt = params_opt

print("\n" + "="*70)
print("PARÁMETROS OPTIMIZADOS")
print("="*70)
print(f"  r       = {r_opt:.6f}  (tasa crecimiento poblacional)")
print(f"  γ_F     = {gamma_F_opt:.6f}  (desigualdad → población)")
print(f"  α_F     = {alpha_F_opt:.6f}  (decaimiento desigualdad)")
print(f"  β_F     = {beta_F_opt:.6f}  (bienestar → desigualdad)")
print(f"  α_W     = {alpha_W_opt:.6f}  (población → bienestar)")
print(f"  β_W     = {beta_W_opt:.6f}  (infraestructura → bienestar)")
print(f"  γ_W     = {gamma_W_opt:.6f}  (auto-regulación bienestar)")
print(f"  δ_W     = {delta_W_opt:.6f}  (desigualdad → bienestar)")

# ------------------------- Simulación con parámetros optimizados -------------------------
y0 = [P_norm[0], F_norm[0], W_norm[0]]
t_sim = np.linspace(0, n_hist-1, n_hist)
sol_opt = odeint(sistema_EDO, y0, t_sim, args=(
    params_opt, t_sim, U_interp, E_interp))

P_sim_norm = sol_opt[:, 0]
F_sim_norm = sol_opt[:, 1]
W_sim_norm = sol_opt[:, 2]

# Desnormalizar
P_sim = P_sim_norm * P_sd + P_mu
F_sim = F_sim_norm * F_sd + F_mu
W_sim = W_sim_norm * W_sd + W_mu
U_sim = U_interp * U_sd + U_mu
E_sim = E_interp * E_sd + E_mu

# ------------------------- Métricas -------------------------
print("\n" + "="*70)
print("MÉTRICAS DE AJUSTE")
print("="*70)


def metricas(real, sim):
    rmse = np.sqrt(np.mean((real - sim)**2))
    mae = np.mean(np.abs(real - sim))
    r2 = 1 - np.sum((real - sim)**2) / np.sum((real - real.mean())**2)
    return rmse, mae, r2


for nombre, real, sim in [('P', P_data, P_sim), ('F', F_data, F_sim), ('W', W_data, W_sim)]:
    rmse, mae, r2 = metricas(real, sim)
    print(f"\n{nombre}:")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")
    print(f"  R²   = {r2:.4f}")

# ------------------------- Gráficos comparativos -------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

variables = [
    ('Población', P_data, P_sim, 'blue'),
    ('Huella Urbana', U_data, U_sim, 'green'),
    ('Infraestructura Ecológica', E_data, E_sim, 'forestgreen'),
    ('Desigualdad', F_data, F_sim, 'orange'),
    ('Bienestar Urbano', W_data, W_sim, 'red')
]

for idx, (nombre, real, sim, color) in enumerate(variables):
    ax = axes[idx]
    ax.plot(t_years_data, real, 'o', label='Datos reales',
            color=color, markersize=7, alpha=0.7)
    ax.plot(t_years_data, sim, '-', label='Simulación optimizada',
            color=color, linewidth=2.5)
    ax.set_xlabel('Año', fontsize=11)
    ax.set_ylabel(nombre, fontsize=11)
    ax.set_title(nombre, fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

axes[5].axis('off')

plt.suptitle('Ajuste del Modelo con Parámetros Optimizados\nBogotá 2012-2024',
             fontsize=16, fontweight='bold')
plt.tight_layout()

output_dir = PROJECT_ROOT / 'Output'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'ajuste_optimizado.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✅ Gráfico guardado: {output_path}")

# ------------------------- Guardar parámetros -------------------------
params_dict = {
    'parametros_optimizados': {
        'r': float(r_opt),
        'gamma_F': float(gamma_F_opt),
        'alpha_F': float(alpha_F_opt),
        'beta_F': float(beta_F_opt),
        'alpha_W': float(alpha_W_opt),
        'beta_W': float(beta_W_opt),
        'gamma_W': float(gamma_W_opt),
        'delta_W': float(delta_W_opt)
    },
    'error_ajuste': float(error_final),
    'normalizacion': {
        'P': {'mu': float(P_mu), 'sd': float(P_sd)},
        'U': {'mu': float(U_mu), 'sd': float(U_sd)},
        'E': {'mu': float(E_mu), 'sd': float(E_sd)},
        'W': {'mu': float(W_mu), 'sd': float(W_sd)},
        'F': {'mu': float(F_mu), 'sd': float(F_sd)}
    }
}

params_file = output_dir / 'parametros_optimizados.json'
with open(params_file, 'w', encoding='utf-8') as f:
    json.dump(params_dict, f, indent=2, ensure_ascii=False)

print(f"✅ Parámetros guardados: {params_file}")

plt.show()

print("\n" + "="*70)
print("✅ PROCESO COMPLETADO")
print("="*70)

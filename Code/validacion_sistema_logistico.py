import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "Output_Correcciones"
params_file = OUT_DIR / "parametros_sistema_logistico.json"
DATA_FILE = PROJECT_ROOT / "Data" / "Consolidado_Bienestar_Urbano.xlsx"

# Cargar parámetros
with open(params_file, 'r', encoding='utf-8') as f:
    resultado = json.load(f)

pars_opt = resultado["parametros_optimizados"]
stats = resultado["normalizacion"]
years = np.array(resultado["years"])

# Cargar datos originales


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return df[n].values.astype(float)
    return None


df = pd.read_excel(DATA_FILE)
P_orig = pick_col(df, ["Poblacion", "Población", "P"])[:len(years)]
U_orig = pick_col(df, ["Huella_Urbana", "Huella Urbana", "U"])[:len(years)]
E_orig = pick_col(df, ["Infraestructura_Ecologica",
                  "Estructura_Ecologica", "E"])[:len(years)]
F_orig = pick_col(df, ["Desigualdad", "Gini", "F"])[:len(years)]
W_orig = pick_col(df, ["Bienestar", "W", "Bienestar_Urbano"])[:len(years)]

# Normalizar


def z(x, mu, sd):
    return (x - mu) / sd


Pz = z(P_orig, stats["P"]["mu"], stats["P"]["sd"])
Uz = z(U_orig, stats["U"]["mu"], stats["U"]["sd"])
Ez = z(E_orig, stats["E"]["mu"], stats["E"]["sd"])
Fz = z(F_orig, stats["F"]["mu"], stats["F"]["sd"])
Wz = z(W_orig, stats["W"]["mu"], stats["W"]["sd"])

# Suavizar U, E


def suavizar(x, deg):
    t = np.linspace(0, 1, len(x))
    coeff = np.polyfit(t, x, deg)
    return np.poly1d(coeff)


U_poly = suavizar(Uz, 2)
E_poly = suavizar(Ez, 3)

# Sistema EDO


def edo(t, y, pars, U_poly, E_poly, T_max):
    P, F, W = y
    s = min(1.0, t / T_max)
    U = U_poly(s)
    E = E_poly(s)
    r, gF, aF, bF, aW, bW, gW, dW, c0, K = [pars[k] for k in
                                            ["r", "gamma_F", "alpha_F", "beta_F", "alpha_W", "beta_W", "gamma_W", "delta_W", "c0", "K"]]
    dP = r * P * (1 - P / K) + gF * F
    dF = -aF * F + bF * W
    dW = aW * P + bW * (U + E) + gW * W - dW * F + c0
    return [dP, dF, dW]


# Simular
N = len(years)
t_eval = np.arange(N, dtype=float)
y0 = [Pz[0], Fz[0], Wz[0]]
sol = solve_ivp(edo, (0, N-1), y0, t_eval=t_eval,
                args=(pars_opt, U_poly, E_poly, N-1),
                rtol=1e-6, atol=1e-8)

P_sim, F_sim, W_sim = sol.y

# Desnormalizar


def desnorm(x_z, mu, sd):
    return x_z * sd + mu


P_sim_real = desnorm(P_sim, stats["P"]["mu"], stats["P"]["sd"])
F_sim_real = desnorm(F_sim, stats["F"]["mu"], stats["F"]["sd"])
W_sim_real = desnorm(W_sim, stats["W"]["mu"], stats["W"]["sd"])

# Métricas


def metricas(real, sim):
    rmse = np.sqrt(np.mean((real - sim)**2))
    mae = np.mean(np.abs(real - sim))
    r2 = 1 - np.sum((real - sim)**2) / np.sum((real - np.mean(real))**2)
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


m_P = metricas(P_orig, P_sim_real)
m_F = metricas(F_orig, F_sim_real)
m_W = metricas(W_orig, W_sim_real)

print("\n" + "="*60)
print("VALIDACIÓN DEL AJUSTE")
print("="*60)
print(
    f"Población:   RMSE={m_P['RMSE']:.2f}, MAE={m_P['MAE']:.2f}, R²={m_P['R2']:.4f}")
print(
    f"Desigualdad: RMSE={m_F['RMSE']:.4f}, MAE={m_F['MAE']:.4f}, R²={m_F['R2']:.4f}")
print(
    f"Bienestar:   RMSE={m_W['RMSE']:.2f}, MAE={m_W['MAE']:.2f}, R²={m_W['R2']:.4f}")
print("="*60)

# Gráfico
fig, axs = plt.subplots(3, 1, figsize=(10, 9), dpi=120)

axs[0].plot(years, P_orig, 'o-', label='Observado',
            color='tab:blue', markersize=5)
axs[0].plot(years, P_sim_real, 's--', label='Simulado',
            color='tab:orange', markersize=4)
axs[0].set_ylabel('Población')
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[0].set_title(f"Ajuste Población (R²={m_P['R2']:.3f})")

axs[1].plot(years, F_orig, 'o-', label='Observado',
            color='tab:blue', markersize=5)
axs[1].plot(years, F_sim_real, 's--', label='Simulado',
            color='tab:orange', markersize=4)
axs[1].set_ylabel('Desigualdad (Gini)')
axs[1].legend()
axs[1].grid(alpha=0.3)
axs[1].set_title(f"Ajuste Desigualdad (R²={m_F['R2']:.3f})")

axs[2].plot(years, W_orig, 'o-', label='Observado',
            color='tab:blue', markersize=5)
axs[2].plot(years, W_sim_real, 's--', label='Simulado',
            color='tab:orange', markersize=4)
axs[2].set_ylabel('Bienestar')
axs[2].set_xlabel('Año')
axs[2].legend()
axs[2].grid(alpha=0.3)
axs[2].set_title(f"Ajuste Bienestar (R²={m_W['R2']:.3f})")

plt.tight_layout()
fig.savefig(OUT_DIR / "validacion_sistema_logistico.png", bbox_inches='tight')
print(f"\nGráfico guardado: {OUT_DIR / 'validacion_sistema_logistico.png'}")

# Guardar métricas
metricas_out = {
    "Poblacion": m_P,
    "Desigualdad": m_F,
    "Bienestar": m_W
}
with open(OUT_DIR / "metricas_validacion.json", "w", encoding="utf-8") as f:
    json.dump(metricas_out, f, indent=2, ensure_ascii=False)
print(f"Métricas guardadas: {OUT_DIR / 'metricas_validacion.json'}")

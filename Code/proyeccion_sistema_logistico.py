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

with open(params_file, 'r', encoding='utf-8') as f:
    resultado = json.load(f)

pars_opt = resultado["parametros_optimizados"]
stats = resultado["normalizacion"]
years_hist = np.array(resultado["years"])

# Si K es muy pequeño, ajustar manualmente
if pars_opt['K'] < 1.0:
    print(
        f"⚠️ K={pars_opt['K']:.2f} muy bajo, ajustando a 2.0 para proyección")
    pars_opt['K'] = 2.0

# Cargar datos para obtener valores finales


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return df[n].values.astype(float)
    return None


df = pd.read_excel(DATA_FILE)
P_orig = pick_col(df, ["Poblacion", "Población", "P"])[:len(years_hist)]
U_orig = pick_col(df, ["Huella_Urbana", "Huella Urbana", "U"])[
    :len(years_hist)]
E_orig = pick_col(df, ["Infraestructura_Ecologica",
                  "Estructura_Ecologica", "E"])[:len(years_hist)]
F_orig = pick_col(df, ["Desigualdad", "Gini", "F"])[:len(years_hist)]
W_orig = pick_col(df, ["Bienestar", "W", "Bienestar_Urbano"])[:len(years_hist)]

# Normalizar últimos valores


def z(x, mu, sd):
    return (x - mu) / sd


y_final = [
    z(P_orig[-1], stats["P"]["mu"], stats["P"]["sd"]),
    z(F_orig[-1], stats["F"]["mu"], stats["F"]["sd"]),
    z(W_orig[-1], stats["W"]["mu"], stats["W"]["sd"])
]

print(
    f"Condiciones iniciales (normalizadas): P={y_final[0]:.4f}, F={y_final[1]:.4f}, W={y_final[2]:.4f}")
print(f"Parámetro K estimado: {pars_opt['K']:.4f}")

# DIAGNÓSTICO: verificar si P está cerca de K
if abs(y_final[0]) > 0.8 * pars_opt['K']:
    print(
        f"⚠️ ADVERTENCIA: P_z ({y_final[0]:.2f}) cercano a K ({pars_opt['K']:.2f}) → inestabilidad potencial")

# Extender U, E con últimos valores observados (escenario constante)
U_last_norm = z(U_orig[-1], stats["U"]["mu"], stats["U"]["sd"])
E_last_norm = z(E_orig[-1], stats["E"]["mu"], stats["E"]["sd"])


def U_fut(t):
    return U_last_norm


def E_fut(t):
    return E_last_norm

# Sistema proyección con PROTECCIÓN contra división por K pequeña


def edo_proj(t, y, pars):
    P, F, W = y
    U = U_fut(t)
    E = E_fut(t)
    r, gF, aF, bF, aW, bW, gW, delta_W_par, c0, K = [pars[k] for k in
                                                     ["r", "gamma_F", "alpha_F", "beta_F", "alpha_W", "beta_W", "gamma_W", "delta_W", "c0", "K"]]

    # CORRECCIÓN: limitar P/K para evitar explosión
    ratio_PK = np.clip(P / K, -5, 5)  # Evitar valores extremos

    dP = r * P * (1 - ratio_PK) + gF * F
    dF = -aF * F + bF * W
    dW = aW * P + bW * (U + E) + gW * W - delta_W_par * F + c0
    return [dP, dF, dW]


# Proyectar 5 años
t_proj = np.arange(0, 6)
sol_proj = solve_ivp(edo_proj, (0, 5), y_final, t_eval=t_proj,
                     args=(pars_opt,), rtol=1e-6, atol=1e-8, method='RK45')

if not sol_proj.success:
    print(f"❌ ERROR en integración: {sol_proj.message}")
    # Intentar con método más robusto
    sol_proj = solve_ivp(edo_proj, (0, 5), y_final, t_eval=t_proj,
                         args=(pars_opt,), rtol=1e-5, atol=1e-7, method='BDF')

P_proj, F_proj, W_proj = sol_proj.y

# Desnormalizar


def desnorm(x_z, mu, sd):
    return x_z * sd + mu


P_proj_real = desnorm(P_proj, stats["P"]["mu"], stats["P"]["sd"])
F_proj_real = desnorm(F_proj, stats["F"]["mu"], stats["F"]["sd"])
W_proj_real = desnorm(W_proj, stats["W"]["mu"], stats["W"]["sd"])

years_proj = years_hist[-1] + t_proj

print("\n" + "="*60)
print("PROYECCIÓN 5 AÑOS (escenario constante U,E)")
print("="*60)
for i, y in enumerate(years_proj):
    print(
        f"Año {int(y)}: P={P_proj_real[i]:>9,.0f}, F={F_proj_real[i]:.4f}, W={W_proj_real[i]:>7.2f}")
print("="*60)

# Gráfico proyección
fig, axs = plt.subplots(3, 1, figsize=(10, 9), dpi=120)

axs[0].plot(years_hist, P_orig, 'o-', label='Histórico',
            color='tab:blue', markersize=5)
axs[0].plot(years_proj, P_proj_real, 's--',
            label='Proyección', color='tab:red', markersize=5)
axs[0].set_ylabel('Población')
axs[0].legend()
axs[0].grid(alpha=0.3)
axs[0].set_title("Proyección Población")

axs[1].plot(years_hist, F_orig, 'o-', label='Histórico',
            color='tab:blue', markersize=5)
axs[1].plot(years_proj, F_proj_real, 's--',
            label='Proyección', color='tab:red', markersize=5)
axs[1].set_ylabel('Desigualdad (Gini)')
axs[1].legend()
axs[1].grid(alpha=0.3)
axs[1].set_title("Proyección Desigualdad")

axs[2].plot(years_hist, W_orig, 'o-', label='Histórico',
            color='tab:blue', markersize=5)
axs[2].plot(years_proj, W_proj_real, 's--',
            label='Proyección', color='tab:red', markersize=5)
axs[2].set_ylabel('Bienestar')
axs[2].set_xlabel('Año')
axs[2].legend()
axs[2].grid(alpha=0.3)
axs[2].set_title("Proyección Bienestar")

plt.tight_layout()
fig.savefig(OUT_DIR / "proyeccion_5anios.png", bbox_inches='tight')
print(f"\nGráfico guardado: {OUT_DIR / 'proyeccion_5anios.png'}")

# Guardar JSON
proj_out = {
    "escenario": "Constante (U,E últimos valores observados)",
    "years": years_proj.tolist(),
    "Poblacion": P_proj_real.tolist(),
    "Desigualdad": F_proj_real.tolist(),
    "Bienestar": W_proj_real.tolist()
}
with open(OUT_DIR / "proyeccion_5anios.json", "w", encoding="utf-8") as f:
    json.dump(proj_out, f, indent=2, ensure_ascii=False)
print(f"Proyección guardada: {OUT_DIR / 'proyeccion_5anios.json'}")

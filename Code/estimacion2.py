import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ============================================================
# 1. Carga de datos
# ============================================================

# Ajusta la ruta si es necesario
file_path = 'Data/Consolidado_Bienestar_Urbano.xlsx'
data = pd.read_excel(file_path)

# Aseguramos que todo sea float
t = data['Año'].values.astype(float)
P = data['Poblacion'].values.astype(float)
U = data['Huella_Urbana'].values.astype(float)
E = data['Infraestructura_Ecologica'].values.astype(float)
F = data['Desigualdad'].values.astype(float)
W = data['Bienestar'].values.astype(float)

# ============================================================
# 2. Ajuste polinómico para cada variable
#    (los mismos grados que ya venías usando)
# ============================================================

deg_P = 3
deg_U = 3
deg_E = 10
deg_F = 4
deg_W = 4

coef_P = np.polyfit(t, P, deg_P)
coef_U = np.polyfit(t, U, deg_U)
coef_E = np.polyfit(t, E, deg_E)
coef_F = np.polyfit(t, F, deg_F)
coef_W = np.polyfit(t, W, deg_W)

pP = np.poly1d(coef_P)
pU = np.poly1d(coef_U)
pE = np.poly1d(coef_E)
pF = np.poly1d(coef_F)
pW = np.poly1d(coef_W)

# Series suavizadas (opcional, para comparar con datos)
P_s = pP(t)
U_s = pU(t)
E_s = pE(t)
F_s = pF(t)
W_s = pW(t)

# ============================================================
# 3. Derivadas dP/dt, dU/dt, dE/dt, dF/dt, dW/dt
#    usando las derivadas de los polinomios
# ============================================================

dpP = np.polyder(pP)
dpU = np.polyder(pU)
dpE = np.polyder(pE)
dpF = np.polyder(pF)
dpW = np.polyder(pW)

dP_dt = dpP(t)
dU_dt = dpU(t)
dE_dt = dpE(t)
dF_dt = dpF(t)
dW_dt = dpW(t)

# ============================================================
# 4. Estimación de parámetros en las EDOs
# ============================================================

# ------------------------------------------------------------
# 4.1. Ecuación de W(t):
# dW/dt = α_W P + β_W (U+E) + γ_W W - δ_W F + λ_U U + λ_E E
#        = [P, U+E, W, -F, U, E] · [α_W, β_W, γ_W, δ_W, λ_U, λ_E]^T
# ------------------------------------------------------------

X_W = np.column_stack([
    P_s,          # para α_W
    U_s + E_s,    # para β_W
    W_s,          # para γ_W
    -F_s,         # para δ_W
    U_s,          # para λ_U
    E_s           # para λ_E
])
y_W = dW_dt

theta_W, residuals_W, rank_W, s_W = np.linalg.lstsq(X_W, y_W, rcond=None)
alpha_W, beta_W, gamma_W, delta_W, lambda_U, lambda_E = theta_W

# ------------------------------------------------------------
# 4.2. Ecuación de F(t):
# dF/dt = -α_F F + β_F W
#        = [-F, W] · [α_F, β_F]^T
# ------------------------------------------------------------

X_F = np.column_stack([
    -F_s,  # para α_F
    W_s    # para β_F
])
y_F = dF_dt

theta_F, residuals_F, rank_F, s_F = np.linalg.lstsq(X_F, y_F, rcond=None)
alpha_F, beta_F = theta_F

# ------------------------------------------------------------
# 4.3. (Opcional) Ecuación de P(t) en versión aproximada:
# dP/dt ≈ r P - (r/K) P^2 + γ_F F
#       = [P, -P^2, F] · [r, r/K, γ_F]^T
# Nota: aquí ignoramos la dependencia detallada de K(U,E,F),
#       así que es una aproximación que puedes comentar en el texto.
# ------------------------------------------------------------

X_P = np.column_stack([
    P_s,          # para r
    -P_s**2,      # para r/K
    F_s           # para γ_F
])
y_P = dP_dt

theta_P, residuals_P, rank_P, s_P = np.linalg.lstsq(X_P, y_P, rcond=None)
r_est, r_over_K_est, gamma_F = theta_P
K_est = r_est / r_over_K_est if r_over_K_est != 0 else np.inf

# ============================================================
# 5. Mostrar resultados
# ============================================================

print("\n===== Parámetros estimados para el modelo modificado =====\n")

print("Ecuación de W(t): dW/dt = α_W P + β_W (U+E) + γ_W W - δ_W F + λ_U U + λ_E E\n")
print(f"  α_W    = {alpha_W:.6e}")
print(f"  β_W    = {beta_W:.6e}")
print(f"  γ_W    = {gamma_W:.6e}")
print(f"  δ_W    = {delta_W:.6e}")
print(f"  λ_U    = {lambda_U:.6e}")
print(f"  λ_E    = {lambda_E:.6e}")
if residuals_W.size > 0:
    print(f"  Suma de cuadrados de residuos (W): {residuals_W[0]:.6e}")
print()

print("Ecuación de F(t): dF/dt = -α_F F + β_F W\n")
print(f"  α_F    = {alpha_F:.6e}")
print(f"  β_F    = {beta_F:.6e}")
if residuals_F.size > 0:
    print(f"  Suma de cuadrados de residuos (F): {residuals_F[0]:.6e}")
print()

print("Ecuación aproximada de P(t): dP/dt ≈ r P - (r/K) P^2 + γ_F F\n")
print(f"  r      = {r_est:.6e}")
print(f"  r/K    = {r_over_K_est:.6e}")
print(f"  K_est  = {K_est:.6e}")
print(f"  γ_F    = {gamma_F:.6e}")
if residuals_P.size > 0:
    print(f"  Suma de cuadrados de residuos (P): {residuals_P[0]:.6e}")
print()

print("===========================================================")

# Guardar parámetros extendidos para el análisis de fase
PROJECT_ROOT = Path(__file__).resolve().parents[1]
out_path = PROJECT_ROOT / 'Output' / 'parametros_estimados_ext.json'
out_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "parametros_ext": {
        "alpha_W": float(alpha_W),
        "beta_W": float(beta_W),
        "gamma_W": float(gamma_W),
        "delta_W": float(delta_W),
        "lambda_U": float(lambda_U),
        "lambda_E": float(lambda_E),
        "alpha_F": float(alpha_F),
        "beta_F": float(beta_F),
        "r": float(r_est),
        "r_over_K": float(r_over_K_est),
        "K_est": float(K_est),
        "gamma_F": float(gamma_F)
    }
}

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"Parámetros extendidos guardados en: {out_path}")

# (Opcional) Puedes guardar estos parámetros en un archivo si lo deseas:
# np.savez('parametros_modelo_modificado.npz',
#          alpha_W=alpha_W, beta_W=beta_W, gamma_W=gamma_W, delta_W=delta_W,
#          lambda_U=lambda_U, lambda_E=lambda_E,
#          alpha_F=alpha_F, beta_F=beta_F,
#          r=r_est, r_over_K=r_over_K_est, K_est=K_est, gamma_F=gamma_F)

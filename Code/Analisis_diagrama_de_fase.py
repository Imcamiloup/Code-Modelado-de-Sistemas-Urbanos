import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import json
from pathlib import Path
import pandas as pd

# Cargar parámetros estimados
PROJECT_ROOT = Path(__file__).resolve().parents[1]
params_file = PROJECT_ROOT / 'Output' / 'parametros_estimados_ext.json'

with open(params_file, 'r', encoding='utf-8') as f:
    params = json.load(f)['parametros_ext']

# Extraer parámetros
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

print("Parámetros cargados:")
print(f"α_W={alpha_W:.6e}, β_W={beta_W:.6e}, γ_W={gamma_W:.6e}")
print(f"δ_W={delta_W:.6e}, λ_U={lambda_U:.6e}, λ_E={lambda_E:.6e}")
print(f"α_F={alpha_F:.6e}, β_F={beta_F:.6e}")
print(f"r={r:.6e}, K={K_est:.6e}, γ_F={gamma_F:.6e}\n")

# Cargar datos para obtener valores promedio de U y E (normalizados)
file_path = PROJECT_ROOT / 'Data' / 'Consolidado_Bienestar_Urbano.xlsx'
data = pd.read_excel(file_path)

# Normalizar U y E
U_mean = data['Huella_Urbana'].mean()
U_std = data['Huella_Urbana'].std()
E_mean = data['Infraestructura_Ecologica'].mean()
E_std = data['Infraestructura_Ecologica'].std()

# Usar valores normalizados promedio como constantes
U_fixed = (data['Huella_Urbana'].mean() - U_mean) / U_std  # ≈ 0
E_fixed = (data['Infraestructura_Ecologica'].mean() - E_mean) / E_std  # ≈ 0

print(f"Valores fijos (normalizados):")
print(f"  U = {U_fixed:.6f}")
print(f"  E = {E_fixed:.6f}\n")

# Definir las variables simbólicas del sistema reducido
P, F, W = sp.symbols('P F W')

# Sistema de ecuaciones dinámicas con U y E como constantes
# dW/dt = α_W*P + β_W*(U+E) + γ_W*W - δ_W*F + λ_U*U + λ_E*E
dWdt = alpha_W*P + beta_W*(U_fixed + E_fixed) + gamma_W*W - \
    delta_W*F + lambda_U*U_fixed + lambda_E*E_fixed

# dF/dt = -α_F*F + β_F*W
dFdt = -alpha_F*F + beta_F*W

# dP/dt = r*P*(1 - P/K) + γ_F*F
dPdt = r*P*(1 - P/K_est) + gamma_F*F

# Crear el sistema de ecuaciones
system = [dPdt, dFdt, dWdt]

# Paso 1: Calcular el Jacobiano simbólico
variables = [P, F, W]
jacobian_matrix = sp.Matrix([[sp.diff(eq, var)
                            for var in variables] for eq in system])

print("Jacobiano simbólico (sistema reducido):")
sp.pprint(jacobian_matrix)
print()

# Convertir sistema y jacobiano a funciones numéricas
system_numeric = [sp.lambdify(variables, eq, 'numpy') for eq in system]
jacobian_numeric = sp.lambdify(variables, jacobian_matrix, 'numpy')


def system_func(x):
    """Evalúa el sistema de ecuaciones"""
    return [float(f(*x)) for f in system_numeric]


# Paso 2: Encontrar MÚLTIPLES puntos fijos con diferentes condiciones iniciales
initial_guesses = [
    [0.0, 0.0, 0.0],     # Trivial
    [0.5, 0.1, 0.5],     # Moderados
    [1.0, 0.5, 1.0],     # Altos
    [2.0, 1.0, 2.0],     # Muy altos
    [5.0, 2.0, 5.0],     # Extremos
    [-0.5, 0.2, -0.5],   # Bajos
    [-1.0, 0.5, -1.0],   # Muy bajos
    [10.0, 5.0, 10.0],   # Outliers
    # Combinaciones asimétricas
    [3.0, 0.1, -2.0],
    [-2.0, 1.5, 3.0],
    [0.0, 2.0, 0.0],
]

fixed_points = []
for i, x0 in enumerate(initial_guesses):
    try:
        sol = fsolve(system_func, x0, full_output=True)
        if sol[2] == 1:  # Convergencia exitosa
            P_fp, F_fp, W_fp = sol[0]
            residual = system_func(sol[0])
            max_residual = max(abs(r) for r in residual)

            if max_residual < 1e-6:  # Verificar que sea realmente un punto fijo
                # Evitar duplicados
                is_duplicate = False
                for existing in fixed_points:
                    if np.allclose([P_fp, F_fp, W_fp], existing, atol=1e-4):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    fixed_points.append([P_fp, F_fp, W_fp])
                    print(f"\n✓ Punto fijo #{len(fixed_points)} encontrado:")
                    print(f"  P = {P_fp:.6f}, F = {F_fp:.6f}, W = {W_fp:.6f}")
                    print(f"  Residual máximo: {max_residual:.2e}")
    except:
        pass

if not fixed_points:
    print("\n⚠ Solo se encontró el punto trivial (0,0,0)")
    fixed_points = [[0.0, 0.0, 0.0]]

print(f"\n{'='*60}")
print(f"Total de puntos fijos encontrados: {len(fixed_points)}")
print(f"{'='*60}\n")

# Analizar estabilidad de TODOS los puntos fijos
for idx, (P0, F0, W0) in enumerate(fixed_points):
    print(f"\n{'─'*60}")
    print(f"ANÁLISIS DEL PUNTO FIJO #{idx+1}")
    print(f"{'─'*60}")
    print(f"  P = {P0:.6f}")
    print(f"  F = {F0:.6f}")
    print(f"  W = {W0:.6f}")

    # Evaluar Jacobiano
    J_eval = np.array(jacobian_numeric(P0, F0, W0), dtype=float)
    eigenvalues = np.linalg.eigvals(J_eval)

    print("\nValores propios:")
    for i, ev in enumerate(eigenvalues):
        if np.isreal(ev):
            print(f"  λ_{i+1} = {ev.real:.6f}")
        else:
            print(f"  λ_{i+1} = {ev.real:.6f} + {ev.imag:.6f}i")

    real_parts = [ev.real if np.iscomplex(ev) else ev for ev in eigenvalues]
    if all(rp < 0 for rp in real_parts):
        stability = "🟢 Estable (atractor)"
    elif all(rp > 0 for rp in real_parts):
        stability = "🔴 Inestable (repulsor)"
    else:
        stability = "🟡 Punto silla (inestable)"

    print(f"\nEstabilidad: {stability}")

# Usar el punto fijo MÁS INTERESANTE para el diagrama (no trivial si existe)
if len(fixed_points) > 1:
    # Elegir el punto con mayor magnitud (no trivial)
    norms = [np.linalg.norm(fp) for fp in fixed_points]
    best_idx = np.argmax(norms)
    P0, F0, W0 = fixed_points[best_idx]
    print(f"\n📊 Usando punto fijo #{best_idx+1} para el diagrama de fase")
else:
    P0, F0, W0 = fixed_points[0]
    print(f"\n📊 Usando punto trivial para el diagrama de fase")

# Paso 5: Generar el diagrama de fase (P vs. W)
delta_P = 0.4
delta_W = 0.4
P_values = np.linspace(max(0, P0 - delta_P), P0 + delta_P, 25)
W_values = np.linspace(W0 - delta_W, W0 + delta_W, 25)
P_grid, W_grid = np.meshgrid(P_values, W_values)

# Fijar F en su valor del punto fijo para el diagrama 2D
dPdt_pw = dPdt.subs({F: F0})
dWdt_pw = dWdt.subs({F: F0})

dPdt_fun = sp.lambdify((P, W), dPdt_pw, 'numpy')
dWdt_fun = sp.lambdify((P, W), dWdt_pw, 'numpy')

dPdt_values = dPdt_fun(P_grid, W_grid)
dWdt_values = dWdt_fun(P_grid, W_grid)

# Determinar estabilidad del punto seleccionado
J_final = np.array(jacobian_numeric(P0, F0, W0), dtype=float)
eigs_final = np.linalg.eigvals(J_final)
real_parts_final = [ev.real if np.iscomplex(ev) else ev for ev in eigs_final]
if all(rp < 0 for rp in real_parts_final):
    stability = "Estable (atractor)"
elif all(rp > 0 for rp in real_parts_final):
    stability = "Inestable (repulsor)"
else:
    stability = "Punto silla"

# Graficar el diagrama de fase
plt.figure(figsize=(12, 8))
plt.streamplot(P_values, W_values, dPdt_values, dWdt_values,
               color='blue', linewidth=1.2, density=1.8, arrowsize=1.5)
plt.xlabel('Población (P) normalizada', fontsize=13)
plt.ylabel('Bienestar Urbano (W) normalizado', fontsize=13)
plt.title(f'Diagrama de Fase: Población vs Bienestar Urbano\n(F={F0:.4f}, U={U_fixed:.4f}, E={E_fixed:.4f})',
          fontsize=14)
plt.grid(True, alpha=0.3)
plt.scatter([P0], [W0], color='red', s=200,
            label=f'Punto fijo: P={P0:.3f}, W={W0:.3f}\n({stability})',
            zorder=5, edgecolors='black', linewidths=2.5)

plt.legend(fontsize=10)
plt.tight_layout()

# Guardar
output_file = PROJECT_ROOT / 'Output' / 'diagrama_fase_P_W.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Diagrama de fase guardado en: {output_file}")
plt.show()

from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.optimize import fsolve
from numpy.linalg import eig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "Output_Correcciones" / "Fase_Completo"
OUT_DIR.mkdir(parents=True, exist_ok=True)

params_file = PROJECT_ROOT / "Output_Correcciones" / \
    "parametros_sistema_logistico.json"
with open(params_file, 'r', encoding='utf-8') as f:
    resultado = json.load(f)

p = resultado["parametros_optimizados"]
r, gF, aF, bF, aW, bW, gW, delta_W_val, c0, K = [p[k] for k in
                                                 ["r", "gamma_F", "alpha_F", "beta_F", "alpha_W", "beta_W", "gamma_W", "delta_W", "c0", "K"]]

print("="*70)
print("ANÁLISIS DE FASE COMPLETO")
print("="*70)
print(f"\nParámetros cargados:")
print(f"r={r:.4f}, γ_F={gF:.4f}, α_F={aF:.4f}, β_F={bF:.6f}")
print(f"α_W={aW:.6f}, β_W={bW:.4f}, γ_W={gW:.4f}, δ_W={delta_W_val:.4f}, c0={c0:.4f}, K={K:.4f}")

# ========== 1. BÚSQUEDA DE PUNTOS FIJOS ==========
print("\n" + "="*70)
print("1. BÚSQUEDA DE PUNTOS FIJOS")
print("="*70)


def sistema_equilibrio(y, U_eq=0.0, E_eq=0.0):
    """Sistema en equilibrio (derivadas = 0)"""
    P, F, W = y
    eq1 = r * P * (1 - P / K) + gF * F
    eq2 = -aF * F + bF * W
    eq3 = aW * P + bW * (U_eq + E_eq) + gW * W - delta_W_val * F + c0
    return [eq1, eq2, eq3]


# Búsqueda desde múltiples condiciones iniciales
initial_guesses = [
    [0, 0, 0],
    [K/2, 0, 0],
    [K, 0, 0],
    [0.5, 0.1, 0],
    [1.0, 0.2, 0.5],
    [-0.5, -0.1, -0.5],
]

puntos_fijos = []
for guess in initial_guesses:
    try:
        sol = fsolve(sistema_equilibrio, guess, args=(0, 0), full_output=True)
        pf = sol[0]
        info = sol[1]
        # Verificar convergencia
        residual = np.linalg.norm(info['fvec'])
        if residual < 1e-6:
            # Verificar si ya existe (evitar duplicados)
            es_nuevo = True
            for pf_prev in puntos_fijos:
                if np.linalg.norm(pf - pf_prev) < 1e-4:
                    es_nuevo = False
                    break
            if es_nuevo:
                puntos_fijos.append(pf)
    except:
        pass

print(f"\nPuntos fijos encontrados: {len(puntos_fijos)}")
for i, pf in enumerate(puntos_fijos):
    print(f"  PF{i+1}: P={pf[0]:7.4f}, F={pf[1]:7.4f}, W={pf[2]:7.4f}")

# ========== 2. JACOBIANO Y ESTABILIDAD ==========
print("\n" + "="*70)
print("2. ANÁLISIS DE ESTABILIDAD (Jacobiano)")
print("="*70)


def jacobiano(P, F, W):
    """Matriz Jacobiana del sistema"""
    J = np.zeros((3, 3))
    # dP/dP, dP/dF, dP/dW
    J[0, 0] = r * (1 - 2*P/K)
    J[0, 1] = gF
    J[0, 2] = 0
    # dF/dP, dF/dF, dF/dW
    J[1, 0] = 0
    J[1, 1] = -aF
    J[1, 2] = bF
    # dW/dP, dW/dF, dW/dW
    J[2, 0] = aW
    J[2, 1] = -delta_W_val
    J[2, 2] = gW
    return J


estabilidad_data = []

for i, pf in enumerate(puntos_fijos):
    P_eq, F_eq, W_eq = pf
    J = jacobiano(P_eq, F_eq, W_eq)
    eigenvalues, eigenvectors = eig(J)

    # Clasificar estabilidad
    partes_reales = np.real(eigenvalues)
    if np.all(partes_reales < 0):
        tipo = "Estable (atractor)"
    elif np.all(partes_reales > 0):
        tipo = "Inestable (repulsor)"
    else:
        tipo = "Silla (inestable)"

    estabilidad_data.append({
        "punto": pf,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "tipo": tipo
    })

    print(f"\n--- Punto Fijo {i+1}: ({P_eq:.4f}, {F_eq:.4f}, {W_eq:.4f}) ---")
    print(f"Tipo: {tipo}")
    print("Valores propios:")
    for j, val in enumerate(eigenvalues):
        if np.iscomplex(val):
            print(f"  λ{j+1} = {val.real:.6f} + {val.imag:.6f}i")
        else:
            print(f"  λ{j+1} = {val.real:.6f}")
    print("Vectores propios (columnas):")
    print(eigenvectors)

# ========== 3. DIAGRAMAS DE FASE 2D (IMÁGENES SEPARADAS) ==========
print("\n" + "="*70)
print("3. GENERANDO DIAGRAMAS DE FASE 2D")
print("="*70)


def campo_vectorial(X1, X2, idx1, idx2, vals_fijos):
    """
    Calcula campo vectorial para un plano 2D
    idx1, idx2: índices de variables (0=P, 1=F, 2=W)
    vals_fijos: valores de la tercera variable
    """
    shape = X1.shape
    dX1 = np.zeros(shape)
    dX2 = np.zeros(shape)

    for i in range(shape[0]):
        for j in range(shape[1]):
            y = [0, 0, 0]
            y[idx1] = X1[i, j]
            y[idx2] = X2[i, j]
            # Tercera variable fija
            tercera_idx = 3 - idx1 - idx2
            y[tercera_idx] = vals_fijos

            P, F, W = y
            dP = r * P * (1 - P / K) + gF * F
            dF = -aF * F + bF * W
            dW = aW * P + bW * 0 + gW * W - delta_W_val * F + c0

            dy = [dP, dF, dW]
            dX1[i, j] = dy[idx1]
            dX2[i, j] = dy[idx2]

    return dX1, dX2


# Configuración de planos
planos = [
    {"idx": (0, 2), "labels": ("P", "W"), "fijo_idx": 1,
     "fijo_val": 0.0, "title": "P-W (F=0)", "filename": "fase_PW.png"},
    {"idx": (0, 1), "labels": ("P", "F"), "fijo_idx": 2,
     "fijo_val": 0.0, "title": "P-F (W=0)", "filename": "fase_PF.png"},
    {"idx": (1, 2), "labels": ("F", "W"), "fijo_idx": 0,
     "fijo_val": K/2, "title": f"F-W (P={K/2:.2f})", "filename": "fase_FW.png"},
]

# Generar cada diagrama en archivo separado
for plano in planos:
    fig, ax = plt.subplots(figsize=(8, 7), dpi=120)

    idx1, idx2 = plano["idx"]
    label1, label2 = plano["labels"]

    # Malla
    x1_range = np.linspace(-2, 2, 30) if idx1 == 0 else np.linspace(-1, 1, 30)
    x2_range = np.linspace(-2, 2, 30) if idx2 == 2 else np.linspace(-1, 1, 30)
    X1, X2 = np.meshgrid(x1_range, x2_range)

    dX1, dX2 = campo_vectorial(X1, X2, idx1, idx2, plano["fijo_val"])

    # Streamplot
    speed = np.sqrt(dX1**2 + dX2**2)
    ax.streamplot(X1, X2, dX1, dX2, color=speed, cmap='viridis',
                  density=1.5, linewidth=1.2, arrowsize=1.5)

    # Plotear puntos fijos proyectados
    for i, pf in enumerate(puntos_fijos):
        pf_proj = [pf[idx1], pf[idx2]]
        color = 'red' if estabilidad_data[i]["tipo"] == "Estable (atractor)" else 'yellow'
        marker = 'o' if estabilidad_data[i]["tipo"] == "Estable (atractor)" else 'x'
        ax.plot(pf_proj[0], pf_proj[1], marker, color=color, markersize=12,
                markeredgewidth=2.5, markeredgecolor='black',
                label=f'PF{i+1}: {estabilidad_data[i]["tipo"]}')

    ax.set_xlabel(f'{label1} (normalizado)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{label2} (normalizado)', fontsize=13, fontweight='bold')
    ax.set_title(plano["title"], fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    if len(puntos_fijos) > 0:
        ax.legend(fontsize=10, loc='best', framealpha=0.9)

    plt.tight_layout()
    output_file = OUT_DIR / plano["filename"]
    fig.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Gráfico guardado: {output_file}")
    plt.close()

# ========== 4. DIAGRAMA 3D ==========
print("\n" + "="*70)
print("4. GENERANDO DIAGRAMA DE FASE 3D")
print("="*70)


fig = plt.figure(figsize=(12, 10), dpi=120)
ax = fig.add_subplot(111, projection='3d')

# Trayectorias desde varios puntos iniciales


def sistema_ode(t, y):
    P, F, W = y
    dP = r * P * (1 - P / K) + gF * F
    dF = -aF * F + bF * W
    dW = aW * P + gW * W - delta_W_val * F + c0
    return [dP, dF, dW]


trayectorias_init = [
    [0.5, 0.1, 0.2],
    [1.0, 0.3, -0.5],
    [K*0.8, 0.2, 0.0],
    [-0.5, -0.2, 0.3],
]

colors_traj = ['blue', 'green', 'purple', 'orange']

for i, y0 in enumerate(trayectorias_init):
    sol = solve_ivp(sistema_ode, (0, 20), y0, t_eval=np.linspace(0, 20, 200),
                    rtol=1e-6, atol=1e-8, method='RK45')
    if sol.success:
        ax.plot(sol.y[0], sol.y[1], sol.y[2], color=colors_traj[i],
                alpha=0.7, linewidth=2, label=f'Trayectoria {i+1}')
        # Marcar inicio
        ax.scatter(y0[0], y0[1], y0[2], color=colors_traj[i],
                   s=80, marker='o', edgecolors='black', linewidths=2)

# Plotear puntos fijos
for i, pf in enumerate(puntos_fijos):
    color = 'red' if estabilidad_data[i]["tipo"] == "Estable (atractor)" else 'yellow'
    marker = 'o' if estabilidad_data[i]["tipo"] == "Estable (atractor)" else 'x'
    ax.scatter(pf[0], pf[1], pf[2], color=color, s=200, marker=marker,
               edgecolors='black', linewidths=2.5, label=f'PF{i+1}: {estabilidad_data[i]["tipo"]}')

ax.set_xlabel('P (normalizado)', fontsize=12, fontweight='bold')
ax.set_ylabel('F (normalizado)', fontsize=12, fontweight='bold')
ax.set_zlabel('W (normalizado)', fontsize=12, fontweight='bold')
ax.set_title('Diagrama de Fase 3D: P-F-W', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3)

output_3d = OUT_DIR / "diagrama_fase_3D.png"
fig.savefig(output_3d, bbox_inches='tight', dpi=150)
print(f"Gráfico guardado: {output_3d}")
plt.close()

# ========== 5. EXPORTAR RESULTADOS ==========
print("\n" + "="*70)
print("5. EXPORTANDO RESULTADOS")
print("="*70)

resultados = {
    "puntos_fijos": [
        {
            "id": i+1,
            "P": float(pf[0]),
            "F": float(pf[1]),
            "W": float(pf[2]),
            "estabilidad": estabilidad_data[i]["tipo"],
            "eigenvalues": [
                {"real": float(np.real(ev)), "imag": float(np.imag(ev))}
                for ev in estabilidad_data[i]["eigenvalues"]
            ]
        }
        for i, pf in enumerate(puntos_fijos)
    ]
}

output_json = OUT_DIR / "analisis_estabilidad.json"
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

print(f"Resultados guardados: {output_json}")
print("\n" + "="*70)
print("ANÁLISIS COMPLETO FINALIZADO")
print("="*70)
print("\nArchivos generados:")
print(f"  - {OUT_DIR / 'fase_PW.png'}")
print(f"  - {OUT_DIR / 'fase_PF.png'}")
print(f"  - {OUT_DIR / 'fase_FW.png'}")
print(f"  - {OUT_DIR / 'diagrama_fase_3D.png'}")
print(f"  - {OUT_DIR / 'analisis_estabilidad.json'}")

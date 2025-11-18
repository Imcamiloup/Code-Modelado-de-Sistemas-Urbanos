import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sympy as sp
from scipy.integrate import odeint

# Paso 1: Cargar los datos desde el archivo Excel
PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / 'Data' / 'Consolidado_Bienestar_Urbano.xlsx'
data = pd.read_excel(file_path)

# Cargar parámetros estimados
params_file = PROJECT_ROOT / 'Output' / 'parametros_estimados_ext.json'

with open(params_file, 'r', encoding='utf-8') as f:
    params_data = json.load(f)

# Extraer parámetros
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

# Normalizar los datos


def normalizar(serie):
    return (serie - serie.mean()) / serie.std()


P_norm = normalizar(data['Poblacion'].values)
U_norm = normalizar(data['Huella_Urbana'].values)
E_norm = normalizar(data['Infraestructura_Ecologica'].values)
F_norm = normalizar(data['Desigualdad'].values)
W_norm = normalizar(data['Bienestar'].values)

# Valores fijos promedio para variables no graficadas
U_fixed = 0.0
E_fixed = 0.0

# Definir las ecuaciones del sistema


def dP_dt(P, F, W, U, E):
    return r * P * (1 - P / K_est) + gamma_F * F


def dF_dt(P, F, W, U, E):
    return -alpha_F * F + beta_F * W


def dW_dt(P, F, W, U, E):
    return alpha_W * P + beta_W * (U + E) + gamma_W * W - delta_W * F + lambda_U * U + lambda_E * E


# Combinaciones de variables para diagramas de fase
combinaciones = [
    ('P', 'W', 'F', dP_dt, dW_dt, 'Población', 'Bienestar'),
    ('P', 'F', 'W', dP_dt, dF_dt, 'Población', 'Desigualdad'),
    ('F', 'W', 'P', dF_dt, dW_dt, 'Desigualdad', 'Bienestar'),
]

# Crear directorio de salida si no existe
output_dir = PROJECT_ROOT / 'Output' / 'Diagramas_Fase'
output_dir.mkdir(parents=True, exist_ok=True)

# Generar diagramas de fase
for var1_name, var2_name, var3_name, func1, func2, label1, label2 in combinaciones:
    print(f"\nGenerando diagrama de fase: {label1} vs {label2}")

    # Determinar valores de las variables
    if var1_name == 'P':
        var1_data = P_norm
    elif var1_name == 'F':
        var1_data = F_norm
    elif var1_name == 'W':
        var1_data = W_norm

    if var2_name == 'P':
        var2_data = P_norm
    elif var2_name == 'F':
        var2_data = F_norm
    elif var2_name == 'W':
        var2_data = W_norm

    if var3_name == 'P':
        var3_fixed = 0.0
    elif var3_name == 'F':
        var3_fixed = 0.0
    elif var3_name == 'W':
        var3_fixed = 0.0

    # Crear malla para el campo vectorial
    delta = 0.4
    x_min, x_max = -delta, delta
    y_min, y_max = -delta, delta

    x_values = np.linspace(x_min, x_max, 25)
    y_values = np.linspace(y_min, y_max, 25)
    X, Y = np.meshgrid(x_values, y_values)

    # Calcular campo vectorial
    U_field = np.zeros_like(X)
    V_field = np.zeros_like(Y)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            x_val = X[i, j]
            y_val = Y[i, j]

            # Asignar valores según las variables
            if var1_name == 'P' and var2_name == 'W':
                P_val, W_val, F_val = x_val, y_val, var3_fixed
            elif var1_name == 'P' and var2_name == 'F':
                P_val, F_val, W_val = x_val, y_val, var3_fixed
            elif var1_name == 'F' and var2_name == 'W':
                F_val, W_val, P_val = x_val, y_val, var3_fixed

            U_field[i, j] = func1(P_val, F_val, W_val, U_fixed, E_fixed)
            V_field[i, j] = func2(P_val, F_val, W_val, U_fixed, E_fixed)

    # Crear figura
    plt.figure(figsize=(12, 8))

    # Dibujar campo vectorial
    plt.streamplot(x_values, y_values, U_field, V_field,
                   color='blue', linewidth=1.2, density=1.8, arrowsize=1.5)

    # Marcar punto fijo en (0, 0)
    plt.scatter([0], [0], color='red', s=200, zorder=5,
                edgecolors='black', linewidths=2.5,
                label='Punto fijo: (0, 0)\n(Estable - atractor)')

    # Configurar gráfico
    plt.xlabel(f'{label1} normalizado', fontsize=13)
    plt.ylabel(f'{label2} normalizado', fontsize=13)
    plt.title(f'Diagrama de Fase: {label1} vs {label2}\n'
              f'({var3_name}=0.0, U=0.0, E=0.0)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Guardar figura
    output_file = output_dir / f'Diagrama_Fase_{var1_name}_vs_{var2_name}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Guardado: {output_file}")
    plt.close()

print(f"\n{'='*60}")
print(f"✅ Todos los diagramas de fase han sido generados y guardados")
print(f"   Directorio: {output_dir}")
print(f"{'='*60}")

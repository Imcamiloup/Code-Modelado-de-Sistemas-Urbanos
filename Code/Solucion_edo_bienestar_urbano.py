import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

# ------------------------- Cargar parámetros estimados -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
params_file = PROJECT_ROOT / 'Output' / 'parametros_estimados.json'

if not params_file.exists():
    raise FileNotFoundError(
        f"No se encontró {params_file}. "
        "Ejecuta primero estimacion_de_parametros_bienestar_urbano.py"
    )

with open(params_file, 'r', encoding='utf-8') as f:
    params_data = json.load(f)

# Extraer parámetros
a = params_data['parametros']['a']
bU = params_data['parametros']['bU']
bE = params_data['parametros']['bE']
c = params_data['parametros']['c']
d = params_data['parametros']['d']
c0 = params_data['parametros']['c0']

# Extraer estadísticos de normalización
P_mu = params_data['normalizacion']['P']['mu']
P_sd = params_data['normalizacion']['P']['sd']
U_mu = params_data['normalizacion']['U']['mu']
U_sd = params_data['normalizacion']['U']['sd']
E_mu = params_data['normalizacion']['E']['mu']
E_sd = params_data['normalizacion']['E']['sd']
W_mu = params_data['normalizacion']['W']['mu']
W_sd = params_data['normalizacion']['W']['sd']
F_mu = params_data['normalizacion']['F']['mu']
F_sd = params_data['normalizacion']['F']['sd']

print(
    f"Parámetros cargados: a={a:.4f}, bU={bU:.4f}, bE={bE:.4f}, c={c:.4f}, d={d:.4f}, c0={c0:.4f}")

# ------------------------- Cargar datos históricos -------------------------
file_path = PROJECT_ROOT / 'Data' / 'Consolidado_Bienestar_Urbano.xlsx'
df = pd.read_excel(file_path)


def pick_col(d, candidates):
    for c in candidates:
        if c in d.columns:
            return d[c].values.astype(float)
    return None


t_years_data = pick_col(df, ['Año', 'Anio', 'Year'])
if t_years_data is None:
    t_years_data = np.arange(len(df), dtype=int) + 2012
else:
    t_years_data = t_years_data.astype(int)

P_data = pick_col(df, ['Poblacion', 'Población', 'P'])
U_data = pick_col(df, ['Huella_Urbana', 'Huella Urbana',
                  'HuellaUrbana', 'Area_Urbanizada', 'Área_Urbanizada', 'U'])
E_data = pick_col(df, ['Infraestructura_Ecologica',
                  'Estructura_Ecologica', 'Area_Protegida', 'Áreas_Protegidas', 'E'])
W_data = pick_col(df, ['Bienestar', 'W', 'Bienestar_Urbano'])
F_data = pick_col(df, ['Desigualdad', 'Gini', 'Inequidad', 'F'])

if any(x is None for x in [P_data, U_data, E_data, W_data, F_data]):
    raise ValueError("Faltan columnas: se requieren P, U, E, W y F.")

n_hist = min(map(len, [P_data, U_data, E_data, W_data, F_data, t_years_data]))
P_data = P_data[:n_hist]
U_data = U_data[:n_hist]
E_data = E_data[:n_hist]
W_data = W_data[:n_hist]
F_data = F_data[:n_hist]
t_years_data = t_years_data[:n_hist]

# ------------------------- Extender series 5 años al futuro (hasta 2029) -------------------------
n_future = 5
t_years_full = np.arange(
    t_years_data[0], t_years_data[-1] + n_future + 1, dtype=int)


def extend_series(data_hist, n_total):
    """Extiende serie con último valor observado."""
    extended = np.empty(n_total, dtype=float)
    n_hist = len(data_hist)
    extended[:n_hist] = data_hist
    extended[n_hist:] = data_hist[-1]  # extrapolación constante
    return extended


P_ext = extend_series(P_data, len(t_years_full))
U_ext = extend_series(U_data, len(t_years_full))
E_ext = extend_series(E_data, len(t_years_full))
F_ext = extend_series(F_data, len(t_years_full))

# Normalizar usando parámetros históricos
Pz_ext = (P_ext - P_mu) / P_sd
Uz_ext = (U_ext - U_mu) / U_sd
Ez_ext = (E_ext - E_mu) / E_sd
Fz_ext = (F_ext - F_mu) / F_sd

# ------------------------- Resolver con método discreto (Euler explícito, dt=1 año) -------------------------


def simulate_W_discrete(params, Pz_arr, Uz_arr, Ez_arr, Fz_arr, W0z, n_steps):
    """Simulación discreta año a año (coherente con estimación)."""
    a, bU, bE, c, d, c0 = params
    Wz = np.empty(n_steps, dtype=float)
    Wz[0] = W0z
    for t in range(n_steps - 1):
        dW = a*Pz_arr[t] + bU*Uz_arr[t] + bE * \
            Ez_arr[t] + c*Wz[t] - d*Fz_arr[t] + c0
        Wz[t + 1] = Wz[t] + dW  # dt = 1 año
    return Wz


# Condición inicial
Wz0 = (W_data[0] - W_mu) / W_sd

# Parámetros
params = [a, bU, bE, c, d, c0]

# Simular hasta 2029
Wz_sim = simulate_W_discrete(
    params, Pz_ext, Uz_ext, Ez_ext, Fz_ext, Wz0, len(t_years_full))

# Desnormalizar
W_sim = Wz_sim * W_sd + W_mu

# ------------------------- Gráfico -------------------------
output_dir = PROJECT_ROOT / 'Output'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'solucion_edo_bienestar_5anios.png'

plt.figure(figsize=(12, 6))
plt.plot(t_years_data, W_data, 'o-',
         label='W real (2012–2024)', color='blue', linewidth=2)
plt.plot(t_years_full, W_sim, '--',
         label='W simulado (2012–2029)', color='red', linewidth=2)
plt.axvline(x=t_years_data[-1], color='gray',
            linestyle=':', linewidth=1.5, label='Inicio proyección')
plt.xlabel('Año', fontsize=12)
plt.ylabel('Bienestar urbano (W)', fontsize=12)
plt.title('Solución de la EDO del Bienestar Urbano: 2012–2029', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Gráfica guardada en: {output_path}')
plt.show()

print("\nValores proyectados de W (2025–2029):")
for year, w in zip(t_years_full[n_hist:], W_sim[n_hist:]):
    print(f"  {year}: {w:.2f}")

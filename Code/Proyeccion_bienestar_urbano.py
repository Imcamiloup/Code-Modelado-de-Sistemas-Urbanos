import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.optimize import curve_fit

# ------------------------- Cargar datos históricos -------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
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

print(
    f"Años históricos: {t_years_data[0]} a {t_years_data[-1]} ({n_hist} observaciones)")

# ------------------------- Cargar parámetros estimados -------------------------
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

print(f"Parámetros cargados desde: {params_file}")
print(f"  a={a:.4f}, bU={bU:.4f}, bE={bE:.4f}, c={c:.4f}, d={d:.4f}, c0={c0:.4f}")

# ------------------------- Normalización -------------------------


def znorm(x, mu, sd):
    return (np.asarray(x, dtype=float) - mu) / sd


def zunscale(z, mu, sd):
    return z * sd + mu


Pz_data = znorm(P_data, P_mu, P_sd)
Uz_data = znorm(U_data, U_mu, U_sd)
Ez_data = znorm(E_data, E_mu, E_sd)
Wz_data = znorm(W_data, W_mu, W_sd)
Fz_data = znorm(F_data, F_mu, F_sd)

# ------------------------- Proyección 5 años al futuro -------------------------
n_future = 5
t_years_full = np.arange(
    t_years_data[0], t_years_data[-1] + n_future + 1, dtype=int)
n_full = len(t_years_full)

# Interpolación lineal para años históricos y extrapolación constante para futuro


def extend_series(data_hist, n_total):
    """Interpola linealmente en años históricos y extiende con último valor."""
    extended = np.empty(n_total, dtype=float)
    n_hist = len(data_hist)
    extended[:n_hist] = data_hist
    # Extrapolación simple: mantener último valor observado
    extended[n_hist:] = data_hist[-1]
    return extended


P_ext = extend_series(P_data, n_full)
U_ext = extend_series(U_data, n_full)
E_ext = extend_series(E_data, n_full)
F_ext = extend_series(F_data, n_full)

# Normalizar series extendidas con los parámetros de la serie histórica
Pz_ext = znorm(P_ext, P_mu, P_sd)
Uz_ext = znorm(U_ext, U_mu, U_sd)
Ez_ext = znorm(E_ext, E_mu, E_sd)
Fz_ext = znorm(F_ext, F_mu, F_sd)

# ------------------------- Simulación con el modelo -------------------------


def simulate_W_extended(params, Pz, Uz, Ez, Fz, W0z):
    """Simula W normalizado usando dWz/dt = a*Pz + bU*Uz + bE*Ez + c*Wz - d*Fz + c0."""
    a, bU, bE, c, d, c0 = params
    n = len(Pz)
    Wz = np.empty(n, dtype=float)
    Wz[0] = W0z
    for t in range(n - 1):
        dW = a*Pz[t] + bU*Uz[t] + bE*Ez[t] + c*Wz[t] - d*Fz[t] + c0
        Wz[t + 1] = Wz[t] + dW
    return Wz


params = [a, bU, bE, c, d, c0]
Wz_sim = simulate_W_extended(
    params, Pz_ext, Uz_ext, Ez_ext, Fz_ext, Wz_data[0])
W_sim = zunscale(Wz_sim, W_mu, W_sd)

# ------------------------- Gráfico: Real vs Simulado + Proyección -------------------------
output_dir = PROJECT_ROOT / 'Output'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'proyeccion_bienestar_5anios.png'

plt.figure(figsize=(12, 6))
plt.plot(t_years_data, W_data, 'o-',
         label='W real (2012–2024)', color='blue', linewidth=2)
plt.plot(t_years_full, W_sim, '--',
         label='W simulado (ajuste + proyección 2025–2029)', color='red', linewidth=2)
plt.axvline(x=t_years_data[-1], color='gray',
            linestyle=':', linewidth=1.5, label='Inicio proyección')
plt.xlabel('Año', fontsize=12)
plt.ylabel('Bienestar urbano (W)', fontsize=12)
plt.title('Proyección del Bienestar Urbano en Bogotá: 2012–2029', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

# Guardar la gráfica
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'Gráfica guardada en: {output_path}')

# -------- Tabla de proyección (CSV y Excel) --------
W_real_full = np.full(len(t_years_full), np.nan)
W_real_full[:n_hist] = W_data

df_proj = pd.DataFrame({
    "Anio": t_years_full,
    "W_real": W_real_full,
    "W_simulado": W_sim,
    "segmento": ["historico"]*n_hist + ["proyeccion"]*(len(t_years_full)-n_hist)
})

csv_path = output_dir / "proyeccion_bienestar_5anios.csv"
xlsx_path = output_dir / "proyeccion_bienestar_5anios.xlsx"
df_proj.to_csv(csv_path, index=False, encoding="utf-8-sig")
df_proj.to_excel(xlsx_path, index=False)
print(f"Tabla de proyección guardada en:\n  - {csv_path}\n  - {xlsx_path}")

# Valores proyectados en consola
print("\nValores proyectados de W (2025–2029):")
for year, w in zip(t_years_full[n_hist:], W_sim[n_hist:]):
    print(f"  {year}: {w:.2f}")

# ------------------------- Función de ajuste exponencial -------------------------


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


# Ajustar la curva a los datos de Infraestructura_Ecologica
popt, pcov = curve_fit(exp_func, t_years_data, E_data)

# Generar valores ajustados
E_fit = exp_func(t_years_full, *popt)

# Graficar los datos originales y el ajuste
plt.figure(figsize=(10, 5))
plt.plot(t_years_data, E_data, 'o',
         label='Datos de Infraestructura Ecológica', color='blue')
plt.plot(t_years_full, E_fit, '-',
         label='Ajuste exponencial', color='red')
plt.xlabel('Año')
plt.ylabel('Infraestructura Ecológica')
plt.title('Ajuste de Curva Exponencial a Infraestructura Ecológica')
plt.legend()
plt.grid()
plt.show()

# Derivada de la función ajustada


def exp_derivative(x, a, b):
    return a * b * np.exp(b * x)


# Calcular el jacobiano
jacobian = exp_derivative(t_years_full, *popt[:2])
print("Jacobian calculado:", jacobian)

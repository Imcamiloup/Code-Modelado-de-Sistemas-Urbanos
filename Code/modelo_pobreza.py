# ============================================================
# ANÁLISIS DE INDICADORES DE POBREZA Y CÁLCULO DE F(t)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

# === 1. Configurar rutas ===
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_input = os.path.join(base_dir, "Data", "Indice_total_de_pobreza.xlsx")
ruta_output = os.path.join(base_dir, "Output")

# Crear carpeta Output si no existe
os.makedirs(ruta_output, exist_ok=True)

# === 2. Cargar los datos ===
df = pd.read_excel(ruta_input, sheet_name="Hoja1")

# === 3. Asegurar tipos numéricos correctos ===
# Reemplazar comas por puntos y convertir a float en las columnas numéricas
for col in ["IPMo", "Gini", "IPC"]:
    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)

# === 4. Normalizar las variables principales ===
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[["IPMo_norm", "Gini_norm", "IPC_norm"]
        ] = scaler.fit_transform(df[["IPMo", "Gini", "IPC"]])

# === 5. Calcular F(t) como índice compuesto ===
w_ipmo = 0.298
w_gini = 0.442
w_ipc = 0.260
df_norm["F_t"] = (
    w_ipmo * df_norm["IPMo_norm"] +
    w_gini * df_norm["Gini_norm"] +
    w_ipc * df_norm["IPC_norm"]
)

# === 6. Tabla final ===
tabla_completa = df.join(
    df_norm[["IPMo_norm", "Gini_norm", "IPC_norm", "F_t"]])
print("\nTabla completa:\n", tabla_completa)

# === 7. Guardar resultados ===
output_excel = os.path.join(ruta_output, "Resultados_F_t.xlsx")
tabla_completa.to_excel(output_excel, index=False)
print(f"\nArchivo guardado en: {output_excel}")

# === 8. Gráfica de indicadores normalizados ===
plt.figure(figsize=(10, 6))
plt.plot(df["Año"], df_norm["IPMo_norm"], label="IPMo (normalizado)")
plt.plot(df["Año"], df_norm["Gini_norm"], label="Gini (normalizado)")
plt.plot(df["Año"], df_norm["IPC_norm"], label="IPC (normalizado)")
plt.title("Indicadores normalizados de pobreza en Bogotá")
plt.xlabel("Año")
plt.ylabel("Valor normalizado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_output, "Indicadores_normalizados.png"))
plt.show()

# === 9. Gráfica F(t) ===
plt.figure(figsize=(8, 5))
plt.plot(df["Año"], df_norm["F_t"], marker='o', color='purple', linewidth=2)
plt.title("Evolución del indicador F(t) de pobreza compuesta")
plt.xlabel("Año")
plt.ylabel("F(t) normalizado")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_output, "F_t.png"))
plt.show()

# === 10. Gráficas combinadas ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

variables = [
    ("IPMo_norm", "IPMo normalizado"),
    ("Gini_norm", "Gini normalizado"),
    ("IPC_norm", "IPC normalizado"),
    ("F_t", "F(t) - Indicador compuesto")
]

for ax, (col, title) in zip(axes, variables):
    ax.plot(df["Año"], df_norm[col], marker='o')
    ax.set_title(title)
    ax.set_xlabel("Año")
    ax.set_ylabel("Valor normalizado")
    ax.grid(True)

plt.tight_layout()
fig.savefig(os.path.join(ruta_output, "Graficas_combinadas.png"))
plt.show()

# === 11. Gráfica comparativa con todos los indicadores y F(t) ===
plt.figure(figsize=(10, 6))
plt.plot(df["Año"], df_norm["IPMo_norm"],
         label="IPMo (normalizado)", linestyle='--')
plt.plot(df["Año"], df_norm["Gini_norm"],
         label="Gini (normalizado)", linestyle='-.')
plt.plot(df["Año"], df_norm["IPC_norm"],
         label="IPC (normalizado)", linestyle=':')
plt.plot(df["Año"], df_norm["F_t"],
         label="F(t) - Indicador compuesto", color='black', linewidth=2)
plt.title("Comparación entre indicadores normalizados y F(t)")
plt.xlabel("Año")
plt.ylabel("Valor normalizado")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_output, "Comparativa_F_t_y_componentes.png"))
plt.show()

print("\n✅ Todas las gráficas y resultados fueron guardados en la carpeta 'Output'.")

# === 11. Gráfica conjunta de los tres indicadores y F(t) ===
plt.figure(figsize=(10, 6))
plt.plot(df["Año"], df_norm["IPMo_norm"],
         label="IPMo (normalizado)", color='blue', linewidth=2)
plt.plot(df["Año"], df_norm["Gini_norm"],
         label="Gini (normalizado)", color='green', linewidth=2)
plt.plot(df["Año"], df_norm["IPC_norm"],
         label="IPC (normalizado)", color='orange', linewidth=2)
plt.plot(df["Año"], df_norm["F_t"], label="F(t) - Índice compuesto",
         color='red', linewidth=3, linestyle='--')

plt.title("Comparación de indicadores normalizados y F(t)")
plt.xlabel("Año")
plt.ylabel("Valor normalizado")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Guardar figura
plt.savefig(os.path.join(ruta_output, "Indicadores_y_Ft.png"))
plt.show()

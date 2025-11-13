# ============================================
#  MODELO DE BIENESTAR URBANO - F(t)
#  Lectura, normalización y visualización
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# --- 1. Leer datos ---
xls = base / "Data" / "Indice_total_de_pobreza.xlsx"
df = pd.read_excel(xls)

# === 3. Normalizar variables ===
scaler = MinMaxScaler()
df_norm = df.copy()
df_norm[["IPM_norm", "Gini_norm", "IPC_norm"]
        ] = scaler.fit_transform(df[["IPM", "Gini", "IPC"]])

# === 4. Calcular F(t) ===
w_ipm = 0.298
w_gini = 0.442
w_ipc = 0.260
df_norm["F_t"] = (
    w_ipm * df_norm["IPM_norm"] +
    w_gini * df_norm["Gini_norm"] +
    w_ipc * df_norm["IPC_norm"]
)

# === 5. Tabla completa ===
tabla_completa = df.join(df_norm[["IPM_norm", "Gini_norm", "IPC_norm", "F_t"]])
print("\nTabla completa:\n", tabla_completa)

# === 6. Guardar tabla en Output ===
output_excel = os.path.join(ruta_output, "Resultados_F_t.xlsx")
tabla_completa.to_excel(output_excel, index=False)
print(f"\nArchivo guardado en: {output_excel}")

# === 7. Graficar cada variable normalizada ===
plt.figure(figsize=(10, 6))
plt.plot(df["Año"], df_norm["IPM_norm"], label="IPM (normalizado)")
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

# === 8. Graficar F(t) ===
plt.figure(figsize=(8, 5))
plt.plot(df["Año"], df_norm["F_t"], marker='o', color='purple', linewidth=2)
plt.title("Evolución del indicador F(t) de pobreza compuesta")
plt.xlabel("Año")
plt.ylabel("F(t) normalizado")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ruta_output, "F_t.png"))
plt.show()

# === 9. Graficar las 4 juntas ===
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

variables = [
    ("IPM_norm", "IPM normalizado"),
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

print("\nTodas las gráficas fueron guardadas en la carpeta 'Output'.")

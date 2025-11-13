import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paso 1: Cargar los datos desde el archivo Excel
# Cambia la ruta según sea necesario
file_path = 'Data/Consolidado_Bienestar_Urbano.xlsx'
data = pd.read_excel(file_path)

# Paso 2: Definir las variables que queremos ajustar
variables = {
    'Poblacion': data['Poblacion'].values,
    'Huella_Urbana': data['Huella_Urbana'].values,
    # Estructura Ecológica
    'Estructura_Ecologica': data['Infraestructura_Ecologica'].values,
    'Desigualdad': data['Desigualdad'].values,
    'Bienestar': data['Bienestar'].values  # Bienestar Urbano
}

# Paso 3: Ajuste polinómico de grado 10 para Estructura Ecológica, grado 4 para Bienestar Urbano y Desigualdad,
# y grado 3 para Población y Huella Urbana
params_cubic = {}
params_quartic = {}
params_deci = {}

# Ajuste de grado 10 para Estructura Ecológica
# Ajuste de grado 4 para Bienestar Urbano y Desigualdad
# Ajuste de grado 3 para Población y Huella Urbana
for var_name, var_data in variables.items():
    if var_name == 'Estructura_Ecologica':
        params_deci[var_name] = np.polyfit(
            data['Año'].values, var_data, 10)  # Ajuste de grado 10
    elif var_name == 'Bienestar' or var_name == 'Desigualdad':
        params_quartic[var_name] = np.polyfit(
            data['Año'].values, var_data, 4)  # Ajuste de grado 4
    else:
        params_cubic[var_name] = np.polyfit(
            data['Año'].values, var_data, 3)  # Ajuste de grado 3

# Paso 4: Graficar y guardar los resultados del ajuste polinómico para cada variable
for i, (var_name, var_data) in enumerate(variables.items(), 1):
    plt.figure(figsize=(8, 6))
    plt.plot(data['Año'], var_data, 'bo', label=f'Datos de {var_name}')

    if var_name == 'Estructura_Ecologica':
        # Evaluar el ajuste polinómico de grado 10
        poly_func = np.poly1d(params_deci[var_name])
    elif var_name == 'Bienestar' or var_name == 'Desigualdad':
        # Evaluar el ajuste polinómico de grado 4
        poly_func = np.poly1d(params_quartic[var_name])
    else:
        # Evaluar el ajuste polinómico de grado 3
        poly_func = np.poly1d(params_cubic[var_name])

    plt.plot(data['Año'], poly_func(data['Año'].values),
             'r-', label=f'Ajuste polinómico')
    plt.xlabel('Año')
    plt.ylabel(var_name)
    plt.title(
        f'Ajuste Polinómico de {var_name} ({10 if var_name == "Estructura_Ecologica" else 4 if var_name == "Bienestar" or var_name == "Desigualdad" else 3}° Grado)')
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica como imagen PNG
    plt.savefig(f'{var_name}_ajuste.png')
    plt.close()  # Cerrar la figura para evitar sobreposición de gráficos

# Imprimir las ecuaciones resultantes


def print_equation(params, degree):
    terms = [f"{coef:.4f}x^{i}" for i, coef in enumerate(reversed(params))]
    equation = " + ".join(terms)
    return equation


print("Ecuaciones resultantes:")
print(
    f"Estructura Ecológica: E(x) = {print_equation(params_deci['Estructura_Ecologica'], 10)}")
print(
    f"Bienestar Urbano: W(x) = {print_equation(params_quartic['Bienestar'], 4)}")
print(
    f"Desigualdad: D(x) = {print_equation(params_quartic['Desigualdad'], 4)}")
print(f"Población: P(x) = {print_equation(params_cubic['Poblacion'], 3)}")
print(
    f"Huella Urbana: H(x) = {print_equation(params_cubic['Huella_Urbana'], 3)}")

print("Las gráficas se han guardado correctamente como imágenes PNG.")

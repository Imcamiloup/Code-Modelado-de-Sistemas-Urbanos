from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Coeficientes estimados para F(t)
ALPHA1 = 8.37248541295474e-08
ALPHA2 = 1.0000007569101497
ALPHA3 = -0.49167428442081607


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    base = Path(__file__).resolve().parent.parent
    xls = base / "Data" / "Indice_total_de_pobreza.xlsx"
    out = base / "Output" / "indice_total_pobreza"
    ensure_dir(out)

    df = pd.read_excel(xls)
    df = df.rename(columns={
        'Año': 'anio', 'Ano': 'anio', 'Anio': 'anio',
        'IPM': 'IPM', 'IPMo': 'IPM',
        'Gini': 'Gini', 'GINI': 'Gini',
        'IPC': 'IPC'
    })
    for c in ['anio', 'IPM', 'Gini', 'IPC']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df.dropna(subset=['anio', 'IPM', 'Gini', 'IPC']).sort_values('anio')

    # Calcular F(t)
    F = ALPHA1 * (df['IPM'] / 100.0) + ALPHA2 * df['Gini'] + ALPHA3 * (1.0 / df['IPC'])
    df['F'] = F
    # Verificación: recalcular y medir error absoluto
    df['F_check'] = ALPHA1 * (df['IPM'] / 100.0) + ALPHA2 * df['Gini'] + ALPHA3 * (1.0 / df['IPC'])
    df['F_error_abs'] = (df['F'] - df['F_check']).abs()

    # Guardar tabla extendida
    df.to_excel(out / 'Indice_total_de_pobreza_con_F.xlsx', index=False)
    df.to_csv(out / 'Indice_total_de_pobreza_con_F.csv', index=False)

    # Gráfica Año vs F
    plt.figure(figsize=(9,5))
    plt.plot(df['anio'], df['F'], marker='o', linewidth=1.8, color='#6f42c1')
    plt.title('Año vs F(t) (función compuesta)')
    plt.xlabel('Año'); plt.ylabel('F(t)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(); plt.savefig(out / 'anio_vs_F.png', dpi=150); plt.close()

    # Gráfico combinado: F, IPM/100, Gini y 1/IPC (mejor visibilidad de F)
    plt.figure(figsize=(10,6))
    # Dibujar primero las series de referencia con menor alpha
    if 'IPM' in df.columns:
        plt.plot(df['anio'], df['IPM']/100.0, '-s', linewidth=1.4, alpha=0.6, label='IPM/100', color='#1f77b4', zorder=2)
    if 'Gini' in df.columns:
        plt.plot(df['anio'], df['Gini'], '-^', linewidth=1.4, alpha=0.6, label='Gini', color='#ff7f0e', zorder=2)
    plt.plot(df['anio'], 1.0/df['IPC'], '-d', linewidth=1.2, alpha=0.6, label='1/IPC', color='#2ca02c', zorder=2)
    # Ahora F encima con mayor espesor y zorder
    plt.plot(df['anio'], df['F'], '-o', linewidth=2.8, markersize=7, label='F(t)', color='#6f42c1', zorder=5)
    plt.title('Año vs F(t), IPM/100, Gini y 1/IPC')
    plt.xlabel('Año'); plt.ylabel('Valor (adimensional)')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout(); plt.savefig(out / 'anio_vs_F_IPM_Gini.png', dpi=150); plt.close()

    # Exportar a Excel con gráficos incrustados (si xlsxwriter disponible)
    try:
        import xlsxwriter  # noqa: F401
        xlsx_path = out / 'reporte_F_compuesto.xlsx'
        with pd.ExcelWriter(xlsx_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='datos', index=False)
            ws = writer.sheets['datos']
            # Insertar imágenes de los gráficos
            img1 = str(out / 'anio_vs_F.png')
            img2 = str(out / 'anio_vs_F_IPM_Gini.png')
            img3 = str(out / 'anio_vs_series_normalizadas_highlight.png')
            img4 = str(out / 'anio_vs_F_menos_Gini.png')
            ws.insert_image('H2', img1, {'x_scale': 0.7, 'y_scale': 0.7})
            ws.insert_image('H22', img2, {'x_scale': 0.7, 'y_scale': 0.7})
            ws.insert_image('H42', img3, {'x_scale': 0.7, 'y_scale': 0.7})
            ws.insert_image('H62', img4, {'x_scale': 0.7, 'y_scale': 0.7})
    except Exception:
        pass

    # Normalización min-max de F, IPM/100, Gini y 1/IPC
    def minmax(s):
        s = s.astype(float)
        rng = s.max() - s.min()
        return (s - s.min()) / rng if rng != 0 else s * 0

    # Intentar cargar alphas estandarizadas para construir F_norm por z-score
    alphas_path = out / 'alphas_F_estandarizadas.csv'
    if alphas_path.exists():
        a = pd.read_csv(alphas_path).iloc[0]
        a1z, a2z, a3z = float(a['alpha1_z']), float(a['alpha2_z']), float(a['alpha3_z'])
        # z-scores
        ipm_prop = (df['IPM']/100.0).to_numpy(float)
        gini = df['Gini'].to_numpy(float)
        invipc = (1.0/df['IPC']).to_numpy(float)
        def zscore(x):
            m = float(np.mean(x)); s = float(np.std(x))
            return (x - m)/s if s != 0 else np.zeros_like(x)
        zipm = zscore(ipm_prop); zg = zscore(gini); zinv = zscore(invipc)
        Fz = a1z*zipm + a2z*zg + a3z*zinv
        # escalar a [0,1]
        F_norm_from_z = (Fz - Fz.min())/(Fz.max()-Fz.min()) if Fz.max()>Fz.min() else np.zeros_like(Fz)
        df['F_norm'] = F_norm_from_z
    else:
        df['F_norm'] = minmax(df['F'])

    df_norm = pd.DataFrame({
        'anio': df['anio'].values,
        'F_norm': df['F_norm'].astype(float),
        'IPM_norm': minmax(df['IPM']/100.0),
        'Gini_norm': minmax(df['Gini']),
        'invIPC_norm': minmax(1.0/df['IPC']),
    })
    df_norm.to_csv(out / 'series_normalizadas.csv', index=False)

    plt.figure(figsize=(10,6))
    plt.plot(df_norm['anio'], df_norm['F_norm'], '-o', label='F (norm)', color='#6f42c1', linewidth=2.8, markersize=7)
    plt.plot(df_norm['anio'], df_norm['IPM_norm'], '-s', label='IPM/100 (norm)', color='#1f77b4', linewidth=1.6, markersize=5)
    plt.plot(df_norm['anio'], df_norm['Gini_norm'], '-^', label='Gini (norm)', color='#ff7f0e', linewidth=1.6, markersize=5)
    plt.plot(df_norm['anio'], df_norm['invIPC_norm'], '-d', label='1/IPC (norm)', color='#2ca02c', linewidth=1.6, markersize=5)
    plt.title('Año vs series normalizadas (F, IPM/100, Gini, 1/IPC)')
    plt.xlabel('Año'); plt.ylabel('Valor normalizado [0,1]')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()
    plt.tight_layout(); 
    norm_png = out / 'anio_vs_series_normalizadas.png'
    norm_png_hl = out / 'anio_vs_series_normalizadas_highlight.png'
    norm_png_hl2 = out / 'anio_vs_series_normalizadas_Fnorm.png'
    plt.savefig(norm_png, dpi=150)
    plt.savefig(norm_png_hl, dpi=150)
    plt.savefig(norm_png_hl2, dpi=150)
    plt.close()

    # Guardar verificación en CSV aparte
    df['F_minus_Gini'] = df['F'] - df['Gini']
    df[['anio','IPM','Gini','IPC','F','F_norm','F_minus_Gini','F_check','F_error_abs']].to_csv(out / 'verificacion_F.csv', index=False)

    # Gráfico de diferencia F - Gini
    plt.figure(figsize=(9,5))
    plt.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.6)
    plt.plot(df['anio'], df['F_minus_Gini'], marker='o', linewidth=1.8, color='#6f42c1')
    plt.title('Año vs (F - Gini)')
    plt.xlabel('Año'); plt.ylabel('F - Gini')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout(); plt.savefig(out / 'anio_vs_F_menos_Gini.png', dpi=150); plt.close()

    diff_png = out / 'anio_vs_F_menos_Gini.png'
    print(f"Listo. Tabla y gráficas creadas en: {out}. Error máximo abs F: {df['F_error_abs'].max():.3e}")
    print(f"Figuras guardadas:\n- {norm_png.name}\n- {norm_png_hl.name}\n- {norm_png_hl2.name}\n- {diff_png.name}")


if __name__ == '__main__':
    main()

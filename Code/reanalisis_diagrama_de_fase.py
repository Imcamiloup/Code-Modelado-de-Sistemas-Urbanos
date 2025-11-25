# Genera diagramas de fase y reporte JSON usando parámetros optimizados
from utilidades_diagrama_de_fase import (
    load_params_opt, field_PW, field_PF, field_FW,
    jacobian_PW, jacobian_PF, jacobian_FW,
    fixed_point_PW, fixed_point_PF, fixed_point_FW,
    classify_eigs, make_phase_plot, add_nullclines_PW, add_nullclines_PF, add_nullclines_FW,
    save_report_json
)
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys

# Habilitar import de utils/fase_utils.py
THIS_DIR = Path(__file__).resolve().parent
sys.path.append(str(THIS_DIR / "utils"))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "Output_Correcciones" / "Diagramas_Fase_Optimizado"
OUT_DIR.mkdir(parents=True, exist_ok=True)

pars = load_params_opt(PROJECT_ROOT)

xr = (-2.5, 2.5)
yr = (-2.5, 2.5)
report = {"parametros": pars, "subsistemas": {}}

# -------- PW (F=0, U+E=0)
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
make_phase_plot(ax, lambda P, W: field_PW(P, W, pars, F_fixed=0.0, C_UE=0.0),
                "P (normalizado)", "W (normalizado)", xr, yr)
add_nullclines_PW(ax, pars, F_fixed=0.0, C_UE=0.0, xr=xr)
P_star, W_star = fixed_point_PW(pars, F_fixed=0.0, C_UE=0.0)
ax.plot([P_star], [W_star], "ro", ms=6, label="Punto fijo")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Plano de fase P–W (F=0, U+E=0)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fase_PW_opt.png", bbox_inches="tight")
plt.close(fig)

J_PW = jacobian_PW(pars)
evals_PW = np.linalg.eigvals(J_PW)
report["subsistemas"]["PW"] = {
    "punto_fijo": {"P*": P_star, "W*": W_star},
    "jacobiano": J_PW,
    "eigenvalores": [complex(e) for e in evals_PW],
    "clasificacion": classify_eigs(evals_PW)
}

# -------- PF (W=0)
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
make_phase_plot(ax, lambda P, F: field_PF(P, F, pars, W_fixed=0.0),
                "P (normalizado)", "F (normalizado)", xr, yr)
add_nullclines_PF(ax, pars, W_fixed=0.0, yr=yr)
P_star, F_star = fixed_point_PF(pars, W_fixed=0.0)
ax.plot([P_star], [F_star], "ro", ms=6, label="Punto fijo")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Plano de fase P–F (W=0)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fase_PF_opt.png", bbox_inches="tight")
plt.close(fig)

J_PF = jacobian_PF(pars)
evals_PF = np.linalg.eigvals(J_PF)
report["subsistemas"]["PF"] = {
    "punto_fijo": {"P*": P_star, "F*": F_star},
    "jacobiano": J_PF,
    "eigenvalores": [complex(e) for e in evals_PF],
    "clasificacion": classify_eigs(evals_PF)
}

# -------- FW (P=0, U+E=0)
fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
make_phase_plot(ax, lambda F, W: field_FW(F, W, pars, P_fixed=0.0, C_UE=0.0),
                "F (normalizado)", "W (normalizado)", xr, yr)
add_nullclines_FW(ax, pars, xr=xr)
F_star, W_star = fixed_point_FW(pars, P_fixed=0.0, C_UE=0.0)
ax.plot([F_star], [W_star], "ro", ms=6, label="Punto fijo")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Plano de fase F–W (P=0, U+E=0)")
fig.tight_layout()
fig.savefig(OUT_DIR / "fase_FW_opt.png", bbox_inches="tight")
plt.close(fig)

J_FW = jacobian_FW(pars)
evals_FW = np.linalg.eigvals(J_FW)
report["subsistemas"]["FW"] = {
    "punto_fijo": {"F*": F_star, "W*": W_star},
    "jacobiano": J_FW,
    "eigenvalores": [complex(e) for e in evals_FW],
    "traza": float(np.trace(J_FW)),
    "determinante": float(np.linalg.det(J_FW)),
    "clasificacion": classify_eigs(evals_FW)
}

# -------- Guardar reporte
save_report_json(OUT_DIR / "reporte_estabilidad_opt.json", report)
print(f"Listo. Figuras y reporte en: {OUT_DIR}")

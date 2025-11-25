# Herramientas para análisis de diagramas de fase con parámetros optimizados
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------- Carga de parámetros -------------------------


def load_params_opt(project_root: Path) -> dict:
    """Lee Output/parametros_optimizados.json y devuelve dict de parámetros."""
    params_path = project_root / "Output" / "parametros_optimizados.json"
    if not params_path.exists():
        raise FileNotFoundError(
            f"No se encontró {params_path}. Ejecuta la re-estimación primero.")
    with open(params_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    p = data["parametros_optimizados"]
    return {
        "r": float(p["r"]),
        "gamma_F": float(p["gamma_F"]),
        "alpha_F": float(p["alpha_F"]),
        "beta_F": float(p["beta_F"]),
        "alpha_W": float(p["alpha_W"]),
        "beta_W": float(p["beta_W"]),
        "gamma_W": float(p["gamma_W"]),
        "delta_W": float(p["delta_W"]),
    }

# ------------------------- Sistemas 2D (modelo continuo simplificado) -------------------------


def field_PW(P, W, pars, F_fixed=0.0, C_UE=0.0):
    r, gamma_F = pars["r"], pars["gamma_F"]
    alpha_W, beta_W, gamma_W, delta_W = pars["alpha_W"], pars["beta_W"], pars["gamma_W"], pars["delta_W"]
    dP = r*P + gamma_F*F_fixed
    dW = alpha_W*P + beta_W*C_UE + gamma_W*W - delta_W*F_fixed
    return dP, dW


def field_PF(P, F, pars, W_fixed=0.0):
    r, gamma_F = pars["r"], pars["gamma_F"]
    alpha_F, beta_F = pars["alpha_F"], pars["beta_F"]
    dP = r*P + gamma_F*F
    dF = -alpha_F*F + beta_F*W_fixed
    return dP, dF


def field_FW(F, W, pars, P_fixed=0.0, C_UE=0.0):
    alpha_F, beta_F = pars["alpha_F"], pars["beta_F"]
    alpha_W, beta_W, gamma_W, delta_W = pars["alpha_W"], pars["beta_W"], pars["gamma_W"], pars["delta_W"]
    dF = -alpha_F*F + beta_F*W
    dW = alpha_W*P_fixed + beta_W*C_UE + gamma_W*W - delta_W*F
    return dF, dW

# ------------------------- Jacobianos y puntos fijos -------------------------


def jacobian_PW(pars):
    return np.array([[pars["r"], 0.0], [pars["alpha_W"], pars["gamma_W"]]], dtype=float)


def jacobian_PF(pars):
    return np.array([[pars["r"], pars["gamma_F"]], [0.0, -pars["alpha_F"]]], dtype=float)


def jacobian_FW(pars):
    return np.array([[-pars["alpha_F"], pars["beta_F"]], [-pars["delta_W"], pars["gamma_W"]]], dtype=float)


def fixed_point_PW(pars, F_fixed=0.0, C_UE=0.0):
    r, gamma_F = pars["r"], pars["gamma_F"]
    alpha_W, beta_W, gamma_W, delta_W = pars["alpha_W"], pars["beta_W"], pars["gamma_W"], pars["delta_W"]
    P_star = 0.0 if abs(r) < 1e-12 else -gamma_F*F_fixed / r
    W_star = -(alpha_W/gamma_W)*P_star - (beta_W/gamma_W) * \
        C_UE + (delta_W/gamma_W)*F_fixed
    return float(P_star), float(W_star)


def fixed_point_PF(pars, W_fixed=0.0):
    r, gamma_F = pars["r"], pars["gamma_F"]
    alpha_F, beta_F = pars["alpha_F"], pars["beta_F"]
    F_star = 0.0 if abs(alpha_F) < 1e-12 else (beta_F/alpha_F)*W_fixed
    P_star = 0.0 if abs(r) < 1e-12 else -(gamma_F/r)*F_star
    return float(P_star), float(F_star)


def fixed_point_FW(pars, P_fixed=0.0, C_UE=0.0):
    alpha_F, beta_F = pars["alpha_F"], pars["beta_F"]
    alpha_W, beta_W, gamma_W, delta_W = pars["alpha_W"], pars["beta_W"], pars["gamma_W"], pars["delta_W"]
    A = np.array([[-alpha_F, beta_F], [-delta_W, gamma_W]], dtype=float)
    b = np.array([0.0, -alpha_W*P_fixed - beta_W*C_UE], dtype=float)
    F_star, W_star = np.linalg.solve(A, b)
    return float(F_star), float(W_star)

# ------------------------- Clasificación de estabilidad -------------------------


def classify_eigs(evals: np.ndarray):
    re = np.real(evals)
    im = np.imag(evals)
    if np.all(re < 0) and np.any(np.abs(im) > 1e-12):
        return "Foco estable"
    if np.all(re < 0):
        return "Nodo estable"
    if np.all(re > 0) and np.any(np.abs(im) > 1e-12):
        return "Foco inestable"
    if np.all(re > 0):
        return "Nodo inestable"
    return "Punto silla"

# ------------------------- Utilidades de gráficos -------------------------


def make_phase_plot(ax, field_fun, xlab, ylab, xr=(-2.5, 2.5), yr=(-2.5, 2.5), density=1.2):
    xs = np.linspace(xr[0], xr[1], 32)
    ys = np.linspace(yr[0], yr[1], 32)
    X, Y = np.meshgrid(xs, ys)
    dX, dY = field_fun(X, Y)
    ax.streamplot(X, Y, dX, dY, color=np.hypot(dX, dY),
                  cmap="viridis", density=density, linewidth=1)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)


def add_nullclines_PW(ax, pars, F_fixed=0.0, C_UE=0.0, xr=(-2.5, 2.5)):
    r, gamma_F = pars["r"], pars["gamma_F"]
    alpha_W, beta_W, gamma_W, delta_W = pars["alpha_W"], pars["beta_W"], pars["gamma_W"], pars["delta_W"]
    xs = np.linspace(xr[0], xr[1], 200)
    if abs(r) > 1e-12:
        Pnc = -gamma_F*F_fixed / r
        ax.axvline(Pnc, color="tab:blue", ls="--", lw=1.6, label="dP/dt=0")
    Wnc = -(alpha_W/gamma_W)*xs - (beta_W/gamma_W) * \
        C_UE + (delta_W/gamma_W)*F_fixed
    ax.plot(xs, Wnc, color="tab:red", ls="--", lw=1.6, label="dW/dt=0")
    ax.legend(fontsize=8)


def add_nullclines_PF(ax, pars, W_fixed=0.0, yr=(-2.5, 2.5)):
    r, gamma_F = pars["r"], pars["gamma_F"]
    alpha_F, beta_F = pars["alpha_F"], pars["beta_F"]
    ys = np.linspace(yr[0], yr[1], 200)
    if abs(r) > 1e-12:
        Pnc = -(gamma_F/r)*ys
        ax.plot(Pnc, ys, color="tab:blue", ls="--", lw=1.6, label="dP/dt=0")
    if abs(alpha_F) > 1e-12:
        Fnc = (beta_F/alpha_F)*W_fixed
        ax.axhline(Fnc, color="tab:red", ls="--", lw=1.6, label="dF/dt=0")
    ax.legend(fontsize=8)


def add_nullclines_FW(ax, pars, xr=(-2.5, 2.5)):
    alpha_F, beta_F = pars["alpha_F"], pars["beta_F"]
    gamma_W, delta_W = pars["gamma_W"], pars["delta_W"]
    xs = np.linspace(xr[0], xr[1], 200)
    if abs(beta_F) > 1e-12:
        W_Fnc = (alpha_F/beta_F)*xs
        ax.plot(xs, W_Fnc, color="tab:orange",
                ls="--", lw=1.6, label="dF/dt=0")
    if abs(gamma_W) > 1e-12:
        W_Wnc = (delta_W/gamma_W)*xs
        ax.plot(xs, W_Wnc, color="tab:purple",
                ls="--", lw=1.6, label="dW/dt=0")
    ax.legend(fontsize=8)

# ------------------------- Guardado robusto -------------------------


def save_report_json(path: Path, report_dict: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convertir eigenvalores complejos a dict {re, im}

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {"re": float(np.real(obj)), "im": float(np.imag(obj))}
        return obj
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2,
                  ensure_ascii=False, default=convert)

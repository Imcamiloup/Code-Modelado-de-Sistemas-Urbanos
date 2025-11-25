import numpy as np
import pandas as pd
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "Data" / "Consolidado_Bienestar_Urbano.xlsx"
OUT_DIR = PROJECT_ROOT / "Output_Correcciones"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def pick_col(df, names):
    for n in names:
        if n in df.columns:
            return df[n].values.astype(float)
    return None


def cargar_normalizado():
    df = pd.read_excel(DATA_FILE)
    P = pick_col(df, ["Poblacion", "Población", "P"])
    U = pick_col(df, ["Huella_Urbana", "Huella Urbana", "U"])
    E = pick_col(df, ["Infraestructura_Ecologica",
                 "Estructura_Ecologica", "E"])
    F = pick_col(df, ["Desigualdad", "Gini", "F"])
    W = pick_col(df, ["Bienestar", "W", "Bienestar_Urbano"])
    years = pick_col(df, ["Año", "Anio", "Year"])
    if any(x is None for x in [P, U, E, F, W, years]):
        raise ValueError("Faltan columnas requeridas.")
    n = min(map(len, [P, U, E, F, W, years]))
    P, U, E, F, W, years = P[:n], U[:n], E[:n], F[:n], W[:n], years[:n]

    def z(x):
        mu, sd = np.mean(x), np.std(x) + 1e-9
        return (x-mu)/sd, mu, sd
    Pz, P_mu, P_sd = z(P)
    Uz, U_mu, U_sd = z(U)
    Ez, E_mu, E_sd = z(E)
    Fz, F_mu, F_sd = z(F)
    Wz, W_mu, W_sd = z(W)
    return {
        "years": years.astype(int),
        "Pz": Pz, "Uz": Uz, "Ez": Ez, "Fz": Fz, "Wz": Wz,
        "stats": {"P": (P_mu, P_sd), "U": (U_mu, U_sd), "E": (E_mu, E_sd),
                  "F": (F_mu, F_sd), "W": (W_mu, W_sd)}
    }


def suavizar(x, deg):
    t = np.linspace(0, 1, len(x))
    coeff = np.polyfit(t, x, deg)
    return np.poly1d(coeff)


def edo(t, y, pars, U_poly, E_poly, T_max):
    P, F, W = y
    # Evaluar U,E normalizados en dominio [0,1]
    s = min(1.0, t / T_max)
    U = U_poly(s)
    E = E_poly(s)
    (r, gF, aF, bF, aW, bW, gW, dW, c0, K) = pars
    # dP/dt = r P (1 - P/K) + γ_F F
    dP = r * P * (1 - P / K) + gF * F
    dF = -aF * F + bF * W
    dW = aW * P + bW * (U + E) + gW * W - dW * F + c0
    return [dP, dF, dW]


def perdida(pars, data, U_poly, E_poly):
    N = len(data["years"])
    t_eval = np.arange(N, dtype=float)
    y0 = [data["Pz"][0], data["Fz"][0], data["Wz"][0]]
    sol = solve_ivp(edo, (0, N-1), y0, t_eval=t_eval,
                    args=(pars, U_poly, E_poly, N-1),
                    rtol=1e-6, atol=1e-8)
    if not sol.success:
        return 1e6
    P_sim, F_sim, W_sim = sol.y
    wP = 1 / np.var(data["Pz"])
    wF = 1 / np.var(data["Fz"])
    wW = 1 / np.var(data["Wz"])
    return (wP * np.mean((P_sim - data["Pz"])**2) +
            wF * 2 * np.mean((F_sim - data["Fz"])**2) +
            wW * np.mean((W_sim - data["Wz"])**2))


def estimar():
    data = cargar_normalizado()
    U_poly = suavizar(data["Uz"], 2)
    E_poly = suavizar(data["Ez"], 3)

    bounds = [
        (-0.5, 0.5),    # r
        (-5, 15),       # gamma_F
        (0, 1),         # alpha_F
        (-0.01, 0.01),  # beta_F
        (-0.01, 0.01),  # alpha_W
        (-10, 10),      # beta_W
        (-1, 0),        # gamma_W
        (-1, 1),        # delta_W
        (-1, 1),        # c0
        (1.5, 10.0)     # K AMPLIADO: permitir mayor capacidad
    ]

    result_de = differential_evolution(
        lambda p: perdida(p, data, U_poly, E_poly),
        bounds, maxiter=150, popsize=25, seed=42, tol=1e-6
    )
    p0 = result_de.x
    result_local = minimize(
        lambda p: perdida(p, data, U_poly, E_poly),
        p0, method="L-BFGS-B", bounds=bounds
    )
    pars = result_local.x if result_local.fun < result_de.fun else p0
    loss_final = min(result_local.fun, result_de.fun)

    out = {
        "parametros_optimizados": {
            "r": pars[0], "gamma_F": pars[1], "alpha_F": pars[2], "beta_F": pars[3],
            "alpha_W": pars[4], "beta_W": pars[5], "gamma_W": pars[6],
            "delta_W": pars[7], "c0": pars[8], "K": pars[9]
        },
        "loss": float(loss_final),
        "normalizacion": {k: {"mu": v[0], "sd": v[1]} for k, v in data["stats"].items()},
        "years": data["years"].tolist()
    }

    out_file = OUT_DIR / "parametros_sistema_logistico.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("Guardado:", out_file)


if __name__ == "__main__":
    estimar()

# Crea sección LaTeX con figuras y estabilidad a partir del JSON generado
from pathlib import Path
import json
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "Output_Correcciones" / "Diagramas_Fase_Optimizado"
json_path = OUT_DIR / "reporte_estabilidad_opt.json"
tex_dir = OUT_DIR / "latex"
tex_dir.mkdir(parents=True, exist_ok=True)
tex_path = tex_dir / "diagrama_fase_optimo.tex"

if not json_path.exists():
    print("No existe el JSON de reporte. Ejecuta primero analisis_diagramas_fase_optimo.py")
    sys.exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    rep = json.load(f)

PW = rep["subsistemas"]["PW"]
PF = rep["subsistemas"]["PF"]
FW = rep["subsistemas"]["FW"]


def fmt_complex(d):
    # d es dict {"re":..., "im":...} o número real
    if isinstance(d, dict) and "re" in d and "im" in d:
        re, im = d["re"], d["im"]
        return f"{re:.4f}" if abs(im) < 1e-10 else f"{re:.4f} {'+' if im >= 0 else '-'} {abs(im):.4f}i"
    return f"{float(d):.4f}"


def get_eval_list(eigs):
    # lista de 2 eigenvalores dict o float
    if isinstance(eigs, list) and len(eigs) >= 2:
        return fmt_complex(eigs[0]), fmt_complex(eigs[1])
    return "—", "—"


PW_L1, PW_L2 = get_eval_list(PW["eigenvalores"])
PF_L1, PF_L2 = get_eval_list(PF["eigenvalores"])
FW_L1, FW_L2 = get_eval_list(FW["eigenvalores"])

latex = r"""
\section{Diagramas de fase con parámetros optimizados}
Los diagramas se generaron usando los parámetros re-estimados mediante optimización directa sobre trayectorias del modelo continuo.

\subsection{Plano P--W (con $F=0$, $U+E=0$)}
\begin{figure}[H]\centering
\includegraphics[width=0.75\textwidth]{Output_Correcciones/Diagramas_Fase_Optimizado/fase_PW_opt.png}
\caption{Plano de fase P--W.}
\end{figure}
Punto fijo: $(P^*,W^*) = \left(%PW_P%,\,%PW_W%\right)$.
\\
Jacobiano:
\[
J_{PW} =
\begin{pmatrix}
%PW_J11% & %PW_J12%\\
%PW_J21% & %PW_J22%
\end{pmatrix},
\quad \lambda = \{%PW_L1%,\ %PW_L2%\}.
\]
Clasificación: \textbf{%PW_CLASS%}.

\subsection{Plano P--F (con $W=0$)}
\begin{figure}[H]\centering
\includegraphics[width=0.75\textwidth]{Output_Correcciones/Diagramas_Fase_Optimizado/fase_PF_opt.png}
\caption{Plano de fase P--F.}
\end{figure}
Punto fijo: $(P^*,F^*) = \left(%PF_P%,\,%PF_F%\right)$.
\\
Jacobiano:
\[
J_{PF} =
\begin{pmatrix}
%PF_J11% & %PF_J12%\\
%PF_J21% & %PF_J22%
\end{pmatrix},
\quad \lambda = \{%PF_L1%,\ %PF_L2%\}.
\]
Clasificación: \textbf{%PF_CLASS%}.

\subsection{Plano F--W (con $P=0$, $U+E=0$)}
\begin{figure}[H]\centering
\includegraphics[width=0.75\textwidth]{Output_Correcciones/Diagramas_Fase_Optimizado/fase_FW_opt.png}
\caption{Plano de fase F--W.}
\end{figure}
Punto fijo: $(F^*,W^*) = \left(%FW_F%,\,%FW_W%\right)$.
\\
Jacobiano:
\[
J_{FW} =
\begin{pmatrix}
%FW_J11% & %FW_J12%\\
%FW_J21% & %FW_J22%
\end{pmatrix},
\quad \lambda = \{%FW_L1%,\ %FW_L2%\}.
\]
Traza$(J)= %FW_TR%$, Determinante$(J)= %FW_DET%$. Clasificación: \textbf{%FW_CLASS%}.
"""

# Reemplazos PW
J_PW = PW["jacobiano"]
latex = latex.replace("%PW_P%", f"{PW['punto_fijo']['P*']:.4f}")
latex = latex.replace("%PW_W%", f"{PW['punto_fijo']['W*']:.4f}")
latex = latex.replace("%PW_J11%", f"{J_PW[0][0]:.4f}")
latex = latex.replace("%PW_J12%", f"{J_PW[0][1]:.4f}")
latex = latex.replace("%PW_J21%", f"{J_PW[1][0]:.4f}")
latex = latex.replace("%PW_J22%", f"{J_PW[1][1]:.4f}")
latex = latex.replace("%PW_L1%", PW_L1)
latex = latex.replace("%PW_L2%", PW_L2)
latex = latex.replace("%PW_CLASS%", PW["clasificacion"])

# Reemplazos PF
J_PF = PF["jacobiano"]
latex = latex.replace("%PF_P%", f"{PF['punto_fijo']['P*']:.4f}")
latex = latex.replace("%PF_F%", f"{PF['punto_fijo']['F*']:.4f}")
latex = latex.replace("%PF_J11%", f"{J_PF[0][0]:.4f}")
latex = latex.replace("%PF_J12%", f"{J_PF[0][1]:.4f}")
latex = latex.replace("%PF_J21%", f"{J_PF[1][0]:.4f}")
latex = latex.replace("%PF_J22%", f"{J_PF[1][1]:.4f}")
latex = latex.replace("%PF_L1%", PF_L1)
latex = latex.replace("%PF_L2%", PF_L2)
latex = latex.replace("%PF_CLASS%", PF["clasificacion"])

# Reemplazos FW
J_FW = FW["jacobiano"]
latex = latex.replace("%FW_F%", f"{FW['punto_fijo']['F*']:.4f}")
latex = latex.replace("%FW_W%", f"{FW['punto_fijo']['W*']:.4f}")
latex = latex.replace("%FW_J11%", f"{J_FW[0][0]:.4f}")
latex = latex.replace("%FW_J12%", f"{J_FW[0][1]:.4f}")
latex = latex.replace("%FW_J21%", f"{J_FW[1][0]:.4f}")
latex = latex.replace("%FW_J22%", f"{J_FW[1][1]:.4f}")
latex = latex.replace("%FW_L1%", FW_L1)
latex = latex.replace("%FW_L2%", FW_L2)
latex = latex.replace("%FW_TR%", f"{FW['traza']:.4f}")
latex = latex.replace("%FW_DET%", f"{FW['determinante']:.6f}")
latex = latex.replace("%FW_CLASS%", FW["clasificacion"])

with open(tex_path, "w", encoding="utf-8") as f:
    f.write(latex)

print("Sección LaTeX creada en:", tex_path)

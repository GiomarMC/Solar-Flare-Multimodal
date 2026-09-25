"""
English compact version of the SHARP permutation-importance figure (BiLSTM, 48 h)

Versión para el camera-ready de SIMBig (formato LNCS).

ÚNICA diferencia: la figura se dibuja al tamaño EXACTO que ocupa en la página
(0.56\textwidth = 194 pt = 2.70 in), de modo que \includegraphics no la
reduce y las fuentes se imprimen a su tamaño nominal. En el original se dibujaba
a 3.7 in y entraba al 73%, así que la letra de 6.5 pt acababa en 4.7 pt.
Springer exige que la rotulación de las figuras no baje de 6 pt.

Los tamaños de fuente NO se tocan; lo que cambia es figsize (y, donde hacía falta,
se acortan etiquetas de leyenda para que quepan en el panel más estrecho).

"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.environ.get("SFMM_FIG", os.path.join(ROOT, "results", "figures"))
os.makedirs(IMG, exist_ok=True)
# Informe de lstm_permutation_importance.py: el de SFMM_OUT si se regeneró, si no el publicado.
TXT = os.path.join(os.environ.get("SFMM_OUT", os.path.join(ROOT, "results", "reports")),
                   "sharp_importance_48h.txt")
if not os.path.exists(TXT):
    TXT = os.path.join(ROOT, "results", "reports", "sharp_importance_48h.txt")

BASE_10 = {"USFLUX", "MEANGBZ", "MEANGBT", "MEANPOT", "SHRGT45",
           "TOTPOT", "SAVNCPP", "ABSNJZH", "AREA_ACR", "NACR"}
C_BASE, C_ADD = "#9aa7b5", "#d62728"

names, mean, sd = [], [], []
row = re.compile(r"^\s*([A-Z0-9_]+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s*$")
with open(TXT) as f:
    for line in f:
        m = row.match(line)
        if m:
            names.append(m.group(1)); mean.append(float(m.group(2))); sd.append(float(m.group(3)))
mean, sd = np.array(mean), np.array(sd)
order = np.argsort(mean)            # ascending -> largest on top with barh
names = [names[i] for i in order]; mean, sd = mean[order], sd[order]
colors = [C_BASE if n in BASE_10 else C_ADD for n in names]

fig, ax = plt.subplots(figsize=(2.70, 2.30))
yy = np.arange(len(names))
ax.barh(yy, mean, xerr=sd, color=colors, alpha=0.9, capsize=2,
        error_kw=dict(ecolor="#555", lw=0.7))
ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=6.5)
for tick, n in zip(ax.get_yticklabels(), names):
    tick.set_color(C_ADD if n not in BASE_10 else "#333")
ax.set_xlabel("TSS drop when permuting (mean $\\pm\\sigma$)", fontsize=7.5)
ax.axvline(0, color="k", lw=0.7); ax.grid(axis="x", alpha=0.3)
ax.tick_params(axis="x", labelsize=7)
# leyenda eliminada: tapaba cuatro barras y el pie de figura ya
# dice que las rojas son las completadas desde JSOC
fig.tight_layout()
fig.savefig(os.path.join(IMG, "fig7_sharp_importance_48h.pdf"), bbox_inches="tight")
plt.close(fig)
print(f"Generated from cache ({len(names)} params): {IMG}/fig7_sharp_importance_48h.pdf")

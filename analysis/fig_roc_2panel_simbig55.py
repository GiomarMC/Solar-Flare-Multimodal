"""
Two-panel ROC figure at 24 h and 48 h, side by side, for the SIMBig/LNCS paper.

Copia paralela de graficos/fig_roc_2panel_en.py para el camera-ready de SIMBig (SIMBig55/).

ÚNICA diferencia: la figura se dibuja al tamaño EXACTO que ocupa en la página
(0.90\textwidth = 312 pt = 4.34 in), de modo que \includegraphics no la
reduce y las fuentes se imprimen a su tamaño nominal. En el original se dibujaba
a 8.2 in y entraba al 53%, así que la letra de 6.4 pt acababa en 3.4 pt.
Springer exige que la rotulación de las figuras no baje de 6 pt.

Los tamaños de fuente NO se tocan; lo que cambia es figsize (y, donde hacía falta,
se acortan etiquetas de leyenda para que quepan en el panel más estrecho).

No modifica graficos/fig_roc_2panel_en.py ni RedaccionSIMBig/images/, que sirven a la versión
IEEE y a la tesis, con otro ancho de columna.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import ROOT, load_split, sigmoid, sweep_tau, auc_point, bootstrap_ci

IMG = os.path.join(ROOT, "SIMBig55", "images")
FOLDS = [0, 1, 2, 3, 4]
MODELS = [("swin3d", "Swin3D (FITS)", "#1f77b4"), ("lstm", "BiLSTM (SHARP)", "#d62728")]
# estilo por modelo: las actas se imprimen en B/N y solo el color no distingue
ESTILOS = {"swin3d": "-", "lstm": (0, (4, 1.6))}


def fold_curves(model, H):
    """Ensemble probs + per-fold ROC curves; tau from each fold's own val (max-TSS)."""
    probs, taus, lab, froc = [], [], None, []
    for k in FOLDS:
        t = load_split(model, H, k, "test")
        v = load_split(model, H, k, "val")
        if t is None:
            continue
        p = sigmoid(t[0]); y = t[1].astype(int); lab = y
        probs.append(p)
        taus.append(sweep_tau(sigmoid(v[0]), v[1], 0.01, 0.99, 0.01)[0])
        fpr, tpr, _ = roc_curve(y, p)
        froc.append((fpr, tpr))
    return np.mean(probs, axis=0), float(np.mean(taus)), lab, froc


def op_point(p, y, tau):
    pred = (p >= tau).astype(int)
    tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
    fp = ((pred == 1) & (y == 0)).sum(); tn = ((pred == 0) & (y == 0)).sum()
    return fp / (fp + tn + 1e-12), tp / (tp + fn + 1e-12)


def draw_panel(ax, H, title):
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, zorder=1, label="chance")
    for model, name, col in MODELS:
        ens, tau, y, froc = fold_curves(model, H)
        auc = auc_point(ens, y)
        ci = bootstrap_ci(ens, y, tau, B=10000)["auc"]
        for fpr, tpr in froc:
            ax.plot(fpr, tpr, color=col, lw=0.6, alpha=0.28, zorder=2)
        fpr, tpr, _ = roc_curve(y, ens)
        ax.plot(fpr, tpr, color=col, lw=1.6, ls=ESTILOS[model], zorder=4,
                label=name.split(chr(32))[0])
        pofd, pod = op_point(ens, y, tau)
        ax.plot(pofd, pod, "o", color=col, ms=6, markeredgecolor="black", zorder=6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.tick_params(labelsize=8)
    ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("False positive rate (POFD)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, frameon=True, handlelength=1.1, handletextpad=0.4, borderpad=0.3, labelspacing=0.25, borderaxespad=0.3)
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=10, fontweight="bold")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(4.40, 2.15), sharey=True)
    draw_panel(axes[0], 24, "24 h")
    draw_panel(axes[1], 48, "48 h")
    axes[0].set_ylabel("Detection rate (POD)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig_roc_24_48h.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Generated: SIMBig55/images/fig_roc_24_48h.pdf")


if __name__ == "__main__":
    main()

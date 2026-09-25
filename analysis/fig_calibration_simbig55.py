"""
English compact version of the reliability/BSS figure (48 h) for the SIMBig/LNCS

Versión para el camera-ready de SIMBig (formato LNCS).

ÚNICA diferencia: la figura se dibuja al tamaño EXACTO que ocupa en la página
(0.68\textwidth = 236 pt = 3.28 in), de modo que \includegraphics no la
reduce y las fuentes se imprimen a su tamaño nominal. En el original se dibujaba
a 6.2 in y entraba al 53%, así que la letra de 6.5 pt acababa en 3.4 pt.
Springer exige que la rotulación de las figuras no baje de 6 pt.

Los tamaños de fuente NO se tocan; lo que cambia es figsize (y, donde hacía falta,
se acortan etiquetas de leyenda para que quepan en el panel más estrecho).

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import ROOT, load_split, sigmoid

IMG = os.environ.get("SFMM_FIG", os.path.join(ROOT, "results", "figures"))
os.makedirs(IMG, exist_ok=True)
H, FOLDS = 48, [0, 1, 2, 3, 4]
MODELS = [("swin3d", "Swin3D (FITS)", "#1f77b4"), ("lstm", "BiLSTM (SHARP)", "#d62728")]


def bss(p, y):
    pbar = float(np.mean(y))
    return 1.0 - np.mean((p - y) ** 2) / np.mean((pbar - y) ** 2)


def reliability(p, y, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            xs.append(p[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def run_model(model):
    raw, cal, labT = [], [], None
    for k in FOLDS:
        v, t = load_split(model, H, k, "val"), load_split(model, H, k, "test")
        if v is None or t is None:
            continue
        lo_v, y_v = v[0], v[1].astype(float); lo_t, y_t = t[0], t[1].astype(float)
        p_v, p_t = sigmoid(lo_v), sigmoid(lo_t)
        platt = LogisticRegression(C=1e6, max_iter=1000).fit(lo_v.reshape(-1, 1), y_v)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_v, y_v)
        b_platt = bss(platt.predict_proba(lo_v.reshape(-1, 1))[:, 1], y_v)
        b_iso = bss(iso.predict(p_v), y_v)
        p_cal = iso.predict(p_t) if b_iso >= b_platt else platt.predict_proba(lo_t.reshape(-1, 1))[:, 1]
        raw.append(p_t); cal.append(p_cal); labT = y_t
    ens_raw, ens_cal = np.mean(raw, axis=0), np.mean(cal, axis=0)
    return ens_raw, ens_cal, labT, bss(ens_cal, labT)


def main():
    results = [run_model(m) for m, _, _ in MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(4.82, 2.10))
    for ax, (ens_raw, ens_cal, y, bss_cal), (_, name, col) in zip(axes, results, MODELS):
        ax.plot([0, 1], [0, 1], "k:", lw=0.9, label="perfect")
        for probs, sty, c, lab in [(ens_raw, "o--", "#888", "raw"),
                                   (ens_cal, "s-", col, "calibrated")]:
            xs, ys = reliability(probs, y)
            ax.plot(xs, ys, sty, color=c, lw=1.6, ms=4, label=lab)
        ax.set_title(f"{name}  ·  BSS {bss_cal:.3f}", fontsize=10)
        ax.set_xlabel("Predicted probability", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.tick_params(labelsize=8)
        ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9,
                  handlelength=1.2, handletextpad=0.4, borderpad=0.3,
                  labelspacing=0.25)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed frequency", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig6_calibracion_48h.pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {IMG}/fig6_calibracion_48h.pdf  "
          f"(BSS cal: Swin3D={results[0][3]:.3f}, BiLSTM={results[1][3]:.3f})")


if __name__ == "__main__":
    main()

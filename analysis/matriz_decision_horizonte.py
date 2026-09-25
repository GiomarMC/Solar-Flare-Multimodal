"""
Matriz de decisión por horizonte — qué modalidad conviene, para qué objetivo.

FUENTE ÚNICA DE VERDAD: recalcula TODO desde los logits de outputs/logits/.
No transcribe valores de ningún .txt previo. Esto responde a la objeción de
critica1 §4 (tres valores distintos de BiLSTM 24 h usados de forma intercambiable
por venir de tres pipelines sin etiquetar).

Produce dos cosas:
  1. Tabla cruda   — por horizonte x rama: TSS, POD, FAR, HSS, F1, AUC, BSS
                     calibrado, nº de parámetros.
  2. Matriz de decisión — por horizonte x objetivo: qué rama gana Y con qué
                     respaldo estadístico (bootstrap pareado + sigma entre folds).

Criterio de respaldo (el mismo de equivalencia_tost.py): una diferencia solo se
declara accionable si es significativa en el bootstrap pareado del test Y además
excede la sigma entre folds — porque el bootstrap remuestrea el test con los
modelos fijos, mientras que la sigma mide qué pasa al reentrenar.

Uso:
    python analysis/matriz_decision_horizonte.py
"""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import (OUTDIR, FIGDIR, DEVICE, load_split, read_tau_opt,
                          sigmoid, sweep_tau, tss_point, auc_point)
from bss_calibration import bss, fit_platt, fit_isotonic
from diversidad_ramas import FOLDS

HORIZONS = [24, 48]
BRANCHES = [("swin3d", "Swin3D (FITS)"), ("lstm", "BiLSTM (SHARP)")]
B = 10000

# paleta categórica validada (dataviz refs/palette.md, slots 1 y 2)
COLOR = {"swin3d": "#2a78d6", "lstm": "#eb6834"}
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


# ------------------------------------------------------------------ recolección
def collect(model, horizon):
    """Ensemble de test (probs crudas y calibradas), tau, labels, TSS por fold."""
    raw, cal, taus, per_fold = [], [], [], []
    y = None
    for k in FOLDS:
        v, t = load_split(model, horizon, k, "val"), load_split(model, horizon, k, "test")
        if v is None or t is None:
            continue
        lo_v, y_v = v[0], v[1].astype(np.float64)
        lo_t, y_t = t[0], t[1].astype(np.float64)
        p_v, p_t = sigmoid(lo_v), sigmoid(lo_t)

        tau = read_tau_opt(model, horizon, k)
        if tau is None:
            tau, _ = sweep_tau(p_v, y_v)
        taus.append(float(tau))
        per_fold.append(tss_point(p_t, y_t, tau))

        # calibración ajustada en val, elegida por BSS en val (igual que bss_calibration.py)
        platt, iso = fit_platt(lo_v, y_v), fit_isotonic(p_v, y_v)
        p_t_cal = iso(p_t) if bss(iso(p_v), y_v)[0] >= bss(platt(lo_v), y_v)[0] else platt(lo_t)

        raw.append(p_t); cal.append(p_t_cal); y = y_t
    return (np.mean(raw, axis=0), np.mean(cal, axis=0), float(np.mean(taus)),
            y, np.array(per_fold))


def metrics(p_raw, p_cal, y, tau):
    pred = (p_raw >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pod = tp / (tp + fn + 1e-12); pofd = fp / (fp + tn + 1e-12)
    far = fp / (tp + fp + 1e-12); pre = tp / (tp + fp + 1e-12)
    f1  = 2 * pre * pod / (pre + pod + 1e-12)
    hss = 2*(tp*tn - fp*fn) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) + 1e-12)
    return dict(TSS=pod - pofd, POD=pod, FAR=far, HSS=hss, F1=f1,
                AUC=auc_point(p_raw, y), BSS=bss(p_cal, y)[0],
                TP=tp, FN=fn, FP=fp, tau=tau)


def paired_metric_ci(fn_metric, arrs_a, arrs_b, y, B=B, seed=42):
    """IC95% de Delta=metric_a-metric_b con remuestreo estratificado PAREADO."""
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    d = np.empty(B)
    for i in range(B):
        idx = np.concatenate([rng.choice(pos, pos.size, replace=True),
                              rng.choice(neg, neg.size, replace=True)])
        d[i] = fn_metric(*[a[idx] for a in arrs_a], y[idx]) - \
               fn_metric(*[a[idx] for a in arrs_b], y[idx])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def verdict(delta, lo, hi, noise, name_a, name_b):
    """Gana quien tenga mayor métrica, con el respaldo estadístico correspondiente.

    Delta se calcula siempre como A-B; si gana B se invierte el signo del intervalo
    para que el IC se lea SIEMPRE a favor del ganador.
    """
    win = name_a if delta > 0 else name_b
    dmin = max(abs(lo), abs(hi))
    sig = (lo > 0 or hi < 0)
    if delta < 0:                       # el ganador es B -> reexpresar como B-A
        lo, hi = -hi, -lo
    if not sig:
        return "empate", f"equivalentes (+-{dmin:.3f}), IC incluye 0"
    if dmin <= noise:
        return win, f"gana por {abs(delta):.3f} [{lo:+.3f},{hi:+.3f}], < sigma folds ({noise:.3f})"
    return win, f"gana por {abs(delta):.3f} [{lo:+.3f},{hi:+.3f}], > sigma folds"


def n_params():
    from scripts.train_lstm_standalone import LSTMStandaloneModel
    from torchvision.models.video import swin3d_t
    nl = sum(p.numel() for p in LSTMStandaloneModel(17, 64, 64, 0.4).parameters())
    ns = sum(p.numel() for p in swin3d_t(weights=None).parameters())
    return {"swin3d": ns, "lstm": nl}


def main():
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lines, cells = [], {}
    def w(s=""):
        print(s); lines.append(s)

    npar = n_params()
    w("=" * 84)
    w("  MATRIZ DE DECISIÓN POR HORIZONTE — recalculada íntegramente desde los logits")
    w("=" * 84)

    for h in HORIZONS:
        data = {}
        for m, nm in BRANCHES:
            p_raw, p_cal, tau, y, pf = collect(m, h)
            data[m] = dict(raw=p_raw, cal=p_cal, tau=tau, y=y, pf=pf,
                           met=metrics(p_raw, p_cal, y, tau))
        y = data["swin3d"]["y"]
        noise = max(data[m]["pf"].std() for m, _ in BRANCHES)

        w()
        w("-" * 84)
        w(f"  HORIZONTE {h} h   (test n={len(y)}, positivos={int(y.sum())}, "
          f"sigma entre folds = {noise:.4f})")
        w("-" * 84)
        w(f"  {'Rama':<18s}{'TSS':>8s}{'POD':>8s}{'FAR':>8s}{'HSS':>8s}{'F1':>8s}"
          f"{'AUC':>8s}{'BSS_cal':>9s}{'params':>12s}{'tau':>6s}")
        for m, nm in BRANCHES:
            e = data[m]["met"]
            w(f"  {nm:<18s}{e['TSS']:>8.4f}{e['POD']:>8.4f}{e['FAR']:>8.4f}{e['HSS']:>8.4f}"
              f"{e['F1']:>8.4f}{e['AUC']:>8.4f}{e['BSS']:>9.4f}{npar[m]:>12,d}{e['tau']:>6.2f}")

        # ---- comparaciones pareadas por objetivo ----
        a, b = data["swin3d"], data["lstm"]
        NA, NB = "Swin3D", "BiLSTM"

        def _tss(p, yy):
            pr = (p >= 0.5).astype(int)
            tp = ((pr == 1) & (yy == 1)).sum(); fn_ = ((pr == 0) & (yy == 1)).sum()
            fp = ((pr == 1) & (yy == 0)).sum(); tn = ((pr == 0) & (yy == 0)).sum()
            return tp/(tp+fn_+1e-12) - fp/(fp+tn+1e-12)

        def _hss(p, yy):
            pr = (p >= 0.5).astype(int)
            tp = ((pr == 1) & (yy == 1)).sum(); fn_ = ((pr == 0) & (yy == 1)).sum()
            fp = ((pr == 1) & (yy == 0)).sum(); tn = ((pr == 0) & (yy == 0)).sum()
            return 2*(tp*tn - fp*fn_)/((tp+fn_)*(fn_+tn) + (tp+fp)*(fp+tn) + 1e-12)

        # decisiones binarias precomputadas en el tau de cada rama -> se comparan como 0/1
        da = (a["raw"] >= a["tau"]).astype(float)
        db = (b["raw"] >= b["tau"]).astype(float)

        objetivos = []
        d, lo, hi = paired_metric_ci(_tss, (da,), (db,), y)
        objetivos.append(("Máximo TSS", *verdict(d, lo, hi, noise, NA, NB)))

        d, lo, hi = paired_metric_ci(_hss, (da,), (db,), y)
        objetivos.append(("Economía de falsas alarmas (HSS)", *verdict(d, lo, hi, noise, NA, NB)))

        d, lo, hi = paired_metric_ci(lambda p, yy: bss(p, yy)[0], (a["cal"],), (b["cal"],), y)
        objetivos.append(("Fiabilidad probabilística (BSS)", *verdict(d, lo, hi, 0.0, NA, NB)))

        ratio = npar["swin3d"] / npar["lstm"]
        objetivos.append(("Costo computacional", NB, f"{ratio:,.0f}x menos parámetros"))

        w()
        w(f"  {'Objetivo':<34s}{'Recomendación':<16s}{'Respaldo'}")
        for obj, win, why in objetivos:
            w(f"  {obj:<34s}{win:<16s}{why}")
        cells[h] = objetivos

    w()
    w("=" * 84)
    w("  Nota: 'gana por X pero < sigma folds' significa que la diferencia es real para")
    w("  ESTE test pero menor que la variabilidad al reentrenar — no sostiene por sí sola")
    w("  una recomendación de modelo. Ver results/reports/equivalencia_tost.txt.")

    out = os.path.join(OUTDIR, "matriz_decision_horizonte.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")
    figura(cells)


def figura(cells):
    objs = [o[0] for o in cells[HORIZONS[0]]]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.set_xlim(0, 2); ax.set_ylim(0, len(objs)); ax.axis("off")

    key = {"Swin3D": "swin3d", "BiLSTM": "lstm"}
    for j, h in enumerate(HORIZONS):
        for i, (obj, win, why) in enumerate(cells[h]):
            yy = len(objs) - 1 - i
            col = COLOR.get(key.get(win, ""), "#9a9a95")
            ax.add_patch(FancyBboxPatch((j + 0.03, yy + 0.12), 0.94, 0.76,
                                        boxstyle="round,pad=0,rounding_size=0.04",
                                        fc=col, ec="none", alpha=0.13))
            ax.add_patch(FancyBboxPatch((j + 0.05, yy + 0.20), 0.035, 0.60,
                                        boxstyle="round,pad=0,rounding_size=0.02",
                                        fc=col, ec="none"))
            ax.text(j + 0.12, yy + 0.62, win, color=INK, fontsize=11, fontweight="bold",
                    va="center", ha="left")
            ax.text(j + 0.12, yy + 0.34, why, color=INK2, fontsize=7.4,
                    va="center", ha="left")
        ax.text(j + 0.5, len(objs) + 0.12, f"{h} h", color=INK, fontsize=12,
                fontweight="bold", ha="center")

    for i, obj in enumerate(objs):
        ax.text(-0.04, len(objs) - 1 - i + 0.5, obj, color=INK2, fontsize=9,
                va="center", ha="right")

    ax.set_title("Qué modalidad conviene, por horizonte y objetivo",
                 color=INK, fontsize=13, fontweight="bold", pad=26, loc="left", x=-0.42)
    for p, lbl in ((COLOR["swin3d"], "Swin3D (FITS)"), (COLOR["lstm"], "BiLSTM (SHARP)")):
        ax.plot([], [], "s", color=p, label=lbl, markersize=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.16), ncol=2, frameon=False,
              fontsize=9, labelcolor=INK2)
    plt.tight_layout()
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIGDIR, f"matriz_decision_horizonte.{ext}"),
                    dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"[guardado] {os.path.join(FIGDIR, 'matriz_decision_horizonte.png')}")


if __name__ == "__main__":
    main()

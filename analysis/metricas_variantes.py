"""
Todas las métricas (no solo TSS) de las ramas individuales y de la fusión tardía,
comparando la rama física del paper (BiLSTM-17) con la variante sin parámetros
derivados del LoS (BiLSTM-16, dataset_temporal_v6).

Métricas, con el mismo protocolo que la tabla de resultados del paper:
  - TSS μ±σ, POD, FAR, HSS, F1 : por fold (τ barrido en el VAL del fold, aplicado
                                  al TEST) y promediados entre folds.
  - TSS_ens, AUC, BSS, TP/FP/FN : sobre el ensemble (promedio de probabilidades de
                                  test entre folds, τ = media de los τ por fold).
  FAR = false alarm ratio = FP / (TP + FP), como en el paper.
  BSS = Brier Skill Score frente a la climatología (graficos/bss_calibration.py),
        con probabilidades crudas. Solo tiene sentido para salidas que son
        probabilidades: en las reglas producto/máximo/mínimo no se reporta.

Los métodos de fusión replican graficos/fusion_variantes.py (que a su vez replica
el código del paper). Control: las filas de BiLSTM-17 a 48 h deben reproducir la
tabla del paper (Swin3D POD 0.985 / FAR 0.752 / HSS 0.348 / F1 0.394, etc.).

Uso:
    python graficos/metricas_variantes.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import OUTDIR, sigmoid, sweep_tau, tss_point, auc_point
from stacking_5fold import train_meta, meta_probs
from late_fusion_rules import RULES, metrics as pod_far_hss_f1
from bss_calibration import bss
from fusion_variantes import load, FOLDS, W_GRID

HORIZONS = [24, 48]
NO_PROB = {"Producto", "Máximo", "Mínimo"}          # salidas que no son probabilidades calibrables
PAPER_48 = {                                        # tabla del paper, 48 h (POD, FAR, HSS, F1)
    "Swin3D (FITS)": (0.985, 0.752, 0.348, 0.394),
    "BiLSTM-17": (0.903, 0.653, 0.464, 0.499),
    "Ens. ponderado · 17": (0.924, 0.716, 0.388, 0.430),
}


# ------------------------------------------------ salidas por fold de cada método
def per_fold_single(d):
    out = []
    for k in FOLDS:
        lv, yv, lt, yt = d[k]
        tau, _ = sweep_tau(sigmoid(lv), yv)
        out.append((sigmoid(lt), yt, tau))
    return out


def per_fold_weighted(sw, ph):
    out = []
    for k in FOLDS:
        s_lv, yv, s_lt, yt = sw[k]
        p_lv, _, p_lt, _ = ph[k]
        ps_va, pl_va = sigmoid(s_lv), sigmoid(p_lv)
        best = (-2.0, 1.0, 0.5)
        for w in W_GRID:
            tau, vtss = sweep_tau(w * pl_va + (1.0 - w) * ps_va, yv)
            if vtss > best[0]:
                best = (vtss, float(w), tau)
        _, w_f, tau_f = best
        out.append((w_f * sigmoid(p_lt) + (1.0 - w_f) * sigmoid(s_lt), yt, tau_f))
    return out


def per_fold_stacking(sw, ph):
    out = []
    for k in FOLDS:
        s_lv, yv, s_lt, yt = sw[k]
        p_lv, _, p_lt, _ = ph[k]
        meta, stats = train_meta(s_lv, p_lv, yv)
        tau, _ = sweep_tau(meta_probs(meta, stats, s_lv, p_lv), yv)
        out.append((meta_probs(meta, stats, s_lt, p_lt), yt, tau))
    return out


def per_fold_rule(rule, sw, ph):
    out = []
    for k in FOLDS:
        s_lv, yv, s_lt, yt = sw[k]
        p_lv, _, p_lt, _ = ph[k]
        tau, _ = sweep_tau(rule(sigmoid(s_lv), sigmoid(p_lv)), yv)
        out.append((rule(sigmoid(s_lt), sigmoid(p_lt)), yt, tau))
    return out


# ------------------------------------------------------------------- agregación
def summarize(folds, prob_output=True):
    tss = [tss_point(p, y, t) for p, y, t in folds]
    m = np.array([pod_far_hss_f1(p, y, t) for p, y, t in folds])   # pod, far, hss, f1
    ens = np.mean([p for p, _, _ in folds], axis=0)
    tau = float(np.mean([t for _, _, t in folds]))
    y = folds[0][1]
    pred = ens >= tau
    return {
        "tss_mu": float(np.mean(tss)), "tss_sd": float(np.std(tss)),
        "pod": m[:, 0].mean(), "far": m[:, 1].mean(), "hss": m[:, 2].mean(), "f1": m[:, 3].mean(),
        "tss_ens": tss_point(ens, y, tau), "auc": auc_point(ens, y),
        "bss": bss(ens, y)[0] if prob_output else None,
        "tp": int((pred & (y == 1)).sum()), "fp": int((pred & (y == 0)).sum()),
        "fn": int((~pred & (y == 1)).sum()),
    }


def main():
    lines = []
    def w(s=""):
        print(s, flush=True); lines.append(s)

    hdr = (f"  {'Modelo':<24}{'TSS μ±σ':>15}{'TSS_ens':>9}{'POD':>7}{'FAR':>7}{'HSS':>7}"
           f"{'F1':>7}{'AUC':>8}{'BSS':>8}{'TP':>6}{'FP':>6}{'FN':>5}")
    w("=" * len(hdr))
    w("  MÉTRICAS COMPLETAS — ramas individuales y fusión tardía (BiLSTM-17 vs BiLSTM-16)")
    w("  POD/FAR/HSS/F1 y TSS μ±σ: media entre folds.  TSS_ens/AUC/BSS/TP/FP/FN: ensemble de folds.")
    w("  FAR = FP/(TP+FP).  POD alto = pocas llamaradas perdidas; FAR bajo = pocas falsas alarmas.")
    w("=" * len(hdr))

    for h in HORIZONS:
        sw, p17, p16 = load("swin3d", h), load("lstm", h), load("lstm16", h)
        y = sw[FOLDS[0]][3]
        rows = [
            ("Swin3D (FITS)", summarize(per_fold_single(sw))),
            ("BiLSTM-17", summarize(per_fold_single(p17))),
            ("BiLSTM-16", summarize(per_fold_single(p16))),
            None,
            ("Ens. ponderado · 17", summarize(per_fold_weighted(sw, p17))),
            ("Stacking · 17", summarize(per_fold_stacking(sw, p17))),
            None,
            ("Ens. ponderado · 16", summarize(per_fold_weighted(sw, p16))),
            ("Stacking · 16", summarize(per_fold_stacking(sw, p16))),
        ]
        for name, rule in RULES.items():
            short = name.split(" ")[0]
            rows.append((f"{short} · 16", summarize(per_fold_rule(rule, sw, p16),
                                                    prob_output=short not in NO_PROB)))

        w()
        w(f"  HORIZONTE {h} h   (test n={len(y)}, positivos={int((y == 1).sum())})")
        w(hdr)
        w("  " + "-" * (len(hdr) - 2))
        for row in rows:
            if row is None:
                w(); continue
            name, r = row
            b = "     —" if r["bss"] is None else f"{r['bss']:>8.4f}"
            w(f"  {name:<24}{r['tss_mu']:>8.4f}±{r['tss_sd']:<6.4f}{r['tss_ens']:>9.4f}"
              f"{r['pod']:>7.3f}{r['far']:>7.3f}{r['hss']:>7.3f}{r['f1']:>7.3f}"
              f"{r['auc']:>8.4f}{b:>8}{r['tp']:>6}{r['fp']:>6}{r['fn']:>5}")
            if h == 48 and name in PAPER_48:
                ref = PAPER_48[name]
                got = (r["pod"], r["far"], r["hss"], r["f1"])
                ok = all(abs(a - b) < 5e-4 for a, b in zip(got, ref))
                w(f"      [control vs paper POD/FAR/HSS/F1 = {ref} -> {'OK' if ok else 'DIFIERE'}]")

    out = os.path.join(OUTDIR, "metricas_variantes.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

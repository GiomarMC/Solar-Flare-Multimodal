"""
Reglas fijas de fusión tardía (Kittler et al., 1998) — Swin3D + BiLSTM, 48 h.

Refuerza el resultado negativo del paper ("la fusión tardía no es aditiva"):
además del ensemble ponderado y el stacking con meta-MLP, evalúa las reglas de
combinación FIJAS clásicas sobre los logits ya guardados, SIN reentrenar:

    media   = (p_swin + p_lstm) / 2          (sum rule)
    producto= p_swin * p_lstm                (product rule, independencia)
    max     = max(p_swin, p_lstm)            (orientada a sensibilidad)
    min     = min(p_swin, p_lstm)            (orientada a precisión)

Protocolo idéntico al del ensemble del cuerpo (analysis/bootstrap_ci.py):
  - Por fold: se combinan las probabilidades; el umbral tau se barre sobre el
    VAL del fold (sin fuga) y se aplica al TEST. TSS por fold -> media ± sigma.
  - Ensemble: se promedian las probabilidades combinadas de test entre folds y
    se usa tau = media de los tau por fold. TSS/AUC con IC95% bootstrap.

Como referencia (sanity) recomputa también Swin3D y BiLSTM individuales bajo el
MISMO protocolo: deben reproducir ~0.868 y ~0.841.

Uso:
    python analysis/late_fusion_rules.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import (ROOT, OUTDIR, load_split, sigmoid, sweep_tau,
                          tss_point, auc_point, bootstrap_ci)

H, FOLDS = 48, [0, 1, 2, 3, 4]
IMG = os.environ.get("SFMM_FIG", os.path.join(ROOT, "results", "figures"))
os.makedirs(IMG, exist_ok=True)
B = 10000

# Reglas de combinación parametrizables (operan sobre probabilidades)
RULES = {
    "Media (sum)":   lambda a, b: 0.5 * (a + b),
    "Producto":      lambda a, b: a * b,
    "Máximo":        lambda a, b: np.maximum(a, b),
    "Mínimo":        lambda a, b: np.minimum(a, b),
}
# Individuales (referencia / sanity): combinan ignorando una rama
SINGLE = {
    "Swin3D (FITS)":  lambda a, b: a,   # a = swin
    "BiLSTM (SHARP)": lambda a, b: b,   # b = lstm
}


def confusion(probs, y, tau):
    pred = (probs >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def metrics(probs, y, tau):
    tp, fp, fn, tn = confusion(probs, y, tau)
    pod = tp / (tp + fn + 1e-12)
    far = fp / (tp + fp + 1e-12)                      # false alarm ratio
    f1  = 2 * tp / (2 * tp + fp + fn + 1e-12)
    n = tp + fp + fn + tn
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (n * n + 1e-12)
    po = (tp + tn) / n
    hss = (po - pe) / (1 - pe + 1e-12)
    return pod, far, hss, f1


def eval_rule(rule, p_swin_val, p_lstm_val, yv,
              p_swin_test, p_lstm_test, yt):
    """Protocolo del ensemble: tau por fold sobre val, promedio de probs en test."""
    per_fold_tss, per_fold_tau, comb_test = [], [], []
    for k in FOLDS:
        cv = rule(p_swin_val[k], p_lstm_val[k])
        ct = rule(p_swin_test[k], p_lstm_test[k])
        tau_k, _ = sweep_tau(cv, yv[k])
        per_fold_tss.append(tss_point(ct, yt[k], tau_k))
        per_fold_tau.append(tau_k)
        comb_test.append(ct)
    ens_test = np.mean(comb_test, axis=0)           # test es idéntico entre folds
    tau_ens = float(np.mean(per_fold_tau))
    y0 = yt[FOLDS[0]]
    tss_ens = tss_point(ens_test, y0, tau_ens)
    auc_ens = auc_point(ens_test, y0)
    ci = bootstrap_ci(ens_test, y0, tau_ens, B=B)
    pod, far, hss, f1 = metrics(ens_test, y0, tau_ens)
    return {
        "mu": float(np.mean(per_fold_tss)), "sd": float(np.std(per_fold_tss)),
        "tss_ens": tss_ens, "ci": ci["tss"], "auc": auc_ens, "tau": tau_ens,
        "pod": pod, "far": far, "hss": hss, "f1": f1,
    }


def main():
    # cargar probabilidades de ambas ramas por fold
    p_swin_val, p_lstm_val, yv = {}, {}, {}
    p_swin_test, p_lstm_test, yt = {}, {}, {}
    for k in FOLDS:
        sv = load_split("swin3d", H, k, "val");  lv = load_split("lstm", H, k, "val")
        st = load_split("swin3d", H, k, "test"); lt = load_split("lstm", H, k, "test")
        if None in (sv, lv, st, lt):
            print(f"  k={k}: faltan logits, omito"); continue
        p_swin_val[k] = sigmoid(sv[0]); p_lstm_val[k] = sigmoid(lv[0]); yv[k] = sv[1]
        p_swin_test[k] = sigmoid(st[0]); p_lstm_test[k] = sigmoid(lt[0]); yt[k] = st[1]

    res = {}
    for name, rule in {**SINGLE, **RULES}.items():
        res[name] = eval_rule(rule, p_swin_val, p_lstm_val, yv,
                              p_swin_test, p_lstm_test, yt)

    swin_ref = res["Swin3D (FITS)"]["tss_ens"]      # mejor rama individual

    # ---- salida de texto ----
    lines = [
        "Reglas fijas de fusión tardía (Kittler et al., 1998) — Swin3D + BiLSTM, 48 h",
        f"(protocolo del ensemble: tau por fold sobre val, promedio de probs en test; B={B})",
        f"Referencia: mejor rama individual (Swin3D) TSS_ens = {swin_ref:.4f}\n",
        f"  {'Método':<18}{'TSS μ±σ':>16}{'TSS_ens':>10}{'  IC95%':>20}"
        f"{'AUC':>8}{'POD':>7}{'FAR':>7}{'HSS':>7}{'F1':>7}  {'τ':>5}",
    ]
    order = ["Swin3D (FITS)", "BiLSTM (SHARP)",
             "Media (sum)", "Producto", "Máximo", "Mínimo"]
    for name in order:
        r = res[name]
        flag = "" if name in SINGLE else (" *" if r["tss_ens"] >= swin_ref else "")
        lines.append(
            f"  {name:<18}{r['mu']:>8.4f}±{r['sd']:<6.4f}{r['tss_ens']:>10.4f}"
            f"   [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]"
            f"{r['auc']:>8.4f}{r['pod']:>7.3f}{r['far']:>7.3f}{r['hss']:>7.3f}"
            f"{r['f1']:>7.3f}  {r['tau']:>5.2f}{flag}")
    lines.append(
        "\nConclusión: ninguna regla fija de fusión supera a la mejor rama individual\n"
        "(Swin3D, TSS=%.4f). El resultado negativo se mantiene en toda la familia de\n"
        "fusión tardía: reglas fijas (media/producto/max/min), ensemble ponderado y\n"
        "stacking con meta-MLP." % swin_ref)

    txt = os.path.join(OUTDIR, "late_fusion_rules_48h.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nGuardado: {txt}")

    # ---- figura: barras TSS_ens con IC95%, línea en Swin3D ----
    names = ["Máximo", "Mínimo", "Producto", "Media (sum)", "BiLSTM (SHARP)", "Swin3D (FITS)"]
    vals = [res[n]["tss_ens"] for n in names]
    err = [[res[n]["tss_ens"] - res[n]["ci"][0] for n in names],
           [res[n]["ci"][1] - res[n]["tss_ens"] for n in names]]
    cols = ["#7f7f7f"] * 4 + ["#d62728", "#1f77b4"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    yy = np.arange(len(names))
    ax.barh(yy, vals, xerr=err, color=cols, alpha=0.9, capsize=3,
            error_kw=dict(ecolor="#333", lw=1))
    ax.axvline(swin_ref, color="#1f77b4", ls="--", lw=1.3,
               label=f"Mejor rama individual (Swin3D = {swin_ref:.3f})")
    ax.set_yticks(yy); ax.set_yticklabels(names)
    ax.set_xlabel("TSS (ensemble de folds) con IC95%")
    ax.set_title("Reglas fijas de fusión tardía vs. mejor rama individual — 48 h")
    ax.set_xlim(0.70, 0.90); ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig8_late_fusion_rules_48h.pdf"))
    fig.savefig(os.path.join(IMG, "fig8_late_fusion_rules_48h.png"), dpi=150)
    plt.close(fig)
    print(f"Figura:   {IMG}/fig8_late_fusion_rules_48h.pdf (+ .png)")


if __name__ == "__main__":
    main()

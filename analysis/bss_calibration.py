"""
Calibración probabilística y Brier Skill Score (BSS) a 48 h — SIN reentrenar.

Trabaja sobre los logits ya guardados (outputs/logits/{model}_48h_k{fold}_{split}.npz),
reusando utilidades de bootstrap_ci.py. Para Swin3D y BiLSTM:

  - Brier Score (BS) y BSS sobre el test fijo, con probabilidades CRUDAS y CALIBRADAS.
  - Calibración ajustada en VALIDACIÓN y aplicada al TEST (sin fuga):
      * Platt  : LogisticRegression sobre el logit (1 feature).
      * Isotónica: IsotonicRegression sobre la probabilidad cruda.
    Se elige por fold el método con mejor BSS de validación.
  - Diagrama de fiabilidad (10 bins) cruda vs calibrada para el ensemble de folds.

Nota: la calibración es monótona → NO altera TSS/AUC ni el orden; es un eje
complementario al TSS. El motivo de la descalibración cruda es el WeightedRandomSampler
(batches 1:1) + pos_weight usados en entrenamiento.

Uso:
    python analysis/bss_calibration.py
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
from bootstrap_ci import ROOT, load_split, sigmoid, read_tau_opt, tss_point, auc_point

H, FOLDS = 48, [0, 1, 2, 3, 4]
OUT = os.environ.get("SFMM_OUT", os.path.join(ROOT, "results", "reports"))
IMG = os.environ.get("SFMM_FIG", os.path.join(ROOT, "results", "figures"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)
MODELS = [("swin3d", "Swin3D (FITS)", "#1f77b4"), ("lstm", "BiLSTM (SHARP)", "#d62728")]


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def bss(p, y):
    """Brier Skill Score frente a la climatología (pronóstico constante = tasa base)."""
    bs = brier(p, y)
    pbar = float(np.mean(y))
    bs_ref = float(np.mean((pbar - y) ** 2))  # = pbar(1-pbar)
    return 1.0 - bs / bs_ref, bs, bs_ref


def fit_platt(logit_val, y_val):
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(logit_val.reshape(-1, 1), y_val)
    return lambda logit: lr.predict_proba(logit.reshape(-1, 1))[:, 1]


def fit_isotonic(p_val, y_val):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_val, y_val)
    return lambda p: iso.predict(p)


def reliability(p, y, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    xs, ys, ns = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean()); ys.append(y[m].mean()); ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)


def run_model(model, lines):
    # ---- cargar por fold ----
    val, test, taus = {}, {}, {}
    labT = None
    for k in FOLDS:
        v, t = load_split(model, H, k, "val"), load_split(model, H, k, "test")
        if v is None or t is None:
            continue
        val[k] = (v[0], v[1].astype(np.float64))
        test[k] = (t[0], t[1].astype(np.float64))
        taus[k] = read_tau_opt(model, H, k) or 0.5
        labT = t[1].astype(np.float64)

    pbar = float(np.mean(labT))
    lines.append(f"\n{'='*70}\n  {model.upper()} — 48 h   (tasa base test = {pbar:.4f})\n{'='*70}")
    lines.append(f"  {'Fold':<5}{'BS_raw':>9}{'BSS_raw':>9}{'BS_cal':>9}{'BSS_cal':>9}{'método':>10}{'TSS':>8}")

    raw_test_probs, cal_test_probs = [], []
    for k in sorted(test):
        lo_v, y_v = val[k]; lo_t, y_t = test[k]
        p_v_raw, p_t_raw = sigmoid(lo_v), sigmoid(lo_t)

        # candidatos de calibración (ajuste en val)
        platt = fit_platt(lo_v, y_v)
        iso = fit_isotonic(p_v_raw, y_v)
        bss_platt_val = bss(platt(lo_v), y_v)[0]
        bss_iso_val = bss(iso(p_v_raw), y_v)[0]
        if bss_iso_val >= bss_platt_val:
            method, p_t_cal = "isotónica", iso(p_t_raw)
        else:
            method, p_t_cal = "Platt", platt(lo_t)

        bss_raw, bs_raw, _ = bss(p_t_raw, y_t)
        bss_cal, bs_cal, _ = bss(p_t_cal, y_t)
        tss = tss_point(p_t_raw, y_t, taus[k])  # invariante a calibración monótona
        lines.append(f"  k={k:<3}{bs_raw:>9.4f}{bss_raw:>9.4f}{bs_cal:>9.4f}{bss_cal:>9.4f}{method:>10}{tss:>8.4f}")
        raw_test_probs.append(p_t_raw); cal_test_probs.append(p_t_cal)

    # ---- ensemble (promedio de probabilidades del test) ----
    ens_raw = np.mean(raw_test_probs, axis=0)
    ens_cal = np.mean(cal_test_probs, axis=0)
    bss_raw_e, bs_raw_e, bs_ref = bss(ens_raw, labT)
    bss_cal_e, bs_cal_e, _ = bss(ens_cal, labT)
    auc = auc_point(ens_raw, labT)
    lines.append(f"  {'-'*64}")
    lines.append(f"  ENSEMBLE   BS_raw={bs_raw_e:.4f}  BSS_raw={bss_raw_e:.4f}   "
                 f"BS_cal={bs_cal_e:.4f}  BSS_cal={bss_cal_e:.4f}   (BS_clim={bs_ref:.4f}, AUC={auc:.4f})")
    return dict(model=model, ens_raw=ens_raw, ens_cal=ens_cal, labels=labT,
                bss_raw=bss_raw_e, bss_cal=bss_cal_e)


def main():
    lines = ["Calibración probabilística y BSS — 48 h (sin reentrenar)"]
    results = [run_model(m, lines) for m, _, _ in MODELS]

    # ---- figura: diagrama de fiabilidad (raw vs calibrado), un panel por modelo ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, res, (_, name, col) in zip(axes, results, MODELS):
        y = res["labels"]
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="calibración perfecta")
        # La curva cruda justifica visualmente la necesidad de calibrar; solo se
        # rotula con su BSS la curva CALIBRADA (la métrica que se reporta).
        for probs, sty, lab in [(res["ens_raw"], "o--", "sin calibrar"),
                                (res["ens_cal"], "s-", f"calibrada (BSS={res['bss_cal']:.3f})")]:
            xs, ys, ns = reliability(probs, y)
            ax.plot(xs, ys, sty, color=col if "calibr" in lab else "#888",
                    lw=2, ms=6, label=lab)
        ax.set_title(name); ax.set_xlabel("Probabilidad media predicha")
        ax.set_ylabel("Frecuencia observada"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Diagramas de fiabilidad — ensemble de folds, 48 h")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig6_calibracion_48h.pdf"))
    fig.savefig(os.path.join(IMG, "fig6_calibracion_48h.png"), dpi=150)
    plt.close(fig)

    txt = os.path.join(OUT, "bss_calibration_48h.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nGuardado: {txt}")
    print(f"Figura:   {IMG}/fig6_calibracion_48h.pdf  (+ .png)")


if __name__ == "__main__":
    main()

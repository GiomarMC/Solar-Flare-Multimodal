"""
Bootstrap PAREADO para comparar modelos sobre el test FIJO (n=9384).

A diferencia de bootstrap_ci.py (que da el IC de cada modelo por separado),
aquí se remuestrean los MISMOS índices del test para los dos modelos en cada
réplica y se calcula Δ = TSS_A − TSS_B. Si el IC95% de Δ no cruza 0, la
diferencia es significativa (solapar los IC marginales NO basta).

Construye, en 5-fold y sobre el test común:
  - Ensemble Swin3D   = promedio de probs de los 5 folds (tau = media de tau_opt).
  - Ensemble BiLSTM   = idem.
  - Fusión cross-modal = por fold se elige (w_phys, tau) que maximiza el TSS en
    el VAL de ese fold (sin fuga), se fusiona el test (w*p_lstm+(1-w)*p_swin) y
    se promedian los 5 folds. tau = media de los tau por fold.

Compara con bootstrap pareado:  Swin3D vs BiLSTM,  Fusión vs Swin3D,  Fusión vs BiLSTM.

Uso:
    python graficos/bootstrap_paired.py --horizon 48 --B 10000
"""

import os
import sys
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import (ROOT, DEVICE, load_split, read_tau_opt,
                          sigmoid, tss_point, sweep_tau)


# ------------------------------------------------- construcción de probabilidades
def model_ensemble(model, horizon, folds):
    """Devuelve (probs_test_ensemble, tau_ensemble, labels_test, dict_por_fold)."""
    test_probs, taus = {}, {}
    val_probs = {}
    labels_test = labels_val = {}
    labels_test_arr = None
    for k in folds:
        t = load_split(model, horizon, k, "test")
        v = load_split(model, horizon, k, "val")
        if t is None:
            continue
        lo, la = t
        test_probs[k] = sigmoid(lo)
        labels_test_arr = la
        if v is not None:
            lvo, lva = v
            val_probs[k] = sigmoid(lvo)
            labels_val[k] = lva
        taus[k] = read_tau_opt(model, horizon, k)
    avail = [k for k in folds if k in test_probs]
    ens = np.mean([test_probs[k] for k in avail], axis=0)
    tau_ens = float(np.mean([taus[k] for k in avail if taus[k] is not None]))
    return ens, tau_ens, labels_test_arr, test_probs, val_probs, labels_val, taus


def fusion_ensemble(horizon, folds, w_grid=None):
    """Fusión cross-modal en 5-fold. w_phys y tau se eligen en el VAL de cada fold."""
    if w_grid is None:
        w_grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    fused_test_per_fold, taus_f, wsel = [], [], []
    labels_test_arr = None
    per_fold_test_tss = []
    for k in folds:
        ts = load_split("swin3d", horizon, k, "test")
        tl = load_split("lstm",   horizon, k, "test")
        vs = load_split("swin3d", horizon, k, "val")
        vl = load_split("lstm",   horizon, k, "val")
        if ts is None or tl is None or vs is None or vl is None:
            continue
        p_swin_te, lab_te = sigmoid(ts[0]), ts[1]
        p_lstm_te         = sigmoid(tl[0])
        p_swin_va, lab_va = sigmoid(vs[0]), vs[1]
        p_lstm_va         = sigmoid(vl[0])
        labels_test_arr = lab_te

        # selección de (w, tau) sobre VAL (sin tocar test)
        best = (-2.0, 1.0, 0.5)  # val_tss, w, tau
        for w in w_grid:
            p_va = w * p_lstm_va + (1.0 - w) * p_swin_va
            tau, vtss = sweep_tau(p_va, lab_va)
            if vtss > best[0]:
                best = (vtss, float(w), tau)
        _, w_f, tau_f = best
        p_te = w_f * p_lstm_te + (1.0 - w_f) * p_swin_te
        fused_test_per_fold.append(p_te)
        taus_f.append(tau_f); wsel.append(w_f)
        per_fold_test_tss.append(tss_point(p_te, lab_te, tau_f))

    ens = np.mean(fused_test_per_fold, axis=0)
    tau_ens = float(np.mean(taus_f))
    return ens, tau_ens, labels_test_arr, np.array(per_fold_test_tss), wsel, taus_f


# ----------------------------------------------------- bootstrap pareado (GPU)
def paired_delta(probs_a, tau_a, probs_b, tau_b, labels, B=10000, chunk=2000, seed=42):
    """IC95% de Δ=TSS_a−TSS_b con remuestreo estratificado pareado (mismos índices)."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    y = torch.as_tensor(labels, dtype=torch.int64, device=DEVICE)
    pa = (torch.as_tensor(probs_a, dtype=torch.float64, device=DEVICE) >= tau_a).double()
    pb = (torch.as_tensor(probs_b, dtype=torch.float64, device=DEVICE) >= tau_b).double()
    pos = (y == 1); neg = (y == 0)
    pa_pos, pa_neg = pa[pos], pa[neg]
    pb_pos, pb_neg = pb[pos], pb[neg]
    n_pos, n_neg = int(pos.sum()), int(neg.sum())

    deltas, tss_a_s, tss_b_s = [], [], []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        ip   = torch.randint(0, n_pos, (b, n_pos), generator=g, device=DEVICE)
        ineg = torch.randint(0, n_neg, (b, n_neg), generator=g, device=DEVICE)
        tss_a = pa_pos[ip].sum(1) / n_pos - pa_neg[ineg].sum(1) / n_neg
        tss_b = pb_pos[ip].sum(1) / n_pos - pb_neg[ineg].sum(1) / n_neg
        deltas.append((tss_a - tss_b).cpu())
        tss_a_s.append(tss_a.cpu()); tss_b_s.append(tss_b.cpu())
        done += b
    d = torch.cat(deltas).numpy()
    return {
        "delta_mean": float(d.mean()),
        "ci": (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))),
        "p_a_gt_b": float((d > 0).mean()),
        "tss_a": float(torch.cat(tss_a_s).mean()),
        "tss_b": float(torch.cat(tss_b_s).mean()),
    }


def fmt(name_a, name_b, r):
    sig = "SIGNIFICATIVO" if (r["ci"][0] > 0 or r["ci"][1] < 0) else "NO significativo"
    return (f"\n  {name_a}  vs  {name_b}\n"
            f"    TSS_A={r['tss_a']:.4f}  TSS_B={r['tss_b']:.4f}\n"
            f"    Δ = {r['delta_mean']:+.4f}   IC95% = [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]\n"
            f"    P(A>B) = {r['p_a_gt_b']*100:.1f}%   →  {sig}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()
    h, folds, B = args.horizon, args.folds, args.B

    print(f"Device: {DEVICE}")
    lines = [f"Bootstrap PAREADO — horizonte {h}h  B={B}  device={DEVICE}"]

    swin_ens, swin_tau, lab, *_ = model_ensemble("swin3d", h, folds)
    lstm_ens, lstm_tau, lab2, *_ = model_ensemble("lstm", h, folds)
    fus_ens, fus_tau, lab3, fus_fold_tss, wsel, taus_f = fusion_ensemble(h, folds)

    hdr = (f"\n{'='*64}\n  COMPARACIÓN PAREADA — {h}h  (test n={lab.shape[0]}, "
           f"pos={int((lab==1).sum())})\n{'='*64}")
    print(hdr); lines.append(hdr)

    info = (f"  Swin3D  ensemble: TSS={tss_point(swin_ens,lab,swin_tau):.4f}  tau={swin_tau:.2f}\n"
            f"  BiLSTM  ensemble: TSS={tss_point(lstm_ens,lab,lstm_tau):.4f}  tau={lstm_tau:.2f}\n"
            f"  Fusión  ensemble: TSS={tss_point(fus_ens,lab,fus_tau):.4f}  tau={fus_tau:.2f}\n"
            f"  Fusión por fold (test): {np.array2string(fus_fold_tss, precision=4)}\n"
            f"  Fusión  CV: {fus_fold_tss.mean():.4f} ± {fus_fold_tss.std():.4f}\n"
            f"  w_phys elegido por fold (val): {wsel}\n"
            f"  tau elegido por fold (val):    {[round(t,2) for t in taus_f]}")
    print(info); lines.append(info)

    r1 = paired_delta(swin_ens, swin_tau, lstm_ens, lstm_tau, lab, B=B)
    r2 = paired_delta(fus_ens,  fus_tau,  swin_ens, swin_tau, lab, B=B)
    r3 = paired_delta(fus_ens,  fus_tau,  lstm_ens, lstm_tau, lab, B=B)
    for s in (fmt("Swin3D", "BiLSTM", r1),
              fmt("Fusión", "Swin3D", r2),
              fmt("Fusión", "BiLSTM", r3)):
        print(s); lines.append(s)

    out = os.path.join(ROOT, "graficos", f"bootstrap_paired_{h}h.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

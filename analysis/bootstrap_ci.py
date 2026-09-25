"""
Intervalos de confianza por bootstrap estratificado para TSS y AUC.

Trabaja sobre los logits/labels ya guardados en results/logits/ (no reentrena).
Pensado para el test FIJO (holdout, n=9384, ~4.4% positivos): el remuestreo es
ESTRATIFICADO por clase para mantener la prevalencia exacta en cada réplica.

Para cada modelo y horizonte produce:
  - Por fold:   TSS (con su tau_opt) y AUC, cada uno con IC95% bootstrap.
  - Ensemble:   promedio de probabilidades de los 5 folds. El tau del ensemble
                se elige barriendo sobre el VAL del ensemble (sin fuga), y se
                aplica al test del ensemble. TSS y AUC con IC95% bootstrap.

El bootstrap corre vectorizado en GPU (CUDA) por bloques.

Uso:
    python analysis/bootstrap_ci.py --model lstm --horizons 24 48 --B 10000
"""

import os
import re
import glob
import argparse
import numpy as np
import torch

ROOT        = os.environ.get("SFMM_ROOT",
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGITS_DIR  = os.environ.get("SFMM_LOGITS", os.path.join(ROOT, "results", "logits"))
METRICS_DIR = os.environ.get("SFMM_METRICS", os.path.join(ROOT, "outputs"))
OUTDIR      = os.environ.get("SFMM_OUT", os.path.join(ROOT, "results", "reports"))
FIGDIR      = os.environ.get("SFMM_FIG", os.path.join(ROOT, "results", "figures"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------- IO
def load_split(model, horizon, fold, split):
    path = os.path.join(LOGITS_DIR, f"{model}_{horizon}h_k{fold}_{split}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["logits"].astype(np.float64), d["labels"].astype(np.int64)


def read_tau_opt(model, horizon, fold):
    """Lee tau_opt del .txt oficial de ese fold.

    El nombre del archivo varía por modelo: el LSTM usa el patrón
    'metrics_{model}_cv_*' y Swin3D usa 'metrics_{model}_fits_*'. Se prueban
    ambos (y la copia local en outputs/) y se toma el primero que exista; si no hay
    ninguno, se recalcula desde los logits de validación.
    """
    candidates = [
        os.path.join(METRICS_DIR, f"metrics_{model}_cv_{horizon}h_k{fold}.txt"),
        os.path.join(METRICS_DIR, f"metrics_{model}_fits_{horizon}h_k{fold}.txt"),
        os.path.join(ROOT, "outputs", f"metrics_{model}_fits_{horizon}h_k{fold}.txt"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                m = re.match(r"\s*tau_opt:\s*([0-9.]+)", line)
                if m:
                    return float(m.group(1))
    # Sin el .txt de métricas (p. ej. en un clon del repo): mismo barrido de τ que el
    # entrenamiento sobre los logits de validación incluidos. Reproduce los 20 τ publicados.
    val = load_split(model, horizon, fold, "val")
    if val is None:
        return None
    return sweep_tau(sigmoid(val[0]), val[1])[0]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# --------------------------------------------------------------- métricas puntuales
def tss_point(probs, labels, tau):
    preds = (probs >= tau).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    pod  = tp / (tp + fn + 1e-12)
    pofd = fp / (fp + tn + 1e-12)
    return pod - pofd


def auc_point(probs, labels):
    """AUC exacta (Mann-Whitney U, maneja empates) en torch."""
    p = torch.as_tensor(probs, dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(labels, dtype=torch.int64, device=DEVICE)
    pos = p[y == 1]
    neg = p[y == 0].sort().values
    left  = torch.searchsorted(neg, pos, right=False).double()   # neg < pos
    right = torch.searchsorted(neg, pos, right=True).double()    # neg <= pos
    u = (left + 0.5 * (right - left)).sum()
    return (u / (pos.numel() * neg.numel())).item()


def sweep_tau(probs, labels, lo=0.01, hi=0.99, step=0.01):
    best_tss, best_tau = -2.0, 0.5
    for tau in np.arange(lo, hi + 1e-9, step):
        t = tss_point(probs, labels, tau)
        if t > best_tss:
            best_tss, best_tau = t, float(tau)
    return best_tau, best_tss


# ------------------------------------------------------ bootstrap estratificado GPU
def bootstrap_ci(probs, labels, tau, B=10000, chunk=2000, seed=42):
    """Devuelve dict con IC95% de TSS y AUC vía bootstrap estratificado por clase."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    p = torch.as_tensor(probs, dtype=torch.float64, device=DEVICE)
    y = torch.as_tensor(labels, dtype=torch.int64, device=DEVICE)

    pos_p = p[y == 1]
    neg_p = p[y == 0]
    n_pos, n_neg = pos_p.numel(), neg_p.numel()

    preds = (p >= tau).double()
    preds_pos = preds[y == 1]            # 1 -> TP en cada copia de un positivo
    preds_neg = preds[y == 0]            # 1 -> FP en cada copia de un negativo

    tss_samples, auc_samples = [], []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        # remuestreo estratificado (con reemplazo) dentro de cada clase
        ip = torch.randint(0, n_pos, (b, n_pos), generator=g, device=DEVICE)
        ineg = torch.randint(0, n_neg, (b, n_neg), generator=g, device=DEVICE)

        # ---- TSS (tau fijo) ----
        tp  = preds_pos[ip].sum(dim=1)
        fp  = preds_neg[ineg].sum(dim=1)
        pod  = tp / n_pos
        pofd = fp / n_neg
        tss_samples.append((pod - pofd).cpu())

        # ---- AUC (libre de umbral, exacta con empates) ----
        bp = pos_p[ip]                                  # [b, n_pos]
        bn = neg_p[ineg].sort(dim=1).values            # [b, n_neg] ordenado
        left  = torch.searchsorted(bn, bp, right=False).double()
        right = torch.searchsorted(bn, bp, right=True).double()
        u = (left + 0.5 * (right - left)).sum(dim=1)
        auc_samples.append((u / (n_pos * n_neg)).cpu())

        done += b

    tss_s = torch.cat(tss_samples).numpy()
    auc_s = torch.cat(auc_samples).numpy()
    return {
        "tss": (np.percentile(tss_s, 2.5), np.percentile(tss_s, 97.5), tss_s.std()),
        "auc": (np.percentile(auc_s, 2.5), np.percentile(auc_s, 97.5), auc_s.std()),
    }


# ---------------------------------------------------------------------------- main
def run_model(model, horizon, folds, B, lines):
    hdr = f"\n{'='*64}\n  {model.upper()} — Horizonte {horizon}h   (B={B}, bootstrap estratificado)\n{'='*64}"
    print(hdr); lines.append(hdr)

    test_probs, val_probs, taus = {}, {}, {}
    labels_test = labels_val = None

    for k in folds:
        t = load_split(model, horizon, k, "test")
        v = load_split(model, horizon, k, "val")
        if t is None:
            print(f"  k={k}: sin test .npz — omito"); continue
        lo, la = t
        test_probs[k] = sigmoid(lo)
        labels_test = la
        if v is not None:
            lvo, lva = v
            val_probs[k] = sigmoid(lvo)
            labels_val = lva
        taus[k] = read_tau_opt(model, horizon, k)

    if not test_probs:
        msg = "  No hay folds disponibles."; print(msg); lines.append(msg); return

    n = labels_test.shape[0]
    npos = int((labels_test == 1).sum()); nneg = n - npos
    info = f"  Test: n={n}  positivos={npos}  negativos={nneg}  prevalencia={npos/n:.4f}"
    print(info); lines.append(info)

    # -------- por fold --------
    head = f"\n  {'Fold':<6}{'tau':>6}{'TSS':>8}{'  IC95% TSS':>20}{'AUC':>9}{'  IC95% AUC':>20}"
    print(head); lines.append(head)
    tss_folds, auc_folds = [], []
    for k in folds:
        if k not in test_probs:
            continue
        tau = taus[k] if taus[k] is not None else 0.5
        probs = test_probs[k]
        tss = tss_point(probs, labels_test, tau)
        auc = auc_point(probs, labels_test)
        ci  = bootstrap_ci(probs, labels_test, tau, B=B)
        tss_folds.append(tss); auc_folds.append(auc)
        row = (f"  k={k:<4}{tau:>6.2f}{tss:>8.4f}"
               f"   [{ci['tss'][0]:.4f}, {ci['tss'][1]:.4f}]"
               f"{auc:>9.4f}   [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]")
        print(row); lines.append(row)

    # -------- resumen CV estándar: media ± σ entre folds (el titular) --------
    summ = (f"\n  CV (media ± σ entre folds):\n"
            f"    TSS = {np.mean(tss_folds):.4f} ± {np.std(tss_folds):.4f}\n"
            f"    AUC = {np.mean(auc_folds):.4f} ± {np.std(auc_folds):.4f}")
    print(summ); lines.append(summ)

    # -------- ensemble soft-vote (promedio de probabilidades del test, mismo n) --------
    # El test ES idéntico entre folds → promediar sus probabilidades es válido.
    # El umbral NO se puede barrer sobre val (cada fold valida ejemplos distintos),
    # así que se usa la media de los tau_opt por fold (cada uno sin fuga sobre su val).
    avail = [k for k in folds if k in test_probs]
    ens_test = np.mean([test_probs[k] for k in avail], axis=0)
    tau_ens = float(np.mean([taus[k] for k in avail if taus[k] is not None]))
    tau_src = "media de tau_opt por fold (sin fuga)"

    tss = tss_point(ens_test, labels_test, tau_ens)
    auc = auc_point(ens_test, labels_test)
    ci  = bootstrap_ci(ens_test, labels_test, tau_ens, B=B)
    blk = (f"\n  ENSEMBLE ({len(avail)} folds)   tau={tau_ens:.2f}  [{tau_src}]\n"
           f"    TSS = {tss:.4f}   IC95% = [{ci['tss'][0]:.4f}, {ci['tss'][1]:.4f}]   (σ_boot={ci['tss'][2]:.4f})\n"
           f"    AUC = {auc:.4f}   IC95% = [{ci['auc'][0]:.4f}, {ci['auc'][1]:.4f}]   (σ_boot={ci['auc'][2]:.4f})")
    print(blk); lines.append(blk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm")
    ap.add_argument("--horizons", nargs="+", type=int, default=[24, 48])
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    lines = [f"Bootstrap estratificado — modelo={args.model}  B={args.B}  device={DEVICE}"]
    for h in args.horizons:
        run_model(args.model, h, args.folds, args.B, lines)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"bootstrap_ci_{args.model}.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

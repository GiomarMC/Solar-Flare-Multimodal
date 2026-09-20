"""
Importancia de parámetros SHARP por permutación — BiLSTM 48 h (SIN reentrenar).

Sobre los checkpoints CV ya entrenados (outputs/checkpoints_lstm_cv_48h_k*), mide
cuánto cae el TSS de test al permutar cada uno de los 17 parámetros SHARP entre
muestras (rompiendo su asociación con la etiqueta, manteniendo su distribución
marginal y la estructura temporal del propio parámetro).

  importancia_j = TSS_base − TSS(permutando el parámetro j)

Se promedia sobre varias semillas de permutación y sobre los 5 folds (media ± σ).
Reusa el patrón de carga de save_logits_lstm.py. Solo inferencia.

Uso:
    python scripts/lstm_permutation_importance.py
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.train_lstm_standalone import LightningModule, SharpOnlyDataset, load_cfg
from scripts.save_logits_lstm import build_split_files, find_best_checkpoint, sweep_tau, compute_metrics
from scripts.dataset_temporal_v3 import SHARP_PARAMS

HORIZON, FOLDS, SEEDS = 48, [0, 1, 2, 3, 4], 5
OUT = os.path.join(ROOT, "graficos")
IMG = os.path.join(ROOT, "RedaccionIEEE", "images")

# Los 10 parámetros del conjunto base (Grim); el resto se completó desde JSOC.
BASE_10 = {"USFLUX", "MEANGBZ", "MEANGBT", "MEANPOT", "SHRGT45",
           "TOTPOT", "SAVNCPP", "ABSNJZH", "AREA_ACR", "NACR"}
C_BASE, C_ADD = "#9aa7b5", "#d62728"   # base (gris-azul) / completado JSOC (rojo)


@torch.no_grad()
def collect_X(loader):
    Xs, ys = [], []
    for _, tab, lab in loader:
        Xs.append(tab.numpy()); ys.append(lab.numpy())
    return np.concatenate(Xs), np.concatenate(ys)   # X[N,T,17], y[N]


@torch.no_grad()
def model_logits(model, X, device, bs=1024):
    outs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i + bs]).float().to(device)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs)


def tss_at(logits, y, tau):
    return compute_metrics(logits, y, tau)['TSS']


def run_fold(base_dir, fold, device):
    cfg = load_cfg(f"configs/lstm_cv_{HORIZON}h_k{fold}.yaml")
    ckpt = find_best_checkpoint(f"outputs/checkpoints_lstm_cv_{HORIZON}h_k{fold}")
    seq_rel = f"Seq_Magnetogram/M{HORIZON}/Seqs16"
    para_path = os.path.join(base_dir, "para_flare_21params.txt")
    train_files, val_files, test_files = build_split_files(base_dir, HORIZON, fold)

    train_ds = SharpOnlyDataset(train_files, base_dir, seq_rel, para_path)
    med = train_ds.sharp_medians
    val_ds = SharpOnlyDataset(val_files, base_dir, seq_rel, para_path, sharp_medians=med)
    test_ds = SharpOnlyDataset(test_files, base_dir, seq_rel, para_path, sharp_medians=med)

    model = LightningModule.load_from_checkpoint(
        ckpt, cfg=cfg, sharp_medians=med, strict=False).model.to(device).eval()

    Xv, yv = collect_X(DataLoader(val_ds, batch_size=256, num_workers=4))
    tau_opt, _ = sweep_tau(model_logits(model, Xv, device), yv)
    Xt, yt = collect_X(DataLoader(test_ds, batch_size=256, num_workers=4))
    base_tss = tss_at(model_logits(model, Xt, device), yt, tau_opt)
    print(f"  k={fold}: base TSS={base_tss:.4f}  τ={tau_opt:.2f}  (ckpt {os.path.basename(ckpt)})")

    imp = np.zeros(len(SHARP_PARAMS))
    for j in range(len(SHARP_PARAMS)):
        drops = []
        for s in range(SEEDS):
            rng = np.random.default_rng(1000 * fold + 10 * j + s)
            perm = rng.permutation(len(Xt))
            Xp = Xt.copy()
            Xp[:, :, j] = Xt[perm, :, j]
            drops.append(base_tss - tss_at(model_logits(model, Xp, device), yt, tau_opt))
        imp[j] = float(np.mean(drops))
    return base_tss, imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=None)
    args = ap.parse_args()
    base_dir = args.base_dir or load_cfg("configs/lstm_cv_48h_k0.yaml")["data"]["base_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\nBase: {base_dir}\n")

    mat = []          # [folds, 17]
    for k in FOLDS:
        _, imp = run_fold(base_dir, k, device)
        mat.append(imp)
    mat = np.array(mat)
    mean, sd = mat.mean(0), mat.std(0)
    order = np.argsort(mean)[::-1]

    lines = ["Importancia de parámetros SHARP por permutación — BiLSTM 48 h",
             f"(caída de TSS al permutar; media ± σ sobre {len(FOLDS)} folds × {SEEDS} semillas)\n",
             f"  {'Parámetro':<18}{'ΔTSS (imp.)':>14}{'σ':>9}"]
    for j in order:
        lines.append(f"  {SHARP_PARAMS[j]:<18}{mean[j]:>14.4f}{sd[j]:>9.4f}")
    txt = os.path.join(OUT, "sharp_importance_48h.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---- figura: barras horizontales ordenadas, coloreadas por origen ----
    fig, ax = plt.subplots(figsize=(7, 6))
    yy = np.arange(len(order))[::-1]
    names = [SHARP_PARAMS[j] for j in order]
    colors = [C_BASE if SHARP_PARAMS[j] in BASE_10 else C_ADD for j in order]
    ax.barh(yy, mean[order], xerr=sd[order], color=colors, alpha=0.9, capsize=3,
            error_kw=dict(ecolor="#555", lw=1))
    ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=9)
    for tick, j in zip(ax.get_yticklabels(), order):       # rótulo en color del origen
        tick.set_color(C_ADD if SHARP_PARAMS[j] not in BASE_10 else "#333")
    ax.set_xlabel("Importancia = caída de TSS al permutar (media ± σ)")
    ax.set_title("Importancia de parámetros SHARP — BiLSTM 48 h")
    ax.axvline(0, color="k", lw=0.8); ax.grid(axis="x", alpha=0.3)
    ax.legend(handles=[Patch(facecolor=C_ADD, label="Completado desde JSOC"),
                       Patch(facecolor=C_BASE, label="Conjunto base (Grim)")],
              loc="lower right", fontsize=8.5, frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig7_sharp_importance_48h.pdf"))
    fig.savefig(os.path.join(OUT, "fig7_sharp_importance_48h.png"), dpi=150)
    plt.close(fig)
    print(f"\nGuardado: {txt}")
    print("Figura:   RedaccionIEEE/images/fig7_sharp_importance_48h.pdf  (+ graficos/.png)")


if __name__ == "__main__":
    main()

"""
Evaluación de Swin3D-T FITS y guardado de logits para curvas ROC.

Busca automáticamente el mejor checkpoint en el directorio del fold.
Guarda logits + labels en .npz para uso posterior (ROC, AUC, etc.).

Usage:
    python scripts/save_logits_swin3d.py \
        --config configs/swin3d_standalone_fits_24h.yaml --fold 3

    python scripts/save_logits_swin3d.py \
        --config configs/swin3d_standalone_fits_48h.yaml --fold 0
"""

import os
import sys
import glob
import argparse
import numpy as np
import yaml
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_swin3d_standalone_fits import LightningModule, build_datasets


def find_best_checkpoint(ckpt_dir: str) -> str:
    pattern = os.path.join(ckpt_dir, "*.ckpt")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints en {ckpt_dir}")
    # El nombre tiene val_tss=X.XXXX — elige el mayor
    def extract_tss(path):
        base = os.path.basename(path)
        try:
            return float(base.split("val_tss=")[1].replace(".ckpt", ""))
        except (IndexError, ValueError):
            return -1.0
    return max(candidates, key=extract_tss)


def compute_metrics(logits, labels, tau):
    preds = (1 / (1 + np.exp(-logits)) >= tau).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    pod  = tp / (tp + fn + 1e-8)
    pofd = fp / (fp + tn + 1e-8)
    tss  = pod - pofd
    far  = fp / (tp + fp + 1e-8)
    pre  = tp / (tp + fp + 1e-8)
    f1   = 2 * pre * pod / (pre + pod + 1e-8)
    hss  = 2*(tp*tn - fp*fn) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) + 1e-8)
    return dict(TSS=tss, HSS=hss, POD=pod, FAR=far, F1=f1,
                TP=tp, FN=fn, FP=fp, TN=tn, tau=tau)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for vid, _, labels in loader:
        vid = vid.permute(0, 2, 1, 3, 4).to(device)
        logits = model(vid).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def sweep_tau(logits, labels, cfg):
    ct = cfg['threshold']
    taus = np.arange(ct['sweep_start'], ct['sweep_end'] + 1e-9, ct['sweep_step'])
    best_tss, best_tau = -1.0, 0.5
    for tau in taus:
        m = compute_metrics(logits, labels, tau)
        if m['TSS'] > best_tss:
            best_tss, best_tau = m['TSS'], tau
    return best_tau, best_tss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold',   type=int, required=True, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--checkpoint', default=None,
                        help='Checkpoint explícito (opcional; si no se da, busca el mejor automáticamente)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    h    = cfg['data']['horizon']
    fold = args.fold

    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt_dir = f"outputs/checkpoints_swin3d_fits_{h}h_k{fold}"
        ckpt = find_best_checkpoint(ckpt_dir)
    print(f"Checkpoint: {ckpt}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | horizonte: {h}h | fold: k={fold}")

    _, val_ds, test_ds = build_datasets(cfg, fold)

    bs = cfg['training']['batch_size']
    nw = cfg['training']['num_workers']
    val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    module = LightningModule.load_from_checkpoint(ckpt, cfg=cfg, strict=False)
    model  = module.model.to(device)

    print("Recolectando logits val...")
    val_logits, val_labels = collect_logits(model, val_loader, device)
    tau_opt, val_tss = sweep_tau(val_logits, val_labels, cfg)
    print(f"τ_opt = {tau_opt:.2f}  val_TSS = {val_tss:.4f}")

    print("Recolectando logits test...")
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_m = compute_metrics(test_logits, test_labels, tau_opt)
    print(f"test_TSS = {test_m['TSS']:.4f}  (τ={tau_opt:.2f})")

    # ── Guardar logits + labels ────────────────────────────────────────────────
    os.makedirs("outputs/logits", exist_ok=True)
    for split, logits, labels in [
        ("val",  val_logits,  val_labels),
        ("test", test_logits, test_labels),
    ]:
        out = f"outputs/logits/swin3d_{h}h_k{fold}_{split}.npz"
        np.savez_compressed(out, logits=logits, labels=labels)
        print(f"Logits guardados: {out}")

    # ── Métricas en texto ──────────────────────────────────────────────────────
    metrics_path = f"outputs/metrics_swin3d_fits_{h}h_k{fold}.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"checkpoint: {ckpt}\n")
        f.write(f"horizon: {h}h | fold: k={fold}\n")
        f.write(f"tau_opt: {tau_opt:.2f}\n\n")
        for split_name, m in [
            ("Val",  compute_metrics(val_logits, val_labels, tau_opt)),
            ("Test", test_m),
        ]:
            f.write(f"[{split_name}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"Métricas: {metrics_path}")


if __name__ == '__main__':
    main()

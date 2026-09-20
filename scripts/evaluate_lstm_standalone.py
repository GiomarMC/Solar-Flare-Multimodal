"""
Evaluación del modelo LSTM standalone (solo parámetros SHARP).

Usage:
    python scripts/evaluate_lstm_standalone.py \
        --config configs/lstm_standalone_k3.yaml \
        --checkpoint outputs/checkpoints_lstm_standalone_k3/<best>.ckpt
"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.train_lstm_standalone import LightningModule, build_datasets


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
    hss  = 2 * (tp*tn - fp*fn) / ((tp+fn)*(fn+tn) + (tp+fp)*(fp+tn) + 1e-8)
    return dict(TSS=tss, HSS=hss, POD=pod, FAR=far, F1=f1,
                TP=tp, FN=fn, FP=fp, TN=tn, tau=tau)


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for _, tab, labels in loader:
        tab = tab.to(device)
        logits = model(tab).cpu().numpy()
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


def print_metrics(title, m):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    print(f"  tau  = {m['tau']:.2f}")
    print(f"  TSS  = {m['TSS']:.4f}")
    print(f"  HSS  = {m['HSS']:.4f}")
    print(f"  POD  = {m['POD']:.4f}")
    print(f"  FAR  = {m['FAR']:.4f}")
    print(f"  F1   = {m['F1']:.4f}")
    print(f"  TP={m['TP']}  FN={m['FN']}  FP={m['FP']}  TN={m['TN']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     default='configs/lstm_standalone_k3.yaml')
    parser.add_argument('--checkpoint', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    _, val_ds, test_ds, _ = build_datasets(cfg)

    bs = cfg['training']['batch_size']
    nw = cfg['training']['num_workers']
    val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    module = LightningModule.load_from_checkpoint(
        args.checkpoint, cfg=cfg, sharp_medians=val_ds.sharp_medians,
    )
    model = module.model.to(device)

    print("Recolectando logits de validación...")
    val_logits, val_labels = collect_logits(model, val_loader, device)
    tau_opt, val_tss = sweep_tau(val_logits, val_labels, cfg)
    print(f"\nτ_opt = {tau_opt:.2f}  (val_TSS = {val_tss:.4f})")
    print_metrics("Validación — τ_opt aplicado",
                  compute_metrics(val_logits, val_labels, tau_opt))

    print("\nRecolectando logits de test...")
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_m = compute_metrics(test_logits, test_labels, tau_opt)
    print_metrics("Test — τ_opt de val (no del test)", test_m)

    ckpt_dir_name = os.path.basename(cfg['output']['checkpoint_dir'].rstrip('/'))
    metrics_name  = ckpt_dir_name.replace('checkpoints_', 'metrics_') + '.txt'
    out_path = os.path.join(cfg['output']['dir'], metrics_name)
    os.makedirs(cfg['output']['dir'], exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"tau_opt: {tau_opt:.2f}\n\n")
        for split, m in [('Val', compute_metrics(val_logits, val_labels, tau_opt)),
                         ('Test', test_m)]:
            f.write(f"[{split}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")
    print(f"\nMétricas guardadas en {out_path}")


if __name__ == '__main__':
    main()

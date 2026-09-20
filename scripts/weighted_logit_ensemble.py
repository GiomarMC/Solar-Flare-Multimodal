"""
Ensemble de promedio ponderado de logits.

  logit_final = w · logit_phys + (1-w) · logit_vis

Barre w en [0, 1] sobre val (fold k=3), aplica el w óptimo al test.

Usage:
    python scripts/weighted_logit_ensemble.py \
        --lstm_config configs/lstm_standalone_k3.yaml \
        --lstm_ckpt outputs/checkpoints_lstm_standalone_k3/sfmm_lstm_standalone-epoch=011-val_tss=0.8254.ckpt \
        --vis_config configs/swin3d_standalone_fits_24h.yaml \
        --vis_ckpt outputs/checkpoints_swin3d_fits_24h_k3/sfmm_swin3d_fits_24h_k3-epoch=002-val_tss=0.7343.ckpt \
        --fold 3
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
import scripts.train_lstm_standalone as lstm_mod
import scripts.train_swin3d_standalone_fits as swin_mod


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def compute_metrics(probs, labels, tau):
    preds = (probs >= tau).astype(int)
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


def best_tau(probs, labels, n=199):
    taus = np.linspace(0.005, 0.995, n)
    best_tss, best_t = -1.0, 0.5
    for t in taus:
        m = compute_metrics(probs, labels, t)
        if m['TSS'] > best_tss:
            best_tss, best_t = m['TSS'], t
    return best_t, best_tss


@torch.no_grad()
def collect_lstm(model, loader, device):
    model.eval()
    logits, labels = [], []
    for _, tab, lbl in loader:
        logits.append(model(tab.to(device)).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(logits), np.concatenate(labels)


@torch.no_grad()
def collect_swin(model, loader, device):
    model.eval()
    logits, labels = [], []
    for vid, _, lbl in loader:
        logits.append(model(vid.permute(0, 2, 1, 3, 4).to(device)).cpu().numpy())
        labels.append(lbl.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_config', required=True)
    parser.add_argument('--lstm_ckpt',   required=True)
    parser.add_argument('--vis_config',  required=True)
    parser.add_argument('--vis_ckpt',    required=True)
    parser.add_argument('--fold',        type=int, default=3, choices=[0,1,2,3,4])
    parser.add_argument('--w_steps',     type=int, default=21,
                        help='Número de valores de w a probar en [0,1]')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | fold: k={args.fold}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    print("\n[1/4] Construyendo datasets...")
    with open(args.lstm_config) as f:
        lstm_cfg = yaml.safe_load(f)
    with open(args.vis_config) as f:
        vis_cfg = yaml.safe_load(f)

    _, lstm_val, lstm_test, sharp_medians = lstm_mod.build_datasets(lstm_cfg)
    _, swin_val, swin_test                = swin_mod.build_datasets(vis_cfg, args.fold)

    assert len(lstm_val)  == len(swin_val),  "Val size mismatch"
    assert len(lstm_test) == len(swin_test), "Test size mismatch"
    print(f"  Val: {len(lstm_val)} | Test: {len(lstm_test)}")

    lstm_bs = lstm_cfg['training']['batch_size']
    lstm_nw = lstm_cfg['training']['num_workers']
    vis_bs  = vis_cfg['training']['batch_size']
    vis_nw  = vis_cfg['training']['num_workers']

    lstm_val_loader  = DataLoader(lstm_val,  batch_size=lstm_bs, shuffle=False, num_workers=lstm_nw)
    lstm_test_loader = DataLoader(lstm_test, batch_size=lstm_bs, shuffle=False, num_workers=lstm_nw)
    swin_val_loader  = DataLoader(swin_val,  batch_size=vis_bs,  shuffle=False, num_workers=vis_nw)
    swin_test_loader = DataLoader(swin_test, batch_size=vis_bs,  shuffle=False, num_workers=vis_nw)

    # ── Modelos ───────────────────────────────────────────────────────────────
    print("\n[2/4] Cargando modelos...")
    lstm_module = lstm_mod.LightningModule.load_from_checkpoint(
        args.lstm_ckpt, cfg=lstm_cfg, sharp_medians=sharp_medians)
    lstm_model = lstm_module.model.to(device)
    for p in lstm_model.parameters():
        p.requires_grad_(False)

    swin_module = swin_mod.LightningModule.load_from_checkpoint(
        args.vis_ckpt, cfg=vis_cfg, strict=False)
    swin_model = swin_module.model.to(device)
    for p in swin_model.parameters():
        p.requires_grad_(False)

    # ── Logits ────────────────────────────────────────────────────────────────
    print("\n[3/4] Recolectando logits...")
    phys_val_logits,  val_labels  = collect_lstm(lstm_model, lstm_val_loader,  device)
    phys_test_logits, test_labels = collect_lstm(lstm_model, lstm_test_loader, device)
    vis_val_logits,   _           = collect_swin(swin_model, swin_val_loader,  device)
    vis_test_logits,  _           = collect_swin(swin_model, swin_test_loader, device)

    # ── Barrido de w ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Barriendo w en {args.w_steps} pasos sobre [0, 1]...")
    ws = np.linspace(0.0, 1.0, args.w_steps)

    results = []
    for w in ws:
        comb_val  = w * phys_val_logits  + (1 - w) * vis_val_logits
        tau_v, val_tss = best_tau(sigmoid(comb_val), val_labels)

        comb_test = w * phys_test_logits + (1 - w) * vis_test_logits
        test_m    = compute_metrics(sigmoid(comb_test), test_labels, tau_v)
        results.append((w, tau_v, val_tss, test_m))

    # Ordenar por val_TSS
    results.sort(key=lambda r: r[2], reverse=True)

    print(f"\n{'w_phys':>7} {'τ':>6} {'val_TSS':>8} {'test_TSS':>9} {'TP':>5} {'FP':>6}")
    print("-" * 50)
    for w, tau_v, val_tss, m in results:
        marker = " ←" if w == results[0][0] else ""
        print(f"  {w:.2f}   {tau_v:.3f}   {val_tss:.4f}   {m['TSS']:.4f}   "
              f"{m['TP']:>3}  {m['FP']:>5}{marker}")

    best_w, best_tau_v, best_val_tss, best_test_m = results[0]

    print(f"\n── Mejor combinación ────────────────────────────────────────")
    print(f"  w_phys = {best_w:.2f}  (BiLSTM) | w_vis = {1-best_w:.2f}  (Swin3D)")
    print(f"  τ_opt  = {best_tau_v:.3f}")
    print(f"  val TSS  = {best_val_tss:.4f}")
    print(f"  test TSS = {best_test_m['TSS']:.4f}")
    print(f"  TP={best_test_m['TP']}  FN={best_test_m['FN']}  "
          f"FP={best_test_m['FP']}  TN={best_test_m['TN']}")

    print(f"\n── Comparativa final ────────────────────────────────────────")
    phys_tau_v, phys_val_tss = best_tau(sigmoid(phys_val_logits), val_labels)
    vis_tau_v,  vis_val_tss  = best_tau(sigmoid(vis_val_logits),  val_labels)
    phys_test_m = compute_metrics(sigmoid(phys_test_logits), test_labels, phys_tau_v)
    vis_test_m  = compute_metrics(sigmoid(vis_test_logits),  test_labels, vis_tau_v)

    rows = [
        ("BiLSTM  (w=1.0)", phys_tau_v, phys_val_tss, phys_test_m),
        ("Swin3D  (w=0.0)", vis_tau_v,  vis_val_tss,  vis_test_m),
        (f"Ensemble(w={best_w:.2f})", best_tau_v, best_val_tss, best_test_m),
    ]
    print(f"\n  {'Modelo':<22} {'τ':>6} {'val_TSS':>8} {'test_TSS':>9}")
    print(f"  {'-'*50}")
    for name, tau_v, val_tss, m in rows:
        print(f"  {name:<22} {tau_v:>6.3f} {val_tss:>8.4f} {m['TSS']:>9.4f}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    h = vis_cfg['data']['horizon']
    out_path = f"outputs/metrics_weighted_ensemble_{h}h_k{args.fold}.txt"
    os.makedirs('outputs', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"lstm_ckpt: {args.lstm_ckpt}\n")
        f.write(f"vis_ckpt:  {args.vis_ckpt}\n")
        f.write(f"horizon: {h}h | fold: k={args.fold}\n\n")
        f.write(f"── Barrido completo ─────────────────────────\n")
        f.write(f"{'w_phys':>7} {'tau':>6} {'val_TSS':>9} {'test_TSS':>10} {'TP':>5} {'FP':>6}\n")
        for w, tau_v, val_tss, m in sorted(results, key=lambda r: r[0]):
            f.write(f"  {w:.2f}   {tau_v:.3f}   {val_tss:.4f}   {m['TSS']:.4f}   "
                    f"{m['TP']:>3}  {m['FP']:>5}\n")
        f.write(f"\n── Mejor combinación ─────────────────────────\n")
        f.write(f"w_phys={best_w:.2f}  tau={best_tau_v:.3f}\n")
        f.write(f"val_TSS={best_val_tss:.4f}  test_TSS={best_test_m['TSS']:.4f}\n")
        for k, v in best_test_m.items():
            f.write(f"  {k}: {v}\n")
    print(f"\nGuardado en {out_path}")


if __name__ == '__main__':
    main()

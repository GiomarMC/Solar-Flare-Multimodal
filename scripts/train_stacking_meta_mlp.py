"""
Stacking meta-MLP: combina predicciones de LSTM standalone y Swin3D FITS standalone.

Modelos base (congelados):
  1. LSTM standalone — SHARP 17 params, BiLSTM → P_phys
  2. Swin3D-T FITS  — magnetogramas float32, 24h → P_vis

Meta-MLP: Linear(2→8) → ReLU → Linear(8→1)  (~25 params)

Procedimiento:
  1. Inferencia de ambos modelos sobre val (fold k=3) → [P_vis, P_phys]
  2. Entrenar meta-MLP sobre val con etiquetas reales
  3. Evaluar sobre test con tau_opt derivado de val

Usage:
    python scripts/train_stacking_meta_mlp.py \
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
import torch.nn as nn
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import scripts.train_lstm_standalone as lstm_mod
import scripts.train_swin3d_standalone_fits as swin_mod


class MetaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


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
    hss  = 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + 1e-8)
    return dict(TSS=tss, HSS=hss, POD=pod, FAR=far, F1=f1,
                TP=tp, FN=fn, FP=fp, TN=tn, tau=tau)


def sweep_tau(probs, labels, n=99):
    taus = np.linspace(0.01, 0.99, n)
    best_tss, best_tau = -1.0, 0.5
    for tau in taus:
        m = compute_metrics(probs, labels, tau)
        if m['TSS'] > best_tss:
            best_tss, best_tau = m['TSS'], tau
    return best_tau, best_tss


@torch.no_grad()
def collect_lstm_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for _, tab, labels in loader:
        logits_all.append(model(tab.to(device)).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


@torch.no_grad()
def collect_swin_logits(model, loader, device):
    model.eval()
    logits_all, labels_all = [], []
    for vid, _, labels in loader:
        # FITSTemporalDataset → [B,T,3,H,W]; VideoSwin espera [B,3,T,H,W]
        logits_all.append(model(vid.permute(0, 2, 1, 3, 4).to(device)).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def train_meta_mlp(vis_val_feats, phys_val_feats, val_labels, epochs, lr, weight_decay):
    n_neg = int((val_labels == 0).sum())
    n_pos = int((val_labels == 1).sum())
    print(f"  Val: {n_pos} flares / {n_neg} no-flares")

    X = torch.from_numpy(
        np.stack([vis_val_feats, phys_val_feats], axis=1).astype(np.float32)
    )
    y = torch.from_numpy(val_labels.astype(np.float32))

    meta = MetaMLP()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(meta.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    loader = DataLoader(TensorDataset(X, y), batch_size=512, shuffle=True)

    meta.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(meta(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
        scheduler.step()
        if epoch % 100 == 0 or epoch == epochs:
            print(f"  [meta-MLP epoch {epoch:03d}/{epochs}] loss={total_loss/len(X):.4f}")

    return meta


def print_metrics(title, m):
    print(f"\n{'='*52}")
    print(f"  {title}")
    print(f"{'='*52}")
    print(f"  tau  = {m['tau']:.3f}")
    print(f"  TSS  = {m['TSS']:.4f}  HSS  = {m['HSS']:.4f}")
    print(f"  POD  = {m['POD']:.4f}  FAR  = {m['FAR']:.4f}  F1 = {m['F1']:.4f}")
    print(f"  TP={m['TP']}  FN={m['FN']}  FP={m['FP']}  TN={m['TN']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lstm_config',   required=True)
    parser.add_argument('--lstm_ckpt',     required=True)
    parser.add_argument('--vis_config',    required=True)
    parser.add_argument('--vis_ckpt',      required=True)
    parser.add_argument('--fold',          type=int, default=3, choices=[0, 1, 2, 3, 4])
    parser.add_argument('--meta_epochs',   type=int, default=300)
    parser.add_argument('--meta_lr',       type=float, default=1e-3)
    parser.add_argument('--meta_wd',       type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | fold: k={args.fold}")

    # ── 1. Build datasets ────────────────────────────────────────────────────
    print("\n[1/5] Construyendo datasets...")
    with open(args.lstm_config) as f:
        lstm_cfg = yaml.safe_load(f)
    with open(args.vis_config) as f:
        vis_cfg = yaml.safe_load(f)

    _, lstm_val_ds, lstm_test_ds, sharp_medians = lstm_mod.build_datasets(lstm_cfg)
    _, swin_val_ds, swin_test_ds = swin_mod.build_datasets(vis_cfg, args.fold)

    assert len(lstm_val_ds) == len(swin_val_ds), \
        f"Val size mismatch: LSTM={len(lstm_val_ds)} vs Swin={len(swin_val_ds)}"
    assert len(lstm_test_ds) == len(swin_test_ds), \
        f"Test size mismatch: LSTM={len(lstm_test_ds)} vs Swin={len(swin_test_ds)}"
    print(f"  Val: {len(lstm_val_ds)} | Test: {len(lstm_test_ds)}")

    lstm_bs = lstm_cfg['training']['batch_size']
    lstm_nw = lstm_cfg['training']['num_workers']
    vis_bs  = vis_cfg['training']['batch_size']
    vis_nw  = vis_cfg['training']['num_workers']

    lstm_val_loader  = DataLoader(lstm_val_ds,  batch_size=lstm_bs, shuffle=False, num_workers=lstm_nw)
    lstm_test_loader = DataLoader(lstm_test_ds, batch_size=lstm_bs, shuffle=False, num_workers=lstm_nw)
    swin_val_loader  = DataLoader(swin_val_ds,  batch_size=vis_bs,  shuffle=False, num_workers=vis_nw)
    swin_test_loader = DataLoader(swin_test_ds, batch_size=vis_bs,  shuffle=False, num_workers=vis_nw)

    # ── 2. Load frozen models ────────────────────────────────────────────────
    print("\n[2/5] Cargando modelos congelados...")
    lstm_module = lstm_mod.LightningModule.load_from_checkpoint(
        args.lstm_ckpt, cfg=lstm_cfg, sharp_medians=sharp_medians,
    )
    lstm_model = lstm_module.model.to(device)
    for p in lstm_model.parameters():
        p.requires_grad_(False)

    swin_module = swin_mod.LightningModule.load_from_checkpoint(
        args.vis_ckpt, cfg=vis_cfg, strict=False,
    )
    swin_model = swin_module.model.to(device)
    for p in swin_model.parameters():
        p.requires_grad_(False)

    n_lstm = sum(p.numel() for p in lstm_model.parameters())
    n_swin = sum(p.numel() for p in swin_model.parameters())
    n_meta = sum(p.numel() for p in MetaMLP().parameters())
    print(f"  LSTM params: {n_lstm:,}  |  Swin3D params: {n_swin:,}  |  Meta-MLP params: {n_meta}")

    # ── 3. Collect base model predictions ───────────────────────────────────
    print("\n[3/5] Recolectando predicciones de modelos base...")
    print("  LSTM  → val...")
    phys_val_logits, phys_val_labels = collect_lstm_logits(lstm_model, lstm_val_loader, device)
    print("  LSTM  → test...")
    phys_test_logits, phys_test_labels = collect_lstm_logits(lstm_model, lstm_test_loader, device)
    print("  Swin3D→ val...")
    vis_val_logits, vis_val_labels = collect_swin_logits(swin_model, swin_val_loader, device)
    print("  Swin3D→ test...")
    vis_test_logits, vis_test_labels = collect_swin_logits(swin_model, swin_test_loader, device)

    assert np.array_equal(phys_val_labels, vis_val_labels),  "Val labels desalineadas entre LSTM y Swin!"
    assert np.array_equal(phys_test_labels, vis_test_labels), "Test labels desalineadas entre LSTM y Swin!"
    val_labels  = phys_val_labels
    test_labels = phys_test_labels

    phys_val_probs  = sigmoid(phys_val_logits)
    phys_test_probs = sigmoid(phys_test_logits)
    vis_val_probs   = sigmoid(vis_val_logits)
    vis_test_probs  = sigmoid(vis_test_logits)

    # Base model individual metrics (for comparison table)
    phys_tau, phys_val_tss = sweep_tau(phys_val_probs, val_labels)
    vis_tau,  vis_val_tss  = sweep_tau(vis_val_probs,  val_labels)
    phys_test_m = compute_metrics(phys_test_probs, test_labels, phys_tau)
    vis_test_m  = compute_metrics(vis_test_probs,  test_labels, vis_tau)

    # Normalizar logits con estadísticas del val (escala y centrado)
    # Preserva la información discriminativa que sigmoid aplasta
    vis_mean,  vis_std  = vis_val_logits.mean(),  vis_val_logits.std()  + 1e-8
    phys_mean, phys_std = phys_val_logits.mean(), phys_val_logits.std() + 1e-8
    vis_val_norm   = (vis_val_logits   - vis_mean)  / vis_std
    vis_test_norm  = (vis_test_logits  - vis_mean)  / vis_std
    phys_val_norm  = (phys_val_logits  - phys_mean) / phys_std
    phys_test_norm = (phys_test_logits - phys_mean) / phys_std
    print(f"  Logits vis  — val: μ={vis_mean:.3f}  σ={vis_std:.3f}")
    print(f"  Logits phys — val: μ={phys_mean:.3f}  σ={phys_std:.3f}")

    # ── 4. Train meta-MLP ────────────────────────────────────────────────────
    print(f"\n[4/5] Entrenando meta-MLP ({args.meta_epochs} epochs, lr={args.meta_lr}, wd={args.meta_wd})...")
    meta = train_meta_mlp(
        vis_val_norm, phys_val_norm, val_labels,
        epochs=args.meta_epochs, lr=args.meta_lr, weight_decay=args.meta_wd,
    )

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    print("\n[5/5] Evaluando...")
    meta.eval()
    with torch.no_grad():
        X_val  = torch.from_numpy(np.stack([vis_val_norm,  phys_val_norm],  axis=1).astype(np.float32))
        X_test = torch.from_numpy(np.stack([vis_test_norm, phys_test_norm], axis=1).astype(np.float32))
        val_meta_probs  = sigmoid(meta(X_val).numpy())
        test_meta_probs = sigmoid(meta(X_test).numpy())

    tau_opt, meta_val_tss = sweep_tau(val_meta_probs, val_labels)
    val_m  = compute_metrics(val_meta_probs,  val_labels,  tau_opt)
    test_m = compute_metrics(test_meta_probs, test_labels, tau_opt)

    print(f"\nτ_opt = {tau_opt:.3f}")
    print_metrics("Validación — meta-MLP", val_m)
    print_metrics("Test — meta-MLP (τ de val)", test_m)

    print("\n\n── Comparativa modelos base vs meta-MLP ───────────────────────────────")
    print(f"  {'Modelo':<30} {'τ':>6} {'val_TSS':>8} {'test_TSS':>9} {'test_TSS':>9}")
    print(f"  {'-'*62}")
    print(f"  {'LSTM standalone':<30} {phys_tau:>6.3f} {phys_val_tss:>8.4f} {phys_test_m['TSS']:>9.4f}")
    print(f"  {'Swin3D FITS standalone':<30} {vis_tau:>6.3f} {vis_val_tss:>8.4f} {vis_test_m['TSS']:>9.4f}")
    print(f"  {'Meta-MLP stacking':<30} {tau_opt:>6.3f} {meta_val_tss:>8.4f} {test_m['TSS']:>9.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    h = vis_cfg['data']['horizon']
    out_path = f"outputs/metrics_stacking_meta_mlp_{h}h_k{args.fold}.txt"
    os.makedirs('outputs', exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f"lstm_ckpt: {args.lstm_ckpt}\n")
        f.write(f"vis_ckpt: {args.vis_ckpt}\n")
        f.write(f"horizon: {h}h | fold: k={args.fold}\n")
        f.write(f"meta_epochs: {args.meta_epochs} | meta_lr: {args.meta_lr} | meta_wd: {args.meta_wd}\n")
        f.write(f"input: normalized logits (z-score per model, val stats)\n")
        f.write(f"tau_opt: {tau_opt:.4f}\n\n")

        f.write("── Base models ─────────────────────────────────────────────\n")
        f.write(f"LSTM standalone         τ={phys_tau:.3f}  val_TSS={phys_val_tss:.4f}  test_TSS={phys_test_m['TSS']:.4f}\n")
        f.write(f"  Test: TP={phys_test_m['TP']} FN={phys_test_m['FN']} FP={phys_test_m['FP']} TN={phys_test_m['TN']}\n")
        f.write(f"Swin3D FITS standalone  τ={vis_tau:.3f}  val_TSS={vis_val_tss:.4f}  test_TSS={vis_test_m['TSS']:.4f}\n")
        f.write(f"  Test: TP={vis_test_m['TP']} FN={vis_test_m['FN']} FP={vis_test_m['FP']} TN={vis_test_m['TN']}\n\n")

        f.write("── Meta-MLP stacking ────────────────────────────────────────\n")
        for split, m in [('Val', val_m), ('Test', test_m)]:
            f.write(f"[{split}]\n")
            for k, v in m.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\nMétricas guardadas en {out_path}")

    # Save meta-MLP weights
    meta_ckpt_path = f"outputs/meta_mlp_{h}h_k{args.fold}.pt"
    torch.save(meta.state_dict(), meta_ckpt_path)
    print(f"Meta-MLP guardado en {meta_ckpt_path}")


if __name__ == '__main__':
    main()

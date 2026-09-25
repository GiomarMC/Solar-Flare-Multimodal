"""
Evaluación de BiLSTM standalone y guardado de logits para curvas ROC.

Itera sobre los 5 folds ya entrenados para 24h y 48h.
Puede correrse en CPU (no necesita GPU).

Usage:
    python scripts/save_logits_lstm.py
    python scripts/save_logits_lstm.py --horizon 24
    python scripts/save_logits_lstm.py --horizon 48 --fold 3
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
from scripts.train_lstm_standalone import (
    LightningModule, SharpOnlyDataset, load_cfg,
)
from scripts.dataset_temporal_v3 import (
    load_para_flare, _parse_split_file, _read_seq_file,
    _log_transform, N_SHARP,
)


def find_best_checkpoint(ckpt_dir: str) -> str:
    candidates = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No hay checkpoints en {ckpt_dir}")
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
    for _, tab, labels in loader:
        tab = tab.to(device)
        logits = model(tab).cpu().numpy()
        all_logits.append(logits)
        all_labels.append(labels.numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


def sweep_tau(logits, labels, sweep_start=0.01, sweep_end=0.99, sweep_step=0.01):
    taus = np.arange(sweep_start, sweep_end + 1e-9, sweep_step)
    best_tss, best_tau = -1.0, 0.5
    for tau in taus:
        m = compute_metrics(logits, labels, tau)
        if m['TSS'] > best_tss:
            best_tss, best_tau = m['TSS'], tau
    return best_tau, best_tss


def build_split_files(base_dir: str, horizon: int, fold: int):
    prefix = os.path.join(base_dir, f"Seq_Magnetogram/M{horizon}/Seq16_flare_Mclass_{horizon}h")
    train_files = [f"{prefix}_TrainVal{k}.txt" for k in range(5) if k != fold]
    val_files   = [f"{prefix}_TrainVal{fold}.txt"]
    test_files  = [f"{prefix}_Test.txt"]
    return train_files, val_files, test_files


def run_fold(base_dir: str, horizon: int, fold: int, device: torch.device):
    # Buscar config y checkpoint
    cfg_path  = f"configs/lstm_cv_{horizon}h_k{fold}.yaml"
    ckpt_dir  = f"outputs/checkpoints_lstm_cv_{horizon}h_k{fold}"

    if not os.path.exists(cfg_path):
        print(f"  Config no encontrado: {cfg_path} — omitiendo")
        return
    if not os.path.isdir(ckpt_dir):
        print(f"  Directorio de checkpoints no encontrado: {ckpt_dir} — omitiendo")
        return

    cfg  = load_cfg(cfg_path)
    ckpt = find_best_checkpoint(ckpt_dir)
    print(f"  Checkpoint: {os.path.basename(ckpt)}")

    para_path = os.path.join(base_dir, "para_flare_21params.txt")
    seq_dir   = os.path.join(base_dir, f"Seq_Magnetogram/M{horizon}/Seqs16")

    train_files, val_files, test_files = build_split_files(base_dir, horizon, fold)

    # Calcular medianas sobre train para imputación
    train_ds = SharpOnlyDataset(
        split_files=train_files,
        base_dir=base_dir,
        seq_dir=f"Seq_Magnetogram/M{horizon}/Seqs16",
        para_flare_path=para_path,
    )
    sharp_medians = train_ds.sharp_medians

    val_ds  = SharpOnlyDataset(val_files,  base_dir, f"Seq_Magnetogram/M{horizon}/Seqs16",
                               para_path, sharp_medians=sharp_medians)
    test_ds = SharpOnlyDataset(test_files, base_dir, f"Seq_Magnetogram/M{horizon}/Seqs16",
                               para_path, sharp_medians=sharp_medians)

    bs = cfg['training'].get('batch_size', 64)
    nw = min(cfg['training'].get('num_workers', 4), 4)
    val_loader  = DataLoader(val_ds,  batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    module = LightningModule.load_from_checkpoint(
        ckpt, cfg=cfg, sharp_medians=sharp_medians, strict=False
    )
    model = module.model.to(device)

    print("  Recolectando logits val...", end=" ", flush=True)
    val_logits, val_labels = collect_logits(model, val_loader, device)
    tau_opt, val_tss = sweep_tau(val_logits, val_labels)
    print(f"val_TSS={val_tss:.4f}  τ={tau_opt:.2f}")

    print("  Recolectando logits test...", end=" ", flush=True)
    test_logits, test_labels = collect_logits(model, test_loader, device)
    test_m = compute_metrics(test_logits, test_labels, tau_opt)
    print(f"test_TSS={test_m['TSS']:.4f}")

    os.makedirs("outputs/logits", exist_ok=True)
    for split, logits, labels in [
        ("val",  val_logits,  val_labels),
        ("test", test_logits, test_labels),
    ]:
        out = f"outputs/logits/lstm_{horizon}h_k{fold}_{split}.npz"
        np.savez_compressed(out, logits=logits, labels=labels)
        print(f"  Guardado: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, choices=[24, 48], default=None,
                        help='Horizonte a procesar (por defecto: ambos)')
    parser.add_argument('--fold', type=int, choices=[0, 1, 2, 3, 4], default=None,
                        help='Fold específico (por defecto: todos)')
    args = parser.parse_args()

    # Detectar base_dir desde el config que exista
    sample_cfg = "configs/lstm_cv_24h_k0.yaml"
    with open(sample_cfg) as f:
        sample = yaml.safe_load(f)
    base_dir = os.path.expandvars(sample['data']['base_dir'])   # admite ${SFMM_DATA}

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    horizons = [args.horizon] if args.horizon else [24, 48]
    folds    = [args.fold]    if args.fold is not None else [0, 1, 2, 3, 4]

    print(f"Device: {device}")
    print(f"Base dir: {base_dir}")
    print(f"Horizontes: {horizons} | Folds: {folds}\n")

    for h in horizons:
        for k in folds:
            print(f"── BiLSTM {h}h k={k} ──────────────────────────")
            run_fold(base_dir, h, k, device)
            print()

    print("Logits LSTM guardados en outputs/logits/")


if __name__ == '__main__':
    main()

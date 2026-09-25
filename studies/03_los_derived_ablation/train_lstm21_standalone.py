"""
Entrenamiento standalone de la BiLSTM sobre los 21 parámetros SHARP (v5).

Copia paralela de train_lstm_standalone.py (17 params) — misma arquitectura,
mismo protocolo, mismos splits. La ÚNICA diferencia es el dataset: v5 en lugar
de v3, es decir 21 parámetros en vez de 17.

Objetivo: comprobar si el techo estructural de la fusión (+0.0004 TSS a 48h,
+0.0082 a 24h, ver results/reports/diversidad_ramas_*.txt) se debe a redundancia
física inevitable o a que la selección univariada de v3 descartó justamente
los 4 parámetros menos recuperables desde la imagen.

Etapa 1 del pre-entrenamiento independiente:
  - Sin X3D — la LSTM es la única señal
  - LSTM bidireccional (1 capa, hidden=64) para capturar evolución temporal
  - El checkpoint guardado se usará en train_pretrained_fusion.py

Usage:
    python studies/03_los_derived_ablation/train_lstm21_standalone.py --config studies/03_los_derived_ablation/configs/lstm21_cv_48h_k3.yaml
    python studies/03_los_derived_ablation/train_lstm21_standalone.py --config studies/03_los_derived_ablation/configs/lstm21_cv_48h_k3.yaml --smoke 5
"""

import os
import sys
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from tqdm import tqdm

_AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_AQUI)))   # raíz del repo
sys.path.insert(0, _AQUI)                                     # módulos de este estudio
from dataset_temporal_v5 import (
    load_para_flare, _parse_split_file, _read_seq_file,
    _log_transform, N_SHARP,
)


class SharpOnlyDataset(torch.utils.data.Dataset):
    """
    Dataset ligero para LSTM standalone — carga solo parámetros SHARP, sin imágenes.

    __getitem__ devuelve (dummy_vid, tab, label) donde dummy_vid es un tensor vacío,
    manteniendo la firma compatible con el resto del pipeline.
    """

    def __init__(self, split_files, base_dir, seq_dir, para_flare_path,
                 sharp_medians=None, num_frames=16):
        self.seq_dir = os.path.join(base_dir, seq_dir)
        self.num_frames = num_frames

        self.para_map = load_para_flare(para_flare_path)

        self.entries = []
        for sp in split_files:
            self.entries.extend(_parse_split_file(sp))

        if sharp_medians is not None:
            self.sharp_medians = sharp_medians
        else:
            self.sharp_medians = self._compute_medians()

    def _compute_medians(self):
        rows = []
        for seq_file, *_ in self.entries:
            for stem in _read_seq_file(os.path.join(self.seq_dir, seq_file)):
                p = self.para_map.get(stem)
                if p is not None:
                    rows.append(p)
        return np.nanmedian(np.stack(rows, axis=0), axis=0).astype(np.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        seq_file, label, _ = self.entries[idx]
        frame_stems = _read_seq_file(os.path.join(self.seq_dir, seq_file))

        all_params = []
        for stem in frame_stems:
            raw = self.para_map.get(stem)
            if raw is None:
                raw = self.sharp_medians.copy()
            else:
                raw = raw.copy()
                nan_mask = np.isnan(raw)
                raw[nan_mask] = self.sharp_medians[nan_mask]
            all_params.append(raw)

        tab = _log_transform(np.stack(all_params, axis=0))  # [T, 21]
        return (
            torch.zeros(1),                                  # dummy_vid — no se usa
            torch.from_numpy(tab).float(),
            torch.tensor(label, dtype=torch.float32),
        )


def load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class PhysicsTemporalEncoderBidir(nn.Module):
    """
    LSTM bidireccional sobre secuencia temporal de parámetros SHARP.

    Input : [B, T, in_dim]
    Output: [B, out_dim]

    Arquitectura:
      input_proj: Linear(in_dim → hidden) + GELU
      lstm: nn.LSTM(hidden → hidden, bidireccional, 1 capa)
      concat(h_forward, h_backward) → Dropout → Linear(hidden*2 → out_dim)
    """

    def __init__(
        self,
        in_dim: int = 21,
        hidden: int = 64,
        out_dim: int = 64,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)                    # [B, T, hidden]
        _, (h_n, _) = self.lstm(x)                # h_n: [2, B, hidden]
        h_fwd = h_n[0]                            # [B, hidden]
        h_bwd = h_n[1]                            # [B, hidden]
        z = torch.cat([h_fwd, h_bwd], dim=-1)     # [B, hidden*2]
        z = self.dropout(z)
        z = self.output_proj(z)                   # [B, out_dim]
        return z


class LSTMStandaloneModel(nn.Module):
    """Modelo standalone: LSTM bidireccional + clasificador lineal."""

    def __init__(self, sharp_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.physics_enc = PhysicsTemporalEncoderBidir(
            in_dim=sharp_dim,
            hidden=hidden,
            out_dim=out_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, tab: torch.Tensor) -> torch.Tensor:
        z = self.physics_enc(tab)
        return self.classifier(z).squeeze(1)


def tss_from_preds(logits: torch.Tensor, labels: torch.Tensor, tau: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= tau).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    pod  = tp / (tp + fn + 1e-8)
    pofd = fp / (fp + tn + 1e-8)
    return pod - pofd


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    labels = [lbl for _, lbl, *_ in dataset.entries]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos, w_neg = 1.0 / n_pos, 1.0 / n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    print(f"WeightedRandomSampler: {n_pos} flares / {n_neg} no-flares → balanced 1:1")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class TrainingProgressCallback(pl.Callback):
    def __init__(self):
        self._pbar = None

    def _close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._close()
        self._pbar = tqdm(
            total=trainer.num_training_batches,
            desc=f"Epoch {trainer.current_epoch + 1:03d}/{trainer.max_epochs} [train]",
            leave=False, dynamic_ncols=True,
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._pbar is None:
            return
        loss = float(outputs) if hasattr(outputs, 'item') else float(outputs.get('loss', float('nan')))
        self._pbar.set_postfix(loss=f"{loss:.4f}")
        self._pbar.update(1)

    def on_train_epoch_end(self, trainer, pl_module):
        self._close()

    def on_validation_epoch_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._close()
        total = trainer.num_val_batches[0] if trainer.num_val_batches else 0
        self._pbar = tqdm(
            total=total,
            desc=f"Epoch {trainer.current_epoch + 1:03d}/{trainer.max_epochs} [val]  ",
            leave=False, dynamic_ncols=True,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.sanity_checking or self._pbar is None:
            return
        self._pbar.update(1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._close()
        m = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        opt = trainer.optimizers[0]
        lr = opt.param_groups[0]['lr']
        tqdm.write(
            f"[Epoch {epoch:03d}/{trainer.max_epochs}]  "
            f"train_loss={float(m.get('train_loss', float('nan'))):.4f}  "
            f"val_loss={float(m.get('val_loss', float('nan'))):.4f}  "
            f"val_TSS={float(m.get('val_tss', float('nan'))):.4f}  "
            f"lr={lr:.2e}"
        )


class LightningModule(pl.LightningModule):
    def __init__(self, cfg: dict, sharp_medians: np.ndarray):
        super().__init__()
        self.save_hyperparameters(ignore=['sharp_medians'])
        self.cfg = cfg
        self.sharp_medians = sharp_medians

        m = cfg['model']
        self.model = LSTMStandaloneModel(
            sharp_dim=m['sharp_dim'],
            hidden=m['hidden'],
            out_dim=m['physics_out'],
            dropout=m['dropout'],
        )
        pos_weight = torch.tensor([cfg['training']['pos_weight']])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        val_pos_weight = torch.tensor([cfg['training'].get('val_pos_weight', 38.0)])
        self.val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pos_weight)

        self._val_logits = []
        self._val_labels = []

    def forward(self, tab):
        return self.model(tab)

    def training_step(self, batch, batch_idx):
        _, tab, labels = batch
        logits = self(tab)
        loss = self.loss_fn(logits, labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, tab, labels = batch
        logits = self(tab)
        loss = self.val_loss_fn(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self._val_logits)
        all_labels = torch.cat(self._val_labels)
        best_tss = -1.0
        cfg_t = self.cfg['threshold']
        taus = np.arange(cfg_t['sweep_start'], cfg_t['sweep_end'] + 1e-9, cfg_t['sweep_step'])
        for tau in taus:
            t = tss_from_preds(all_logits, all_labels, tau)
            if t > best_tss:
                best_tss = t
        self.log('val_tss', best_tss, prog_bar=True)
        self._val_logits.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        t = self.cfg['training']
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=t['learning_rate'],
            weight_decay=t['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=t.get('lr_reduce_factor', 0.5),
            patience=t.get('lr_patience', 5),
            min_lr=1e-7,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'},
        }


def build_datasets(cfg: dict):
    d = cfg['data']
    base = os.path.expandvars(d['base_dir'])   # admite ${SFMM_DATA}
    para_path = os.path.join(base, d['para_flare'])

    train_files = [os.path.join(base, p) for p in d['splits']['train']]
    val_files   = [os.path.join(base, p) for p in d['splits']['val']]
    test_files  = [os.path.join(base, p) for p in d['splits']['test']]

    train_ds = SharpOnlyDataset(
        split_files=train_files, base_dir=base, seq_dir=d['seq_dir'],
        para_flare_path=para_path, num_frames=cfg['model']['num_frames'],
    )
    sharp_medians = train_ds.sharp_medians

    val_ds = SharpOnlyDataset(
        split_files=val_files, base_dir=base, seq_dir=d['seq_dir'],
        para_flare_path=para_path, sharp_medians=sharp_medians,
        num_frames=cfg['model']['num_frames'],
    )
    test_ds = SharpOnlyDataset(
        split_files=test_files, base_dir=base, seq_dir=d['seq_dir'],
        para_flare_path=para_path, sharp_medians=sharp_medians,
        num_frames=cfg['model']['num_frames'],
    )
    return train_ds, val_ds, test_ds, sharp_medians


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=os.path.join(_AQUI, 'configs', 'lstm21_cv_48h_k3.yaml'))
    parser.add_argument('--smoke', type=int, default=0, metavar='N')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg['training']['seed'], workers=True)

    train_ds, val_ds, test_ds, sharp_medians = build_datasets(cfg)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    nw = cfg['training']['num_workers']
    bs = cfg['training']['batch_size']

    train_sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    module = LightningModule(cfg, sharp_medians)

    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output']['log_dir'], exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=cfg['output']['checkpoint_dir'],
        filename='sfmm_lstm21_standalone-{epoch:03d}-{val_tss:.4f}',
        monitor='val_tss',
        mode='max',
        save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss',
        patience=cfg['training']['patience'],
        mode='min',
    )
    progress_cb = TrainingProgressCallback()
    logger = CSVLogger(cfg['output']['log_dir'], name='sfmm_lstm21_standalone')

    use_gpu = torch.cuda.is_available()
    precision = 'bf16-mixed' if use_gpu else '32-true'

    trainer = pl.Trainer(
        max_epochs=2 if args.smoke else cfg['training']['epochs'],
        limit_train_batches=args.smoke if args.smoke else 1.0,
        limit_val_batches=max(1, args.smoke // 4) if args.smoke else 1.0,
        callbacks=[checkpoint_cb, early_stop_cb, progress_cb],
        logger=logger,
        log_every_n_steps=1 if args.smoke else 10,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        enable_progress_bar=False,
    )

    smoke_str = f"[smoke] {args.smoke} batches x 2 epochs" if args.smoke else f"[train] {cfg['training']['epochs']} epochs"
    print(f"{smoke_str} — precision={precision} — {'GPU' if use_gpu else 'CPU'}")
    trainer.fit(module, train_loader, val_loader)

    best_ckpt = checkpoint_cb.best_model_path
    print(f"\nBest checkpoint: {best_ckpt}")
    print("Usa este checkpoint en train_pretrained_fusion.py (--lstm_ckpt)")


if __name__ == '__main__':
    main()

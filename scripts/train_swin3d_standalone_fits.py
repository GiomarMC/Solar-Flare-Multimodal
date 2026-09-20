"""
Swin3D-T standalone sobre FITS float32 (.npy cache). Solo visual, sin SHARP.

Idéntica arquitectura al Swin3D standalone JPEG pero carga magnetogramas desde
la caché .npy (float32, escala física en Gauss, normalizado a [-1,1]).
El fold de validación se elige en tiempo de ejecución con --fold.

Usage:
    python scripts/train_swin3d_standalone_fits.py \
        --config configs/swin3d_standalone_fits_24h.yaml --fold 3
    python scripts/train_swin3d_standalone_fits.py \
        --config configs/swin3d_standalone_fits_48h.yaml --fold 3 --smoke 5
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.multimodal_videoswin import VideoSwinEncoder
from scripts.dataset_fits_temporal import FITSTemporalDataset


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_split_files(cfg, fold):
    base = cfg['data']['base_dir']
    h = cfg['data']['horizon']
    prefix = os.path.join(base, f"Seq_Magnetogram/M{h}/Seq16_flare_Mclass_{h}h")
    train_files = [f"{prefix}_TrainVal{k}.txt" for k in range(5) if k != fold]
    val_files   = [f"{prefix}_TrainVal{fold}.txt"]
    test_files  = [f"{prefix}_Test.txt"]
    return train_files, val_files, test_files


class Swin3DStandaloneModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.video_enc = VideoSwinEncoder(video_out_dim=768, pretrained=pretrained)
        self.classifier = nn.Linear(768, 1)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, vid):
        return self.classifier(self.video_enc(vid)).squeeze(1)


def tss_from_preds(logits, labels, tau=0.5):
    preds = (torch.sigmoid(logits) >= tau).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    return tp / (tp + fn + 1e-8) - fp / (fp + tn + 1e-8)


def make_weighted_sampler(dataset):
    labels = [lbl for _, lbl, *_ in dataset.entries]
    n_pos, n_neg = sum(labels), len(labels) - sum(labels)
    weights = [1.0 / n_pos if l == 1 else 1.0 / n_neg for l in labels]
    print(f"WeightedRandomSampler: {n_pos} flares / {n_neg} no-flares → balanced 1:1")
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class ProgressCallback(pl.Callback):
    def __init__(self):
        self._pbar = None

    def _close(self):
        if self._pbar:
            self._pbar.close()
            self._pbar = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._close()
        self._pbar = tqdm(
            total=trainer.num_training_batches,
            desc=f"Epoch {trainer.current_epoch+1:03d}/{trainer.max_epochs} [train]",
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
            desc=f"Epoch {trainer.current_epoch+1:03d}/{trainer.max_epochs} [val]  ",
            leave=False, dynamic_ncols=True,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.sanity_checking and self._pbar:
            self._pbar.update(1)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        self._close()
        m = trainer.callback_metrics
        epoch = trainer.current_epoch + 1
        lr = trainer.optimizers[0].param_groups[0]['lr']
        tqdm.write(
            f"[Epoch {epoch:03d}/{trainer.max_epochs}]  "
            f"train_loss={float(m.get('train_loss', float('nan'))):.4f}  "
            f"val_loss={float(m.get('val_loss', float('nan'))):.4f}  "
            f"val_TSS={float(m.get('val_tss', float('nan'))):.4f}  "
            f"lr={lr:.2e}"
        )


class LightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = Swin3DStandaloneModel(pretrained=cfg['model'].get('pretrained', True))
        pos_weight = torch.tensor([cfg['training']['pos_weight']])
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        val_pos_weight = torch.tensor([cfg['training'].get('val_pos_weight', 38.0)])
        self.val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pos_weight)
        self._val_logits = []
        self._val_labels = []

    def forward(self, vid):
        # FITSTemporalDataset → [B,T,3,H,W]; VideoSwin espera [B,3,T,H,W]
        return self.model(vid.permute(0, 2, 1, 3, 4))

    def training_step(self, batch, batch_idx):
        vid, _, labels = batch
        loss = self.loss_fn(self(vid), labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vid, _, labels = batch
        logits = self(vid)
        loss = self.val_loss_fn(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self._val_logits)
        all_labels = torch.cat(self._val_labels)
        cfg_t = self.cfg['threshold']
        taus = np.arange(cfg_t['sweep_start'], cfg_t['sweep_end'] + 1e-9, cfg_t['sweep_step'])
        best_tss = max(tss_from_preds(all_logits, all_labels, tau) for tau in taus)
        self.log('val_tss', best_tss, prog_bar=True)
        self._val_logits.clear()
        self._val_labels.clear()

    def configure_optimizers(self):
        t = self.cfg['training']
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=t['learning_rate'],
            weight_decay=t['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=t.get('lr_reduce_factor', 0.5),
            patience=t.get('lr_patience', 5),
            min_lr=1e-8,
        )
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}}


def build_datasets(cfg, fold):
    d = cfg['data']
    base = os.path.expandvars(d['base_dir'])   # admite ${SFMM_DATA}
    h = d['horizon']
    para_path = os.path.join(base, 'para_flare_21params.txt')
    m = cfg['model']
    t = cfg['training']

    train_files, val_files, test_files = build_split_files(cfg, fold)
    seq_dir = f"Seq_Magnetogram/M{h}/Seqs16"
    aug = dict(aug_rotation=t.get('aug_rotation', 180.0),
               aug_noise_std=t.get('aug_noise_std', 0.02),
               aug_crop_scale=t.get('aug_crop_scale', 0.8))

    train_ds = FITSTemporalDataset(
        split_files=train_files, base_dir=base, seq_dir=seq_dir,
        fits_dir=d['fits_dir'], para_flare_path=para_path,
        augment=True, img_size=m['img_size'], num_frames=m['num_frames'],
        clip_gauss=m.get('clip_gauss', 500.0), **aug,
    )
    sharp_medians = train_ds.sharp_medians
    val_ds = FITSTemporalDataset(
        split_files=val_files, base_dir=base, seq_dir=seq_dir,
        fits_dir=d['fits_dir'], para_flare_path=para_path,
        sharp_medians=sharp_medians, augment=False,
        img_size=m['img_size'], num_frames=m['num_frames'],
        clip_gauss=m.get('clip_gauss', 500.0),
    )
    test_ds = FITSTemporalDataset(
        split_files=test_files, base_dir=base, seq_dir=seq_dir,
        fits_dir=d['fits_dir'], para_flare_path=para_path,
        sharp_medians=sharp_medians, augment=False,
        img_size=m['img_size'], num_frames=m['num_frames'],
        clip_gauss=m.get('clip_gauss', 500.0),
    )
    return train_ds, val_ds, test_ds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--fold',   type=int, required=True, choices=[0, 1, 2, 3, 4],
                        help='Fold de validación (0-4). Train = todos los demás folds.')
    parser.add_argument('--smoke',  type=int, default=0, metavar='N')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    h = cfg['data']['horizon']

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg['training']['seed'], workers=True)

    train_ds, val_ds, _ = build_datasets(cfg, args.fold)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    nw  = cfg['training']['num_workers']
    bs  = cfg['training']['batch_size']
    acc = cfg['training']['accumulate_grad_batches']

    train_loader = DataLoader(train_ds, batch_size=bs, sampler=make_weighted_sampler(train_ds),
                              num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              num_workers=nw, pin_memory=True)

    module = LightningModule(cfg)

    ckpt_dir = f"outputs/checkpoints_swin3d_fits_{h}h_k{args.fold}"
    log_dir  = f"outputs/logs_swin3d_fits_{h}h_k{args.fold}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f'sfmm_swin3d_fits_{h}h_k{args.fold}-{{epoch:03d}}-{{val_tss:.4f}}',
        monitor='val_tss', mode='max', save_top_k=3,
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss', patience=cfg['training']['patience'], mode='min',
    )
    logger = CSVLogger(log_dir, name=f'swin3d_fits_{h}h')

    use_gpu = torch.cuda.is_available()
    precision = 'bf16-mixed' if use_gpu else '32-true'

    trainer = pl.Trainer(
        max_epochs=2 if args.smoke else cfg['training']['epochs'],
        accumulate_grad_batches=1 if args.smoke else acc,
        limit_train_batches=args.smoke if args.smoke else 1.0,
        limit_val_batches=max(1, args.smoke // 4) if args.smoke else 1.0,
        callbacks=[checkpoint_cb, early_stop_cb, ProgressCallback()],
        logger=logger,
        log_every_n_steps=1 if args.smoke else 10,
        precision=precision,
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        enable_progress_bar=False,
    )

    mode = f"[smoke {args.smoke} batches]" if args.smoke else f"[{cfg['training']['epochs']} epochs]"
    print(f"[swin3d-fits-{h}h k={args.fold}] {mode} — precision={precision} — {'GPU' if use_gpu else 'CPU'}")
    trainer.fit(module, train_loader, val_loader)
    print(f"\nBest checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == '__main__':
    main()

"""
Ablación de arquitectura de la BiGRU standalone — 24 h (parametriza hidden y num_layers).

Archivo PARALELO a train_lstm_standalone.py (no lo modifica): reusa su dataset,
sampler y callbacks, y redefine SOLO las clases del modelo para que la LSTM lea
`num_layers` (con dropout entre capas cuando num_layers>1). Con num_layers=1 e
hidden=64 reproduce el baseline.

Toma el config base lstm_cv_48h_k{fold}.yaml en SOLO LECTURA y sobreescribe en
memoria hidden/num_layers/physics_out y los directorios de salida (paralelos).
Tras entrenar, guarda logits val+test en outputs/logits_ablation/.

Uso:
    python scripts/train_gru_ablation.py --fold 3 --hidden 64 --num_layers 1
    python scripts/train_gru_ablation.py --fold 3 --hidden 128 --num_layers 2 --smoke 5
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.train_lstm_standalone import (
    SharpOnlyDataset, load_cfg, make_weighted_sampler,
    TrainingProgressCallback, tss_from_preds, build_datasets,
)
from scripts.save_logits_lstm import find_best_checkpoint, collect_logits, sweep_tau, compute_metrics

CACHE_DIR = os.path.join(ROOT, "outputs/cache_sharp")


# ----- caché de tensores SHARP en memoria (idéntico entre arquitecturas, por fold) -----
class CachedDS(torch.utils.data.Dataset):
    """Devuelve (dummy_vid, tab[T,17], label) desde tensores en memoria."""
    def __init__(self, X, y):
        self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return torch.zeros(1), self.X[i], self.y[i]


def _materialize(ds, nw=8):
    Xs, ys = [], []
    for _, tab, lab in DataLoader(ds, batch_size=256, num_workers=nw):
        Xs.append(tab.numpy()); ys.append(lab.numpy())
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


def build_or_load_cache(cfg, fold):
    """Materializa train/val/test del fold una sola vez y los reusa (npz en disco)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"fold{fold}_24h.npz")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    print(f"[cache] materializando fold {fold} (una vez)...", flush=True)
    train_ds, val_ds, test_ds, med = build_datasets(cfg)
    Xtr, ytr = _materialize(train_ds)
    Xv, yv = _materialize(val_ds)
    Xt, yt = _materialize(test_ds)
    np.savez_compressed(path, X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv,
                        X_test=Xt, y_test=yt, sharp_medians=med)
    return dict(X_train=Xtr, y_train=ytr, X_val=Xv, y_val=yv,
                X_test=Xt, y_test=yt, sharp_medians=med)


def sampler_from_labels(y):
    n_pos = float(y.sum()); n_neg = float(len(y) - n_pos)
    w = np.where(y == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
    print(f"WeightedRandomSampler: {int(n_pos)} flares / {int(n_neg)} no-flares → 1:1")
    return WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)


# ----- modelo con num_layers parametrizado (única diferencia con el standalone) -----
class PhysicsTemporalEncoderBidirGRU(nn.Module):
    def __init__(self, in_dim=17, hidden=64, out_dim=64, dropout=0.4, num_layers=1):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU())
        self.gru = nn.GRU(
            input_size=hidden, hidden_size=hidden, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=(dropout if num_layers > 1 else 0.0),
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden * 2, out_dim)

    def forward(self, x):
        x = self.input_proj(x)
        _, h_n = self.gru(x)          # h_n: [2*num_layers, B, hidden]
        h_fwd, h_bwd = h_n[-2], h_n[-1]     # última capa, ambos sentidos
        z = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.output_proj(self.dropout(z))


class GRUStandaloneModel(nn.Module):
    def __init__(self, sharp_dim, hidden, out_dim, dropout, num_layers):
        super().__init__()
        self.physics_enc = PhysicsTemporalEncoderBidirGRU(
            in_dim=sharp_dim, hidden=hidden, out_dim=out_dim,
            dropout=dropout, num_layers=num_layers)
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, tab):
        return self.classifier(self.physics_enc(tab)).squeeze(1)


class LightningModule(pl.LightningModule):
    def __init__(self, cfg, sharp_medians):
        super().__init__()
        self.save_hyperparameters(ignore=['sharp_medians'])
        self.cfg = cfg
        self.sharp_medians = sharp_medians
        m = cfg['model']
        self.model = GRUStandaloneModel(
            sharp_dim=m['sharp_dim'], hidden=m['hidden'], out_dim=m['physics_out'],
            dropout=m['dropout'], num_layers=m.get('num_layers', 1))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg['training']['pos_weight']]))
        self.val_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([cfg['training'].get('val_pos_weight', 38.0)]))
        self._val_logits, self._val_labels = [], []

    def forward(self, tab):
        return self.model(tab)

    def training_step(self, batch, _):
        _, tab, labels = batch
        loss = self.loss_fn(self(tab), labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        _, tab, labels = batch
        logits = self(tab)
        loss = self.val_loss_fn(logits, labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_labels.append(labels.detach().cpu())
        return loss

    def on_validation_epoch_end(self):
        all_logits = torch.cat(self._val_logits); all_labels = torch.cat(self._val_labels)
        t = self.cfg['threshold']
        best = max(tss_from_preds(all_logits, all_labels, float(tau))
                   for tau in np.arange(t['sweep_start'], t['sweep_end'] + 1e-9, t['sweep_step']))
        self.log('val_tss', best, prog_bar=True)
        self._val_logits.clear(); self._val_labels.clear()

    def configure_optimizers(self):
        t = self.cfg['training']
        opt = torch.optim.AdamW(self.model.parameters(), lr=t['learning_rate'],
                                weight_decay=t['weight_decay'])
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='min', factor=t.get('lr_reduce_factor', 0.5),
            patience=t.get('lr_patience', 5), min_lr=1e-7)
        return {'optimizer': opt, 'lr_scheduler': {'scheduler': sch, 'monitor': 'val_loss'}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--hidden', type=int, required=True)
    ap.add_argument('--num_layers', type=int, required=True)
    ap.add_argument('--physics_out', type=int, default=None, help='default = hidden')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()

    H, k = 24, args.fold
    tag = f"h{args.hidden}_l{args.num_layers}_24h_k{k}"
    cfg = load_cfg(f"configs/lstm_cv_{H}h_k{k}.yaml")     # solo lectura
    cfg['model']['hidden'] = args.hidden
    cfg['model']['num_layers'] = args.num_layers
    cfg['model']['physics_out'] = args.physics_out or args.hidden
    if args.epochs:
        cfg['training']['epochs'] = args.epochs
    cfg['output']['checkpoint_dir'] = f"outputs/checkpoints_gru_abl_{tag}"
    cfg['output']['log_dir'] = f"outputs/logs_gru_abl_{tag}"

    torch.set_float32_matmul_precision('medium')
    pl.seed_everything(cfg['training']['seed'], workers=True)

    cache = build_or_load_cache(cfg, k)
    sharp_medians = cache['sharp_medians']
    train_ds = CachedDS(cache['X_train'], cache['y_train'])
    val_ds = CachedDS(cache['X_val'], cache['y_val'])
    test_ds = CachedDS(cache['X_test'], cache['y_test'])
    print(f"[{tag}] Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")
    bs = cfg['training']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=sampler_from_labels(cache['y_train']),
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    module = LightningModule(cfg, sharp_medians)
    os.makedirs(cfg['output']['checkpoint_dir'], exist_ok=True)
    os.makedirs(cfg['output']['log_dir'], exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=cfg['output']['checkpoint_dir'],
                              filename='abl-{epoch:03d}-{val_tss:.4f}',
                              monitor='val_tss', mode='max', save_top_k=1)
    es_cb = EarlyStopping(monitor='val_loss', patience=cfg['training']['patience'], mode='min')
    logger = CSVLogger(cfg['output']['log_dir'], name='abl')
    use_gpu = torch.cuda.is_available()

    # LSTM diminuta: fp32 usa el kernel cuDNN optimizado (bf16 cae a un camino lento);
    # sin TrainingProgressCallback para no inundar el log redirigido con tqdm.
    trainer = pl.Trainer(
        max_epochs=2 if args.smoke else cfg['training']['epochs'],
        limit_train_batches=args.smoke if args.smoke else 1.0,
        limit_val_batches=max(1, args.smoke // 4) if args.smoke else 1.0,
        callbacks=[ckpt_cb, es_cb],
        logger=logger, log_every_n_steps=50,
        precision='32-true',
        gradient_clip_val=1.0, gradient_clip_algorithm='norm',
        enable_progress_bar=False)
    trainer.fit(module, train_loader, val_loader)

    # ---- inferencia + guardado de logits val/test (paralelo) ----
    device = torch.device('cuda' if use_gpu else 'cpu')
    model = LightningModule.load_from_checkpoint(
        find_best_checkpoint(cfg['output']['checkpoint_dir']),
        cfg=cfg, sharp_medians=sharp_medians, strict=False).model.to(device)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    vl, vy = collect_logits(model, val_loader, device)
    tau_opt, val_tss = sweep_tau(vl, vy)
    tl, ty = collect_logits(model, test_loader, device)
    test_m = compute_metrics(tl, ty, tau_opt)

    out_dir = os.path.join(ROOT, "outputs/logits_ablation")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, f"gru_abl_{tag}_val.npz"), logits=vl, labels=vy)
    np.savez_compressed(os.path.join(out_dir, f"gru_abl_{tag}_test.npz"), logits=tl, labels=ty)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{tag}] params={n_params}  val_TSS={val_tss:.4f}  τ={tau_opt:.2f}  test_TSS={test_m['TSS']:.4f}")


if __name__ == '__main__':
    main()

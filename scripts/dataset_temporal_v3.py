"""
Dataset temporal v3 — 17 parámetros SHARP, sin hflip ni vflip.

Cambios respecto a v2 (dataset_temporal_v2.py):
  - 17 parámetros SHARP en lugar de 21: elimina los 4 de baja señal discriminativa
    (MEANGBH d=0.14, MEANJZH d=0.11, MEANJZD d=0.53, MEANALP d=0.17).
  - Lee los 21 del archivo y selecciona los 17 índices relevantes.
  - Sin hflip ni vflip (Ley de Hale, igual que v2).
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# Orden completo en para_flare_21params.txt
_SHARP_21_ALL = [
    'TOTUSJH', 'TOTPOT',  'TOTUSJZ', 'USFLUX',  'USFLUXL',   # 0-4
    'MEANGBH', 'MEANGBL', 'MEANGBT', 'MEANGBZ',               # 5-8
    'MEANJZH', 'MEANJZD', 'ABSNJZH', 'SAVNCPP', 'MEANALP',   # 9-13
    'MEANPOT', 'MEANSHR', 'SHRGT45', 'AREA_ACR', 'NACR',      # 14-18
    'MEANGAM', 'R_VALUE',                                       # 19-20
]

# Índices a conservar (eliminados: 5=MEANGBH, 9=MEANJZH, 10=MEANJZD, 13=MEANALP)
_KEEP_IDX = [0, 1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20]

SHARP_PARAMS = [_SHARP_21_ALL[i] for i in _KEEP_IDX]
N_SHARP = len(SHARP_PARAMS)  # 17


def _log_transform(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log10(np.abs(x) + 1.0)


def load_para_flare(para_path: str):
    """Returns dict: stem_name -> np.array(17,) — los 17 params seleccionados."""
    mapping = {}
    keep = np.array(_KEEP_IDX)
    with open(para_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 23:
                continue
            stem = parts[0].removesuffix('.fits')
            all21 = np.array(
                [float(p) if p != 'nan' else np.nan for p in parts[1:22]],
                dtype=np.float32,
            )
            mapping[stem] = all21[keep]
    return mapping


def _parse_split_file(split_path: str):
    entries = []
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            seq_file = parts[0]
            label = int(parts[1])
            aug_tag = parts[2] if len(parts) >= 3 else 'nf'
            entries.append((seq_file, label, aug_tag))
    return entries


def _read_seq_file(seq_path: str):
    frames = []
    with open(seq_path, 'r') as f:
        for line in f:
            name = line.strip()
            if name:
                frames.append(name)
    return frames


class MultimodalTemporalDataset(Dataset):
    """
    Returns (X_vid, X_tab, label) para SF-MM con 17 parámetros SHARP.

    X_vid : [3, T, H, W]   — T=16 magnetogramas, replicados a 3 canales
    X_tab : [T, 17]         — T=16 timestamps × 17 SHARP (log-transformados)
    label : escalar int     — 0/1

    v3: 17 params (sin MEANGBH/MEANJZH/MEANJZD/MEANALP), sin hflip/vflip.
    """

    def __init__(
        self,
        split_files: list,
        base_dir: str,
        seq_dir: str,
        img_dir: str,
        para_flare_path: str,
        sharp_medians: np.ndarray = None,
        augment: bool = False,
        img_size: int = 160,
        num_frames: int = 16,
        aug_rotation: float = 180.0,
        aug_noise_std: float = 0.02,
        aug_brightness: float = 0.10,
        aug_contrast: float = 0.10,
        aug_crop_scale: float = 0.80,
    ):
        self.base_dir = base_dir
        self.seq_dir = os.path.join(base_dir, seq_dir)
        self.img_dir = os.path.join(base_dir, img_dir)
        self.augment = augment
        self.img_size = img_size
        self.num_frames = num_frames
        self.aug_rotation = aug_rotation
        self.aug_noise_std = aug_noise_std
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast
        self.aug_crop_scale = aug_crop_scale

        self.para_map = load_para_flare(para_flare_path)

        self.entries = []
        for sp in split_files:
            self.entries.extend(_parse_split_file(sp))

        if sharp_medians is not None:
            self.sharp_medians = sharp_medians
        else:
            self.sharp_medians = self._compute_medians()

    def _compute_medians(self) -> np.ndarray:
        rows = []
        for seq_file, *_ in self.entries:
            seq_path = os.path.join(self.seq_dir, seq_file)
            for stem in _read_seq_file(seq_path):
                params = self.para_map.get(stem)
                if params is not None:
                    rows.append(params)
        arr = np.stack(rows, axis=0)
        return np.nanmedian(arr, axis=0).astype(np.float32)

    def __len__(self):
        return len(self.entries)

    def _load_frame(self, stem: str) -> np.ndarray:
        jpg_name = stem + '.fits.jpg'
        jpg_path = os.path.join(self.img_dir, jpg_name)
        img = Image.open(jpg_path).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0

    def _augment_frames(self, frames: list) -> list:
        # sin hflip ni vflip — violarían la Ley de Hale
        angle = random.uniform(-self.aug_rotation, self.aug_rotation)
        crop_scale = random.uniform(self.aug_crop_scale, 1.0)
        crop_size = int(self.img_size * crop_scale)
        crop_i = random.randint(0, self.img_size - crop_size)
        crop_j = random.randint(0, self.img_size - crop_size)
        brightness = random.uniform(1.0 - self.aug_brightness, 1.0 + self.aug_brightness)
        contrast = random.uniform(1.0 - self.aug_contrast, 1.0 + self.aug_contrast)

        augmented = []
        for arr in frames:
            pil = Image.fromarray((arr * 255).astype(np.uint8))
            pil = TF.rotate(pil, angle)
            pil = TF.crop(pil, crop_i, crop_j, crop_size, crop_size)
            pil = pil.resize((self.img_size, self.img_size), Image.BILINEAR)
            pil = TF.adjust_brightness(pil, brightness)
            pil = TF.adjust_contrast(pil, contrast)
            augmented.append(np.array(pil, dtype=np.float32) / 255.0)
        return augmented

    def _add_noise(self, frames: list) -> list:
        return [
            np.clip(f + np.random.normal(0, self.aug_noise_std, f.shape).astype(np.float32), 0, 1)
            for f in frames
        ]

    def _load_temporal_sharp(self, frame_stems: list) -> np.ndarray:
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
        return np.stack(all_params, axis=0)  # [T, 17]

    def __getitem__(self, idx):
        seq_file, label, _ = self.entries[idx]
        seq_path = os.path.join(self.seq_dir, seq_file)
        frame_stems = _read_seq_file(seq_path)

        frames = [self._load_frame(s) for s in frame_stems]

        if self.augment:
            frames = self._augment_frames(frames)
            frames = self._add_noise(frames)

        vid = np.stack(frames, axis=0)
        vid = torch.from_numpy(vid).unsqueeze(0)
        vid = vid.expand(3, -1, -1, -1).float()       # [3, T, H, W]

        raw_params = self._load_temporal_sharp(frame_stems)  # [T, 17]
        tab = _log_transform(raw_params)
        tab = torch.from_numpy(tab).float()

        return vid, tab, torch.tensor(label, dtype=torch.float32)

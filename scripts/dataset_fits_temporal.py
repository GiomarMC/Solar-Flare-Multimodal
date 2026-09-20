"""
Dataset temporal con FITS float32 para Plan B (ConvNeXt + BiLSTM).

Diferencias vs dataset_temporal.py:
  - Lee FITS directamente con astropy (preserva escala física en Gauss)
  - Normalización: flipud + clip ±clip_gauss → [-1, 1] + repeat a 3 canales
  - Devuelve X_vid con shape [T, 3, H, W] (no [3, T, H, W] de VideoSwin)
  - Sin augmentación de brillo/contraste (cambia la escala física)
  - SHARP: log-transform + imputación con medianas del training set
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

try:
    from astropy.io import fits as astropy_fits
    _ASTROPY_AVAILABLE = True
except ImportError:
    _ASTROPY_AVAILABLE = False


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


def _load_fits_frame(fits_path: str, img_size: int, clip_gauss: float) -> np.ndarray:
    """
    Carga un frame de magnetograma. Prioriza .npy preprocesado (float16, 224×224).
    Fallback: leer FITS con astropy si el .npy no existe.
    Devuelve array [H, W] float32 en [-1, 1].
    """
    npy_path = fits_path.replace('/magnetogram_fits/', '/magnetogram_npy/').replace('.fits', '.npy')
    if os.path.exists(npy_path):
        arr = np.load(npy_path).astype(np.float32)
        if arr.shape != (img_size, img_size):
            img = Image.fromarray(arr, mode='F')
            img = img.resize((img_size, img_size), Image.BILINEAR)
            arr = np.array(img, dtype=np.float32)
        return arr

    if not _ASTROPY_AVAILABLE:
        raise ImportError("astropy requerido: pip install astropy")
    with astropy_fits.open(fits_path) as hdul:
        data = hdul[1].data.astype(np.float32)
    data = np.flipud(data)
    data = np.nan_to_num(data, nan=0.0, posinf=clip_gauss, neginf=-clip_gauss)
    data = np.clip(data, -clip_gauss, clip_gauss) / clip_gauss
    img = Image.fromarray(data, mode='F')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


class FITSTemporalDataset(Dataset):
    """
    Returns (X_vid, X_tab, label) para predicción multimodal con FITS float32.

    X_vid : [T, 3, H, W]   — T=16 frames FITS, normalizados a [-1,1], repetidos a 3 canales
    X_tab : [T, 17]         — T=16 timestamps × 17 SHARP (log-transformados)
    label : float tensor    — 0.0 / 1.0
    """

    def __init__(
        self,
        split_files: list,
        base_dir: str,
        seq_dir: str,
        fits_dir: str,
        para_flare_path: str,
        sharp_medians: np.ndarray = None,
        augment: bool = False,
        img_size: int = 224,
        num_frames: int = 16,
        clip_gauss: float = 500.0,
        aug_rotation: float = 180.0,
        aug_noise_std: float = 0.02,
        aug_crop_scale: float = 0.80,
    ):
        self.base_dir = base_dir
        self.seq_dir = os.path.join(base_dir, seq_dir)
        self.fits_dir = fits_dir
        self.augment = augment
        self.img_size = img_size
        self.num_frames = num_frames
        self.clip_gauss = clip_gauss
        self.aug_rotation = aug_rotation
        self.aug_noise_std = aug_noise_std
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
        """Mediana por columna sobre todos los frames de los entries (training set)."""
        rows = []
        for seq_file, *_ in self.entries:
            seq_path = os.path.join(self.seq_dir, seq_file)
            frames = _read_seq_file(seq_path)
            for stem in frames:
                params = self.para_map.get(stem)
                if params is not None:
                    rows.append(params)
        arr = np.stack(rows, axis=0)
        return np.nanmedian(arr, axis=0).astype(np.float32)

    def compute_pos_weight(self) -> float:
        """Calcula pos_weight = N_neg / N_pos para BCEWithLogitsLoss."""
        labels = [lbl for _, lbl, *_ in self.entries]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        return float(n_neg) / max(n_pos, 1)

    def __len__(self):
        return len(self.entries)

    def _augment_frame(self, arr: np.ndarray, angle: float, crop_params, do_noise: bool) -> np.ndarray:
        """Augmenta un frame [H, W] float32 con los parámetros pre-generados."""
        img = Image.fromarray(arr, mode='F')
        if angle is not None:
            img = TF.rotate(img, angle, fill=0.0)
        if crop_params is not None:
            i, j, h, w = crop_params
            img = TF.crop(img, i, j, h, w)
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        if do_noise:
            arr = arr + np.random.normal(0, self.aug_noise_std, arr.shape).astype(np.float32)
            arr = np.clip(arr, -1.0, 1.0)
        return arr

    def _load_temporal_sharp(self, frame_stems: list) -> np.ndarray:
        """Carga SHARP params para los T frames. Imputa NaN con sharp_medians."""
        rows = []
        for stem in frame_stems:
            raw = self.para_map.get(stem)
            if raw is None:
                raw = self.sharp_medians.copy()
            else:
                nan_mask = np.isnan(raw)
                if nan_mask.any():
                    raw = raw.copy()
                    raw[nan_mask] = self.sharp_medians[nan_mask]
            rows.append(raw)
        return np.stack(rows, axis=0).astype(np.float32)  # [T, 17]

    def __getitem__(self, idx):
        seq_file, label, _ = self.entries[idx]
        seq_path = os.path.join(self.seq_dir, seq_file)
        frame_stems = _read_seq_file(seq_path)[:self.num_frames]

        # Pre-generar parámetros de augmentación (los mismos para todos los frames)
        angle = None
        crop_params = None
        do_noise = False
        if self.augment:
            if random.random() < 0.5:
                angle = random.uniform(-self.aug_rotation, self.aug_rotation)
            if random.random() < 0.5:
                scale = random.uniform(self.aug_crop_scale, 1.0)
                h = int(self.img_size * scale)
                w = int(self.img_size * scale)
                # Calculamos sobre el tamaño original del FITS después de resize
                max_i = self.img_size - h
                max_j = self.img_size - w
                i = random.randint(0, max(0, max_i))
                j = random.randint(0, max(0, max_j))
                crop_params = (i, j, h, w)
            do_noise = random.random() < 0.5

        vid_frames = []
        for stem in frame_stems:
            fits_path = os.path.join(self.fits_dir, stem + '.fits')
            npy_path = fits_path.replace('/magnetogram_fits/', '/magnetogram_npy/').replace('.fits', '.npy')
            if os.path.exists(npy_path) or os.path.exists(fits_path):
                arr = _load_fits_frame(fits_path, self.img_size, self.clip_gauss)
                if self.augment:
                    arr = self._augment_frame(arr, angle, crop_params, do_noise)
            else:
                arr = np.zeros((self.img_size, self.img_size), dtype=np.float32)

            arr_3ch = np.stack([arr, arr, arr], axis=0)  # [3, H, W]
            vid_frames.append(arr_3ch)

        X_vid = torch.from_numpy(np.stack(vid_frames, axis=0))   # [T, 3, H, W]

        raw_tab = self._load_temporal_sharp(frame_stems)          # [T, 10]
        X_tab = torch.from_numpy(_log_transform(raw_tab))         # [T, 10]

        return X_vid, X_tab, torch.tensor(float(label), dtype=torch.float32)

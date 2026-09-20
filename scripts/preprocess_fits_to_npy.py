"""
Preprocesa todos los FITS de magnetograma a arrays numpy float16.

Operaciones por archivo:
  1. Leer FITS con astropy (extensión 1)
  2. flipud — corrige convención astronómica
  3. nan_to_num + clip ±500G + normalizar → [-1, 1]
  4. Resize a img_size × img_size (BILINEAR)
  5. Guardar como .npy float16

Resultado: carga ~20x más rápida durante entrenamiento.
Reanuda automáticamente si se interrumpe.

Usage:
    python scripts/preprocess_fits_to_npy.py --workers 8
"""

import os
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from astropy.io import fits as afits
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    os.system("pip install tqdm -q")
    from tqdm import tqdm


def process_one(args):
    src_path, dst_path, img_size, clip_gauss = args
    try:
        with afits.open(src_path) as hdul:
            data = hdul[1].data.astype(np.float32)
        data = np.flipud(data)
        data = np.nan_to_num(data, nan=0.0, posinf=clip_gauss, neginf=-clip_gauss)
        data = np.clip(data, -clip_gauss, clip_gauss) / clip_gauss
        img = Image.fromarray(data, mode='F')
        img = img.resize((img_size, img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float16)
        np.save(dst_path, arr)
        return True
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fits-dir',  default='/mnt/almacenamiento/magnetogram_fits')
    parser.add_argument('--out-dir',   default='/mnt/almacenamiento/magnetogram_npy')
    parser.add_argument('--img-size',  type=int, default=224)
    parser.add_argument('--clip-gauss', type=float, default=500.0)
    parser.add_argument('--workers',   type=int, default=8)
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(Path(args.fits_dir).glob('*.fits'))
    total = len(fits_files)

    tasks = []
    skipped = 0
    for src in fits_files:
        dst = out_path / (src.stem + '.npy')
        if dst.exists() and dst.stat().st_size > 0:
            skipped += 1
        else:
            tasks.append((str(src), str(dst), args.img_size, args.clip_gauss))

    print(f"Total FITS:      {total:,}")
    print(f"Ya procesados:   {skipped:,}")
    print(f"Pendientes:      {len(tasks):,}")
    print(f"Workers:         {args.workers}")
    print(f"Destino:         {args.out_dir}\n")

    if not tasks:
        print("Todo ya procesado.")
        return

    ok = fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, t): t for t in tasks}
        with tqdm(total=len(tasks), unit='archivo', ncols=72) as pbar:
            for fut in as_completed(futures):
                if fut.result():
                    ok += 1
                else:
                    fail += 1
                pbar.update(1)
                pbar.set_postfix(ok=ok, fail=fail)

    print(f"\nCompletado: {ok:,} OK | {fail:,} fallos")
    if fail > 0:
        print("Vuelve a ejecutar para reintentar los fallos.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
download_jsoc_fits.py — Descarga FITS de magnetograma desde JSOC sin cola de exportación.

Usa c.query(..., seg='magnetogram') para obtener URLs directas del servidor JSOC
y las descarga con HTTP — mucho más rápido que el sistema de export.

Uso:
  python scripts/download_jsoc_fits.py \\
      --jpg-dir  /ruta/a/magnetogram_jpg \\
      --out-dir  data/magnetogram_fits \\
      --workers  8

Reanuda automáticamente: los archivos ya descargados se saltan.
No requiere email registrado (solo lectura, sin export).
"""

import os
import re
import sys
import time
import argparse
import shutil
import urllib.request
import threading
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import drms
except ImportError:
    print("ERROR: drms no instalado.  pip install drms")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    os.system(f"{sys.executable} -m pip install tqdm -q")
    from tqdm import tqdm

JSOC_BASE = "http://jsoc.stanford.edu"
FNAME_RE  = re.compile(
    r'(hmi\.sharp_720s\.(\d+)\.(\d{8})_(\d{6})_TAI\.magnetogram\.fits)(?:\.jpg)?$'
)


def parse_fname(fname: str):
    """Devuelve (fits_name, harpnum, T_REC_jsoc) o None."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    fits_name = m.group(1)
    harpnum   = int(m.group(2))
    d, t      = m.group(3), m.group(4)
    trec      = f"{d[:4]}.{d[4:6]}.{d[6:8]}_{t[:2]}:{t[2:4]}:{t[4:6]}_TAI"
    return fits_name, harpnum, trec


def collect_pending(jpg_dir: str, out_dir: str):
    """
    Devuelve dict: harpnum → list of (fits_name, trec, dest_path)
    Solo incluye archivos que aún no existen en out_dir.
    """
    out_path = Path(out_dir)
    harps    = defaultdict(list)
    skipped  = 0

    fnames = [f for f in os.listdir(jpg_dir)
              if f.endswith('.jpg') and f != 'desktop.ini']

    for fname in fnames:
        parsed = parse_fname(fname)
        if not parsed:
            continue
        fits_name, harpnum, trec = parsed
        dest = out_path / fits_name
        if dest.exists() and dest.stat().st_size > 0:
            skipped += 1
            continue
        harps[harpnum].append((fits_name, trec, dest))

    return harps, skipped, len(fnames)


def query_harp_urls(client, harpnum: int, trecs: list,
                    max_retries: int = 3) -> dict:
    """
    Consulta JSOC para obtener las URLs directas de los FITS de un HARPNUM.
    Devuelve dict: trec → url (o vacío si falla).
    """
    trec_sorted = sorted(trecs)
    t_min, t_max = trec_sorted[0], trec_sorted[-1]
    query = f"hmi.sharp_720s[{harpnum}][{t_min}-{t_max}]"

    for attempt in range(max_retries):
        try:
            keys, segs = client.query(query, key='T_REC', seg='magnetogram')
            result = {}
            for i, row in keys.iterrows():
                trec = row['T_REC']
                path = segs.loc[i, 'magnetogram']
                if path and path != 'NA':
                    result[trec] = JSOC_BASE + path
            return result
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return {}


def download_file(url: str, dest: Path, max_retries: int = 3) -> bool:
    """Descarga un archivo con reintentos. Devuelve True si tuvo éxito."""
    tmp = dest.with_suffix('.tmp')
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size > 0:
                shutil.move(str(tmp), str(dest))
                return True
        except Exception:
            if tmp.exists():
                tmp.unlink()
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return False


def process_harp(client, harpnum: int, entries: list) -> tuple:
    """Procesa un HARPNUM: consulta URLs y descarga. Devuelve (ok, fail)."""
    trecs    = [e[1] for e in entries]
    trec_map = {e[1]: (e[0], e[2]) for e in entries}

    url_map = query_harp_urls(client, harpnum, trecs)

    ok = fail = 0
    for trec, (fits_name, dest) in trec_map.items():
        if dest.exists() and dest.stat().st_size > 0:
            ok += 1
            continue
        url = url_map.get(trec)
        if not url:
            fail += 1
            continue
        if download_file(url, dest):
            ok += 1
        else:
            fail += 1

    return ok, fail


def main():
    parser = argparse.ArgumentParser(
        description='Descarga FITS de magnetograma desde JSOC (sin cola de export)'
    )
    parser.add_argument('--jpg-dir',
        default=os.path.join(os.environ.get('SFMM_DATA', ''), 'magnetogram_jpg'))
    parser.add_argument('--out-dir',
        default=os.environ.get('SFMM_FITS', 'data/magnetogram_fits'))
    parser.add_argument('--workers', type=int, default=8,
        help='HARPNUMs procesados en paralelo (default: 8)')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sep = '=' * 62
    print(sep)
    print('  JSOC FITS Downloader — descarga directa sin export')
    print(sep)

    print(f'\n[1/3] Escaneando {args.jpg_dir} ...')
    harps, skipped, total_files = collect_pending(args.jpg_dir, args.out_dir)
    total_pending = sum(len(v) for v in harps.values())

    print(f'      Total archivos en dataset:    {total_files:,}')
    print(f'      Ya descargados (saltar):       {skipped:,}')
    print(f'      Pendientes de descarga:        {total_pending:,}')
    print(f'      HARPNUMs pendientes:           {len(harps):,}')
    print(f'      Destino:                       {args.out_dir}')
    est_gb = total_pending * 135 / 1024 / 1024
    print(f'      Espacio estimado:              ~{est_gb:.1f} GB  (~135 KB/archivo)')

    if args.dry_run:
        print('\n[dry-run] No se descarga nada.')
        return

    if total_pending == 0:
        print('\nTodos los archivos ya están descargados.')
        return

    print(f'\n[2/3] Conectando a jsoc.stanford.edu ...')
    try:
        client = drms.Client()
        print('      Conexión OK')
    except Exception as e:
        print(f'      ERROR: {e}')
        sys.exit(1)

    print(f'\n[3/3] Descargando {len(harps):,} HARPNUMs con {args.workers} workers ...\n')

    total_ok = total_fail = 0
    lock = threading.Lock()

    pbar = tqdm(total=total_pending, unit='archivo', ncols=72,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}  {rate_fmt}]')

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_harp, client, harpnum, entries): harpnum
            for harpnum, entries in harps.items()
        }
        for fut in as_completed(futures):
            ok, fail = fut.result()
            with lock:
                total_ok   += ok
                total_fail += fail
            pbar.update(ok + fail)
            pbar.set_postfix(ok=total_ok, fail=total_fail)

    pbar.close()

    print(f'\n{sep}')
    print(f'  COMPLETADO')
    print(f'  Descargados OK:  {total_ok:,}')
    print(f'  Fallos:          {total_fail:,}')
    print(f'  Ya existían:     {skipped:,}')
    du = shutil.disk_usage(args.out_dir)
    print(f'  Espacio usado:   {du.used / 1e9:.1f} GB  (libre: {du.free / 1e9:.1f} GB)')
    print(sep)

    if total_fail > 0:
        print(f'\n  AVISO: {total_fail} archivos fallaron.')
        print('  Vuelve a ejecutar el mismo comando para reintentar.')


if __name__ == '__main__':
    main()

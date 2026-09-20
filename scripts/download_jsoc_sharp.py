#!/usr/bin/env python3
"""
download_jsoc_sharp.py — Descarga los 21 parámetros SHARP físicos de JSOC/HMI
y genera para_flare_21params.txt a partir del para_flare.txt existente.

Resultado: mismo formato que para_flare.txt pero con 21 columnas de parámetros
en lugar de 10, añadiendo los top-predictores faltantes (TOTUSJH, TOTUSJZ, etc.)

Uso:
  python scripts/download_jsoc_sharp.py \\
      --para-flare /ruta/a/para_flare.txt \\
      --output     /ruta/a/para_flare_21params.txt \\
      --cache      outputs/jsoc_cache.csv \\
      --delay      0.3
"""

import os
import re
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

try:
    import drms
except ImportError:
    print("ERROR: drms no instalado.")
    print("  Ejecuta: pip install drms")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    os.system(f"{sys.executable} -m pip install tqdm -q")
    from tqdm import tqdm

# ── Parámetros físicos disponibles en hmi.sharp_720s ──────────────────────
# Verificados contra la serie real en JSOC (junio 2026).
# NOTA: TOTBSQ (#2 Bobra 2015) NO está disponible como keyword en esta serie.
# NOTA: MEANGBR (radial) tampoco está disponible.
SHARP_PARAMS = [
    # Top predictores por F-score de Fisher (Bobra & Couvidat 2015)
    'TOTUSJH',   # #1  Helicidad de corriente total sin signo
    'TOTPOT',    # #3  Energía libre magnética fotosférica total
    'TOTUSJZ',   # #4  Corriente vertical total sin signo

    # Flujo magnético
    'USFLUX',    #     Flujo magnético total sin signo (vectorial)
    'USFLUXL',   #     Flujo sin signo (línea de visión)

    # Gradientes de campo
    'MEANGBH',   #     Gradiente medio del campo horizontal
    'MEANGBL',   #     Gradiente medio del campo longitudinal
    'MEANGBT',   #     Gradiente medio del campo total
    'MEANGBZ',   #     Gradiente medio del campo vertical

    # Helicidad y corriente
    'MEANJZH',   #     Helicidad de corriente media (contribución Bz)
    'MEANJZD',   #     Densidad de corriente vertical media
    'ABSNJZH',   #     |Helicidad de corriente neta|
    'SAVNCPP',   #     Corriente neta por polaridad (suma de módulos)
    'MEANALP',   #     Parámetro de twist α medio

    # Energía
    'MEANPOT',   #     Densidad de energía magnética libre media

    # Cizallamiento (shear)
    'MEANSHR',   #     Ángulo de cizallamiento medio
    'SHRGT45',   #     Área con shear > 45°

    # Geometría de la región activa
    'AREA_ACR',  #     Área de píxeles de campo fuerte
    'NACR',      #     Número de píxeles de campo fuerte
    'MEANGAM',   #     Ángulo medio del campo respecto a la vertical

    # Línea de inversión de polaridad
    'R_VALUE',   #     Flujo magnético cerca de la PIL
]
# Total: 21 parámetros físicos
N_PARAMS = len(SHARP_PARAMS)

# Los 10 actuales en para_flare.txt (para referencia y verificación)
CURRENT_10 = ['USFLUX', 'MEANGBZ', 'MEANGBT', 'MEANPOT', 'SHRGT45',
              'TOTPOT', 'SAVNCPP', 'ABSNJZH', 'AREA_ACR', 'NACR']

NEW_PARAMS = [p for p in SHARP_PARAMS if p not in CURRENT_10]


def parse_filename(filename: str):
    """Extrae (harpnum, T_REC en formato JSOC) del nombre de archivo SHARP."""
    m = re.search(r'hmi\.sharp_720s\.(\d+)\.(\d{8})_(\d{6})_TAI', filename)
    if not m:
        return None, None
    harpnum = int(m.group(1))
    d, t = m.group(2), m.group(3)
    # Formato JSOC confirmado: 2010.05.04_16:00:00_TAI
    trec = f"{d[:4]}.{d[4:6]}.{d[6:8]}_{t[:2]}:{t[2:4]}:{t[4:6]}_TAI"
    return harpnum, trec


def load_para_flare(path: str) -> pd.DataFrame:
    """Carga para_flare.txt y extrae harpnum + T_REC de los filenames."""
    rows = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            filename = parts[0]
            label = int(parts[11])
            harpnum, trec = parse_filename(filename)
            if harpnum is None:
                continue
            rows.append({
                'filename': filename,
                'harpnum': harpnum,
                'T_REC': trec,
                'label': label,
            })
    df = pd.DataFrame(rows)
    return df


def query_harpnum(client, harpnum: int, trec_set: set,
                  max_retries: int = 3, delay: float = 0.3) -> pd.DataFrame:
    """
    Consulta JSOC para un HARPNUM en el rango de tiempo que cubre los T_RECs
    solicitados. Devuelve DataFrame con T_REC + SHARP_PARAMS.
    """
    trec_sorted = sorted(trec_set)
    t_min = trec_sorted[0]
    t_max = trec_sorted[-1]
    keys = ['T_REC'] + SHARP_PARAMS
    query_str = f"hmi.sharp_720s[{harpnum}][{t_min}-{t_max}]"

    for attempt in range(max_retries):
        try:
            df = client.query(query_str, key=','.join(keys))
            if df is not None and len(df) > 0:
                return df
            return pd.DataFrame(columns=keys)
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return pd.DataFrame(columns=keys)

    return pd.DataFrame(columns=keys)


def main():
    parser = argparse.ArgumentParser(
        description='Descarga 21 parámetros SHARP físicos de JSOC y genera para_flare_21params.txt'
    )
    parser.add_argument('--para-flare', required=True,
                        help='Ruta al para_flare.txt original (10 params)')
    parser.add_argument('--output', required=True,
                        help='Ruta del archivo de salida (21 params)')
    parser.add_argument('--cache', default='outputs/jsoc_cache.csv',
                        help='Archivo CSV de caché para reanudar (default: outputs/jsoc_cache.csv)')
    parser.add_argument('--delay', type=float, default=0.3,
                        help='Segundos entre consultas JSOC (default: 0.3)')
    args = parser.parse_args()

    sep = '=' * 62
    print(sep)
    print('  JSOC SHARP Downloader')
    print(f'  Descargando {N_PARAMS} parámetros físicos de hmi.sharp_720s')
    print(sep)

    # ── 1. Cargar para_flare.txt original ──────────────────────────────────
    print(f'\n[1/5] Leyendo {args.para_flare} ...')
    df_orig = load_para_flare(args.para_flare)
    n_harps = df_orig['harpnum'].nunique()
    print(f'      {len(df_orig):,} filas  |  {n_harps:,} HARPNUMs únicos')
    print(f'\n      Parámetros actuales  (10): {", ".join(CURRENT_10)}')
    print(f'      Parámetros nuevos    ({len(NEW_PARAMS):2d}): {", ".join(NEW_PARAMS)}')
    print(f'      Total tras descarga  ({N_PARAMS:2d}): {", ".join(SHARP_PARAMS)}')

    # ── 2. Caché ────────────────────────────────────────────────────────────
    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    df_cache = pd.DataFrame()
    cached_harps = set()

    if cache_path.exists():
        print(f'\n[2/5] Caché encontrada: {cache_path}')
        df_cache = pd.read_csv(cache_path)
        cached_harps = set(df_cache['harpnum'].unique())
        print(f'      {len(cached_harps):,} HARPNUMs ya descargados  '
              f'({len(cached_harps)*100//n_harps}% completado)')
    else:
        print(f'\n[2/5] Sin caché previa — descarga desde cero')

    # ── 3. Conectar a JSOC ─────────────────────────────────────────────────
    print(f'\n[3/5] Conectando a jsoc.stanford.edu ...')
    try:
        client = drms.Client()
        print('      Conexión OK')
    except Exception as e:
        print(f'      ERROR de conexión: {e}')
        sys.exit(1)

    # ── 4. Descarga por HARPNUM ─────────────────────────────────────────────
    harp_trecs = defaultdict(set)
    for _, row in df_orig.iterrows():
        if row['harpnum'] not in cached_harps:
            harp_trecs[row['harpnum']].add(row['T_REC'])

    pending = dict(harp_trecs)
    n_pending = len(pending)
    eta_min = n_pending * args.delay / 60
    eta_max = n_pending * 1.5 / 60

    print(f'\n[4/5] Descargando {n_pending:,} HARPNUMs pendientes ...')
    print(f'      ETA estimado: {eta_min:.0f}–{eta_max:.0f} minutos\n')

    new_rows = []
    errors = 0
    save_interval = 100  # guardar caché cada N HARPNUMs

    pbar = tqdm(
        total=n_pending,
        unit='HARP',
        ncols=72,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}  {rate_fmt}]'
    )

    for i, (harpnum, trec_set) in enumerate(pending.items()):
        df_jsoc = query_harpnum(client, harpnum, trec_set, delay=args.delay)

        if len(df_jsoc) == 0:
            # Sin datos JSOC: crear filas con NaN para este HARPNUM
            df_sub = df_orig[df_orig['harpnum'] == harpnum][
                ['filename', 'harpnum', 'T_REC', 'label']
            ].copy()
            for p in SHARP_PARAMS:
                df_sub[p] = np.nan
            new_rows.append(df_sub)
            errors += 1
            pbar.set_postfix(errors=errors, status=f'HARP {harpnum} sin datos')
        else:
            # Filtrar a los T_RECs exactos que necesitamos
            df_jsoc = df_jsoc[df_jsoc['T_REC'].isin(trec_set)].copy()

            # Merge con el df_orig para recuperar filename y label
            df_sub = df_orig[df_orig['harpnum'] == harpnum][
                ['filename', 'harpnum', 'T_REC', 'label']
            ].copy()
            merged = df_sub.merge(
                df_jsoc[['T_REC'] + SHARP_PARAMS],
                on='T_REC',
                how='left'
            )
            new_rows.append(merged)
            pbar.set_postfix(
                errors=errors,
                rows=sum(len(r) for r in new_rows)
            )

        time.sleep(args.delay)
        pbar.update(1)

        # Guardar caché incremental
        if (i + 1) % save_interval == 0 and new_rows:
            df_partial = pd.concat(new_rows, ignore_index=True)
            if len(df_cache) > 0:
                df_partial = pd.concat([df_cache, df_partial], ignore_index=True)
            df_partial.to_csv(cache_path, index=False)
            pbar.write(f'  → Caché guardada: {len(df_partial):,} filas en {cache_path}')

    pbar.close()

    # ── 5. Combinar, guardar caché final y escribir output ──────────────────
    print(f'\n[5/5] Generando {args.output} ...')

    df_new = pd.concat(new_rows, ignore_index=True) if new_rows else pd.DataFrame()
    df_final = pd.concat([df_cache, df_new], ignore_index=True) if len(df_cache) > 0 else df_new

    # Guardar caché completa
    df_final.to_csv(cache_path, index=False)

    # Estadísticas de NaN
    nan_total = df_final[SHARP_PARAMS].isna().sum().sum()
    nan_pct = 100 * nan_total / (len(df_final) * N_PARAMS)

    # Escribir para_flare_21params.txt
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, 'w') as fout:
        for _, row in df_final.iterrows():
            vals = []
            for p in SHARP_PARAMS:
                v = row.get(p, np.nan)
                vals.append('nan' if pd.isna(v) else f'{float(v):.6g}')
            fout.write(f"{row['filename']}  {'  '.join(vals)}  {int(row['label'])}\n")
            written += 1

    print(f'\n{sep}')
    print(f'  COMPLETADO')
    print(f'  Filas escritas:        {written:,}')
    print(f'  HARPNUMs procesados:   {n_harps:,}')
    print(f'  HARPNUMs con error:    {errors:,}  ({100*errors//n_harps}%)')
    print(f'  Valores NaN totales:   {nan_total:,}  ({nan_pct:.1f}%)')
    print(f'  Archivo generado:      {args.output}')
    print(f'  Caché guardada:        {cache_path}')
    print(f'{sep}')
    bobra_rank = {'TOTUSJH': '#1 Bobra 2015', 'TOTPOT': '#3 Bobra 2015',
                  'TOTUSJZ': '#4 Bobra 2015'}
    print()
    print('  Parámetros en el archivo de salida (en orden):')
    for i, p in enumerate(SHARP_PARAMS, 1):
        tag = '← NEW   ' if p in NEW_PARAMS else '        '
        rank = f'← {bobra_rank[p]}' if p in bobra_rank else ''
        print(f'  {i:2d}. {p:<12} {tag} {rank}')
    print(f'{sep}')


if __name__ == '__main__':
    main()

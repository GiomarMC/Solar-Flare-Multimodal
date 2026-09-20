"""
¿Cuánto de los parámetros SHARP es recuperable desde el magnetograma que ve la rama visual?

Motivación: imagen y escalares comparten identidad de archivo exacta — mismo HARP, mismos
16 T_REC, la misma clave indexa ambos (dataset_temporal_v*.py: frame_stems es una sola
lista; imagen = stem + '.fits.jpg', escalar = para_map[stem]). Pero la imagen es solo la
componente LOS, mientras que los SHARP se calculan del campo VECTORIAL. La pregunta
empírica es cuánto pesa esa diferencia.

Método: se calculan dos agregados TRIVIALES del JPG (sin aprender nada) y se correlacionan
con cada uno de los 21 SHARP mediante Spearman:

    sum|B|  = suma de |2p - 1|            proxy de flujo total sin signo
    area    = numero de pixeles |2p-1|>0.3 proxy de area activa

Si un parametro correlaciona alto con estos agregados, la rama visual ya lo tiene "gratis"
y no puede aportar nada nuevo a la fusion.

Uso:
    python graficos/redundancia_sharp_imagen.py
"""
import os
import sys
import numpy as np
from PIL import Image
from scipy.stats import spearmanr

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import ROOT, OUTDIR

BASE = os.environ.get("SFMM_DATA", "")
N_SAMPLE = 500
HORIZON = 48

# Orden de columnas en para_flare_21params.txt (dataset_temporal_v3.py:21)
SHARP_21 = ['TOTUSJH', 'TOTPOT', 'TOTUSJZ', 'USFLUX', 'USFLUXL',
            'MEANGBH', 'MEANGBL', 'MEANGBT', 'MEANGBZ',
            'MEANJZH', 'MEANJZD', 'ABSNJZH', 'SAVNCPP', 'MEANALP',
            'MEANPOT', 'MEANSHR', 'SHRGT45', 'AREA_ACR', 'NACR',
            'MEANGAM', 'R_VALUE']
# Los 17 que usa el modelo v3 (elimina 5=MEANGBH, 9=MEANJZH, 10=MEANJZD, 13=MEANALP)
KEEP_17 = [0, 1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20]


def main():
    lines = []
    def w(s=""):
        print(s); lines.append(s)

    pmap = {}
    with open(f"{BASE}/para_flare_21params.txt") as f:
        for line in f:
            p = line.split()
            if len(p) < 23:
                continue
            pmap[p[0].removesuffix('.fits')] = p[1:22]

    sd = f"{BASE}/Seq_Magnetogram/M{HORIZON}/Seqs16"
    split = f"{BASE}/Seq_Magnetogram/M{HORIZON}/Seq16_flare_Mclass_{HORIZON}h_Test.txt"
    entries = [l.split() for l in open(split) if len(l.split()) >= 2]

    rng = np.random.default_rng(0)
    stems = set()
    for i in rng.choice(len(entries), min(len(entries), N_SAMPLE + 200), replace=False):
        frames = [l.strip() for l in open(os.path.join(sd, entries[i][0])) if l.strip()]
        stems.add(frames[-1])
    stems = sorted(stems)[:N_SAMPLE]

    isum, iarea, sharp = [], [], []
    for s in stems:
        jpg = f"{BASE}/magnetogram_jpg/{s}.fits.jpg"
        if not os.path.exists(jpg) or s not in pmap:
            continue
        a = np.asarray(Image.open(jpg).convert("L"), dtype=np.float32) / 255.0
        sig = np.abs(2 * a - 1.0)
        isum.append(sig.sum())
        iarea.append((sig > 0.3).sum())
        sharp.append([float(x) if x != 'nan' else np.nan for x in pmap[s]])
    isum, iarea, sharp = np.array(isum), np.array(iarea), np.array(sharp)

    w("=" * 78)
    w("  REDUNDANCIA SHARP <-> IMAGEN")
    w("  ¿Cuánto de cada parámetro físico es recuperable desde el magnetograma LOS?")
    w("=" * 78)
    w(f"  n = {len(isum)} frames (último frame de secuencias del test {HORIZON}h)")
    w("  Agregados de la imagen: sum|B| = suma de |2p-1| ; area = #pixeles |2p-1|>0.3")
    w("  (dos líneas de numpy, sin aprender nada)")
    w()
    w(f"  {'#':>3s} {'SHARP':<10s}{'en modelo v3':>14s}{'rho(sum|B|)':>13s}{'rho(area)':>11s}{'|rho| max':>11s}")
    rows = []
    for j, nm in enumerate(SHARP_21):
        v = sharp[:, j]; m = ~np.isnan(v)
        if m.sum() < 50:
            continue
        r1, _ = spearmanr(isum[m], v[m]); r2, _ = spearmanr(iarea[m], v[m])
        mx = max(abs(r1), abs(r2))
        rows.append((j, nm, j in KEEP_17, r1, r2, mx))
    for j, nm, k, r1, r2, mx in sorted(rows, key=lambda r: -r[5]):
        w(f"  {j:>3d} {nm:<10s}{('sí' if k else 'NO'):>14s}{r1:>13.3f}{r2:>11.3f}{mx:>11.3f}")

    used = [r for r in rows if r[2]]
    drop = [r for r in rows if not r[2]]
    w()
    w("-" * 78)
    w(f"  De los {len(used)} SHARP que SÍ entran al BiLSTM de 17:")
    for thr in (0.7, 0.8, 0.9):
        c = sum(1 for r in used if r[5] >= thr)
        w(f"    |rho| >= {thr}: {c}/{len(used)}  ({100*c/len(used):.0f} %)")
    w(f"    mediana |rho| = {np.median([r[5] for r in used]):.3f}")
    w()
    w(f"  De los {len(drop)} que v3 DESCARTA por baja d de Cohen univariada:")
    for j, nm, k, r1, r2, mx in sorted(drop, key=lambda r: r[5]):
        w(f"    {nm:<10s} |rho| = {mx:.3f}")
    w(f"    mediana |rho| = {np.median([r[5] for r in drop]):.3f}")
    w()
    w("  CONCLUSIÓN: los 4 parámetros excluidos del modelo son justamente los MENOS")
    w("  recuperables desde la imagen — es decir, la selección univariada descartó")
    w("  precisamente la información potencialmente NO redundante con la rama visual.")
    w("  La redundancia observada es en parte consecuencia de esa selección, no solo")
    w("  de que ambas modalidades compartan fuente física.")
    w()
    w("  Contraste a testear: BiLSTM-21 (dataset_temporal_v5) y recálculo del techo")
    w("  con LSTM_MODEL=lstm21 python graficos/diversidad_ramas.py")

    out = os.path.join(OUTDIR, "redundancia_sharp_imagen.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

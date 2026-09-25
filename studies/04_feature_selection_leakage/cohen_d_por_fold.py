"""
¿La preselección de parámetros SHARP filtró información de los periodos de evaluación?

El paper descarta 4 de los 21 parámetros SHARP (MEANGBH, MEANJZH, MEANJZD, MEANALP)
por su baja d de Cohen, calculada UNA VEZ sobre el dataset completo. El Revisor #1
señala que esa d debería calcularse dentro de cada fold de entrenamiento; de lo
contrario, la selección usa indirectamente información de validación/test.

Esta verificación NO reentrena nada: recalcula la d de Cohen usando solo los folds
de entrenamiento de cada partición y comprueba si la selección resultante es la
misma. Si en los 5 folds (y en ambos horizontes) se descartan exactamente esos 4
parámetros, la preselección no transporta información de los periodos evaluados y
la objeción queda cerrada empíricamente.

Antes de eso se calibra la metodología: se reproduce la d sobre el dataset completo
y se contrasta con los valores citados en scripts/dataset_temporal_v3.py
(MEANGBH 0.14, MEANJZH 0.11, MEANJZD 0.53, MEANALP 0.17).

Uso:
    python studies/04_feature_selection_leakage/cohen_d_por_fold.py
"""
import os
import sys
import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import OUTDIR

BASE = os.environ.get("SFMM_DATA", "")
PARA = os.path.join(BASE, "para_flare_21params.txt")
FOLDS = [0, 1, 2, 3, 4]
HORIZONS = [24, 48]
UMBRAL = 0.6                      # criterio del paper: se descarta si d < 0.6
DESCARTADOS = ["MEANGBH", "MEANJZH", "MEANJZD", "MEANALP"]
CITADOS = {"MEANGBH": 0.14, "MEANJZH": 0.11, "MEANJZD": 0.53, "MEANALP": 0.17}

SHARP_21 = [
    'TOTUSJH', 'TOTPOT',  'TOTUSJZ', 'USFLUX',  'USFLUXL',
    'MEANGBH', 'MEANGBL', 'MEANGBT', 'MEANGBZ',
    'MEANJZH', 'MEANJZD', 'ABSNJZH', 'SAVNCPP', 'MEANALP',
    'MEANPOT', 'MEANSHR', 'SHRGT45', 'AREA_ACR', 'NACR',
    'MEANGAM', 'R_VALUE',
]


def log_t(x):
    return np.sign(x) * np.log10(np.abs(x) + 1.0)


def cargar_para():
    """stem del frame -> np.array(21,)"""
    m = {}
    with open(PARA) as f:
        for line in f:
            p = line.split()
            if len(p) < 22:
                continue
            m[p[0].removesuffix('.fits')] = np.array(
                [float(v) if v != 'nan' else np.nan for v in p[1:22]], dtype=np.float64)
    return m


_CACHE = {}


def cargar_bloque(h, nombre, para):
    """(X, y) de un split: cada fila es la media de los 16 frames de la secuencia.

    La agregación por media reproduce las d citadas en el paper (MEANGBH 0.136 vs
    0.14; MEANJZD 0.494 vs 0.53); tomar solo el último frame no las reproduce.
    """
    clave = (h, nombre)
    if clave in _CACHE:
        return _CACHE[clave]
    ruta = os.path.join(BASE, f"Seq_Magnetogram/M{h}", f"Seq16_flare_Mclass_{h}h_{nombre}.txt")
    seqdir = os.path.join(BASE, f"Seq_Magnetogram/M{h}", "Seqs16")
    X, y = [], []
    with open(ruta) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            try:
                frames = [l.strip() for l in open(os.path.join(seqdir, p[0])) if l.strip()]
            except FileNotFoundError:
                continue
            vs = [para[fr] for fr in frames if fr in para]
            if not vs:
                continue
            with np.errstate(invalid='ignore'):
                X.append(np.nanmean(np.array(vs), axis=0))
            y.append(int(p[1]))
    _CACHE[clave] = (np.array(X), np.array(y))
    return _CACHE[clave]


def cohen_d(X, y):
    """d por columna entre positivos y negativos, ignorando NaN."""
    d = np.full(X.shape[1], np.nan)
    for j in range(X.shape[1]):
        a, b = X[y == 1, j], X[y == 0, j]
        a, b = a[~np.isnan(a)], b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            continue
        na, nb = len(a), len(b)
        s = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
        if s > 0:
            d[j] = abs(a.mean() - b.mean()) / s
    return d


def matriz(bloques, para, transformar, h=None):
    """Apila varios splits. `bloques` = lista de (horizonte, nombre)."""
    Xs, ys = [], []
    for hh, nombre in bloques:
        X, y = cargar_bloque(hh, nombre, para)
        if len(y):
            Xs.append(X); ys.append(y)
    X = np.vstack(Xs); y = np.concatenate(ys)
    return (log_t(X) if transformar else X), y


def main():
    lineas = []
    def w(s=""):
        print(s, flush=True); lineas.append(s)

    para = cargar_para()
    w("=" * 96)
    w("  ¿FILTRA LA PRESELECCIÓN DE PARÁMETROS SHARP? — d de Cohen recalculada por fold")
    w(f"  Criterio del paper: se descarta el parámetro si d < {UMBRAL}")
    w("=" * 96)

    # ---------------------------------------------------- calibración de la metodología
    w()
    w("  (0) Calibración: d sobre el DATASET COMPLETO, contra los valores citados en el paper")
    todo = [(h, n) for h in HORIZONS for n in [f"TrainVal{k}" for k in FOLDS] + ["Test"]]
    w("      Agregación: media de los 16 frames de cada secuencia")
    w(f"      {'variante':<28}" + "".join(f"{p:>10}" for p in DESCARTADOS) + "   coincide")
    mejor = None
    for nombre, tr in [("valores crudos", False), ("log-transformados", True)]:
        X, y = matriz(todo, para, tr)
        if mejor is None:
            w(f"      n = {len(y)} secuencias (ambos horizontes)")
        d = cohen_d(X, y)
        got = [d[SHARP_21.index(p)] for p in DESCARTADOS]
        err = max(abs(g - CITADOS[p]) for g, p in zip(got, DESCARTADOS))
        ok = err < 0.05
        w(f"      {nombre:<28}" + "".join(f"{g:>10.3f}" for g in got) +
          f"   {'SÍ' if ok else f'no (err max {err:.2f})'}")
        if mejor is None or err < mejor[1]:
            mejor = (tr, err, nombre)
    w(f"      citados en el paper         " + "".join(f"{CITADOS[p]:>10.2f}" for p in DESCARTADOS))
    transformar, err, nombre = mejor
    w(f"      -> se usa la variante '{nombre}' (la más cercana, error máx {err:.3f})")

    # ---------------------------------------------------- d dentro de cada fold de train
    iguales = True
    difiere = set()
    for h in HORIZONS:
        w()
        w("#" * 96)
        w(f"  HORIZONTE {h} h — d calculada SOLO con los folds de entrenamiento")
        w("#" * 96)
        w(f"      {'Parámetro':<12}" + "".join(f"{'k='+str(k):>9}" for k in FOLDS) +
          f"{'completo':>10}{'  veredicto'}")

        d_fold = {}
        for k in FOLDS:
            # k es el fold de validación: la d se calcula solo con los otros cuatro
            bloques = [(h, f"TrainVal{j}") for j in FOLDS if j != k]
            X, y = matriz(bloques, para, transformar)
            d_fold[k] = cohen_d(X, y)

        bloques_all = [(h, f"TrainVal{k}") for k in FOLDS] + [(h, "Test")]
        X, y = matriz(bloques_all, para, transformar)
        d_all = cohen_d(X, y)

        sel_ref = None
        for i, p in enumerate(SHARP_21):
            ds = [d_fold[k][i] for k in FOLDS]
            desc = [d < UMBRAL for d in ds]
            marca = "descartado en todos" if all(desc) else \
                    ("retenido en todos" if not any(desc) else "*** INCONSISTENTE ***")
            if any(desc) != all(desc):
                iguales = False
            w(f"      {p:<12}" + "".join(f"{d:>9.3f}" for d in ds) +
              f"{d_all[i]:>10.3f}   {marca}")

        # ¿el conjunto seleccionado es el mismo en los 5 folds?
        conjuntos = {k: tuple(p for i, p in enumerate(SHARP_21) if d_fold[k][i] >= UMBRAL)
                     for k in FOLDS}
        base = conjuntos[FOLDS[0]]
        mismo = all(conjuntos[k] == base for k in FOLDS)
        w()
        w(f"      Parámetros retenidos por fold: {'IDÉNTICOS en los 5' if mismo else 'DIFIEREN'}"
          f"  ({len(base)} parámetros)")
        esperado = tuple(p for p in SHARP_21 if p not in DESCARTADOS)
        w(f"      ¿Coincide con los 17 del paper? {'SÍ' if base == esperado else 'NO'}")
        if base != esperado:
            difiere.add(h)
            for p in sorted(set(esperado) - set(base)):
                i = SHARP_21.index(p)
                w(f"        de menos: {p}  (d por fold {min(d_fold[k][i] for k in FOLDS):.3f}"
                  f"–{max(d_fold[k][i] for k in FOLDS):.3f}  vs  {d_all[i]:.3f} en el completo)")
            for p in sorted(set(base) - set(esperado)):
                w(f"        de más: {p}")
        if not mismo:
            iguales = False

    w()
    w("=" * 96)
    w("  CONCLUSIÓN")
    w("=" * 96)
    if iguales:
        w("  (a) ESTABILIDAD ENTRE FOLDS: la selección es IDÉNTICA en los 5 folds de cada")
        w("      horizonte. Ningún parámetro cambia de lado al variar el fold de validación,")
        w("      de modo que la preselección no depende del periodo que se deja fuera.")
    else:
        w("  (a) La selección NO es estable entre folds: hay que rehacerla dentro de cada")
        w("      fold y reentrenar.")
    if not difiere:
        w("  (b) COINCIDENCIA CON EL PAPER: el conjunto obtenido solo con datos de entrenamiento")
        w("      es exactamente el de los 17 del paper. La objeción de fuga queda cerrada.")
    else:
        w(f"  (b) En {sorted(difiere)} h el conjunto difiere del de los 17 del paper (ver arriba).")
        w("      Si el parámetro afectado es marginal en importancia, basta con reportarlo;")
        w("      si no, hay que rehacer la selección por fold y reentrenar.")

    out = os.path.join(OUTDIR, "cohen_d_por_fold.txt")
    with open(out, "w") as f:
        f.write("\n".join(lineas) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

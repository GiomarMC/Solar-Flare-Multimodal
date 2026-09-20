"""
Utilidades compartidas para agregar los 5 folds y etiquetar significancia.

Estaban definidas dentro de studies/03_los_derived_ablation/ablacion_hijos.py, pero
las usan también otros estudios, así que viven aquí para no acoplar una carpeta de
estudio con otra.
"""
import numpy as np

from bootstrap_ci import tss_point


def ens(folds):
    """Ensemble de folds: (probabilidades de test promediadas, tau medio, etiquetas)."""
    return (np.mean([p for p, _, _ in folds], axis=0),
            float(np.mean([t for _, _, t in folds])), folds[0][1])


def tss_e(e):
    """TSS de una tupla (probs, tau, etiquetas) tal como la devuelve ens()."""
    probs, tau, y = e
    return tss_point(probs, y, tau)


def sig_label(lo, hi, better="MEJOR", worse="peor"):
    """Veredicto a partir de un intervalo de confianza de la diferencia."""
    return f"{better} (sig.)" if lo > 0 else f"{worse} (sig.)" if hi < 0 else "sin diferencia"

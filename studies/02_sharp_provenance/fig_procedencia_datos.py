"""
Procedencia de los datos: dónde se bifurcan la rama visual (FITS de magnetograma)
y la rama física (parámetros SHARP).

El punto clave que ninguna figura publicada muestra: la bifurcación NO ocurre
"después de construir las imágenes", sino aguas arriba, en el vector de Stokes.
La rama visual usa solo polarización circular (I±V); la rama física usa el Stokes
completo (I,Q,U,V) e incluye la polarización lineal Q,U, que mide el campo
transversal y está físicamente ausente del magnetograma de línea de visión.

Dos dependencias cruzadas se marcan en naranja:
  (a) la máscara `bitmap` que decide qué píxeles entran al cálculo de los índices
      SHARP se deriva del campo LoS -> soporte espacial compartido;
  (b) 5 de los 17 parámetros del BiLSTM no son índices vectoriales: AREA_ACR y
      NACR (keywords HARP, Bobra et al. 2014 Tabla A.8) y USFLUXL, MEANGBL,
      R_VALUE (Bobra et al. 2021 §3.2) se calculan sobre el campo LoS.

Fuentes: Hoeksema et al. 2014 (SoPh 289, 3483; Fig. 2 y §2.3),
         Bobra et al. 2014 (SoPh 289, 3549; §4, Tabla 3, Tablas A.7/A.8),
         Couvidat et al. 2012/2016 (algoritmo MDI-like).

Genera RedaccionIEEE/images/procedencia_datos.pdf (+ graficos/procedencia_datos.png).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "RedaccionIEEE", "images")
OUT = os.path.join(ROOT, "graficos")

C_TRUNK = "#6a3d9a"   # tronco común (fotones -> Stokes)
C_VIS   = "#1f77b4"   # rama visual (LoS)
C_PHY   = "#d62728"   # rama física (vectorial)
C_DEP   = "#e08214"   # dependencias cruzadas entre ramas
C_TXT   = "#1a1a1a"


def box(ax, x, y, w, h, text, edge, face, fs=8.5, weight="normal", lw=1.4, z=4):
    """Caja redondeada centrada en (x, y)."""
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.16",
        facecolor=face, edgecolor=edge, lw=lw, zorder=z))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            color=C_TXT, weight=weight, zorder=z + 1, linespacing=1.45)


def arrow(ax, p0, p1, color, lw=1.6, style="-", rad=0.0, z=3):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=13,
        color=color, lw=lw, linestyle=style, zorder=z,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2, shrinkB=2))


def label(ax, x, y, text, color, fs=7.8, weight="bold", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            weight=weight, zorder=8, linespacing=1.4,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92))


def main():
    fig, ax = plt.subplots(figsize=(11.4, 8.6))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(-0.4, 13.8)
    ax.axis("off")

    XL, XR = 4.0, 12.2          # centros de columna
    FL, FR = "#eef5fb", "#fdeeee"

    # ---------------------------------------------------------- tronco común
    ax.add_patch(Circle((7.0, 13.0), 0.40, facecolor="#fdd49e",
                        edgecolor="#b35806", lw=1.6, zorder=5))
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        ax.plot([7.0 + 0.48 * np.cos(ang), 7.0 + 0.68 * np.cos(ang)],
                [13.0 + 0.48 * np.sin(ang), 13.0 + 0.68 * np.sin(ang)],
                color="#b35806", lw=1.1, zorder=4)
    ax.text(8.0, 13.0, "Sol — fotosfera\nlínea Fe I 6173 Å", ha="left", va="center",
            fontsize=9, weight="bold", color=C_TXT, linespacing=1.4)

    arrow(ax, (7.0, 12.45), (7.0, 12.05), C_TRUNK, lw=1.8)
    box(ax, 8.1, 11.58, 7.6, 0.92,
        "SDO/HMI — filtrogramas\n6 longitudes de onda × 6 estados de polarización",
        C_TRUNK, "#f4eefa", fs=9)

    arrow(ax, (8.1, 11.10), (8.1, 10.62), C_TRUNK, lw=1.8)
    box(ax, 8.1, 10.10, 7.6, 1.0,
        "hmi.S_720s — vector de Stokes  [ I , Q , U , V ]\n"
        "(calibración polarimétrica; Hoeksema+2014, Fig. 2)",
        C_TRUNK, "#ead9f5", fs=9, weight="bold", lw=2.2)

    ax.annotate("LA BIFURCACIÓN\nOCURRE AQUÍ\n— antes de existir\nninguna imagen —",
                xy=(11.98, 10.10), xytext=(14.9, 10.10),
                ha="center", va="center", fontsize=8.2, weight="bold", color=C_TRUNK,
                arrowprops=dict(arrowstyle="-|>", color=C_TRUNK, lw=1.6),
                zorder=9, linespacing=1.5)

    # ------------------------------------------------------- rama izq (LoS)
    arrow(ax, (6.3, 9.58), (XL, 8.62), C_VIS, lw=2.0)
    label(ax, 2.55, 9.14, "solo $I \\pm V$\n(polarización CIRCULAR)", C_VIS)

    box(ax, XL, 8.10, 7.0, 0.98,
        "Algoritmo MDI-like  (Couvidat+2012)\n"
        "$B_{los}\\ \\propto\\ \\Delta$Doppler( RCP $-$ LCP )",
        C_VIS, FL)
    arrow(ax, (XL, 7.61), (XL, 7.22), C_VIS)

    box(ax, XL, 6.78, 7.0, 0.80,
        "hmi.M_720s — magnetograma de línea de visión (disco completo)", C_VIS, FL)
    arrow(ax, (XL, 6.38), (XL, 5.99), C_VIS)

    box(ax, XL, 5.55, 7.0, 0.80,
        "recorte a la caja HARP  $\\rightarrow$  segmento  magnetogram", C_VIS, FL)
    arrow(ax, (XL, 5.15), (XL, 4.76), C_VIS)

    box(ax, XL, 4.24, 7.0, 0.96,
        "hmi.sharp_720s.NNNN.YYYYMMDD_HHMMSS_TAI.magnetogram.fits\n"
        "$\\bf{tus\\ imágenes}$  (float32 → JPG 8 bits)",
        C_VIS, "#d6e6f4", lw=2.0)
    arrow(ax, (XL, 3.76), (XL, 3.36), C_VIS, lw=2.0)

    box(ax, XL, 2.95, 5.4, 0.82, "RAMA VISUAL — Swin3D-T",
        C_VIS, "#bcd9ef", fs=10, weight="bold", lw=2.0)

    # ---------------------------------------------------- rama der (vector)
    arrow(ax, (9.9, 9.58), (XR, 8.62), C_PHY, lw=2.0)
    label(ax, 14.0, 9.14, "$I,\\ Q,\\ U,\\ V$ completo\n"
                          "(incluye $Q,U$ = polariz. LINEAL)", C_PHY)

    box(ax, XR, 8.10, 7.0, 0.98,
        "Inversión Milne–Eddington  VFISV\n"
        "$\\rightarrow$ $|B|$,  inclinación $\\gamma$,  acimut $\\phi$",
        C_PHY, FR)
    arrow(ax, (XR, 7.61), (XR, 7.22), C_PHY)

    box(ax, XR, 6.78, 7.0, 0.80,
        "desambiguación del acimut a 180°  (minimum energy)", C_PHY, FR)
    arrow(ax, (XR, 6.38), (XR, 5.99), C_PHY)

    box(ax, XR, 5.55, 7.0, 0.80,
        "$B_r,\\ B_\\theta,\\ B_\\phi$  $\\rightarrow$  remapeo CEA  (hmi.sharp_cea_720s)",
        C_PHY, FR)
    arrow(ax, (XR, 5.15), (XR, 4.76), C_PHY)

    box(ax, XR, 4.24, 7.0, 0.96,
        "módulo SHARP — 16 índices vectoriales (Bobra+2014, Tabla 3)\n"
        "USFLUX · TOTPOT · MEANPOT · SHRGT45 · SAVNCPP · ABSNJZH · MEANGB*",
        C_PHY, "#f6d5d5", fs=8.0, lw=2.0)
    arrow(ax, (XR, 3.76), (XR, 3.36), C_PHY, lw=2.0)

    box(ax, XR, 2.95, 5.4, 0.82, "RAMA FÍSICA — BiLSTM",
        C_PHY, "#f2bcbc", fs=10, weight="bold", lw=2.0)

    # ------------------------------------ dependencias cruzadas (naranja)
    # (a) la máscara bitmap sale del campo LoS y filtra el cálculo vectorial
    arrow(ax, (7.60, 5.45), (9.35, 4.70), C_DEP, lw=1.8, style=(0, (4, 2.2)), rad=-0.30)
    ax.text(8.45, 5.30, "(a)", ha="center", va="center", fontsize=9,
            weight="bold", color=C_DEP, zorder=9)

    # (b) AREA_ACR y NACR nacen en la rama LoS y alimentan la rama física
    box(ax, 8.1, 1.72, 6.6, 0.88,
        "USFLUXL · MEANGBL · R_VALUE · AREA_ACR · NACR   (5 de 17)\n"
        "calculados sobre el campo $\\bf{LoS}$, NO sobre el vectorial",
        C_DEP, "#fdf0dd", fs=8.2, lw=1.8, z=6)
    arrow(ax, (7.58, 5.28), (7.95, 2.22), C_DEP, lw=1.8, style=(0, (4, 2.2)), rad=-0.16)
    arrow(ax, (10.6, 2.10), (11.3, 2.52), C_DEP, lw=1.8, style=(0, (4, 2.2)), rad=-0.20)
    ax.text(8.62, 3.40, "(b)", ha="center", va="center", fontsize=9,
            weight="bold", color=C_DEP, zorder=9)

    # ------------------------------------------------------------- pie
    ax.plot([0.45, 16.15], [1.12, 1.12], color="#bbbbbb", lw=0.9, zorder=2)
    ax.text(0.45, 0.22,
            "(a)  La máscara  bitmap  que decide qué píxeles entran al cálculo de los índices vectoriales se umbraliza sobre el campo LoS "
            "$\\Rightarrow$ ambas ramas comparten soporte espacial.\n"
            "(b)  5 de los 17 parámetros del BiLSTM se derivan del mismo campo LoS que observa la rama visual (Bobra+2014, Tabla A.8; Bobra+2021, §3.2); los otros 12 son vectoriales.\n"
            "Redundancia medida (test 48 h, $n=500$; $\\rho$ de Spearman contra agregados triviales de la imagen):\n"
            "NACR 0.990 · AREA_ACR 0.989 · USFLUXL 0.989 · USFLUX 0.923 · "
            "ABSNJZH 0.739 · SHRGT45 0.528 · MEANJZH 0.038.",
            ha="left", va="center", fontsize=7.3, color="#3a3a3a", linespacing=1.85)

    fig.tight_layout(pad=0.4)
    os.makedirs(IMG, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(IMG, "procedencia_datos.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, "procedencia_datos.png"), dpi=200, bbox_inches="tight")
    print("OK ->", os.path.join(IMG, "procedencia_datos.pdf"))
    print("OK ->", os.path.join(OUT, "procedencia_datos.png"))


if __name__ == "__main__":
    main()

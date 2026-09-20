"""
Figura 1 (arquitectura SF-MM) para el camera-ready de SIMBig — versión VECTORIAL.

Reemplaza images/pipeline.png, que tenía tres problemas:
  1. Dibujaba mal el ensemble: decía "Fusion (combine logits)" y una sigmoide
     FINAL, cuando el código (graficos/bootstrap_paired.py::fusion_ensemble)
     aplica la sigmoide a CADA rama y promedia PROBABILIDADES. La sigmoide final
     solo vale para el stacking.
  2. Su rotulación quedaba en 2.8 pt impresa (Springer exige >= 6 pt), porque el
     PNG medía 1920 px de ancho y entraba en una columna de 347 pt (escala 0.22).
  3. Era ráster (~433 dpi) para un diagrama; Springer pide vectorial y >= 800 dpi
     en dibujos de línea.

Se dibuja al ancho EXACTO que ocupa en la página (\\textwidth = 347 pt = 4.82 in),
así que la escala es ~1.0 y las fuentes salen a su tamaño nominal (>= 6.3 pt).
El `dpi=600` del savefig no afecta a lo vectorial: es por el magnetograma
incrustado, que si no se guardaba a ~102 ppi.

Uso:
    python graficos/fig_pipeline_simbig55.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.patches import (Polygon, FancyBboxPatch, FancyArrowPatch,
                                Rectangle, Circle)

BASE_SEQ = os.environ.get("SFMM_DATA", "")
SEQ_EJEMPLO = "hmi.sharp_720s.5692.20150622_142400_TAI.to.20150623_142400_TAI"
# los 17 del paper dentro de los 21 de para_flare_21params.txt (dataset_temporal_v3)
KEEP_17 = [0, 1, 2, 3, 4, 6, 7, 8, 11, 12, 14, 15, 16, 17, 18, 19, 20]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "SIMBig55", "images")
MAG = os.path.join(ROOT, "RedaccionSIMBig", "images", "magnetograma_ejemplo.jpeg")

C_VIS, C_PHY, C_CMB, C_OUT = "#1f77b4", "#d62728", "#2ca02c", "#6a51a3"
F_VIS, F_PHY, F_CMB = "#eaf3fb", "#fdeeee", "#eef8ee"
FS_T, FS_L, FS_S, FS_X = 7.6, 7.0, 6.7, 6.6      # título / etiqueta / pequeña / mínima


def shade(c, f):
    r, g, b = mcolors.to_rgb(c)
    return (min(1, r * f), min(1, g * f), min(1, b * f))


def caja(ax, x, y, w, h, borde, relleno, lw=0.8, r=1.2, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad=0.25,rounding_size={r}",
                                facecolor=relleno, edgecolor=borde, lw=lw, zorder=z))


def cuboid(ax, x, y, w, h, d, color, lw=0.6, z=4):
    dx, dy = d * 0.5, d * 0.42
    for pts, f in [([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], 0.72),
                   ([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], 0.88)]:
        ax.add_patch(Polygon(pts, closed=True, facecolor=shade(color, f),
                             edgecolor="black", lw=lw, zorder=z))
    ax.add_patch(Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True,
                         facecolor=color, edgecolor="black", lw=lw, zorder=z + 1))


def flecha(ax, p0, p1, color="#777", lw=0.9, z=7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6,
                                 lw=lw, color=color, zorder=z, shrinkA=0, shrinkB=0))


def columna(ax, x, y0, y1, n, color, r=0.62, z=5):
    """Columna de neuronas (un vector de activaciones)."""
    for yy in np.linspace(y0, y1, n):
        ax.add_patch(Circle((x, yy), r, facecolor=color, edgecolor="black",
                            lw=0.35, zorder=z))


def mini_mlp(ax, x0, x1, y_c, n_in, n_h, n_out, c_in, alto=6.0, z=5, c_out="#2f6f2f"):
    """Grafito de una capa densa: entradas -> ocultas -> salida."""
    caps = []
    for i, n in enumerate([n_in, n_h, n_out]):
        xx = x0 + (x1 - x0) * i / 2
        ys = np.linspace(y_c - alto / 2, y_c + alto / 2, n) if n > 1 else np.array([y_c])
        caps.append((xx, ys))
    for (xa, ya), (xb, yb) in zip(caps[:-1], caps[1:]):
        for u in ya:
            for v in yb:
                ax.plot([xa, xb], [u, v], color="#c8c8c8", lw=0.25, zorder=z - 1)
    for j, (xx, ys) in enumerate(caps):
        col = c_in if j == 0 else (c_out if j == 2 else "#aaa")
        for yy in ys:
            ax.add_patch(Circle((xx, yy), 0.5, facecolor=col, edgecolor="black",
                                lw=0.3, zorder=z))


def sharp_reales():
    """Los 17 SHARP de una secuencia REAL con llamarada, normalizados a [0,1].

    Antes aquí había ruido aleatorio, que parecía un dato sin serlo.
    Devuelve None si el dataset no está montado (la figura usa entonces un
    marcador neutro).
    """
    try:
        seq = os.path.join(BASE_SEQ, "Seq_Magnetogram/M48/Seqs16", SEQ_EJEMPLO)
        frames = [l.strip() for l in open(seq) if l.strip()]
        vals = {}
        with open(os.path.join(BASE_SEQ, "para_flare_21params.txt")) as f:
            for line in f:
                c = line.split()
                if len(c) >= 22:
                    vals[c[0].removesuffix(".fits")] = c[1:22]
        M = np.array([[float(v) if v != "nan" else np.nan
                       for v in vals[fr]] for fr in frames], dtype=float)
        M = M[:, KEEP_17]
        M = np.sign(M) * np.log10(np.abs(M) + 1.0)          # igual que el dataset
        C = M - np.nanmean(M, axis=0)                        # centrar cada parámetro
        esc = np.nanstd(C)                                   # UNA escala para todos:
        if not np.isfinite(esc) or esc < 1e-9:               # así los parámetros que
            esc = 1.0                                        # apenas varían siguen
        return np.clip(0.5 + C / (6 * esc), 0.03, 0.97)      # planos, como en los datos
    except Exception:
        return None


def main():
    fig, ax = plt.subplots(figsize=(4.82, 3.15))
    ax.set_xlim(0, 100); ax.set_ylim(0, 68); ax.axis("off")

    # =========================================================== RAMA VISUAL
    caja(ax, 1, 37, 59, 29.5, C_VIS, F_VIS)
    ax.text(3.2, 63.4, "Visual branch", fontsize=FS_T, fontweight="bold",
            color=C_VIS, zorder=6)

    bx, by, bw, bh = 3.2, 43.0, 10.0, 11.5
    for i in range(4, 0, -1):
        o = i * 0.7
        ax.add_patch(Rectangle((bx + o, by + o), bw, bh, facecolor="#cfe2f5",
                               edgecolor=C_VIS, lw=0.35, zorder=2))
    try:
        ax.imshow(mpimg.imread(MAG)[45:245, 140:340],
                  extent=[bx, bx + bw, by, by + bh], zorder=3, aspect="auto")
    except FileNotFoundError:
        pass
    ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor=C_VIS, lw=0.7, zorder=4))
    ax.text(10.2, 42.2, "16 FITS", ha="center", va="top",
            fontsize=FS_X, color="#444", zorder=6)
    ax.text(10.2, 39.6, r"$224\times224$ px", ha="center", va="top",
            fontsize=FS_X, color="#888", zorder=6)

    flecha(ax, (15.2, 49.0), (19.0, 49.0))

    # -- Swin3D-T: 4 etapas jerárquicas
    caja(ax, 19.4, 41.0, 24.2, 21.0, "#9ecae1", "white", lw=0.6, r=0.9, z=2)
    ax.text(31.5, 59.9, "Swin3D-T", ha="center", fontsize=FS_S,
            fontweight="bold", color=C_VIS, zorder=6)
    etapas = [(20.5, 45.0, 3.4, 10.0, 2.2), (27.0, 46.2, 2.7, 7.6, 2.6),
              (32.8, 47.2, 2.1, 5.6, 3.0), (38.0, 48.0, 1.6, 4.0, 3.4)]

    azules = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5"]
    prev = None
    for (sx, sy, sw, sh, sd), col in zip(etapas, azules):
        cuboid(ax, sx, sy, sw, sh, sd, col)
        if prev is not None:
            flecha(ax, (prev, 50.0), (sx - 0.2, 50.0), color="#aaa", lw=0.6)
        prev = sx + sw + sd * 0.5

    ax.text(31.5, 40.2, r"4 stages:  $56^3\!\to\!7^3$,  $C\!\to\!8C$", ha="center",
            va="top", fontsize=FS_X, color="#777", zorder=6)

    # -- pooling + cabeza lineal
    flecha(ax, (44.2, 49.5), (47.6, 49.5), color="#aaa", lw=0.6)
    columna(ax, 48.6, 45.8, 53.2, 5, "#6baed6")
    ax.text(48.6, 43.6, "GAP", ha="center", va="top", fontsize=FS_X, color="#666", zorder=6)
    mini_mlp(ax, 51.5, 55.5, 49.5, 4, 3, 1, "#6baed6", alto=7.0, c_out=C_VIS)
    ax.add_patch(Circle((58.0, 49.5), 1.15, facecolor=C_VIS, edgecolor="black",
                        lw=0.5, zorder=6))
    ax.text(58.0, 46.2, r"$\ell_{\mathrm{vis}}$", ha="center", fontsize=FS_S,
            color=C_VIS, zorder=6)

    # ========================================================== RAMA FÍSICA
    caja(ax, 1, 2.5, 59, 29.5, C_PHY, F_PHY)
    ax.text(3.2, 28.9, "Physical branch", fontsize=FS_T, fontweight="bold",
            color=C_PHY, zorder=6)

    px0, py0, pw, ph = 3.2, 8.5, 10.0, 11.5
    ax.add_patch(Rectangle((px0, py0), pw, ph, facecolor="white",
                           edgecolor=C_PHY, lw=0.7, zorder=3))
    S = sharp_reales()
    if S is not None:
        xs = px0 + np.linspace(0.6, pw - 0.6, S.shape[0])
        rojos = plt.get_cmap("Reds")(np.linspace(0.35, 0.95, S.shape[1]))
        for j in range(S.shape[1]):
            ax.plot(xs, py0 + 0.8 + S[:, j] * (ph - 1.6), color=rojos[j],
                    lw=0.45, zorder=4, solid_capstyle="round")
    ax.text(10.2, 7.6, "17 SHARP", ha="center", va="top", fontsize=FS_X,
            color="#444", zorder=6)
    ax.text(10.2, 5.0, "16 steps", ha="center", va="top", fontsize=FS_X,
            color="#888", zorder=6)

    flecha(ax, (15.2, 14.5), (19.0, 14.5))

    # -- BiLSTM desenrollada
    caja(ax, 19.4, 6.5, 26.0, 21.0, "#f6b8b8", "white", lw=0.6, r=0.9, z=2)
    ax.text(32.4, 25.4, "BiLSTM encoder", ha="center", fontsize=FS_S,
            fontweight="bold", color=C_PHY, zorder=6)
    cx, cw, ch, cy = [20.3, 26.7, 33.1, 39.5], 5.2, 6.0, 12.5
    etiq = [r"$t_1$", r"$t_2$", r"$\cdots$", r"$t_{16}$"]
    for i, xx in enumerate(cx):
        caja(ax, xx, cy, cw, ch, C_PHY, "#fde2e2", lw=0.6, r=0.5, z=4)
        ax.text(xx + cw / 2, cy + ch / 2, "LSTM", ha="center", va="center",
                fontsize=FS_X, color=C_PHY, zorder=6)
        ax.text(xx + cw / 2, 9.7, etiq[i], ha="center", fontsize=FS_X,
                color="#888", zorder=6)
        if i < len(cx) - 1:
            flecha(ax, (xx + cw + 0.4, cy + ch * 0.72), (cx[i + 1] - 0.4, cy + ch * 0.72),
                   color="#999", lw=0.55)
            flecha(ax, (cx[i + 1] - 0.4, cy + ch * 0.28), (xx + cw + 0.4, cy + ch * 0.28),
                   color="#e0669a", lw=0.55)
    ax.text(19.8, 20.0, "forward", fontsize=FS_X, color="#999", zorder=6)
    ax.text(44.9, 20.0, "backward", fontsize=FS_X, color="#e0669a", ha="right", zorder=6)

    flecha(ax, (46.2, 15.0), (47.6, 15.0), color="#aaa", lw=0.6)
    columna(ax, 48.6, 11.3, 18.7, 5, "#f08a8a")
    ax.text(49.3, 9.6, "concat", ha="center", va="top", fontsize=FS_X, color="#666", zorder=6)
    mini_mlp(ax, 51.5, 55.5, 15.0, 4, 3, 1, "#f08a8a", alto=7.0, c_out=C_PHY)
    ax.add_patch(Circle((58.0, 15.0), 1.15, facecolor=C_PHY, edgecolor="black",
                        lw=0.5, zorder=6))
    ax.text(58.0, 11.7, r"$\ell_{\mathrm{phy}}$", ha="center", fontsize=FS_S,
            color=C_PHY, zorder=6)

    # ========================================================= FUSIÓN TARDÍA
    fx, fy, fw, fh = 63.5, 21.0, 35.5, 38.0
    caja(ax, fx, fy, fw, fh, C_CMB, F_CMB, lw=0.9)
    ax.text(fx + fw / 2, fy + fh - 3.4, "Late fusion", ha="center",
            fontsize=FS_T, fontweight="bold", color=C_CMB, zorder=6)

    flecha(ax, (59.3, 49.5), (fx - 0.2, fy + fh - 8.0), color=C_VIS, lw=0.85)
    flecha(ax, (59.3, 15.0), (fx - 0.2, fy + 7.0), color=C_PHY, lw=0.85)

    # -- ensemble ponderado: la sigmoide va en CADA rama, luego se promedia
    caja(ax, fx + 1.6, fy + 18.5, fw - 3.2, 13.5, "#bcdfbc", "white", lw=0.5, r=0.7, z=3)
    ax.text(fx + fw / 2, fy + 29.6, "Weighted ensemble", ha="center", fontsize=FS_X,
            fontweight="bold", color="#2f6f2f", zorder=6)
    for yy, c in [(fy + 25.8, C_VIS), (fy + 21.4, C_PHY)]:
        ax.add_patch(Circle((fx + 5.2, yy), 1.3, facecolor=c, edgecolor="black",
                            lw=0.4, zorder=6))
        ax.text(fx + 5.2, yy, r"$\sigma$", ha="center", va="center", fontsize=FS_X,
                color="white", zorder=7)
        flecha(ax, (fx + 6.7, yy), (fx + 9.6, fy + 23.6), color="#999", lw=0.55)
    ax.add_patch(Circle((fx + 11.2, fy + 23.6), 1.6, facecolor="white",
                        edgecolor="#2f6f2f", lw=0.7, zorder=6))
    ax.text(fx + 11.2, fy + 23.6, "+", ha="center", va="center", fontsize=FS_L,
            color="#2f6f2f", zorder=7)
    ax.text(fx + 14.0, fy + 23.6,
            r"$w\,p_{\mathrm{phy}}\!+\!(1\!-\!w)\,p_{\mathrm{vis}}$",
            ha="left", va="center", fontsize=FS_X, color="#222", zorder=6)

    # -- stacking: el meta-MLP recibe los LOGITS, la sigmoide va al final
    caja(ax, fx + 1.6, fy + 3.5, fw - 3.2, 13.5, "#bcdfbc", "white", lw=0.5, r=0.7, z=3)
    ax.text(fx + fw / 2, fy + 14.6, "Meta-MLP stacking", ha="center", fontsize=FS_X,
            fontweight="bold", color="#2f6f2f", zorder=6)
    for yy, c in [(fy + 10.8, C_VIS), (fy + 6.4, C_PHY)]:
        ax.add_patch(Circle((fx + 5.2, yy), 1.3, facecolor=c, edgecolor="black",
                            lw=0.4, zorder=6))
        ax.text(fx + 5.2, yy, r"$\ell$", ha="center", va="center", fontsize=FS_X,
                color="white", zorder=7)
    mini_mlp(ax, fx + 8.8, fx + 15.6, fy + 8.6, 2, 4, 1, "#aaa", alto=6.6)
    ax.text(fx + 17.6, fy + 8.6, r"$\sigma(\mathrm{MLP})$", ha="left", va="center",
            fontsize=FS_X, color="#222", zorder=6)

    # =============================================================== SALIDA
    ox, oy, ow, oh = 63.5, 2.5, 35.5, 14.5
    caja(ax, ox, oy, ow, oh, C_OUT, "#f5f2fa", lw=0.8)
    flecha(ax, (ox + ow / 2, fy - 0.4), (ox + ow / 2, oy + oh + 0.4), color="#555", lw=0.9)
    ax.text(ox + ow / 2 + 2.6, oy + oh + 2.2, r"$p$", fontsize=FS_X, color="#555", zorder=6)

    t = np.linspace(-6, 6, 120)
    ax.plot(ox + 3.2 + (t + 6) * 0.62, oy + 4.2 + 7.6 / (1 + np.exp(-t)),
            color=C_OUT, lw=1.0, zorder=6)
    ax.text(ox + 7.0, oy + 1.6, r"$p\!\geq\!\tau$", ha="center", fontsize=FS_X,
            color=C_OUT, zorder=6)
    for yy, txt, col in [(oy + 10.2, r"$\geq$M flare", "#e6550d"),
                         (oy + 4.0, r"no flare", "#808080")]:
        caja(ax, ox + 16.0, yy - 2.2, 16.5, 4.4, col, "white", lw=0.6, r=0.5, z=5)
        ax.text(ox + 24.3, yy, txt, ha="center", va="center", fontsize=FS_X,
                color=col, zorder=7)
    flecha(ax, (ox + 13.2, oy + 8.6), (ox + 15.6, oy + 10.2), color="#aaa", lw=0.6)
    flecha(ax, (ox + 13.2, oy + 5.6), (ox + 15.6, oy + 4.0), color="#aaa", lw=0.6)

    ax.text(30.5, 34.5, "branches trained independently", ha="center", va="center",
            fontsize=FS_X, style="italic", color="#aaa", zorder=6)

    fig.tight_layout(pad=0.1)
    # dpi alto solo por el magnetograma incrustado; lo vectorial no cambia
    fig.savefig(os.path.join(IMG, "pipeline.pdf"), bbox_inches="tight", dpi=600)
    fig.savefig(os.path.join(ROOT, "graficos", "pipeline_simbig55.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Generated: SIMBig55/images/pipeline.pdf")


if __name__ == "__main__":
    main()

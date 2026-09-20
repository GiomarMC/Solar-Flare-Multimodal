"""
Contraste de las tres variantes de la rama física frente a Swin3D:

    BiLSTM-17 (v3)  — 12 vectoriales + 5 derivados del LoS          [modelo del paper]
    BiLSTM-21 (v5)  — los 21 SHARP completos
    BiLSTM-16 (v6)  — solo los 16 vectoriales (sin los 5 derivados del LoS)

Pregunta que responde (objeción "padre-hijo"): ¿la redundancia entre ramas se
debe a que parte de la rama física se calcula sobre el mismo magnetograma LoS
que ve la rama visual?

  (1) ¿Quitar los 5 derivados del LoS cambia la RAMA FÍSICA sola?
      -> bootstrap pareado BiLSTM-16 vs BiLSTM-17 y vs BiLSTM-21.
  (2) ¿Cambia el TECHO ESTRUCTURAL de la fusión con Swin3D?
      -> si con BiLSTM-16 el techo sigue en ~0, la redundancia NO se explica por
         derivación de los datos sino por causa física común.

Requiere outputs/logits/lstm16_*.npz (los genera scripts/save_logits_lstm16.py).

Uso:
    python graficos/comparar_lstm_variantes.py
"""
import os
import sys

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import OUTDIR
from diversidad_ramas import tss_from_pred, RULE_ALIAS, paired_delta_preds
from comparar_lstm17_vs_21 import ens, ceiling

VARIANTS = [
    ("BiLSTM-17", "lstm",   "12 vect + 5 LoS"),
    ("BiLSTM-21", "lstm21", "16 vect + 5 LoS"),
    ("BiLSTM-16", "lstm16", "16 vect, sin LoS"),
]


def main():
    lines = []
    def w(s=""):
        print(s); lines.append(s)

    w("=" * 78)
    w("  RAMA FÍSICA: BiLSTM-17  vs  BiLSTM-21  vs  BiLSTM-16 (solo vectoriales)")
    w("=" * 78)

    for h in (24, 48):
        p_swin, t_swin, y = ens("swin3d", h)
        ssw = tss_from_pred((p_swin >= t_swin).astype(int), y)[0]

        preds = {}
        w()
        w("-" * 78)
        w(f"  HORIZONTE {h} h")
        w("-" * 78)
        w(f"    {'Swin3D (FITS)':<14s}{'':<20s} TSS = {ssw:.4f}   (tau={t_swin:.2f})")
        for name, key, desc in VARIANTS:
            p, t, _ = ens(key, h)
            preds[name] = (p, (p >= t).astype(int))
            s = tss_from_pred(preds[name][1], y)[0]
            w(f"    {name:<14s}{desc:<20s} TSS = {s:.4f}   (tau={t:.2f})")

        w()
        w("    (1) ¿Cambia la rama física sola? — bootstrap pareado B=10000")
        for ref in ("BiLSTM-17", "BiLSTM-21"):
            r = paired_delta_preds(preds["BiLSTM-16"][1], preds[ref][1], y)
            lo, hi = r["ci"]
            sig = "SIGNIFICATIVO" if (lo > 0 or hi < 0) else "no significativo"
            w(f"        BiLSTM-16 - {ref}:  D={r['delta']:+.4f}  "
              f"IC95%=[{lo:+.4f}, {hi:+.4f}]  P(16>{ref[-2:]})={100*r['p']:.1f}%  -> {sig}")

        w()
        w("    (2) ¿Cambia el techo estructural de la fusión con Swin3D?")
        w(f"        {'rama física':<14s}{'mejor individual':>18s}{'techo':>10s}{'margen':>10s}   regla")
        for name, _, _ in VARIANTS:
            bsg, cel, mar, rule = ceiling(p_swin, preds[name][0], y)
            w(f"        {name:<14s}{bsg:>18.4f}{cel:>10.4f}{mar:>+10.4f}   {rule} {RULE_ALIAS.get(rule, '')}")

    w()
    w("  Lectura: el margen es una cota TRAMPOSA (umbrales y regla elegidos sobre test).")
    w("  - Margen de BiLSTM-16 en ~0     -> la redundancia NO viene de que la rama física")
    w("                                    sea 'hija' del magnetograma: es causa física común.")
    w("  - Margen de BiLSTM-16 claramente -> los 5 derivados del LoS eran la fuente de la")
    w("    mayor que el de BiLSTM-17        redundancia; la fusión sí tiene espacio sin ellos.")

    out = os.path.join(OUTDIR, "comparar_lstm_variantes.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

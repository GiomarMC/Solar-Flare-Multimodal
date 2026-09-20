"""
Contraste directo: BiLSTM-17 (v3) vs BiLSTM-21 (v5) — rama física y techo de fusión.

Responde dos preguntas separadas:

  (1) ¿Reincorporar los 4 parámetros descartados mejora la RAMA FÍSICA sola?
      -> bootstrap pareado BiLSTM-21 vs BiLSTM-17 sobre el test fijo.

  (2) ¿Cambia el TECHO ESTRUCTURAL de la fusión con Swin3D?
      -> si el techo sigue en ~0, la redundancia es estructural y no un artefacto
         de la selección univariada de v3 (que descartó justo los 4 parámetros
         menos recuperables desde la imagen — ver redundancia_sharp_imagen.txt).

Requiere outputs/logits/lstm21_*.npz (los genera scripts/save_logits_lstm21.py).

Uso:
    python graficos/comparar_lstm17_vs_21.py
"""
import os
import sys
import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import OUTDIR, load_split, sigmoid, sweep_tau
from diversidad_ramas import (FOLDS, tss_from_pred, enumerate_rules, RULE_ALIAS,
                              paired_delta_preds)


def ens(model, horizon):
    pt, taus = [], []
    for k in FOLDS:
        v = load_split(model, horizon, k, "val")
        t = load_split(model, horizon, k, "test")
        if v is None or t is None:
            raise FileNotFoundError(f"faltan logits: {model} {horizon}h k{k}")
        tau, _ = sweep_tau(sigmoid(v[0]), v[1]); taus.append(tau)
        pt.append(sigmoid(t[0])); y = t[1]
    return np.mean(pt, axis=0), float(np.mean(taus)), y


def ceiling(p_swin, p_phys, y):
    grid = np.arange(0.05, 1.0, 0.05)
    bs = max(tss_from_pred((p_swin >= t).astype(int), y)[0] for t in grid)
    bp = max(tss_from_pred((p_phys >= t).astype(int), y)[0] for t in grid)
    best_single = max(bs, bp)
    top = (-9.0, None)
    for a in grid:
        pa = (p_swin >= a).astype(int)
        for b in grid:
            rules, _ = enumerate_rules(pa, (p_phys >= b).astype(int), y)
            if rules[0][0] > top[0]:
                top = (rules[0][0], rules[0][1])
    return best_single, top[0], top[0] - best_single, top[1]


def main():
    lines = []
    def w(s=""):
        print(s); lines.append(s)

    w("=" * 78)
    w("  BiLSTM-17 (v3, 17 params)  vs  BiLSTM-21 (v5, 21 params)")
    w("=" * 78)

    for h in (24, 48):
        p_swin, t_swin, y = ens("swin3d", h)
        p17, t17, _ = ens("lstm",   h)
        p21, t21, _ = ens("lstm21", h)

        s17 = tss_from_pred((p17 >= t17).astype(int), y)[0]
        s21 = tss_from_pred((p21 >= t21).astype(int), y)[0]
        ssw = tss_from_pred((p_swin >= t_swin).astype(int), y)[0]

        w()
        w("-" * 78)
        w(f"  HORIZONTE {h} h")
        w("-" * 78)
        w(f"    Swin3D (FITS)        TSS = {ssw:.4f}   (tau={t_swin:.2f})")
        w(f"    BiLSTM-17 (SHARP)    TSS = {s17:.4f}   (tau={t17:.2f})")
        w(f"    BiLSTM-21 (SHARP)    TSS = {s21:.4f}   (tau={t21:.2f})")

        w()
        w("    (1) ¿Mejora la rama física sola? — bootstrap pareado B=10000")
        r = paired_delta_preds((p21 >= t21).astype(int), (p17 >= t17).astype(int), y)
        lo, hi = r["ci"]
        sig = "SIGNIFICATIVO" if (lo > 0 or hi < 0) else "no significativo"
        w(f"        BiLSTM-21 - BiLSTM-17:  D={r['delta']:+.4f}  "
          f"IC95%=[{lo:+.4f}, {hi:+.4f}]  P(21>17)={100*r['p']:.1f}%  -> {sig}")

        w()
        w("    (2) ¿Cambia el techo estructural de la fusión con Swin3D?")
        w(f"        {'rama física':<14s}{'mejor individual':>18s}{'techo':>10s}{'margen':>10s}   regla")
        for nm, pp in (("BiLSTM-17", p17), ("BiLSTM-21", p21)):
            bsg, cel, mar, rule = ceiling(p_swin, pp, y)
            w(f"        {nm:<14s}{bsg:>18.4f}{cel:>10.4f}{mar:>+10.4f}   {rule} {RULE_ALIAS.get(rule,'')}")

    w()
    w("  Lectura: el margen es una cota TRAMPOSA (umbrales y regla elegidos sobre test).")
    w("  Si sigue en ~0 con 21 params, la redundancia con la rama visual es estructural.")

    out = os.path.join(OUTDIR, "comparar_lstm17_vs_21.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

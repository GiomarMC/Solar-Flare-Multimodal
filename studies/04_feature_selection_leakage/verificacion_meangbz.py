"""
Verificación para el Revisor #1 de SIMBig: ¿depende algún resultado de MEANGBZ?

Contexto. El paper preselecciona 17 de 21 parámetros SHARP con la d de Cohen
calculada sobre el dataset completo. El revisor objeta que eso usa indirectamente
información de los periodos de evaluación. Al recalcular la d solo con los folds
de entrenamiento (studies/04_feature_selection_leakage/cohen_d_por_fold.py), la selección sale IDÉNTICA en los
5 folds y en ambos horizontes, con un único caso en la frontera:

    MEANGBZ   48h: 0.634 (dataset completo)  vs  0.494-0.580 (por fold)
              24h: 0.589 (dataset completo)  vs  0.484-0.558 (por fold)

Es decir, un criterio estrictamente interno al fold también lo habría descartado.
Este informe entrena esa alternativa y compara:

    BiLSTM-17  (lstm)     — modelo del paper
    BiLSTM-16b (lstm16b)  — los mismos 17 sin MEANGBZ  [dataset_temporal_v8]

Si el TSS no se mueve de forma significativa, la preselección no afecta a lo
publicado y la objeción queda cerrada empíricamente.

Reporta, por horizonte: métricas completas de la rama física, bootstrap pareado
del TSS (16b − 17), y si la fusión sigue sin superar a la mejor rama individual.

NO confundir con studies/03_los_derived_ablation/ablacion_hijos.py (BiLSTM-12, otra pregunta).

Requiere outputs/logits/lstm16b_*.npz (scripts/save_logits_lstm16b.py).

Uso:
    python studies/04_feature_selection_leakage/verificacion_meangbz.py
"""
import os
import sys

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import OUTDIR, tss_point
from bootstrap_paired import paired_delta
from late_fusion_rules import RULES
from fold_ensemble import ens, tss_e, sig_label
from metricas_variantes import (load, per_fold_single, per_fold_weighted,
                                per_fold_stacking, per_fold_rule, summarize, NO_PROB)

HORIZONS = [24, 48]
B = 10000
KEY, LABEL = "lstm16b", "BiLSTM-16b"


def main():
    lines = []
    def w(s=""):
        print(s, flush=True); lines.append(s)

    w("=" * 100)
    w("  ¿DEPENDE ALGÚN RESULTADO DE MEANGBZ? — BiLSTM-17 (paper) vs BiLSTM-16b (sin MEANGBZ)")
    w("  Única diferencia entre ambos: el parámetro MEANGBZ")
    w("=" * 100)

    synth = []
    for h in HORIZONS:
        sw, p17, pab = load("swin3d", h), load("lstm", h), load(KEY, h)
        f_sw, f_17, f_ab = per_fold_single(sw), per_fold_single(p17), per_fold_single(pab)
        e_sw, e_17, e_ab = ens(f_sw), ens(f_17), ens(f_ab)
        y = e_sw[2]

        w()
        w("#" * 100)
        w(f"  HORIZONTE {h} h   (test n={len(y)}, positivos={int((y == 1).sum())})")
        w("#" * 100)

        # ------------------------------------------------------- (1) rama física
        w()
        w("  (1) Rama física sola")
        w(f"      {'Modelo':<16}{'TSS_ens':>9}{'POD':>7}{'FAR':>7}{'HSS':>7}{'F1':>7}"
          f"{'AUC':>8}{'TP':>6}{'FP':>6}{'FN':>5}")
        for name, f in (("Swin3D (ref.)", f_sw), ("BiLSTM-17", f_17), (LABEL, f_ab)):
            r = summarize(f)
            w(f"      {name:<16}{r['tss_ens']:>9.4f}{r['pod']:>7.3f}{r['far']:>7.3f}{r['hss']:>7.3f}"
              f"{r['f1']:>7.3f}{r['auc']:>8.4f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>5}")
        d = paired_delta(e_ab[0], e_ab[1], e_17[0], e_17[1], y, B=B)
        lo, hi = d["ci"]
        veredicto = ("sin diferencia significativa" if lo <= 0 <= hi
                     else ("mejora (sig.)" if lo > 0 else "empeora (sig.)"))
        w(f"      Δ TSS ({LABEL} − BiLSTM-17) = {d['delta_mean']:+.4f}"
          f"  IC95%=[{lo:+.4f}, {hi:+.4f}]  -> {veredicto}")
        synth.append(f"  {h}h  rama física: Δ TSS = {d['delta_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                     f"  {veredicto}")

        # ------------------------------------------------- (2) ¿cambia la conclusión?
        best_name, best = ("Swin3D", e_sw) if tss_e(e_sw) >= tss_e(e_ab) else (LABEL, e_ab)
        w()
        w(f"  (2) Fusión Swin3D + {LABEL} — Δ contra la mejor rama individual: {best_name} "
          f"(TSS {tss_e(best):.4f})")
        w(f"      {'Método':<20}{'TSS_ens':>9}{'HSS':>7}{'F1':>7}"
          f"{'Δ TSS':>10}{'IC95%':>22}   veredicto")
        methods = [("Ensemble ponderado", per_fold_weighted(sw, pab)),
                   ("Stacking meta-MLP", per_fold_stacking(sw, pab))]
        methods += [(f"Regla: {n.split(' ')[0]}", per_fold_rule(rule, sw, pab))
                    for n, rule in RULES.items()]
        top = None
        for name, f in methods:
            r, e = summarize(f, prob_output=name.split(" ")[-1] not in NO_PROB), ens(f)
            dd = paired_delta(e[0], e[1], best[0], best[1], y, B=B)
            lo2, hi2 = dd["ci"]
            w(f"      {name:<20}{r['tss_ens']:>9.4f}{r['hss']:>7.3f}{r['f1']:>7.3f}"
              f"{dd['delta_mean']:>+10.4f}   [{lo2:+.4f}, {hi2:+.4f}]   {sig_label(lo2, hi2)}")
            if top is None or dd["delta_mean"] > top[1]["delta_mean"]:
                top = (name, dd)
        lo2, hi2 = top[1]["ci"]
        synth.append(f"  {h}h  mejor fusión = {top[0]:<20} Δ vs mejor rama = "
                     f"{top[1]['delta_mean']:+.4f} [{lo2:+.4f}, {hi2:+.4f}]  "
                     f"{'SUPERA' if lo2 > 0 else 'NO supera'}")

    w()
    w("=" * 100)
    w("  SÍNTESIS")
    w("=" * 100)
    for s in synth:
        w(s)
    w()
    w("  Lectura: si los Δ de (1) no son significativos, quitar MEANGBZ no cambia la rama")
    w("  física y la preselección señalada por el revisor no afecta a los resultados.")

    out = os.path.join(OUTDIR, "verificacion_meangbz.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

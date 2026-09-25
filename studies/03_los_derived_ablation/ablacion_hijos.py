"""
Ablación PURA de los parámetros derivados del magnetograma LoS ("hijos" de la imagen):

    BiLSTM-17 (lstm)    — modelo del paper: 12 vectoriales + 5 derivados del LoS
    BiLSTM-12 (lstm12)  — los mismos 12 vectoriales, sin los 5 derivados del LoS

La única diferencia entre ambos es la presencia de USFLUXL, MEANGBL, R_VALUE,
AREA_ACR y NACR (mismos hiperparámetros, protocolo y splits). A diferencia de la
comparación con BiLSTM-16, aquí el efecto es atribuible solo a esos 5 parámetros.

Responde, por horizonte:
  (1) ¿Qué pierde la rama física sola sin los hijos?  Métricas completas (TSS, POD,
      FAR, HSS, F1, AUC, TP/FP/FN) + bootstrap pareado del TSS (12 − 17).
  (2) ¿Se abre el techo de fusión a nivel de decisión?  (cota tramposa sobre test)
  (3) ¿Alguna fusión por probabilidades supera a la mejor rama individual?
      Ensemble ponderado, stacking meta-MLP y reglas media/producto/máximo/mínimo,
      con bootstrap pareado contra la mejor rama.

Los métodos replican exactamente el código del paper (ver metricas_variantes.py y
fusion_variantes.py, cuyos controles reproducen la tabla publicada).

Requiere outputs/logits/lstm12_*.npz (scripts/save_logits_lstm12.py).

Uso:
    python studies/03_los_derived_ablation/ablacion_hijos.py
"""
import os
import sys
import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import OUTDIR, tss_point
from fold_ensemble import ens, tss_e, sig_label
from bootstrap_paired import paired_delta
from late_fusion_rules import RULES
from comparar_lstm17_vs_21 import ceiling
from metricas_variantes import (load, per_fold_single, per_fold_weighted,
                                per_fold_stacking, per_fold_rule, summarize, NO_PROB)

HORIZONS = [24, 48]
B = 10000




def main(key="lstm12", label="BiLSTM-12", out_name="ablacion_hijos.txt"):
    lines = []
    def w(s=""):
        print(s, flush=True); lines.append(s)

    w("=" * 100)
    w(f"  ABLACIÓN PURA DE LOS 5 PARÁMETROS DERIVADOS DEL LoS — BiLSTM-17 (paper) vs {label}")
    w("  Única diferencia: USFLUXL, MEANGBL, R_VALUE, AREA_ACR, NACR")
    w("=" * 100)

    synth = []
    for h in HORIZONS:
        sw, p17, pab = load("swin3d", h), load("lstm", h), load(key, h)
        f_sw, f_17, f_ab = per_fold_single(sw), per_fold_single(p17), per_fold_single(pab)
        e_sw, e_17, e_ab = ens(f_sw), ens(f_17), ens(f_ab)
        y = e_sw[2]

        w()
        w("#" * 100)
        w(f"  HORIZONTE {h} h   (test n={len(y)}, positivos={int((y == 1).sum())})")
        w("#" * 100)

        # ---------------------------------------------- (1) rama física sola
        w()
        w("  (1) Rama física sola — ¿qué se pierde sin los hijos?")
        hdr = (f"      {'Modelo':<16}{'TSS_ens':>9}{'POD':>7}{'FAR':>7}{'HSS':>7}{'F1':>7}"
               f"{'AUC':>8}{'TP':>6}{'FP':>6}{'FN':>5}")
        w(hdr)
        for name, f in (("Swin3D (ref.)", f_sw), ("BiLSTM-17", f_17), (label, f_ab)):
            r = summarize(f)
            w(f"      {name:<16}{r['tss_ens']:>9.4f}{r['pod']:>7.3f}{r['far']:>7.3f}{r['hss']:>7.3f}"
              f"{r['f1']:>7.3f}{r['auc']:>8.4f}{r['tp']:>6}{r['fp']:>6}{r['fn']:>5}")
        d = paired_delta(e_ab[0], e_ab[1], e_17[0], e_17[1], y, B=B)
        lo, hi = d["ci"]
        w(f"      Δ TSS ({label} − BiLSTM-17) = {d['delta_mean']:+.4f}  IC95%=[{lo:+.4f}, {hi:+.4f}]"
          f"  -> {sig_label(lo, hi, 'mejora', 'empeora')}")
        synth.append(f"  {h}h  rama física sola: Δ TSS = {d['delta_mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                     f"  {sig_label(lo, hi, 'mejora', 'empeora')}")

        # ---------------------------------------------- (2) techo de decisión
        w()
        w("  (2) Techo estructural de la fusión a nivel de decisión (cota tramposa: τ y regla vistos en test)")
        w(f"      {'rama física':<16}{'mejor individual':>18}{'techo':>10}{'margen':>10}")
        for name, e in (("BiLSTM-17", e_17), (label, e_ab)):
            bsg, cel, mar, _ = ceiling(e_sw[0], e[0], y)
            w(f"      {name:<16}{bsg:>18.4f}{cel:>10.4f}{mar:>+10.4f}")
            if name == label:
                synth.append(f"  {h}h  techo de decisión con {label}: margen {mar:+.4f}")

        # ---------------------------------------------- (3) fusión por probabilidades
        best_name, best = ("Swin3D", e_sw) if tss_e(e_sw) >= tss_e(e_ab) else (label, e_ab)
        w()
        w(f"  (3) Fusión Swin3D + {label} — Δ contra la mejor rama individual: {best_name} "
          f"(TSS {tss_e(best):.4f})")
        w(f"      {'Método':<20}{'TSS_ens':>9}{'HSS':>7}{'F1':>7}{'FP':>6}{'FN':>5}"
          f"{'Δ TSS':>10}{'IC95%':>22}   veredicto")
        methods = [("Ensemble ponderado", per_fold_weighted(sw, pab)),
                   ("Stacking meta-MLP", per_fold_stacking(sw, pab))]
        methods += [(f"Regla: {n.split(' ')[0]}", per_fold_rule(rule, sw, pab)) for n, rule in RULES.items()]
        top = None
        for name, f in methods:
            r, e = summarize(f, prob_output=name.split(" ")[-1] not in NO_PROB), ens(f)
            d = paired_delta(e[0], e[1], best[0], best[1], y, B=B)
            lo, hi = d["ci"]
            w(f"      {name:<20}{r['tss_ens']:>9.4f}{r['hss']:>7.3f}{r['f1']:>7.3f}{r['fp']:>6}{r['fn']:>5}"
              f"{d['delta_mean']:>+10.4f}   [{lo:+.4f}, {hi:+.4f}]   {sig_label(lo, hi)}")
            if top is None or d["delta_mean"] > top[1]["delta_mean"]:
                top = (name, d)
        lo, hi = top[1]["ci"]
        synth.append(f"  {h}h  mejor fusión = {top[0]:<20} Δ vs mejor rama = {top[1]['delta_mean']:+.4f} "
                     f"[{lo:+.4f}, {hi:+.4f}]  {'SUPERA' if lo > 0 else 'NO supera'} a la mejor rama")

    w()
    w("=" * 100)
    w("  SÍNTESIS")
    w("=" * 100)
    for s in synth:
        w(s)

    out = os.path.join(OUTDIR, out_name) if not os.path.isabs(out_name) else out_name
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

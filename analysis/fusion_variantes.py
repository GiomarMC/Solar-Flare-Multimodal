"""
Fusión tardía Swin3D + rama física con las tres variantes de la BiLSTM:

    BiLSTM-17 (lstm)    — 12 vectoriales + 5 derivados del LoS   [modelo del paper]
    BiLSTM-21 (lstm21)  — los 21 SHARP
    BiLSTM-16 (lstm16)  — solo los 16 vectoriales (sin derivados del LoS)

Pregunta: sin los parámetros que se calculan sobre el mismo magnetograma que ve
la rama visual, ¿alguna fusión supera a la mejor rama individual? El techo de
analysis/diversidad_ramas.py solo acota la fusión a nivel de DECISIÓN; aquí se
evalúan los métodos que combinan PROBABILIDADES, que no están acotados por él.

Métodos (réplica exacta del código que produjo los números del paper, importando
sus funciones sin modificarlas):
  - Ensemble ponderado  -> analysis/bootstrap_paired.py::fusion_ensemble
                           (w_phys y tau por fold sobre VAL; w en pasos de 0.05)
  - Stacking meta-MLP   -> analysis/stacking_5fold.py::train_meta / meta_probs
  - Reglas de Kittler   -> analysis/late_fusion_rules.py::RULES / eval_rule
                           (media, producto, máximo, mínimo)
Protocolo común: tau por fold barrido sobre VAL, promedio de probabilidades de
test entre folds, tau = media de los tau por fold. Cada fusión se compara con la
MEJOR rama individual de su variante mediante bootstrap pareado (B=10000).

tau de las ramas individuales: se barre sobre VAL (sweep_tau). Para lstm y swin3d
se verificó que coincide con el tau_opt de los metrics_*.txt oficiales en los 20
pares fold/horizonte; las variantes lstm21/lstm16 no tienen metrics_*.txt.

Control: la fila BiLSTM-17 debe reproducir el paper (48 h: ensemble 0.8335,
stacking 0.8168).

Uso:
    python analysis/fusion_variantes.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import OUTDIR, load_split, sigmoid, sweep_tau, tss_point
from bootstrap_paired import paired_delta
from stacking_5fold import train_meta, meta_probs
from late_fusion_rules import RULES, eval_rule

FOLDS = [0, 1, 2, 3, 4]
HORIZONS = [24, 48]
B = 10000
W_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)   # igual que fusion_ensemble
VARIANTS = [
    ("BiLSTM-17", "lstm",   "12 vect + 5 LoS"),
    ("BiLSTM-21", "lstm21", "16 vect + 5 LoS"),
    ("BiLSTM-16", "lstm16", "16 vect, sin LoS"),
]
PAPER_CHECK = {("lstm", 48): {"Ensemble ponderado": 0.8335, "Stacking meta-MLP": 0.8168}}


def load(model, h):
    """{fold: (logit_val, y_val, logit_test, y_test)}"""
    out = {}
    for k in FOLDS:
        v, t = load_split(model, h, k, "val"), load_split(model, h, k, "test")
        if v is None or t is None:
            raise FileNotFoundError(f"faltan logits: {model} {h}h k{k}")
        out[k] = (v[0], v[1], t[0], t[1])
    return out


def pack(fold_tss, fold_tau, fold_test_probs, y):
    ens = np.mean(fold_test_probs, axis=0)
    tau = float(np.mean(fold_tau))
    return {"mu": float(np.mean(fold_tss)), "sd": float(np.std(fold_tss)),
            "tss_ens": tss_point(ens, y, tau), "tau": tau, "probs": ens}


def single(d):
    tss, taus, probs = [], [], []
    for k in FOLDS:
        lv, yv, lt, yt = d[k]
        tau, _ = sweep_tau(sigmoid(lv), yv)
        pt = sigmoid(lt)
        tss.append(tss_point(pt, yt, tau)); taus.append(tau); probs.append(pt)
    return pack(tss, taus, probs, yt)


def weighted(sw, ph):
    """Réplica de bootstrap_paired.fusion_ensemble con la rama física parametrizada."""
    tss, taus, probs, ws = [], [], [], []
    for k in FOLDS:
        s_lv, yv, s_lt, yt = sw[k]
        p_lv, _, p_lt, _ = ph[k]
        ps_va, pl_va = sigmoid(s_lv), sigmoid(p_lv)
        best = (-2.0, 1.0, 0.5)
        for w in W_GRID:
            tau, vtss = sweep_tau(w * pl_va + (1.0 - w) * ps_va, yv)
            if vtss > best[0]:
                best = (vtss, float(w), tau)
        _, w_f, tau_f = best
        pt = w_f * sigmoid(p_lt) + (1.0 - w_f) * sigmoid(s_lt)
        tss.append(tss_point(pt, yt, tau_f)); taus.append(tau_f); probs.append(pt); ws.append(w_f)
    r = pack(tss, taus, probs, yt)
    r["extra"] = f"w_phys por fold = {ws}"
    return r


def stacking(sw, ph):
    """Réplica del bucle de stacking_5fold.main con la rama física parametrizada."""
    tss, taus, probs = [], [], []
    for k in FOLDS:
        s_lv, yv, s_lt, yt = sw[k]
        p_lv, _, p_lt, _ = ph[k]
        meta, stats = train_meta(s_lv, p_lv, yv)
        p_val = meta_probs(meta, stats, s_lv, p_lv)
        p_te = meta_probs(meta, stats, s_lt, p_lt)
        tau, _ = sweep_tau(p_val, yv)
        tss.append(tss_point(p_te, yt, tau)); taus.append(tau); probs.append(p_te)
    return pack(tss, taus, probs, yt)


def kittler(sw, ph):
    """Reglas fijas vía late_fusion_rules.eval_rule (sin modificar)."""
    psv = {k: sigmoid(sw[k][0]) for k in FOLDS}; plv = {k: sigmoid(ph[k][0]) for k in FOLDS}
    pst = {k: sigmoid(sw[k][2]) for k in FOLDS}; plt_ = {k: sigmoid(ph[k][2]) for k in FOLDS}
    yv = {k: sw[k][1] for k in FOLDS}; yt = {k: sw[k][3] for k in FOLDS}
    out = {}
    for name, rule in RULES.items():
        r = eval_rule(rule, psv, plv, yv, pst, plt_, yt)
        r["probs"] = np.mean([rule(pst[k], plt_[k]) for k in FOLDS], axis=0)
        out[f"Regla: {name}"] = r
    return out


def main():
    lines = []
    def w(s=""):
        print(s, flush=True); lines.append(s)

    w("=" * 92)
    w("  FUSIÓN TARDÍA Swin3D + rama física — variantes 17 / 21 / 16 (sin derivados del LoS)")
    w("  Δ = TSS(fusión) − TSS(mejor rama individual de esa variante), bootstrap pareado B=10000")
    w("=" * 92)

    verdict = []
    for h in HORIZONS:
        sw = load("swin3d", h)
        y = sw[FOLDS[0]][3]
        r_sw = single(sw)
        w()
        w("#" * 92)
        w(f"  HORIZONTE {h} h   (test n={len(y)}, positivos={int((y == 1).sum())})")
        w("#" * 92)
        w(f"  Swin3D (FITS)                         TSS_ens = {r_sw['tss_ens']:.4f}   "
          f"CV {r_sw['mu']:.4f}±{r_sw['sd']:.4f}")

        for vname, key, desc in VARIANTS:
            ph = load(key, h)
            for k in FOLDS:   # los mismos ejemplos en el mismo orden
                assert np.array_equal(sw[k][1], ph[k][1]) and np.array_equal(sw[k][3], ph[k][3]), \
                    f"etiquetas desalineadas swin3d/{key} {h}h k{k}"
            r_ph = single(ph)
            best_name, best = ("Swin3D", r_sw) if r_sw["tss_ens"] >= r_ph["tss_ens"] else (vname, r_ph)

            methods = {"Ensemble ponderado": weighted(sw, ph),
                       "Stacking meta-MLP": stacking(sw, ph),
                       **kittler(sw, ph)}

            w()
            w(f"  ── {vname} ({desc})   TSS_ens = {r_ph['tss_ens']:.4f}   CV {r_ph['mu']:.4f}±{r_ph['sd']:.4f}")
            w(f"     mejor rama individual: {best_name} ({best['tss_ens']:.4f})")
            w(f"     {'Método':<22}{'TSS_ens':>9}{'CV μ±σ':>18}{'Δ vs mejor':>12}{'IC95%':>22}{'P(>)':>8}   veredicto")
            top = None
            for mname, r in methods.items():
                d = paired_delta(r["probs"], r["tau"], best["probs"], best["tau"], y, B=B)
                lo, hi = d["ci"]
                sig = ("MEJOR (sig.)" if lo > 0 else "peor (sig.)" if hi < 0 else "sin diferencia")
                w(f"     {mname:<22}{r['tss_ens']:>9.4f}{r['mu']:>11.4f}±{r['sd']:<6.4f}"
                  f"{d['delta_mean']:>+12.4f}   [{lo:+.4f}, {hi:+.4f}]{100*d['p_a_gt_b']:>7.1f}%   {sig}")
                if "extra" in r:
                    w(f"        {r['extra']}")
                chk = PAPER_CHECK.get((key, h), {}).get(mname)
                if chk is not None:
                    w(f"        [control vs paper: {r['tss_ens']:.4f} vs {chk:.4f} "
                      f"-> {'OK' if abs(r['tss_ens'] - chk) < 5e-4 else 'DIFIERE'}]")
                if top is None or d["delta_mean"] > top[1]["delta_mean"]:
                    top = (mname, d)
            lo, hi = top[1]["ci"]
            verdict.append(f"  {h}h  {vname:<10} mejor fusión = {top[0]:<20} Δ={top[1]['delta_mean']:+.4f} "
                           f"[{lo:+.4f}, {hi:+.4f}]  "
                           f"{'SUPERA a la mejor rama' if lo > 0 else 'NO supera a la mejor rama'}")

    w()
    w("=" * 92)
    w("  SÍNTESIS — ¿alguna fusión supera significativamente a la mejor rama individual?")
    w("=" * 92)
    for v in verdict:
        w(v)

    out = os.path.join(OUTDIR, "fusion_variantes.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

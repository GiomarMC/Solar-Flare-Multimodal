"""
Test de equivalencia (TOST) — Swin3D vs BiLSTM, por horizonte.

Motivación: "no significativo" es ambiguo — puede significar que las ramas son
equivalentes, o que faltó poder estadístico. El TOST distingue las dos cosas.
En vez de fijar un margen arbitrario, reportamos el MARGEN MÍNIMO DE EQUIVALENCIA

    delta_min = max(|IC_inf|, |IC_sup|)

es decir, el delta más pequeño tal que el IC95% de la diferencia cae entero dentro
de +-delta. La lectura es directa: "las ramas son equivalentes dentro de +-delta_min".

PUNTO CLAVE DE MÉTODO — dos escalas de incertidumbre que NO miden lo mismo:

  (a) Bootstrap pareado del TEST: cuánto variaría la diferencia si el test hubiera
      sido otra muestra. NO captura variabilidad de entrenamiento.
  (b) Sigma entre folds: cuánto varía el TSS al reentrenar con otra partición.
      Esta es la que refleja la reproducibilidad real del resultado.

A 24 h ambas ramas clasifican IDÉNTICAMENTE los 195 positivos en su punto de
operación (ver diversidad_ramas_24h.txt), así que el término positivo se cancela en
cada réplica pareada y el IC del bootstrap se estrecha mucho. El bootstrap dice
"significativo", pero la diferencia es MENOR que la sigma entre folds del propio
estudio. Ambas cosas son ciertas y hay que reportarlas juntas: significativa frente
al muestreo del test, pero por debajo del ruido de reentrenamiento.

(Nota: 'tau_opt' de los .txt oficiales se calculó barriendo el val, de modo que
coincide con el barrido en val — no son dos criterios distintos. Se verificó.)

Uso:
    python analysis/equivalencia_tost.py
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import (OUTDIR, DEVICE, load_split, read_tau_opt,
                          sigmoid, sweep_tau, tss_point, auc_point)
from diversidad_ramas import paired_delta_preds, FOLDS

HORIZONS = [24, 48]
MODELS = [("swin3d", "Swin3D (FITS)"), ("lstm", "BiLSTM (SHARP)")]
B_TSS, B_AUC = 10000, 2000


def ensemble(model, horizon, tau_mode):
    """Probs de test promediadas entre folds + tau segun el criterio pedido.

    Devuelve tambien el TSS por fold (para el piso de ruido sigma).
    """
    ptest, taus, per_fold = [], [], []
    y = None
    for k in FOLDS:
        v = load_split(model, horizon, k, "val")
        t = load_split(model, horizon, k, "test")
        if v is None or t is None:
            continue
        if tau_mode == "val_sweep":
            tau, _ = sweep_tau(sigmoid(v[0]), v[1])
        else:
            tau = read_tau_opt(model, horizon, k)
            if tau is None:
                tau, _ = sweep_tau(sigmoid(v[0]), v[1])
        taus.append(float(tau))
        p_t = sigmoid(t[0])
        ptest.append(p_t)
        y = t[1]
        per_fold.append(tss_point(p_t, y, tau))
    return np.mean(ptest, axis=0), float(np.mean(taus)), y, np.array(per_fold)


def paired_delta_auc(pa, pb, y, B=B_AUC, seed=42):
    """IC95% de Delta AUC con remuestreo estratificado pareado."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    yt = torch.as_tensor(y, dtype=torch.int64, device=DEVICE)
    pos_idx = torch.where(yt == 1)[0]
    neg_idx = torch.where(yt == 0)[0]
    n_pos, n_neg = pos_idx.numel(), neg_idx.numel()
    deltas = []
    for _ in range(B):
        ip = pos_idx[torch.randint(0, n_pos, (n_pos,), generator=g, device=DEVICE)]
        ineg = neg_idx[torch.randint(0, n_neg, (n_neg,), generator=g, device=DEVICE)]
        idx = torch.cat([ip, ineg]).cpu().numpy()
        deltas.append(auc_point(pa[idx], y[idx]) - auc_point(pb[idx], y[idx]))
    d = np.array(deltas)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    lines = []
    def w(s=""):
        print(s); lines.append(s)

    w("=" * 78)
    w("  TEST DE EQUIVALENCIA (TOST) — Swin3D vs BiLSTM por horizonte")
    w("=" * 78)
    w("  delta_min = margen mas pequeño dentro del cual las ramas son equivalentes")
    w("            = max(|IC_inf|, |IC_sup|) del bootstrap pareado (B=10000, estratificado)")
    w()

    summary = {}
    for h in HORIZONS:
        w("-" * 78)
        w(f"  HORIZONTE {h} h")
        w("-" * 78)

        # piso de ruido del propio estudio
        sig = {}
        for m, nm in MODELS:
            _, _, _, pf = ensemble(m, h, "val_sweep")
            sig[m] = (pf.mean(), pf.std())
            w(f"    {nm:<18s} TSS por fold: {pf.mean():.4f} +- {pf.std():.4f}")
        noise = max(sig[m][1] for m, _ in MODELS)
        w(f"    -> piso de ruido del estudio (mayor sigma entre folds) = {noise:.4f}")
        w()

        res_h = {}
        for mode, label in (("tau_opt", "tau barrido en val (== tau_opt de los .txt)"),):
            ps, ts, y, _ = ensemble("swin3d", h, mode)
            pl, tl, _, _ = ensemble("lstm",   h, mode)
            pa = (ps >= ts).astype(int)
            pb = (pl >= tl).astype(int)
            r = paired_delta_preds(pa, pb, y, B=B_TSS)
            lo, hi = r["ci"]
            dmin = max(abs(lo), abs(hi))
            sig_flag = (lo > 0 or hi < 0)
            w(f"    {label}   (tau_swin={ts:.2f}, tau_lstm={tl:.2f})")
            w(f"       TSS: Swin3D {tss_point(ps,y,ts):.4f}  |  BiLSTM {tss_point(pl,y,tl):.4f}")
            w(f"       Delta = {r['delta']:+.4f}   IC95% = [{lo:+.4f}, {hi:+.4f}]")
            w(f"       significancia: {'SI (IC excluye 0)' if sig_flag else 'NO (IC incluye 0)'}")
            w(f"       delta_min de equivalencia = +-{dmin:.4f} TSS"
              f"   ({dmin/noise:.2f}x el piso de ruido)")
            w()
            res_h[mode] = dict(delta=r["delta"], lo=lo, hi=hi, dmin=dmin, sig=sig_flag)

        # AUC como metrica secundaria (invariante al umbral)
        ps, _, y, _ = ensemble("swin3d", h, "val_sweep")
        pl, _, _, _ = ensemble("lstm",   h, "val_sweep")
        da, alo, ahi = paired_delta_auc(ps, pl, y)
        w(f"    AUC (secundaria, invariante al umbral; B={B_AUC})")
        w(f"       Swin3D {auc_point(ps,y):.4f}  |  BiLSTM {auc_point(pl,y):.4f}")
        w(f"       Delta = {da:+.4f}   IC95% = [{alo:+.4f}, {ahi:+.4f}]"
          f"   delta_min = +-{max(abs(alo),abs(ahi)):.4f}")
        w()
        summary[h] = (res_h, noise, (da, alo, ahi))

    # ------------------------------------------------------------------ veredicto
    w("=" * 78)
    w("  VEREDICTO POR HORIZONTE  (redactado para citar)")
    w("=" * 78)
    for h in HORIZONS:
        res_h, noise, _ = summary[h]
        r = res_h["tau_opt"]
        d, lo, hi, dmin, sg = r["delta"], r["lo"], r["hi"], r["dmin"], r["sig"]
        mejor = "BiLSTM (SHARP)" if d < 0 else "Swin3D (FITS)"
        w()
        if sg and dmin > noise:
            w(f"  {h} h — DIFIEREN. Delta = {d:+.4f} TSS [{lo:+.4f}, {hi:+.4f}];")
            w(f"        gana {mejor}. La diferencia ({dmin:.4f}) EXCEDE la sigma entre")
            w(f"        folds ({noise:.4f}), asi que es reproducible al reentrenar.")
        elif sg:
            w(f"  {h} h — Delta = {d:+.4f} TSS [{lo:+.4f}, {hi:+.4f}]: significativa frente al")
            w(f"        muestreo del test (gana {mejor}), PERO equivalentes dentro de")
            w(f"        +-{dmin:.4f}, que es solo {dmin/noise:.2f}x la sigma entre folds ({noise:.4f}).")
            w(f"        => Reportar como diferencia POR DEBAJO del ruido de reentrenamiento:")
            w(f"           no sostiene una recomendacion de modelo por si sola.")
        else:
            w(f"  {h} h — EQUIVALENTES dentro de +-{dmin:.4f} TSS; el IC incluye 0")
            w(f"        (sigma entre folds = {noise:.4f}).")
    w()
    w("  Nota de método: el margen NO se eligió a conveniencia — se reporta el delta_min")
    w("  que se deduce del IC, y se contrasta con la sigma entre folds del propio estudio.")
    w()
    w("  ADVERTENCIA: un IC de bootstrap estrecho NO implica un resultado reproducible.")
    w("  El bootstrap remuestrea el test con los modelos FIJOS; la sigma entre folds mide")
    w("  qué pasa al reentrenar. Cuando delta_min < sigma, la diferencia es real para ESTE")
    w("  test pero no sobrevive necesariamente a otra partición de entrenamiento.")

    out = os.path.join(OUTDIR, "equivalencia_tost.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[guardado] {out}")


if __name__ == "__main__":
    main()

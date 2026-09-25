"""
Diagnóstico de diversidad / redundancia entre ramas — Swin3D (FITS) vs BiLSTM (SHARP).

Responde la pregunta que ninguna tabla de fusión contesta: ¿la fusión no suma
porque no hay nada que sumar (errores correlacionados), o porque las reglas
probadas no supieron explotar una complementariedad que sí existe?

Mide, a nivel de INSTANCIA, sobre los logits ya guardados (sin reentrenar):

  1. Tabla de contingencia 2x2 (aciertos/fallos de cada rama), global y
     estratificada por clase (positivos = fulgores; negativos = quietud).
  2. Medidas de diversidad de ensembles (Kuncheva & Whitaker 2003):
       Q de Yule, desacuerdo D, correlacion rho de los vectores oraculo.
     Q -> 1 significa errores totalmente correlacionados: bajo esa condicion
     esta establecido que ninguna regla de combinacion aporta ganancia.
  3. Correlacion de rangos de Spearman entre las probabilidades de ambas ramas
     (complementa el AUC casi identico: mide si ORDENAN igual).
  4. TECHO EXACTO de la fusion a nivel de decision: las dos decisiones binarias
     parten el test en 4 celdas; cualquier regla que use solo esas decisiones es
     una de las 2^4 = 16 asignaciones posibles. Se enumeran las 16 y se reporta
     la mejor. Si la mejor coincide con "usar solo una rama", la fusion a nivel
     de decision es demostrablemente inutil, sea cual sea la arquitectura.

Protocolo identico al del ensemble del cuerpo (analysis/bootstrap_ci.py):
  por fold se barre tau sobre VAL (sin fuga); el ensemble promedia las
  probabilidades de test entre folds y usa tau = media de los tau por fold.
  Sanity: Swin3D y BiLSTM deben reproducir ~0.868 y ~0.841 a 48 h.

Uso:
    python analysis/diversidad_ramas.py
"""
import os
import sys
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import ROOT, OUTDIR, load_split, sigmoid, sweep_tau

FOLDS = [0, 1, 2, 3, 4]
HORIZONS = [24, 48]

# Rama fisica a comparar contra Swin3D:
#   'lstm'   -> 17 parametros SHARP (dataset_temporal_v3)  [por defecto]
#   'lstm21' -> 21 parametros SHARP (dataset_temporal_v5)
LSTM_MODEL = os.environ.get("LSTM_MODEL", "lstm")
TAG = "" if LSTM_MODEL == "lstm" else "_" + LSTM_MODEL


def ensemble_probs(model, horizon):
    """Promedia probs de test entre folds; tau = media de los tau barridos en val."""
    ptest, taus = [], []
    for k in FOLDS:
        val = load_split(model, horizon, k, "val")
        tst = load_split(model, horizon, k, "test")
        if val is None or tst is None:
            raise FileNotFoundError(f"faltan logits: {model} {horizon}h k{k}")
        tau, _ = sweep_tau(sigmoid(val[0]), val[1])
        taus.append(tau)
        ptest.append(sigmoid(tst[0]))
        y = tst[1]
    return np.mean(ptest, axis=0), float(np.mean(taus)), y


def tss_from_pred(pred, y):
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    pod  = tp / (tp + fn + 1e-12)
    pofd = fp / (fp + tn + 1e-12)
    return pod - pofd, tp, fp, fn, tn


def diversity(ok_a, ok_b):
    """Q de Yule, desacuerdo y rho sobre los vectores oraculo (acierto/fallo)."""
    n11 = int((ok_a & ok_b).sum())        # ambos aciertan
    n10 = int((ok_a & ~ok_b).sum())       # solo A acierta
    n01 = int((~ok_a & ok_b).sum())       # solo B acierta
    n00 = int((~ok_a & ~ok_b).sum())      # ambos fallan
    n = n11 + n10 + n01 + n00
    num = n11 * n00 - n01 * n10
    q   = num / (n11 * n00 + n01 * n10 + 1e-12)
    d   = (n01 + n10) / (n + 1e-12)
    den = np.sqrt(float((n11 + n10)) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    rho = num / (den + 1e-12)
    return dict(n11=n11, n10=n10, n01=n01, n00=n00, n=n, Q=q, D=d, rho=rho)


def enumerate_rules(pa, pb, y):
    """Las 16 funciones booleanas de (pred_A, pred_B). Devuelve lista ordenada."""
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    masks = {c: ((pa == c[0]) & (pb == c[1])) for c in cells}
    out = []
    for bits in range(16):
        assign = {c: (bits >> i) & 1 for i, c in enumerate(cells)}
        pred = np.zeros_like(y)
        for c in cells:
            if assign[c]:
                pred[masks[c]] = 1
        tss, tp, fp, fn, tn = tss_from_pred(pred, y)
        name = "".join(str(assign[c]) for c in cells)   # orden: 00,01,10,11
        out.append((tss, name, assign, tp, fp, fn, tn))
    out.sort(key=lambda t: -t[0])
    return out, masks


RULE_ALIAS = {
    "0001": "AND (consenso)",
    "0111": "OR (union)",
    "0011": "solo Swin3D",
    "0101": "solo BiLSTM",
    "0110": "XOR (desacuerdo)",
    "0000": "todo negativo",
    "1111": "todo positivo",
}


def run(horizon, lines):
    def w(s=""):
        print(s); lines.append(s)

    p_swin, tau_swin, y = ensemble_probs("swin3d", horizon)
    p_lstm, tau_lstm, _ = ensemble_probs(LSTM_MODEL, horizon)

    pred_swin = (p_swin >= tau_swin).astype(int)
    pred_lstm = (p_lstm >= tau_lstm).astype(int)

    tss_s, tp_s, fp_s, fn_s, tn_s = tss_from_pred(pred_swin, y)
    tss_l, tp_l, fp_l, fn_l, tn_l = tss_from_pred(pred_lstm, y)

    npos, nneg = int((y == 1).sum()), int((y == 0).sum())

    w()
    w("=" * 78)
    w(f"  DIVERSIDAD ENTRE RAMAS — Horizonte {horizon} h   (rama fisica: {LSTM_MODEL})")
    w("=" * 78)
    w(f"  Test: {len(y)} ejemplos — {npos} positivos ({100*npos/len(y):.2f} %), {nneg} negativos")
    w(f"  Punto de operacion honesto (tau barrido en val, promediado entre folds):")
    w(f"    Swin3D (FITS) : tau={tau_swin:.2f}  TSS={tss_s:.4f}  TP={tp_s} FN={fn_s} FP={fp_s}")
    w(f"    BiLSTM (SHARP): tau={tau_lstm:.2f}  TSS={tss_l:.4f}  TP={tp_l} FN={fn_l} FP={fp_l}")

    # ---------------------------------------------------------------- 1. contingencia
    ok_s = (pred_swin == y)
    ok_l = (pred_lstm == y)
    pos, neg = (y == 1), (y == 0)

    w()
    w("-" * 78)
    w("  1. TABLAS DE CONTINGENCIA (acierto/fallo por instancia)")
    w("-" * 78)
    for tag, m in [("POSITIVOS (fulgores >=M)", pos), ("NEGATIVOS (quietud)", neg), ("GLOBAL", np.ones_like(pos))]:
        c = diversity(ok_s[m.astype(bool)], ok_l[m.astype(bool)])
        w()
        w(f"  {tag}  (n={c['n']})")
        w(f"    {'':22s}{'BiLSTM acierta':>16s}{'BiLSTM falla':>16s}")
        w(f"    {'Swin3D acierta':22s}{c['n11']:>16d}{c['n10']:>16d}")
        w(f"    {'Swin3D falla':22s}{c['n01']:>16d}{c['n00']:>16d}")
        w(f"    -> solo Swin3D acierta: {c['n10']}   solo BiLSTM acierta: {c['n01']}   ambos fallan: {c['n00']}")

    # ------------------------------------------------------------------ 2. diversidad
    w()
    w("-" * 78)
    w("  2. MEDIDAS DE DIVERSIDAD (Kuncheva & Whitaker 2003)")
    w("-" * 78)
    w(f"    {'Estrato':<26s}{'Q de Yule':>12s}{'Desacuerdo D':>16s}{'rho oraculo':>14s}")
    for tag, m in [("Global", np.ones_like(pos)), ("Solo positivos", pos), ("Solo negativos", neg)]:
        c = diversity(ok_s[m.astype(bool)], ok_l[m.astype(bool)])
        w(f"    {tag:<26s}{c['Q']:>12.4f}{c['D']:>16.4f}{c['rho']:>14.4f}")
    w()
    w("    Lectura: Q -> +1  errores totalmente correlacionados (sin diversidad);")
    w("             Q ->  0  ramas independientes; D = fraccion de ejemplos en desacuerdo.")
    w("    NOTA: la fila 'Global' esta inflada por el acuerdo trivial en verdaderos")
    w("          negativos (desbalance ~1:22). Las filas estratificadas son las informativas.")

    # -------------------------------------------------------------------- 3. spearman
    rho_all, _ = spearmanr(p_swin, p_lstm)
    rho_pos, _ = spearmanr(p_swin[pos], p_lstm[pos])
    rho_neg, _ = spearmanr(p_swin[neg], p_lstm[neg])
    w()
    w("-" * 78)
    w("  3. CORRELACION DE RANGOS DE SPEARMAN (probabilidades, no decisiones)")
    w("-" * 78)
    w(f"    Global        : rho = {rho_all:+.4f}")
    w(f"    Solo positivos: rho = {rho_pos:+.4f}")
    w(f"    Solo negativos: rho = {rho_neg:+.4f}")
    w("    Lectura: rho alta => ambas ramas ORDENAN el test de forma parecida,")
    w("             i.e. usan la misma senal para la tarea aunque la entrada difiera.")

    # ----------------------------------------------------------------------- 4. techo
    rules, masks = enumerate_rules(pred_swin, pred_lstm, y)
    w()
    w("-" * 78)
    w("  4. TECHO EXACTO DE LA FUSION A NIVEL DE DECISION")
    w("-" * 78)
    w("    Las 2 decisiones binarias parten el test en 4 celdas. CUALQUIER regla que")
    w("    use solo esas decisiones es una de las 16 asignaciones. Enumeradas todas:")
    w()
    w(f"    Ocupacion de celdas (pred_Swin, pred_BiLSTM):")
    for c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        m = masks[c]
        w(f"      ({c[0]},{c[1]}): n={int(m.sum()):6d}   positivos={int((m & pos).sum()):5d}   negativos={int((m & neg).sum()):6d}")
    w()
    w(f"    {'Rank':>4s}  {'Regla(00,01,10,11)':<22s}{'TSS':>9s}{'TP':>7s}{'FP':>8s}{'FN':>6s}   alias")
    for i, (tss, name, _, tp, fp, fn, tn) in enumerate(rules[:6], 1):
        w(f"    {i:>4d}  {name:<22s}{tss:>9.4f}{tp:>7d}{fp:>8d}{fn:>6d}   {RULE_ALIAS.get(name,'')}")
    best_tss, best_name = rules[0][0], rules[0][1]
    w()
    w(f"    TECHO = {best_tss:.4f}  (regla '{best_name}' {RULE_ALIAS.get(best_name,'')})")
    w(f"    Mejor rama individual = {max(tss_s, tss_l):.4f}"
      f"  ({'Swin3D' if tss_s >= tss_l else 'BiLSTM'})")
    gain = best_tss - max(tss_s, tss_l)
    w(f"    MARGEN DISPONIBLE = {gain:+.4f} TSS")
    if best_name in ("0101", "0011"):
        w("    => La mejor regla ES una rama sola: la fusion a nivel de decision es")
        w("       demostrablemente inutil aqui, sea cual sea la arquitectura.")
    else:
        w("    => Existe una regla de decision mejor que ambas ramas: hay margen que")
        w("       las fusiones probadas (reglas fijas, ensemble ponderado, stacking) no tomaron.")

    # techo de recall
    union_tp = int(((pred_swin == 1) | (pred_lstm == 1))[pos].sum())
    w()
    w(f"    Techo de recall: positivos capturados por AL MENOS una rama = {union_tp}/{npos}"
      f"  (POD_union = {union_tp/npos:.4f})")
    w(f"    Swin3D solo captura {tp_s}/{npos} (POD = {tp_s/npos:.4f});"
      f" BiLSTM solo {tp_l}/{npos} (POD = {tp_l/npos:.4f})")
    inter_fp = int(((pred_swin == 1) & (pred_lstm == 1))[neg].sum())
    union_fp = int(((pred_swin == 1) | (pred_lstm == 1))[neg].sum())
    w(f"    Falsas alarmas: Swin3D={fp_s}  BiLSTM={fp_l}  interseccion={inter_fp}  union={union_fp}")
    jac = inter_fp / (union_fp + 1e-12)
    w(f"    Solapamiento de FP (Jaccard) = {jac:.4f}"
      f"   [1.0 = mismas falsas alarmas; 0.0 = disjuntas]")
    return dict(horizon=horizon, tss_s=tss_s, tss_l=tss_l, ceiling=best_tss,
                best_rule=best_name, gain=gain, jaccard_fp=jac)




# ---------------------------------------------------------------- 5. honestidad
def paired_delta_preds(pred_a, pred_b, y, B=10000, chunk=2000, seed=42):
    """IC95% de D=TSS_a-TSS_b con remuestreo estratificado pareado sobre DECISIONES."""
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator(device=dev).manual_seed(seed)
    yt = torch.as_tensor(y, dtype=torch.int64, device=dev)
    pa = torch.as_tensor(pred_a, dtype=torch.float64, device=dev)
    pb = torch.as_tensor(pred_b, dtype=torch.float64, device=dev)
    pos, neg = (yt == 1), (yt == 0)
    pa_pos, pa_neg = pa[pos], pa[neg]
    pb_pos, pb_neg = pb[pos], pb[neg]
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    deltas, done = [], 0
    while done < B:
        b = min(chunk, B - done)
        ip   = torch.randint(0, n_pos, (b, n_pos), generator=g, device=dev)
        ineg = torch.randint(0, n_neg, (b, n_neg), generator=g, device=dev)
        ta = pa_pos[ip].sum(1) / n_pos - pa_neg[ineg].sum(1) / n_neg
        tb = pb_pos[ip].sum(1) / n_pos - pb_neg[ineg].sum(1) / n_neg
        deltas.append((ta - tb).cpu()); done += b
    import numpy as _np
    d = torch.cat(deltas).numpy()
    return dict(delta=float(d.mean()),
                ci=(float(_np.percentile(d, 2.5)), float(_np.percentile(d, 97.5))),
                p=float((d > 0).mean()))


def apply_rule(name, pa, pb):
    cells = [(0, 0), (0, 1), (1, 0), (1, 1)]
    pred = np.zeros(len(pa), dtype=int)
    for bit, c in zip(name, cells):
        if bit == "1":
            pred[(pa == c[0]) & (pb == c[1])] = 1
    return pred


def honest_check(horizon, lines):
    """La regla elegida sobre TEST es un techo optimista. Aqui se hace lo honesto:
    (a) seleccion de regla por fold sobre VAL, aplicada al test de ese fold;
    (b) bootstrap pareado del consenso AND (regla a priori, Kittler) vs cada rama."""
    def w(s=""):
        print(s); lines.append(s)

    w()
    w("-" * 78)
    w("  5. CONTROL DE HONESTIDAD (el techo de la seccion 4 se eligio viendo el test)")
    w("-" * 78)

    # (a) seleccion de regla por fold sobre VAL
    per_fold = []
    for k in FOLDS:
        vs, vl = load_split("swin3d", horizon, k, "val"),  load_split(LSTM_MODEL, horizon, k, "val")
        ts, tl = load_split("swin3d", horizon, k, "test"), load_split("lstm", horizon, k, "test")
        tau_s, _ = sweep_tau(sigmoid(vs[0]), vs[1])
        tau_l, _ = sweep_tau(sigmoid(vl[0]), vl[1])
        va = (sigmoid(vs[0]) >= tau_s).astype(int); vb = (sigmoid(vl[0]) >= tau_l).astype(int)
        ta = (sigmoid(ts[0]) >= tau_s).astype(int); tb = (sigmoid(tl[0]) >= tau_l).astype(int)
        rules_val, _ = enumerate_rules(va, vb, vs[1])
        best = rules_val[0][1]
        tss_test, *_ = tss_from_pred(apply_rule(best, ta, tb), ts[1])
        tss_s_k, *_  = tss_from_pred(ta, ts[1])
        tss_l_k, *_  = tss_from_pred(tb, ts[1])
        per_fold.append((k, best, tss_test, tss_s_k, tss_l_k))

    w()
    w("  (a) Regla seleccionada sobre VAL de cada fold, aplicada al TEST de ese fold:")
    w(f"      {'Fold':<7s}{'regla val':<12s}{'alias':<18s}{'TSS test':>10s}{'Swin3D':>10s}{'BiLSTM':>10s}")
    for k, best, t, ts_, tl_ in per_fold:
        w(f"      k={k:<5d}{best:<12s}{RULE_ALIAS.get(best,''):<18s}{t:>10.4f}{ts_:>10.4f}{tl_:>10.4f}")
    arr = np.array([p[2] for p in per_fold])
    aS  = np.array([p[3] for p in per_fold]); aL = np.array([p[4] for p in per_fold])
    w(f"      {'mu+-sigma':<37s}{arr.mean():>10.4f}{aS.mean():>10.4f}{aL.mean():>10.4f}")
    w(f"      {'':<37s}{arr.std():>10.4f}{aS.std():>10.4f}{aL.std():>10.4f}")
    chosen = set(p[1] for p in per_fold)
    w(f"      Reglas distintas elegidas entre folds: {len(chosen)}  -> {sorted(chosen)}")
    if len(chosen) > 1:
        w("      => la regla optima NO es estable entre folds: no hay sinergia reproducible.")

    # (b) bootstrap pareado del consenso AND (regla a priori)
    p_swin, tau_swin, y = ensemble_probs("swin3d", horizon)
    p_lstm, tau_lstm, _ = ensemble_probs(LSTM_MODEL, horizon)
    pa = (p_swin >= tau_swin).astype(int); pb = (p_lstm >= tau_lstm).astype(int)
    and_pred = apply_rule("0001", pa, pb)
    t_and, *_ = tss_from_pred(and_pred, y)
    t_s, *_   = tss_from_pred(pa, y)
    t_l, *_   = tss_from_pred(pb, y)

    w()
    w("  (b) Bootstrap pareado del consenso AND (regla A PRIORI de Kittler, no elegida")
    w("      sobre test) contra cada rama individual — B=10000, estratificado:")
    for nm, pred_other, t_other in [("Swin3D", pa, t_s), ("BiLSTM", pb, t_l)]:
        r = paired_delta_preds(and_pred, pred_other, y)
        lo, hi = r["ci"]
        sig = "SIGNIFICATIVO" if (lo > 0 or hi < 0) else "no significativo"
        w(f"      AND vs {nm:<8s} TSS {t_and:.4f} vs {t_other:.4f}   "
          f"D={r['delta']:+.4f}  IC95%=[{lo:+.4f}, {hi:+.4f}]  P(AND>{nm})={100*r['p']:.1f}%  -> {sig}")
    w()
    w("      Recordatorio: la regla 'Minimo' sobre PROBABILIDADES (aprox. AND, con tau")
    w("      re-barrido en val) ya fue evaluada en late_fusion_rules.py y quedo por")
    w("      debajo de ambas ramas. El consenso solo aparece competitivo cuando se")
    w("      congelan los tau individuales y se opera a nivel de decision.")



# --------------------------------------------- 6. techo estructural (tau libre)
def matched_pod(p_swin, p_lstm, y, lines):
    """Capturas exclusivas con POD IGUALADO: aisla la diversidad del punto de operacion.

    En la seccion 1 las capturas exclusivas se miden en el tau honesto, donde una
    rama puede tener mucho mas POD que la otra y contener trivialmente sus aciertos.
    Aqui se fija el mismo POD en ambas y se cuenta la diversidad real.
    """
    def w(s=""):
        print(s); lines.append(s)
    pos = (y == 1)
    w()
    w("-" * 78)
    w("  6. DIVERSIDAD CON POD IGUALADO (control del punto de operacion)")
    w("-" * 78)
    w(f"    {'POD fijado':<12s}{'solo Swin3D':>14s}{'solo BiLSTM':>14s}{'ambos':>9s}{'FP swin':>10s}{'FP lstm':>10s}")
    for target in (0.80, 0.85, 0.90, 0.95):
        qs = np.quantile(p_swin[pos], 1 - target); ql = np.quantile(p_lstm[pos], 1 - target)
        a = (p_swin >= qs) & pos; b = (p_lstm >= ql) & pos
        fs = int(((p_swin >= qs) & (y == 0)).sum()); fl = int(((p_lstm >= ql) & (y == 0)).sum())
        w(f"    {target:<12.2f}{int((a & ~b).sum()):>14d}{int((b & ~a).sum()):>14d}"
          f"{int((a & b).sum()):>9d}{fs:>10d}{fl:>10d}")
    w("    (a POD igualado las capturas exclusivas son simetricas por construccion)")
    w("    Lectura: SI existe diversidad a nivel de instancia cuando se igualan los POD;")
    w("             lo que la seccion 7 muestra es que esa diversidad NO es productiva en TSS.")


def structural_ceiling(p_swin, p_lstm, y, lines):
    """Cota superior ABSOLUTA: optimiza los DOS umbrales Y la regla sobre el TEST.

    Es deliberadamente tramposa a favor de la fusion (selecciona sobre el test, algo
    inalcanzable en la practica). Si aun asi el margen es ~0, ninguna fusion a nivel
    de decision puede aportar aqui, con cualquier arquitectura o calibracion.
    """
    def w(s=""):
        print(s); lines.append(s)
    grid = np.arange(0.05, 1.0, 0.05)
    bs = max(tss_from_pred((p_swin >= t).astype(int), y)[0] for t in grid)
    bl = max(tss_from_pred((p_lstm >= t).astype(int), y)[0] for t in grid)
    best_single = max(bs, bl)
    top = (-9.0, None, None, None)
    for a in grid:
        pa = (p_swin >= a).astype(int)
        for b in grid:
            pb = (p_lstm >= b).astype(int)
            rules, _ = enumerate_rules(pa, pb, y)
            if rules[0][0] > top[0]:
                top = (rules[0][0], rules[0][1], a, b)
    w()
    w("-" * 78)
    w("  7. TECHO ESTRUCTURAL — umbrales Y regla optimizados sobre el TEST")
    w("-" * 78)
    w("    Cota superior deliberadamente TRAMPOSA a favor de la fusion: elige los dos")
    w("    umbrales y la regla viendo las etiquetas de test. Inalcanzable en la practica.")
    w()
    w(f"    Mejor rama individual, con el mejor tau posible : {best_single:.4f}"
      f"   (Swin3D {bs:.4f} / BiLSTM {bl:.4f})")
    w(f"    Mejor regla de decision, con los mejores tau    : {top[0]:.4f}"
      f"   regla={top[1]} {RULE_ALIAS.get(top[1],'')}  tau_swin={top[2]:.2f} tau_lstm={top[3]:.2f}")
    w(f"    MARGEN ESTRUCTURAL MAXIMO = {top[0]-best_single:+.4f} TSS")
    w()
    w("    => Este es el limite superior de lo que CUALQUIER fusion a nivel de decision")
    w("       podria aportar sobre estas dos ramas, incluso con informacion del test.")
    return top[0] - best_single


def main():
    summary = []
    for h in HORIZONS:
        lines = []
        s = run(h, lines)
        honest_check(h, lines)
        _ps, _ts, _y = ensemble_probs('swin3d', h)
        _pl, _tl, _  = ensemble_probs(LSTM_MODEL, h)
        matched_pod(_ps, _pl, _y, lines)
        s['struct'] = structural_ceiling(_ps, _pl, _y, lines)
        summary.append(s)
        out = os.path.join(OUTDIR, f"diversidad_ramas{TAG}_{h}h.txt")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n[guardado] {out}")

    print("\n" + "=" * 78)
    print("  RESUMEN")
    print("=" * 78)
    print(f"  {'Horiz.':<8s}{'Swin3D':>10s}{'BiLSTM':>10s}{'Techo':>10s}{'Margen':>10s}{'Estructural':>12s}{'Jaccard FP':>13s}  regla")
    for s in summary:
        print(f"  {str(s['horizon'])+' h':<8s}{s['tss_s']:>10.4f}{s['tss_l']:>10.4f}"
              f"{s['ceiling']:>10.4f}{s['gain']:>+10.4f}{s['struct']:>+12.4f}{s['jaccard_fp']:>13.4f}  "
              f"{s['best_rule']} {RULE_ALIAS.get(s['best_rule'],'')}")


if __name__ == "__main__":
    main()

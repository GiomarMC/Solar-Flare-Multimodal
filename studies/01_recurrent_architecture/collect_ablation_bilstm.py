"""
Colector de la ablación de arquitectura BiLSTM 48 h.

Lee outputs/logits_ablation/lstm_abl_h{H}_l{L}_48h_k{fold}_{val,test}.npz, calcula
por fold el τ óptimo sobre val (sweep) y el TSS de test, y agrega media ± σ entre
folds por configuración (hidden × capas). Reusa tss_point/sweep_tau de bootstrap_ci.

Uso:
    python graficos/collect_ablation_bilstm.py
"""
import os
import sys
import numpy as np

_RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_RAIZ, "analysis"))   # módulos compartidos
from bootstrap_ci import ROOT, sigmoid, tss_point, sweep_tau
sys.path.insert(0, ROOT)   # para importar scripts.train_lstm_ablation

ABL = os.path.join(ROOT, "outputs/logits_ablation")
OUT = os.path.join(ROOT, "graficos")
HIDDENS, LAYERS, FOLDS = [32, 64, 128], [1, 2], [0, 1, 2, 3, 4]


def n_params(hidden, num_layers, sharp_dim=17, out_dim=None):
    """Cuenta de parámetros del LSTMStandaloneModel (sin instanciar torch)."""
    import torch
    from scripts.train_lstm_ablation import LSTMStandaloneModel
    m = LSTMStandaloneModel(sharp_dim=sharp_dim, hidden=hidden,
                            out_dim=out_dim or hidden, dropout=0.4, num_layers=num_layers)
    return sum(p.numel() for p in m.parameters())


def load(tag, split):
    p = os.path.join(ABL, f"lstm_abl_{tag}_{split}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return d["logits"].astype(np.float64), d["labels"].astype(np.int64)


def config_tss(hidden, layers):
    tss = []
    for k in FOLDS:
        tag = f"h{hidden}_l{layers}_48h_k{k}"
        v, t = load(tag, "val"), load(tag, "test")
        if v is None or t is None:
            return None
        tau, _ = sweep_tau(sigmoid(v[0]), v[1])
        tss.append(tss_point(sigmoid(t[0]), t[1], tau))
    return np.array(tss)


def main():
    lines = ["Ablación de arquitectura BiLSTM — 48 h (TSS test, τ por val, media ± σ entre folds)\n",
             f"  {'hidden':>7}{'capas':>7}{'#params':>10}{'TSS μ±σ':>16}{'  TSS por fold':>28}"]
    rows = []
    for h in HIDDENS:
        for l in LAYERS:
            arr = config_tss(h, l)
            if arr is None:
                lines.append(f"  {h:>7}{l:>7}{'—':>10}{'(incompleto)':>16}")
                continue
            np_ = n_params(h, l)
            star = "  *baseline" if (h == 64 and l == 1) else ""
            perfold = " ".join(f"{x:.3f}" for x in arr)
            lines.append(f"  {h:>7}{l:>7}{np_:>10}{arr.mean():>9.3f}±{arr.std():.3f}   [{perfold}]{star}")
            rows.append((h, l, np_, arr.mean(), arr.std()))

    txt = os.path.join(OUT, "ablation_bilstm_48h.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nGuardado: {txt}")


if __name__ == "__main__":
    main()

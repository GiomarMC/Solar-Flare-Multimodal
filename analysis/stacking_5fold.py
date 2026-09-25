"""
Stacking (meta-MLP) en 5-fold a partir de los logits guardados, + bootstrap pareado.

Replica EXACTO el meta-MLP de scripts/train_stacking_meta_mlp.py:
  - MetaMLP: Linear(2->8) -> ReLU -> Linear(8->1)
  - entrada: logits CRUDOS z-scoreados con stats del VAL de cada modelo
  - BCEWithLogitsLoss(pos_weight=n_neg/n_pos), Adam lr=1e-3 wd=1e-4, Cosine, 300 ep, bs=512
  - tau_opt elegido en val, aplicado a test
Para cada fold usa los .npz de swin3d_{h}h_k{f} y lstm_{h}h_k{f} (val/test), idénticos a los
del bootstrap. Reporta por fold + CV media±σ + ensemble, y el pareado Stacking vs Swin3D / BiLSTM.

Uso: python analysis/stacking_5fold.py --horizon 48 --B 10000
"""
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bootstrap_ci import ROOT, OUTDIR, DEVICE, load_split, read_tau_opt, sigmoid, tss_point, sweep_tau
from bootstrap_paired import model_ensemble, paired_delta, fmt


class MetaMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x):
        return self.net(x).squeeze(1)


def train_meta(vis_logit_val, phys_logit_val, y_val, epochs=300, lr=1e-3, wd=1e-4, seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    # z-score con stats del val (igual que el script original)
    vm, vs = vis_logit_val.mean(),  vis_logit_val.std()  + 1e-8
    pm, ps = phys_logit_val.mean(), phys_logit_val.std() + 1e-8
    Xv = np.stack([(vis_logit_val - vm)/vs, (phys_logit_val - pm)/ps], axis=1).astype(np.float32)
    X = torch.from_numpy(Xv); y = torch.from_numpy(y_val.astype(np.float32))
    n_pos = float((y_val == 1).sum()); n_neg = float((y_val == 0).sum())
    meta = MetaMLP()
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg/max(n_pos,1)]))
    opt = torch.optim.Adam(meta.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    ds = torch.utils.data.TensorDataset(X, y)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=512, shuffle=True, generator=g)
    meta.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(meta(xb), yb); loss.backward(); opt.step()
        sch.step()
    return meta, (vm, vs, pm, ps)


def meta_probs(meta, stats, vis_logit, phys_logit):
    vm, vs, pm, ps = stats
    X = np.stack([(vis_logit - vm)/vs, (phys_logit - pm)/ps], axis=1).astype(np.float32)
    with torch.no_grad():
        z = meta(torch.from_numpy(X)).numpy()
    return sigmoid(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=48)
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--B", type=int, default=10000)
    args = ap.parse_args()
    h, folds, B = args.horizon, args.folds, args.B
    lines = [f"Stacking meta-MLP 5-fold — horizonte {h}h  B={B}"]

    fold_tss, taus_f, stack_test_probs = [], [], []
    labels_test = None
    hdr = f"\n  Fold   tau   stack_test_TSS   (base: swin / lstm)"
    print(hdr); lines.append(hdr)
    for k in folds:
        sv = load_split("swin3d", h, k, "val");  st = load_split("swin3d", h, k, "test")
        lv = load_split("lstm",   h, k, "val");  lt = load_split("lstm",   h, k, "test")
        if any(x is None for x in (sv, st, lv, lt)):
            msg = f"  k={k}: faltan logits — omito"; print(msg); lines.append(msg); continue
        vis_lv, y_val_v = sv; vis_lt, y_te_v = st
        phys_lv, y_val_p = lv; phys_lt, y_te_p = lt
        if not np.array_equal(y_val_v, y_val_p) or not np.array_equal(y_te_v, y_te_p):
            msg = (f"  k={k}: ETIQUETAS NO ALINEADAS entre swin y lstm "
                   f"(val {np.array_equal(y_val_v,y_val_p)}, test {np.array_equal(y_te_v,y_te_p)}) — omito")
            print(msg); lines.append(msg); continue
        y_val, labels_test = y_val_v, y_te_v
        meta, stats = train_meta(vis_lv, phys_lv, y_val)
        p_val = meta_probs(meta, stats, vis_lv, phys_lv)
        p_te  = meta_probs(meta, stats, vis_lt, phys_lt)
        tau, _ = sweep_tau(p_val, y_val)
        tss = tss_point(p_te, labels_test, tau)
        sw_tss = tss_point(sigmoid(st[0]), labels_test, read_tau_opt("swin3d", h, k) or 0.5)
        ls_tss = tss_point(sigmoid(lt[0]), labels_test, read_tau_opt("lstm",   h, k) or 0.5)
        fold_tss.append(tss); taus_f.append(tau); stack_test_probs.append(p_te)
        row = f"  k={k}  {tau:.2f}   {tss:.4f}        ({sw_tss:.3f} / {ls_tss:.3f})"
        print(row); lines.append(row)

    fold_tss = np.array(fold_tss)
    summ = f"\n  Stacking CV: {fold_tss.mean():.4f} ± {fold_tss.std():.4f}"
    print(summ); lines.append(summ)

    # Ensemble de stacking (promedio de probs del meta sobre el test, mismo test)
    stack_ens = np.mean(stack_test_probs, axis=0)
    tau_ens = float(np.mean(taus_f))
    ens_tss = tss_point(stack_ens, labels_test, tau_ens)
    blk = f"  Stacking ENSEMBLE: TSS={ens_tss:.4f}  (tau={tau_ens:.2f})"
    print(blk); lines.append(blk)

    # Pareado: Stacking vs Swin3D / BiLSTM (ensembles)
    swin_ens, swin_tau, lab, *_ = model_ensemble("swin3d", h, folds)
    lstm_ens, lstm_tau, _,  *_ = model_ensemble("lstm",   h, folds)
    r1 = paired_delta(stack_ens, tau_ens, swin_ens, swin_tau, lab, B=B)
    r2 = paired_delta(stack_ens, tau_ens, lstm_ens, lstm_tau, lab, B=B)
    for s in (fmt("Stacking", "Swin3D", r1), fmt("Stacking", "BiLSTM", r2)):
        print(s); lines.append(s)

    out = os.path.join(OUTDIR, f"stacking_5fold_{h}h.txt")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()

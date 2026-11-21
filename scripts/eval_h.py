"""6 s forecast for a model pair of any horizon H: 300 / H chained passes from the simulated context, contact tokens rebuilt
from the predicted states before every pass.  usage: eval_h.py device name ckpt0,ckpt1 [float32]"""
import json, os, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from dlo import soft_contact as sc
dev = torch.device(sys.argv[1]); name = sys.argv[2]; F6 = 300
models = [sc.load(p, dev)[0] for p in sys.argv[3].split(",")]; amp = False if len(sys.argv) > 4 else None

@torch.no_grad()
def forecast(ds, s, batch=64):
    K, H = ds.K, ds.H; out = []
    for i in range(0, ds.E, batch):
        es = list(range(i, min(i + batch, ds.E)))
        track = torch.from_numpy(np.stack([ds.pos[e] for e in es])).to(dev)
        for t0 in range(s, s + F6, H):
            its = [ds.item(e, t0) for e in es]
            b = {k: torch.stack([it[k] for it in its]).to(dev) for k in its[0]}
            track[:, :, t0 + 1:t0 + H + 1] = sc.predict(models, track[:, :, t0 - K + 1:t0 + 1], b, amp=amp)
        out.append(track[:, :, s + 1:s + F6 + 1].cpu().numpy())
    return np.concatenate(out)

def wander(path):
    """How far the cable-cable contact travels along both cables in the simulation [m] per episode (-1: no cable contact)."""
    d = np.load(path); f = np.linalg.norm(d["force_cable"].astype(np.float32), axis=-1); pos = d["pos"]      # (E,2,T,N)
    on = f[:, 1].sum(-1) > 0; w = np.zeros(len(f))
    for c in (0, 1):
        L = np.linalg.norm(np.diff(pos[:, c, 0], axis=1), axis=-1).sum(-1)
        arc = (f[:, c] * np.arange(f.shape[-1])).sum(-1) / np.maximum(f[:, c].sum(-1), 1e-9) / (f.shape[-1] - 1) * L[:, None]
        w += (np.abs(np.diff(arc, axis=1)) * (on[:, 1:] & on[:, :-1])).sum(1)
    return np.where(on.any(1), w, -1.0)

WBINS = (("kein Kabelkontakt", -2, 0), ("Kontakt wandert < 0,3 m", 0, 0.3), ("Kontakt wandert 0,3-1,2 m", 0.3, 1.2), ("Kontakt wandert > 1,2 m", 1.2, 1e9))
res = {}; steps = [1, 2, 5, 10, 25, 50, 100, 150, 200, 300]
for dn, path in {"val": "data/val_u2.npz", "test": "data/test_u2.npz", "unseen": "data/val_u2_ood.npz", "motion": "data/val_motion.npz", "motion_test": "data/test_motion.npz"}.items():
    if not os.path.exists(path): continue
    ds = sc.SceneWindows(path, models[0].cfg); g = np.asarray(ds.groups() if dn in ("val", "test") else ds.mode)
    keep = np.ones(ds.E, bool)                                # wrap scenes are already removed from the motion sets (data/val_motion.npz, test_motion.npz)
    wa = wander(path)
    err = np.stack([np.linalg.norm(forecast(ds, s) - ds.pos[:, :, s + 1:s + F6 + 1], axis=-1) * 100 for s in (ds.K - 1, ds.T - 1 - F6)], 1)[keep]
    per = err.mean((1, 2, 3, 4)); q = {str(k): float(per[g == k].mean()) for k in sorted(set(g))}
    q.update(alle=float(per.mean()), sem=float(per.std() / np.sqrt(len(per))), gezogen=float(err[:, :, 0].mean()), U=float(err[:, :, 1].mean()),
             at_step={k: float(err[:, :, :, k - 1].mean()) for k in steps}, pro_sekunde=[float(err[:, :, :, k * 50:(k + 1) * 50].mean()) for k in range(6)])
    q["wander"] = {n: [float(per[(wa >= a) & (wa < b)].mean()), int(((wa >= a) & (wa < b)).sum())] for n, a, b in WBINS if ((wa >= a) & (wa < b)).any()}
    q["n"] = int(keep.sum()); q["ohne_umwickeln_entfernt"] = int((~keep).sum())
    res[dn] = q
    print(name, dn, "%.2f cm" % q["alle"], {k: round(v, 2) for k, v in q.items() if isinstance(v, float)}, "| at step", {k: round(v, 3) for k, v in q["at_step"].items()}, "| wander", {k: round(v[0], 2) for k, v in q["wander"].items()}, flush=True)
json.dump(res, open(f"stats/u3/h_{name}.json", "w"), indent=1)

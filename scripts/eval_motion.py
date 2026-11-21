"""6-s forecast on robot motions of a kind that never occurs in training (data/val_motion.npz: circle, sine, jerk).
usage: eval_motion.py device"""
import json, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from dlo import soft_contact as sc
dev = torch.device(sys.argv[1]); F6 = 300
runs = {"2 s": "runs/u3/hz/H100_chain6k", "1 s": "runs/u3/hz/H50_chain6k", "0.5 s": "runs/u3/hz/H25_chain2", "20 ms": "runs/u3/one_long64"}

@torch.no_grad()
def forecast(models, ds, s, batch=64):
    K, H = ds.K, ds.H; out = []
    for i in range(0, ds.E, batch):
        es = list(range(i, min(i + batch, ds.E)))
        track = torch.from_numpy(np.stack([ds.pos[e] for e in es])).to(dev)
        for t0 in range(s, s + F6, H):
            its = [ds.item(e, t0) for e in es]
            b = {k: torch.stack([it[k] for it in its]).to(dev) for k in its[0]}
            track[:, :, t0 + 1:t0 + H + 1] = sc.predict(models, track[:, :, t0 - K + 1:t0 + 1], b)
        out.append(track[:, :, s + 1:s + F6 + 1].cpu().numpy())
    return np.concatenate(out)

res = {}
for name, d in runs.items():
    models = [sc.load(f"{d}/best_{c}.pt", dev)[0] for c in (0, 1)]
    for dn, path in {"val_motion": "data/val_motion.npz", "test_motion": "data/test_motion.npz"}.items():
        ds = sc.SceneWindows(path, models[0].cfg); g = np.array(ds.mode)
        err = np.stack([np.linalg.norm(forecast(models, ds, s) - ds.pos[:, :, s + 1:s + F6 + 1], axis=-1) * 100 for s in (ds.K - 1, ds.T - 1 - F6)], 1)
        per = err.mean((1, 2, 3, 4))
        q = {str(k): float(per[g == k].mean()) for k in sorted(set(g))}
        q.update(alle=float(per.mean()), sem={str(k): float(per[g == k].std() / np.sqrt((g == k).sum())) for k in sorted(set(g))},
                 gezogen=float(err[:, :, 0].mean()), U=float(err[:, :, 1].mean()), per_episode=[float(x) for x in per])
        res[f"{name} | {dn}"] = q
        print(name, dn, {k: round(v, 2) for k, v in q.items() if isinstance(v, float)}, flush=True)
json.dump(res, open("stats/u3/motion.json", "w"), indent=1)

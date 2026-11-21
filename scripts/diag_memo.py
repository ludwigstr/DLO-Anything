"""Does the model merely map the robot path to a cable motion? 6-s forecasts on val (both starts) with
  normal   : everything as usual
  material : bend/twist/damping/density of both cables replaced by those of another episode (robot path, state untouched)
  contact  : soft- and rigid-contact tokens switched off
If the forecast came from the robot path alone, these changes would not matter."""
import json, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from dlo import soft_contact as sc
dev = torch.device(sys.argv[1]); F = 300
runs = {"2 s": "runs/u3/hz/H100_chain6k", "0.5 s": "runs/u3/hz/H25_chain2"}

@torch.no_grad()
def forecast(models, ds, s, mode, perm, batch=74):
    K, H = ds.K, ds.H; out = []
    for i in range(0, ds.E, batch):
        es = list(range(i, min(i + batch, ds.E)))
        track = torch.from_numpy(np.stack([ds.pos[e] for e in es])).to(dev).float()
        for t0 in range(s, s + F, H):
            its = [ds.item(e, t0) for e in es]
            b = {k: torch.stack([it[k] for it in its]).to(dev) for k in its[0]}
            if mode == "material":
                other = torch.stack([ds.item(int(perm[e]), t0)["mat"] for e in es]).to(dev)
                b["mat"] = torch.cat([other[..., :4], b["mat"][..., 4:]], -1)
            track[:, :, t0 + 1:t0 + H + 1] = sc.predict(models, track[:, :, t0 - K + 1:t0 + 1], b, use_contact=mode != "contact")
        out.append(track[:, :, s + 1:s + F + 1].cpu().numpy())
    return np.concatenate(out)

res = {}
for name, d in runs.items():
    models = [sc.load(f"{d}/best_{c}.pt", dev)[0] for c in (0, 1)]
    ds = sc.SceneWindows("data/val_u2.npz", models[0].cfg); g = np.array(ds.groups())
    perm = np.roll(np.arange(ds.E), 37)
    for mode in ("normal", "material", "contact"):
        e = np.stack([np.linalg.norm(forecast(models, ds, s, mode, perm) - ds.pos[:, :, s + 1:s + F + 1], axis=-1).mean((1, 2, 3)) * 100 for s in (ds.K - 1, ds.T - 1 - F)], 1).mean(1)
        r = {"alle": float(e.mean()), **{str(q): float(e[g == q].mean()) for q in sorted(set(g))}}
        res[f"{name} {mode}"] = r; print(name, mode, {k: round(v, 2) for k, v in r.items()}, flush=True)
json.dump(res, open("stats/u3/diag_memo.json", "w"), indent=1)

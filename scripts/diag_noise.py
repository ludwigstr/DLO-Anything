"""Why do short steps drift more? Sensitivity of the chained forecast to small errors in the context.
Every model forecasts 1 s from the simulated context, once clean and once with a small perturbation of the context:
  jitter : independent noise per context frame (corrupts the velocity seen by the model)
  offset : the same noise in all context frames (shifts the positions, velocity untouched)
Reported: deviation of the perturbed from the clean forecast, relative to the size of the perturbation."""
import json, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from dlo import soft_contact as sc
dev = torch.device(sys.argv[1]); F = 50; SIG = 0.0005                    # 1 s, 0.5 mm per coordinate
runs = {"2 s": "runs/u3/hz/H100_chain6k", "1 s": "runs/u3/hz/H50_chain6k", "0.5 s": "runs/u3/hz/H25_chain2", "0.1 s": "runs/u3/hz/H5_chain", "20 ms": "runs/u3/one_long64"}

@torch.no_grad()
def forecast(models, ds, s, noise, batch=74):
    K, H = ds.K, ds.H; out = []
    for i in range(0, ds.E, batch):
        es = list(range(i, min(i + batch, ds.E)))
        track = torch.from_numpy(np.stack([ds.pos[e][:, :s + F + H + 1] for e in es])).to(dev).float()
        track[:, :, s - K + 1:s + 1] += noise[i:i + len(es)]
        for t0 in range(s, s + F, H):
            its = [ds.item(e, t0) for e in es]
            b = {k: torch.stack([it[k] for it in its]).to(dev) for k in its[0]}
            track[:, :, t0 + 1:t0 + H + 1] = sc.predict(models, track[:, :, t0 - K + 1:t0 + 1], b)
        out.append(track[:, :, s + 1:s + F + 1])
    return torch.cat(out)

res = {}; steps = [1, 5, 10, 25, 50]
for name, d in runs.items():
    models = [sc.load(f"{d}/best_{c}.pt", dev)[0] for c in (0, 1)]
    ds = sc.SceneWindows("data/val_u2.npz", models[0].cfg); s = 100
    g = torch.Generator(device=dev).manual_seed(0)
    shape = (ds.E, 2, ds.K, ds.pos.shape[-2], 3)
    jit = torch.randn(shape, device=dev, generator=g) * SIG
    off = (torch.randn((ds.E, 2, 1, shape[3], 3), device=dev, generator=g) * SIG).expand(shape)
    gt = torch.from_numpy(np.stack([ds.pos[e][:, s + 1:s + F + 1] for e in range(ds.E)])).to(dev).float()
    clean = forecast(models, ds, s, torch.zeros(shape, device=dev))
    r = {"error": {k: float((clean - gt).norm(dim=-1)[:, :, k - 1].mean() * 100) for k in steps}}
    for nn, nz in (("jitter", jit), ("offset", off)):
        p = forecast(models, ds, s, nz)
        dv = (p - clean).norm(dim=-1)                                     # (E,2,F,N)
        r[nn] = {k: float(dv[:, :, k - 1].mean() / (nz.norm(dim=-1).mean())) for k in steps}
        r[nn + "_err"] = {k: float((p - gt).norm(dim=-1)[:, :, k - 1].mean() * 100) for k in steps}
    res[name] = r
    print(name, "| error cm", {k: round(v, 3) for k, v in r["error"].items()}, "| jitter gain", {k: round(v, 2) for k, v in r["jitter"].items()},
          "| offset gain", {k: round(v, 2) for k, v in r["offset"].items()}, flush=True)
json.dump(res, open("stats/u3/diag_noise.json", "w"), indent=1)

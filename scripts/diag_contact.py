"""Does the model capture the cable-cable contact?  0.5-s model, forecast over the whole episode (as in the videos), frames of the first 6 s.
Per episode: how far the contact wanders in the simulation (path of the force-weighted contact position along both cables, in m),
whether the predicted cables touch where/when the simulated ones do (gap of the predicted cables < 1 cm), and whether the predicted
cables pass through each other (the pulled cable ends up on the other side of the U than in the simulation)."""
import json, sys, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from dlo import soft_contact as sc
dev = torch.device(sys.argv[1]); GAP = 0.01
models = [sc.load(f"runs/u3/hz/H25_chain2/best_{c}.pt", dev)[0] for c in (0, 1)]

def gaps(P):                                             # P (2,T,N,3) -> min distance node of cable 0 to polyline of cable 1, (T,N), and arc on cable 1
    p = torch.from_numpy(P).to(dev).float()
    q, arc, gap, i, ti = sc.closest_on_cable(p[0], p[1])
    return gap.cpu().numpy(), arc.cpu().numpy()

out = {}
for dn, path in {"val": "data/val_u2.npz", "unseen": "data/val_u2_ood.npz", "motion": "data/val_motion.npz"}.items():
    ds = sc.SceneWindows(path, models[0].cfg); d = np.load(path); K = ds.K; sl = slice(K, K + 300)
    with torch.no_grad(): P = sc.rollout(models, ds, list(range(ds.E)), dev)
    rows = []
    for e in range(ds.E):
        f = np.linalg.norm(d["force_cable"][e].astype(np.float32), axis=-1)[:, sl]         # (2,T,N)
        on = f[1].sum(-1) > 0
        L = [float(np.linalg.norm(np.diff(ds.pos[e, c, 0], axis=0), axis=1).sum()) for c in (0, 1)]
        wander = 0.0
        for c in (0, 1):
            a = (f[c] * np.arange(f.shape[-1])).sum(-1) / np.maximum(f[c].sum(-1), 1e-9) / (f.shape[-1] - 1) * L[c]
            both = on[1:] & on[:-1]; wander += float(np.abs(np.diff(a))[both].sum())
        err = float(np.linalg.norm(P[e][:, sl] - ds.pos[e][:, sl], axis=-1).mean() * 100)
        gs, arc_s = gaps(ds.pos[e][:, sl]); gp, arc_p = gaps(P[e][:, sl])
        seen = gp.min(-1) < GAP                                                             # model: predicted cables touch
        both = on & seen
        loc = float(np.abs(arc_p[np.arange(300), gp.argmin(-1)] - arc_s[np.arange(300), gs.argmin(-1)])[both].mean() * L[1] * 100) if both.any() else float("nan")
        rows.append(dict(mode=str(ds.mode[e]), err=err, contact_s=float(on.mean() * 6), wander=wander, recall=float(both.sum() / max(on.sum(), 1)) if on.any() else float("nan"),
                         false=float((seen & ~on).mean() * 6), loc_cm=loc, min_gap_pred=float(gp.min() * 100), phases=int((np.diff(on.astype(int)) == 1).sum() + on[0])))
    out[dn] = rows
    r = lambda k, m=None: np.array([x[k] for x in rows if m is None or m(x)])
    con = lambda x: x["contact_s"] > 0
    print(f"{dn}: episodes {len(rows)}, with contact {int(sum(con(x) for x in rows))} | contact frames also seen by the model {np.nanmean(r('recall', con)) * 100:.0f} % | "
          f"contact seen without contact in the simulation {r('false').mean():.2f} s per episode | position of the contact on the U off by {np.nanmean(r('loc_cm', con)):.1f} cm")
    w, er = r("wander", con), r("err", con)
    edges = [0, 0.1, 0.3, 0.6, 1.2, 99]
    print("   error by contact wander [m]:", {f"{a}-{b}": (round(float(er[(w >= a) & (w < b)].mean()), 2), int(((w >= a) & (w < b)).sum())) for a, b in zip(edges[:-1], edges[1:]) if ((w >= a) & (w < b)).any()},
          "| no contact:", round(float(r("err", lambda x: not con(x)).mean()), 2), "| rank corr %.2f" % np.corrcoef(np.argsort(np.argsort(w)), np.argsort(np.argsort(er)))[0, 1])
json.dump(out, open("stats/u3/diag_contact.json", "w"), indent=1)
m = out["motion"]
for e in (162, 288, 5, 38, 196, 187): print(e, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m[e].items()})

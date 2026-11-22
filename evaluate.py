"""Evaluate a model pair on the 6-s forecast, and fit the scaling law of the paper.

    python evaluate.py --models runs/F_12000/model_cable0.pt,runs/F_12000/model_cable1.pt --data data/test.npz
    python evaluate.py --fit

Forecast: 0.2 s of simulated context, then twelve chained 0.5-s predictions without any further access to the simulation.
The error is the distance between predicted and simulated node positions, averaged over the nodes of both cables, the
300 predicted steps and two start times per scene (start and end of the scene).

Scaling law for held-out robot motions (N training scenes, k motion kinds):   E(N, k) = a k^-beta + c (N / 1000)^-gamma"""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from model import SceneWindows, load, predict


FORECAST = 300        # steps of the forecast (6 s at 50 Hz)
# 6-s error [cm] on the test set with held-out motions (circle, sine, jerk; 341 scenes): set -> {training scenes: error}
KINDS = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 8}
RESULTS = {
    "A": {371: 5.26, 743: 4.53, 1486: 4.10, 2971: 3.67, 6000: 3.48, 12000: 3.32, 24000: 3.42},
    "B": {371: 4.65, 743: 3.94, 1486: 3.43, 2971: 3.18, 6000: 2.86, 12000: 2.77, 24000: 2.75},
    "C": {12000: 2.68}, "D": {12000: 2.52}, "E": {12000: 2.50},
    "F": {371: 3.80, 743: 3.47, 1486: 3.16, 2968: 2.66, 6000: 2.47, 12000: 2.33, 24000: 2.22},
}


@torch.no_grad()
def forecast(models, ds, start, dev, batch=64):
    """Chained forecast of FORECAST steps from step `start` for all scenes -> (E,2,FORECAST,N,3) [m]."""
    K, H, out = ds.K, ds.H, []
    for i in range(0, ds.E, batch):
        es = list(range(i, min(i + batch, ds.E)))
        track = torch.from_numpy(np.stack([ds.pos[e] for e in es])).to(dev)
        for t0 in range(start, start + FORECAST, H):
            its = [ds.item(e, t0) for e in es]
            b = {k: torch.stack([it[k] for it in its]).to(dev) for k in its[0]}
            track[:, :, t0 + 1:t0 + H + 1] = predict(models, track[:, :, t0 - K + 1:t0 + 1], b)
        out.append(track[:, :, start + 1:start + FORECAST + 1].cpu().numpy())
    return np.concatenate(out)


def evaluate(args):
    dev = torch.device(args.device); models = [load(p, dev)[0] for p in args.models.split(",")]
    ds = SceneWindows(args.data, models[0].cfg); starts = (ds.K - 1, ds.T - 1 - FORECAST)
    err = np.stack([np.linalg.norm(forecast(models, ds, s, dev) - ds.pos[:, :, s + 1:s + FORECAST + 1], axis=-1) * 100 for s in starts], 1)   # (E,2 starts,2 cables,steps,N)
    per = err.mean((1, 2, 3, 4))
    res = dict(scenes=int(ds.E), error_cm=float(per.mean()), standard_error_cm=float(per.std() / np.sqrt(len(per))),
               pulled_cable_cm=float(err[:, :, 0].mean()), fixed_cable_cm=float(err[:, :, 1].mean()),
               per_motion={str(k): float(per[ds.mode == k].mean()) for k in sorted(set(ds.mode))},
               per_second=[float(err[:, :, :, k * 50:(k + 1) * 50].mean()) for k in range(FORECAST // 50)])
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


def fit():
    from scipy.optimize import least_squares
    k = np.array([KINDS[s] for s, r in RESULTS.items() for _ in r], float); n = np.array([n for r in RESULTS.values() for n in r], float)
    y = np.array([e for r in RESULTS.values() for e in r.values()])
    law = lambda q, k, n: q[0] * k ** -q[1] + q[2] * (n / 1000) ** -q[3]
    solve = lambda m, q0: least_squares(lambda q: np.log(law(q, k[m], n[m])) - np.log(y[m]), q0, bounds=([0, 0, 0, 0], [20, 3, 50, 5]))
    s = solve(np.ones(len(y), bool), [3, 0.2, 1, 0.8])
    r2 = 1 - (s.fun ** 2).sum() / ((np.log(y) - np.log(y).mean()) ** 2).sum()
    loo = [np.log(law(solve(np.arange(len(y)) != i, s.x).x, k[i], n[i])) - np.log(y[i]) for i in range(len(y))]
    print(f"E(N, k) = a k^-beta + c (N / 1000)^-gamma     fitted to {len(y)} training runs")
    print(f"a = {s.x[0]:.2f} cm, beta = {s.x[1]:.2f}, c = {s.x[2]:.2f} cm, gamma = {s.x[3]:.2f}   R2 = {r2:.3f}, leave-one-out error {100 * np.sqrt(np.mean(np.square(loo))):.1f} %")
    print("set  kinds  scenes  measured  law [cm]")
    for name, r in RESULTS.items():
        for scenes, e in r.items():
            print(f"{name:>3s} {KINDS[name]:6d} {scenes:7d} {e:9.2f} {law(s.x, KINDS[name], scenes):9.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="checkpoints 'pulled cable,fixed U'"); ap.add_argument("--data", help="scene file of simulate.py")
    ap.add_argument("--out", default=None, help="write the result as json"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fit", action="store_true", help="fit the scaling law to the results of the paper")
    args = ap.parse_args()
    fit() if args.fit else evaluate(args)


if __name__ == "__main__":
    main()

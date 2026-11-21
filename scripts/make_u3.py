"""Nested training subsets for the contact scaling study.
A: scenes of the original generator (train_u2 + new chunks).  B: every second episode is a broad random-waypoint scene."""
import numpy as np, glob, os, sys
KEYS = ["pos", "pose", "action", "material", "force_cable", "mode"]
NS = [371, 743, 1486, 2971, 6000, 12000, 24000]
def pool(files):
    ds = [np.load(f) for f in files]
    return ds, np.cumsum([0] + [len(d["mode"]) for d in ds])
def take(ds, cum, n):
    out = {k: [] for k in KEYS}
    for d, a in zip(ds, cum[:-1]):
        m = max(0, min(n - a, len(d["mode"])))
        if m:
            for k in KEYS: out[k].append(d[k][:m])
    return {k: np.concatenate(v) for k, v in out.items()}
na, ca = pool(["data/train_u2.npz"] + sorted(glob.glob("data/u3/narrow_*.npz")))
nb, cb = pool(sorted(glob.glob("data/u3/broad_*.npz")))
for n in NS:
    f = f"data/u3/A_{n}.npz"
    if not os.path.exists(f) and ca[-1] >= n:
        np.savez(f + ".tmp.npz", **take(na, ca, n)); os.rename(f + ".tmp.npz", f); print("wrote", f, flush=True)
    f = f"data/u3/B_{n}.npz"; h = n // 2
    if not os.path.exists(f) and ca[-1] >= n - h and len(nb) and cb[-1] >= h:
        a, b = take(na, ca, n - h), take(nb, cb, h)
        out = {}
        for k in KEYS:                                   # interleave: narrow, broad, narrow, ...
            x = np.empty((n,) + a[k].shape[1:], a[k].dtype); x[0::2] = a[k]; x[1::2] = b[k]; out[k] = x
        np.savez(f + ".tmp.npz", **out); os.rename(f + ".tmp.npz", f); print("wrote", f, flush=True)

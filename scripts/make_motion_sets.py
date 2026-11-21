"""Motion validation / test sets: remove the scenes in which the pulled cable wraps around the U.
usage: make_motion_sets.py raw.npz out.npz      (raw.npz.wrap.npy must exist: python wrap.py <device> raw.npz)
e.g.   python -m dlo.generate_u_ood --families circle,sine,jerk --per-family 110 --seed 995000 --out data/val_motion_raw.npz
       python -m dlo.generate_u_ood --families circle,sine,jerk --per-family 125 --seed 996000 --out data/test_motion_raw.npz
These files are validation and test data only; circle, sine and jerk motions are never used for training."""
import sys, numpy as np
src, dst = sys.argv[1], sys.argv[2]
d = np.load(src); w = np.load(src + ".wrap.npy"); keep = w <= 360
out = {k: (d[k][keep] if d[k].ndim and len(d[k]) == len(w) else d[k]) for k in d.files}; out["winding"] = w[keep]
np.savez_compressed(dst, **out)
print(dst, int(keep.sum()), "scenes kept,", int((~keep).sum()), "wrap scenes removed", dict(zip(*np.unique(out["mode"], return_counts=True))))

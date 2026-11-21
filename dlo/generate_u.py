"""Generate a dataset of the U scenario: cable 2 fixed at both ends, cable 1 pulled through it.

Usage:
    python -m dlo.generate_u --episodes 2500 --out data/train_u.npz --seed 410000 --workers 24

Arrays (E = stable episodes, c = cable 0/1, T steps, N nodes):
    pos (E,2,T,N,3)  pose (E,2,T,3)  action (E,2,T,3)  material (E,2,6)
    force (E,2,T,N,3) float16 [N]: total contact force on each node (world frame) = force_cable + force_floor
    force_cable: force of the other cable, force_floor: force of the floor (u2 data; u data: force = cable only)
    n_contact (E,T) int16, mode (E) str, seed (E)
"""
from __future__ import annotations
import argparse, time, os
from multiprocessing import Pool
import numpy as np

from .config import SimConfig
from . import u_env

_CFG = SimConfig()
_CFG.style = "mix"


def _worker(seed: int):
    try:
        return u_env.generate_episode_u(_CFG, seed)
    except Exception as e:
        return dict(ok=False, seed=seed, error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--out", default="data/train_two_1.npz")
    ap.add_argument("--seed", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    if args.steps:
        _CFG.episode_steps = args.steps
    seeds = list(range(args.seed, args.seed + args.episodes))
    t0 = time.time()
    eps = []
    with Pool(args.workers) as pool:
        for i, ep in enumerate(pool.imap_unordered(_worker, seeds, chunksize=1)):
            eps.append(ep)
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{args.episodes} episodes, {time.time() - t0:.0f}s", flush=True)
    n_bad = sum(not e["ok"] for e in eps)
    eps = [e for e in eps if e["ok"]]
    eps.sort(key=lambda e: e["seed"])
    touch = np.array([(e["n_contact"] > 0).mean() for e in eps])
    print(f"done: {len(eps)} ok, {n_bad} unstable, {time.time() - t0:.0f}s | steps in contact: mean {touch.mean() * 100:.1f} %, "
          f"episodes with contact {np.mean(touch > 0) * 100:.0f} %")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        pos=np.stack([e["pos"] for e in eps]), pose=np.stack([e["pose"] for e in eps]),
        action=np.stack([e["action"] for e in eps]), material=np.stack([e["material"] for e in eps]),
        force=np.stack([e["force"] + e["force_floor"] for e in eps]).astype(np.float16),       # total: cable + floor
        force_cable=np.stack([e["force"] for e in eps]).astype(np.float16),
        force_floor=np.stack([e["force_floor"] for e in eps]).astype(np.float16),
        n_contact=np.stack([e["n_contact"] for e in eps]), anchor_b=np.stack([e["anchor_b"] for e in eps]), mode=np.array([e["mode"] for e in eps]),
        seed=np.array([e["seed"] for e in eps]), control_dt=_CFG.control_dt, style=_CFG.style)
    print("saved", args.out)


if __name__ == "__main__":
    main()

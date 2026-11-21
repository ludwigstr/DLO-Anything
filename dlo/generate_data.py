"""Generate a dataset of random cable episodes in parallel.

Usage:
    python -m dlo.generate_data --episodes 4000 --out data/train.npz --seed 0
"""
from __future__ import annotations
import argparse, time, os
from multiprocessing import Pool
import numpy as np

from .config import SimConfig
from . import cable_env


_CFG = SimConfig()


def _worker(seed: int):
    try:
        return cable_env.generate_episode(_CFG, seed)
    except Exception as e:   # e.g. mujoco.FatalError from the integrator -> count as unstable
        return dict(ok=False, seed=seed, error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--out", type=str, default="data/train.npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 4))
    ap.add_argument("--style", default=None, help="trajectory style: long | diverse | mix (default: SimConfig)")
    ap.add_argument("--steps", type=int, default=None, help="control steps per episode (default: SimConfig)")
    args = ap.parse_args()
    if args.style:
        _CFG.style = args.style        # set before the Pool is forked
    if args.steps:
        _CFG.episode_steps = args.steps

    seeds = list(range(args.seed, args.seed + args.episodes))
    t0 = time.time()
    eps = []
    with Pool(args.workers) as pool:
        for i, ep in enumerate(pool.imap_unordered(_worker, seeds, chunksize=2)):
            eps.append(ep)
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{args.episodes} episodes, {time.time() - t0:.0f}s", flush=True)
    n_bad = sum(not e["ok"] for e in eps)
    eps = [e for e in eps if e["ok"]]
    eps.sort(key=lambda e: e["seed"])
    print(f"done: {len(eps)} ok, {n_bad} unstable, {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        pos=np.stack([e["pos"] for e in eps]),            # (E, T, N, 3)
        pose=np.stack([e["pose"] for e in eps]),          # (E, T, 3)  x, y, yaw
        action=np.stack([e["action"] for e in eps]),      # (E, T, 3)  dx, dy, dyaw
        material=np.stack([e["material"] for e in eps]),  # (E, 5)
        seed=np.array([e["seed"] for e in eps]),
        control_dt=_CFG.control_dt, gripper_height=_CFG.gripper_height,
        cable_length=_CFG.cable_length, style=_CFG.style,
    )
    print("saved", args.out)


if __name__ == "__main__":
    main()

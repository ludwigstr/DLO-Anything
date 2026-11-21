"""Out-of-distribution validation scenes for the U scenario: layouts that never occur in train_u2.

Training scenes (dlo.u_env): the gripper starts 0.3-0.7 m in front of the U, crosses it once (or there and back)
close to the normal of the U plane (at most 35 degrees off) and within the middle 70 % of the span.

Families here (stored in `mode`):
    far      start 1.0-1.5 m in front of the U: contact much later than in training
    near     start 0.08-0.2 m in front of the U: contact almost immediately
    miss     training-like straight/angled path, shifted sideways past the end of the U: no cable contact
    oblique  crossing 55-72 degrees off the normal of the U plane
    multi    three or four crossings with pauses of up to 1.5 s

Motion families (robot motions of a kind that the training generator never produces; --families circle,sine,jerk):
    circle   one to several full circles (radius 0.15-0.45 m) that cut through the plane of the U
    sine     slow crossing with a sideways sine oscillation in time (0.4-1.2 Hz, amplitude 0.1-0.3 m)
    jerk     many short moves (5-30 cm) that start and stop within one control step, with short pauses

Usage:
    python -m dlo.generate_u_ood --per-family 80 --out data/val_u2_ood.npz --seed 990000 --workers 40
"""
from __future__ import annotations
import argparse, time, os
from multiprocessing import Pool
import numpy as np

from .config import SimConfig, sample_material
from . import u_env

FAMILIES = ["far", "near", "miss", "oblique", "multi"]
_CFG = SimConfig()
_CFG.style = "mix"


def _path(fam, g, rng, side):
    """Gripper path pieces in U coordinates (x along the span, y along the normal), speed range, pause range."""
    w = g["span"]
    xc = rng.uniform(-0.35, 0.35) * w
    line = lambda p, q: np.linspace(p, q, 600)
    if fam == "far":
        d0, d1 = rng.uniform(1.0, 1.5), rng.uniform(0.3, 0.6)
        t = np.tan(rng.uniform(-0.3, 0.3))
        return [line([xc - t * d0, -side * d0], [xc + t * d1, side * d1])], (0.4, 0.8), (0, 0)
    if fam == "near":
        d0, d1 = rng.uniform(0.08, 0.2), rng.uniform(0.3, 0.9)
        t = np.tan(rng.uniform(-0.5, 0.5))
        return [line([xc - t * d0, -side * d0], [xc + t * d1, side * d1])], (0.15, 0.8), (0, 0)
    if fam == "miss":
        xs = rng.choice([-1, 1]) * (0.5 * w + rng.uniform(0.25, 0.6))
        d0, d1 = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.9)
        t = np.tan(rng.uniform(-0.3, 0.3)) * (-np.sign(xs))          # never steers towards the U
        return [line([xs + t * d0, -side * d0], [xs - t * d1, side * d1])], (0.15, 0.8), (0, 0)
    if fam == "oblique":
        d0, d1 = rng.uniform(0.3, 0.6), rng.uniform(0.3, 0.6)
        t = np.tan(np.radians(rng.uniform(55, 72))) * rng.choice([-1, 1])
        return [line([xc - t * d0, -side * d0], [xc + t * d1, side * d1])], (0.3, 0.8), (0, 0)
    if fam == "multi":
        n = int(rng.integers(3, 5)); pts = [[xc, -side * rng.uniform(0.3, 0.6)]]
        for i in range(n):
            s = side if i % 2 == 0 else -side
            pts.append([rng.uniform(-0.35, 0.35) * w, s * rng.uniform(0.25, 0.6)])
        return [line(pts[i], pts[i + 1]) for i in range(n)], (0.5, 0.9), (0.2, 1.5)
    if fam == "random":
        # broad training scenes: random waypoints anywhere around the U, no designed crossing
        r = 0.5 * w + 0.3
        ang = rng.uniform(-np.pi, np.pi); d = rng.uniform(0.2, 1.6)
        pts = [[d * np.cos(ang), d * np.sin(ang)]]
        for _ in range(int(rng.integers(3, 7))):
            a2, d2 = rng.uniform(-np.pi, np.pi), r * np.sqrt(rng.uniform())
            pts.append([d2 * np.cos(a2), d2 * np.sin(a2)])
        return [line(pts[i], pts[i + 1]) for i in range(len(pts) - 1)], (0.15, 1.0), (0.0, 1.5)
    raise ValueError(fam)


MOTION = ["circle", "sine", "jerk"]


def _motion(fam, g, rng, side, cfg):
    """Gripper positions (T,2) in U coordinates for the motion families, sampled directly in time."""
    w, dt, n = g["span"], cfg.control_dt, cfg.episode_steps
    xc = rng.uniform(-0.3, 0.3) * w
    wait = int(round(rng.uniform(0.2, 0.8) / dt))
    t = np.clip(np.arange(n) - wait, 0, None) * dt
    ramp = rng.uniform(0.3, 0.6)
    tau = np.where(t < ramp, ramp * (t / ramp) ** 2 / 2 * 2 * 0.5, t - ramp / 2)      # time with a smooth start (speed ramps linearly)
    if fam == "circle":
        r, v = rng.uniform(0.15, 0.45), rng.uniform(0.3, 0.9)
        yc = -side * rng.uniform(0.0, 0.7) * r                                         # the circle cuts the plane of the U
        phi = rng.choice([-1, 1]) * v / r * tau
        return np.column_stack([xc + r * np.sin(phi), yc - side * r * np.cos(phi)])     # starts at the point farthest in front of the U
    if fam == "sine":
        f = rng.uniform(0.4, 1.2); a = min(rng.uniform(0.1, 0.3), 0.16 / f)            # peak sideways speed <= 1 m/s
        d0, v = rng.uniform(0.25, 0.5), rng.uniform(0.1, 0.3)
        grow = np.clip(t / (2 * ramp), 0, 1)
        return np.column_stack([xc + a * grow * np.sin(2 * np.pi * f * t), side * (-d0 + v * tau)])
    if fam == "jerk":
        p = np.array([xc, -side * rng.uniform(0.3, 0.6)]); seq = [np.repeat(p[None], wait, axis=0)]
        while sum(len(q) for q in seq) < n:
            ang = rng.uniform(-1.2, 1.2) + (np.pi if rng.uniform() < 0.25 else 0.0)    # mostly towards the U, sometimes back
            q = p + rng.uniform(0.05, 0.3) * np.array([np.sin(ang), side * np.cos(ang)])
            q[0] = np.clip(q[0], -0.5 * w - 0.2, 0.5 * w + 0.2); q[1] = np.clip(q[1], -0.8, 0.8)
            seq.append(u_env._follow(np.linspace(p, q, 50), rng.uniform(0.5, 1.2), dt, dt))   # full speed within one control step
            seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(0.04, 0.4) / dt)), axis=0)); p = seq[-1][-1]
        return np.concatenate(seq)[:n]
    raise ValueError(fam)


def sample(cfg, rng, fam):
    mats = [sample_material(rng), sample_material(rng)]
    g = u_env.u_geometry(cfg, mats[1], rng)
    H1 = float(rng.uniform(g["z_bottom"] + 0.05, max(mats[1]["height"] - 0.02, g["z_bottom"] + 0.06)))
    mats[0]["height"] = H1
    mats[0]["length"] = float(rng.uniform(max(0.6, H1 - g["z_bottom"] + 0.05), 2.0))
    side = float(rng.choice([-1.0, 1.0]))
    if fam in MOTION:
        local = _motion(fam, g, rng, side, cfg)
        xy = g["centre"] + local[:, 0, None] * g["e"] + local[:, 1, None] * g["n"]
        return mats, g, np.column_stack([xy, np.zeros(len(xy))]), -side * g["n"]
    pieces, sp, pause = _path(fam, g, rng, side)
    speed, ramp, dt = rng.uniform(*sp), rng.uniform(0.2, 0.8), cfg.control_dt
    seq = [np.repeat(pieces[0][:1], int(round(rng.uniform(0.2, 1.0) / dt)), axis=0)]
    for p in pieces:
        seq.append(u_env._follow(p, speed, ramp, dt))
        seq.append(np.repeat(seq[-1][-1:], int(round(rng.uniform(*pause) / dt)), axis=0))
    local = np.concatenate(seq); n = cfg.episode_steps
    if len(local) < n:
        local = np.concatenate([local, np.repeat(local[-1:], n - len(local), axis=0)])
    local = local[:n]
    xy = g["centre"] + local[:, 0, None] * g["e"] + local[:, 1, None] * g["n"]
    away = -side * g["n"]
    if fam == "random":                                   # cable starts pointing away from the centre of the U
        v = xy[0] - g["centre"]; away = v / (np.linalg.norm(v) + 1e-9)
    return mats, g, np.column_stack([xy, np.zeros(n)]), away


def _worker(job):
    seed, fam = job
    try:
        rng = np.random.default_rng(seed)
        mats, g, traj, away = sample(_CFG, rng, fam)
        out, _ = u_env.run_episode_u(_CFG, mats, g, traj, away)
        out["seed"] = seed; out["mode"] = fam
        return out
    except Exception as e:
        return dict(ok=False, seed=seed, mode=fam, error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=80)
    ap.add_argument("--out", default="data/val_u2_ood.npz")
    ap.add_argument("--seed", type=int, default=990000)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--families", default=",".join(FAMILIES), help="'random' = broad training scenes with random waypoints")
    ap.add_argument("--slim", action="store_true", help="training file: keep force_cable only")
    args = ap.parse_args()
    fams = args.families.split(",")
    jobs = [(args.seed + i, fams[i % len(fams)]) for i in range(args.per_family * len(fams))]
    t0 = time.time()
    with Pool(args.workers) as pool:
        eps = list(pool.imap_unordered(_worker, jobs, chunksize=1))
    bad = [e for e in eps if not e["ok"]]
    eps = sorted([e for e in eps if e["ok"]], key=lambda e: e["seed"])
    for f in fams:
        es = [e for e in eps if e["mode"] == f]
        print(f"{f:8s} ok {len(es):3d}, rejected {sum(b['mode'] == f for b in bad):2d}, with cable contact "
              f"{np.mean([(np.abs(e['force'][1]).sum() > 0) for e in es]) * 100:.0f} %", flush=True)
    print(f"done in {time.time() - t0:.0f}s")
    np.savez_compressed(
        args.out,
        pos=np.stack([e["pos"] for e in eps]), pose=np.stack([e["pose"] for e in eps]),
        action=np.stack([e["action"] for e in eps]), material=np.stack([e["material"] for e in eps]),
        **({} if args.slim else dict(force=np.stack([e["force"] + e["force_floor"] for e in eps]).astype(np.float16),
                                     force_floor=np.stack([e["force_floor"] for e in eps]).astype(np.float16))),
        force_cable=np.stack([e["force"] for e in eps]).astype(np.float16),
        n_contact=np.stack([e["n_contact"] for e in eps]), anchor_b=np.stack([e["anchor_b"] for e in eps]),
        mode=np.array([e["mode"] for e in eps]), seed=np.array([e["seed"] for e in eps]), control_dt=_CFG.control_dt, style=_CFG.style)
    print("saved", args.out)


if __name__ == "__main__":
    main()

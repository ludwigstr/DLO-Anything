"""Videos of two cables: MuJoCo simulation (ghost) vs. the coupled prediction of two model
instances that exchange contact forces (dlo.couple).

Left panel: the two models predict independently (no force exchange, as the single-cable model
would). Right panel: the coupled prediction; estimated contact points are drawn as red dots.
The context-force channel is fed with the algorithm's own estimates from the previous windows,
so the prediction uses no simulator force at all.

Usage:
    python -m dlo.make_videos_two --ckpt runs/force1/best.pt --data data/test_two.npz --out videos/two --episodes 6
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch
import mujoco
import imageio
from PIL import ImageFont

from .config import SimConfig, material_dict, pad_material
from .two_cable_env import TwoCableSim
from .couple import CoupledPredictor, load_model
from .make_videos import Viz, compose, FONT, FONT_B

COL_PRED = ([0.95, 0.35, 0.1, 1.0], [0.95, 0.80, 0.1, 1.0])     # prediction cable 1 (orange), cable 2 (yellow)
COL_CONTACT = [0.9, 0.05, 0.05, 1.0]


class TwoViz(Viz):
    def __init__(self, sim: TwoCableSim, width=960, height=540):
        self.sim = sim
        self.r = mujoco.Renderer(sim.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 1.5; self.cam.elevation = -70; self.cam.azimuth = 140
        self.cam.lookat[:] = [0, 0, 0.10]
        self.geoms = sim.cable_geoms[0] + sim.cable_geoms[1]
        self.rgba_cable = sim.model.geom_rgba[self.geoms].copy()
        self.radius = sim.cfg.cable_radius
        self.width, self.height = width, height

    def _add_sphere(self, p, rgba, radius=0.012):
        scn = self.r.scene
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([radius, 0, 0]), np.asarray(p, np.float64),
                            np.eye(3).flatten(), np.asarray(rgba, np.float32))
        scn.ngeom += 1

    def render_pred(self, preds, contact_pts=None):
        m = self.sim.model
        rgba = self.rgba_cable.copy(); rgba[:, 3] = 0.25
        m.geom_rgba[self.geoms] = rgba                      # ghosts of the true cables
        self.r.update_scene(self.sim.data, self.cam)
        m.geom_rgba[self.geoms] = self.rgba_cable
        for c, p in enumerate(preds):
            if p is not None and len(p):
                self._add_chain(p, COL_PRED[c], radius=max(self.radius, 0.005))
        if contact_pts is not None:
            for p in contact_pts:
                self._add_sphere(p, COL_CONTACT)
        return self.r.render()


def render_episode(cp: CoupledPredictor, ep, cfg: SimConfig, out_path, fps=25, width=960, height=540,
                   frame_stride=1, true_ctx_forces=False):
    pos, pose, action = ep["pos"].astype(np.float64), ep["pose"], ep["action"]
    material = pad_material(ep["material"]); force_true = ep["force"].astype(np.float32)
    mats = [material_dict(material[c]) for c in range(2)]
    T, N = pos.shape[1], pos.shape[2]
    K, H = cp.mcfg.context_len, cp.mcfg.horizon

    # ---- consecutive open-loop windows, coupled and uncoupled -------------------------------
    pred_c = np.full(pos.shape, np.nan); pred_u = np.full(pos.shape, np.nan)
    force_est = np.zeros_like(force_true)                      # the algorithm's own force estimates (context channel)
    contact_pts = [None] * T
    window_start = np.full(T, -1)
    starts = list(range(K - 1, T - H, H))
    if starts[-1] + H < T - 1:
        starts.append(T - 1 - H)                               # last window overlaps so the prediction never disappears
    for t0 in starts:
        r = cp.predict(pos, pose, action, material, force_true if true_ctx_forces else force_est, t0)
        sl = slice(t0 + 1, t0 + H + 1)
        new = window_start[sl] < 0
        pred_c[:, sl][:, new] = r["pred"][:, new]; pred_u[:, sl][:, new] = r["pred_uncoupled"][:, new]
        force_est[:, sl][:, new] = r["force"][:, new]
        for k in np.where(new)[0]:
            contact_pts[t0 + 1 + k] = r["points"][k][r["active"][k]]
        window_start[sl][new] = t0
    err_c = np.linalg.norm(pred_c - pos, axis=-1); err_u = np.linalg.norm(pred_u - pos, axis=-1)   # (2,T,N)

    # ---- re-simulate and render ---------------------------------------------------------------
    sim = TwoCableSim(cfg, mats, start_xy=(pose[0, 0, :2], pose[1, 0, :2]))
    viz = TwoViz(sim, width, height)
    hands = np.concatenate([np.column_stack([pose[c, :, 0], pose[c, :, 1], np.full(T, mats[c]["height"])]) for c in range(2)])
    viz.frame_points(np.concatenate([pos.reshape(-1, 3), pred_c[~np.isnan(pred_c[..., 0])], hands]))
    for c in range(2):
        sim.set_gripper(c, *pose[c, 0])
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(cfg.settle_steps):
        sim.step_control()
    fs = max(15, int(round(width / 64)))
    fonts = (ImageFont.truetype(FONT, fs), ImageFont.truetype(FONT_B, fs + 2))
    mat_lines = [f"cable {c + 1}: bend {m['bend']:.1e} twist {m['twist']:.1e} damp {m['damping']:.3f} dens {m['density']:.0f} "
                 f"L {m['length']:.2f} m H {m['height']:.2f} m" for c, m in enumerate(mats)]
    n_true = ep["n_contact"]
    frames = []
    for t in range(T):
        for c in range(2):
            sim.set_gripper(c, *pose[c, t])
        if t > 0:
            sim.step_control()
        else:
            mujoco.mj_forward(sim.model, sim.data)
        if t % frame_stride:
            continue
        if window_start[t] >= 0:
            k = t - window_start[t]
            left = viz.render_pred([pred_u[0, t], pred_u[1, t]])
            right = viz.render_pred([pred_c[0, t], pred_c[1, t]], contact_pts[t])
            n_est = 0 if contact_pts[t] is None else len(contact_pts[t])
            status = (f"step {k:3d}/{H} (open-loop from t={window_start[t] * cfg.control_dt:.2f}s)   "
                      f"mean err: independent {np.nanmean(err_u[:, t]) * 100:5.2f} cm | coupled {np.nanmean(err_c[:, t]) * 100:5.2f} cm   "
                      f"contacts: sim {int(n_true[t]):2d} | estimated {n_est:2d}")
        else:
            left = right = viz.render_pred([None, None])
            status = f"collecting context ({t + 1}/{K} steps)"
        info = [f"t = {t * cfg.control_dt:5.2f} s   mode {ep['mode']}   " + status] + mat_lines
        frames.append(compose([left, right], ["Two independent model instances (no force exchange)",
                                              "Coupled: contact forces exchanged between the two instances (red = estimated contacts)"],
                              info, fonts, title_colors=((200, 200, 200), (120, 180, 255))))
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    touch = np.linalg.norm(force_true, axis=-1) > 0
    return dict(mean_err_independent_cm=float(np.nanmean(err_u) * 100), mean_err_coupled_cm=float(np.nanmean(err_c) * 100),
                contact_err_independent_cm=float(np.nanmean(err_u[touch]) * 100) if touch.any() else None,
                contact_err_coupled_cm=float(np.nanmean(err_c[touch]) * 100) if touch.any() else None,
                steps_in_contact=int((n_true > 0).sum()), mode=str(ep["mode"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/test_two.npz")
    ap.add_argument("--out", default="videos/two")
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--index", default=None, help="comma-separated episode indices")
    ap.add_argument("--min-contact", type=int, default=40, help="without --index: pick the first episodes with at least this many contact steps")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--kc", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--true-ctx-forces", action="store_true", help="feed simulator forces into the context channel instead of own estimates")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, mcfg = load_model(args.ckpt, device)
    cp = CoupledPredictor(model, mcfg, device, k_c=args.kc, n_iter=args.iters, margin=args.margin)
    cfg = SimConfig()
    d = np.load(args.data)
    os.makedirs(args.out, exist_ok=True)
    if args.index:
        indices = [int(i) for i in args.index.split(",")]
    else:
        n_c = (d["n_contact"] > 0).sum(1)
        indices = [int(i) for i in np.where(n_c >= args.min_contact)[0][:args.episodes]]
    results = {}
    for e in indices:
        ep = {k: d[k][e] for k in ["pos", "pose", "action", "material", "force", "n_contact", "mode"]}
        out_path = os.path.join(args.out, f"episode_{e:03d}.mp4")
        res = render_episode(cp, ep, cfg, out_path, fps=args.fps, frame_stride=args.frame_stride, width=args.width,
                             height=args.height, true_ctx_forces=args.true_ctx_forces)
        results[f"episode_{e:03d}"] = res
        print(f"{out_path}: {res}", flush=True)
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

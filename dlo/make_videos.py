"""Render videos of the transformer prediction over the MuJoCo simulation.

For each test episode the simulation runs along its recorded random trajectory. Every
`horizon` control steps the model receives the last `context_len` simulated states, the material
parameters and the next `horizon` gripper actions, and predicts the next `horizon` cable states
open-loop. The video shows that prediction (orange) over the true cable as a translucent ghost
in one large panel; `--split` adds the plain simulation as a second panel on the left.

Usage:
    python -m dlo.make_videos --ckpt runs/base/best.pt --data data/test.npz --out videos --episodes 8
"""
from __future__ import annotations
import argparse, os, json
import numpy as np
import torch
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from .config import SimConfig, ModelConfig, MATERIAL_NAMES, material_dict, pad_material
from .cable_env import CableSim, build_xml
from .dataset import make_sample, from_gripper_frame, POS_SCALE
from .model import PhysicsTransformer

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    mcfg = ModelConfig(**ck["cfg"])
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, mcfg


@torch.no_grad()
def predict_window(model, mcfg, pos, pose, action, material, t0, device):
    """Predict world-frame node positions for steps t0+1 .. t0+H. Returns (H, N, 3)."""
    s = make_sample(pos, pose, action, material, t0, mcfg.context_len, mcfg.horizon)
    b = {k: v[None].to(device) for k, v in s.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred = model(b["node"], b["act"], b["mat"], b["cur"]).float()[0].cpu().numpy()
    return from_gripper_frame(pred / POS_SCALE, pose[t0])


class Viz:
    """Renders the simulation, and a 'ghost' scene with user-drawn cables for predictions."""

    def __init__(self, sim: CableSim, width=640, height=480):
        self.sim = sim
        self.r = mujoco.Renderer(sim.model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 1.5
        self.cam.elevation = -70
        self.cam.azimuth = 140
        self.cam.lookat[:] = [0, 0, 0.10]
        self.rgba_cable = sim.model.geom_rgba[sim.cable_geoms].copy()
        self.radius = sim.cfg.cable_radius
        self.width, self.height = width, height

    def follow(self, xy, alpha=0.08):
        tgt = np.array([xy[0], xy[1], 0.10])
        self.cam.lookat[:] = (1 - alpha) * self.cam.lookat + alpha * tgt

    def camera_axes(self):
        """Forward / right / up unit vectors of the free camera (MuJoCo azimuth/elevation convention)."""
        az, el = np.deg2rad(self.cam.azimuth), np.deg2rad(self.cam.elevation)
        f = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        r = np.cross(f, [0.0, 0.0, 1.0]); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        return f, r, u

    def frame_points(self, pts, margin=0.15):
        """Set lookat and distance (keeping azimuth/elevation) so that every point in `pts` lies
        inside the image with a relative margin of `margin` of the half-width / half-height on
        every side. The look-at point is the centre of the points' extent in the image plane, so
        the margins are symmetric left/right and top/bottom."""
        pts = np.asarray(pts, np.float64).reshape(-1, 3)
        f, r, u = self.camera_axes()
        fovy = np.deg2rad(float(self.sim.model.vis.global_.fovy))
        ty = np.tan(fovy / 2) * (1 - margin)                 # usable half-height per unit depth
        tx = ty * self.width / self.height                   # usable half-width per unit depth
        x, y, z = pts @ r, pts @ u, pts @ f
        centre = (0.5 * (x.max() + x.min()) * r + 0.5 * (y.max() + y.min()) * u
                  + 0.5 * (z.max() + z.min()) * f)
        dx, dy, dz = x - centre @ r, y - centre @ u, z - centre @ f
        # a point at depth (d + dz) must satisfy |dx| <= (d + dz) * tx  (same for y)
        d = max(float((np.abs(dx) / tx - dz).max()), float((np.abs(dy) / ty - dz).max()), 0.3)
        self.cam.lookat[:] = centre
        self.cam.distance = d

    def _add_chain(self, nodes, rgba, radius=None):
        scn = self.r.scene
        radius = radius or self.radius
        for a, b in zip(nodes[:-1], nodes[1:]):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.asarray(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                                 np.asarray(a, np.float64), np.asarray(b, np.float64))
            scn.ngeom += 1

    def render_sim(self):
        self.r.update_scene(self.sim.data, self.cam)
        return self.r.render()

    def render_pred(self, pred_nodes, true_nodes):
        m = self.sim.model
        m.geom_rgba[self.sim.cable_geoms] = [0.0, 0.0, 1.0, 0.25]   # ghost of the true cable
        self.r.update_scene(self.sim.data, self.cam)
        m.geom_rgba[self.sim.cable_geoms] = self.rgba_cable
        self._add_chain(pred_nodes, [0.95, 0.35, 0.1, 1.0], radius=max(self.radius, 0.005))
        return self.r.render()


def compose(panels, titles, info, fonts, title_colors=((235, 235, 235), (120, 180, 255))):
    """Stack one or two rendered panels side by side under a header that holds the panel titles
    and all status / material lines (nothing below the image)."""
    h, w = panels[0].shape[:2]
    f, fb = fonts
    lh = f.size + 11                       # info line height
    top = fb.size + 17 + lh * len(info)    # header: title line + info lines
    top += (h + top) % 2                   # libx264 needs even dimensions
    gap = 10 if len(panels) > 1 else 0
    canvas = Image.new("RGB", (len(panels) * w + gap * (len(panels) - 1), h + top), (24, 26, 30))
    for i, pnl in enumerate(panels):
        canvas.paste(Image.fromarray(pnl), (i * (w + gap), top))
    dr = ImageDraw.Draw(canvas)
    for i, ttl in enumerate(titles):
        dr.text((i * (w + gap) + 10, 8), ttl, font=fb, fill=title_colors[i])
    for i, ln in enumerate(info):
        dr.text((10, fb.size + 17 + lh * i), ln, font=f, fill=(210, 210, 210))
    return np.asarray(canvas)


def render_episode(model, mcfg, ep, cfg: SimConfig, device, out_path, fps=25, width=1280, height=720,
                   frame_stride=1, split=False):
    """split=False: one large panel with the prediction (orange) over the simulation ghost.
    split=True: simulation left, prediction right (each width x height)."""
    pos, pose, action, material = ep["pos"], ep["pose"], ep["action"], pad_material(ep["material"])
    mat = material_dict(material)
    T = pos.shape[0]
    K, H = mcfg.context_len, mcfg.horizon

    # ---- model predictions for consecutive open-loop windows ------------------------------
    pred = np.full_like(pos, np.nan)
    window_start = np.full(T, -1)
    t0 = K - 1
    while t0 + H < T:
        pred[t0 + 1:t0 + H + 1] = predict_window(model, mcfg, pos, pose, action, material, t0, device)
        window_start[t0 + 1:t0 + H + 1] = t0
        t0 += H
    # tail: the remaining steps (fewer than H) come from one last window that starts earlier and
    # overlaps the previous one, so the prediction never disappears before the episode ends
    if t0 + 1 < T:
        t_last = T - 1 - H
        full = predict_window(model, mcfg, pos, pose, action, material, t_last, device)
        pred[t0 + 1:T] = full[t0 - t_last:]
        window_start[t0 + 1:T] = t_last
    err = np.linalg.norm(pred - pos, axis=-1)          # (T, N)

    # ---- re-simulate the same episode and render ----------------------------------------
    sim = CableSim(cfg, mat)
    viz = Viz(sim, width, height)
    # fixed camera per episode: every simulated / predicted node and the hand stay inside the
    # image with a margin on all sides for the whole episode
    hand = np.column_stack([pose[:, 0], pose[:, 1], np.full(T, mat["height"])])
    viz.frame_points(np.concatenate([pos.reshape(-1, 3), pred[~np.isnan(pred[..., 0])], hand]))
    sim.set_gripper(*pose[0]); mujoco.mj_forward(sim.model, sim.data)
    for _ in range(cfg.settle_steps):
        sim.step_control()
    fs = max(17, int(round(width / 64)))             # scale fonts with the panel width (20 px at 1280)
    fonts = (ImageFont.truetype(FONT, fs), ImageFont.truetype(FONT_B, fs + 2))
    mat_str = (f"bend {mat['bend']:.2e}  twist {mat['twist']:.2e}  damping {mat['damping']:.4f}  "
               f"density {mat['density']:.0f}  length {mat['length']:.2f} m  height {mat['height']:.2f} m")
    frames = []
    for t in range(T):
        sim.set_gripper(*pose[t])
        if t > 0:
            sim.step_control()
        else:
            mujoco.mj_forward(sim.model, sim.data)
        if t % frame_stride:
            continue
        true_nodes = sim.node_positions()
        if window_start[t] >= 0:
            right = viz.render_pred(pred[t], true_nodes)
            k = t - window_start[t]
            e_now = np.nanmean(err[t]) * 100
            e_max = np.nanmax(err[t]) * 100
            status = (f"prediction step {k:3d}/{H}  (open-loop from t={window_start[t] * cfg.control_dt:.2f}s)   "
                      f"mean err {e_now:5.2f} cm   max err {e_max:5.2f} cm")
        else:
            right = viz.render_pred(true_nodes[:0], true_nodes)
            status = f"collecting context ({t + 1}/{K} steps)"
        info = [f"t = {t * cfg.control_dt:5.2f} s   " + status, "material: " + mat_str]
        pred_title = "Transformer prediction (orange)  |  ghost = MuJoCo simulation"
        if split:
            frames.append(compose([viz.render_sim(), right], ["MuJoCo simulation", pred_title], info, fonts))
        else:
            frames.append(compose([right], [pred_title], info, fonts, title_colors=((120, 180, 255),)))
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    valid = ~np.isnan(err[:, 0])
    return dict(mean_err_cm=float(np.nanmean(err) * 100),
                err_h100_cm=float(np.nanmean(err[(window_start >= 0) & (np.arange(T) - window_start == H)]) * 100),
                max_err_cm=float(np.nanmax(err) * 100), material=mat,
                n_windows=int(len(np.unique(window_start[window_start >= 0]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/base/best.pt")
    ap.add_argument("--data", default="data/test.npz")
    ap.add_argument("--out", default="videos")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--index", default=None, help="comma-separated episode indices to render instead of the first --episodes")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frame-stride", type=int, default=1, help="render every n-th control step (1 = half speed at 25 fps)")
    ap.add_argument("--split", action="store_true", help="side-by-side simulation | prediction instead of one large prediction panel")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    device = torch.device(args.device)
    model, mcfg = load_model(args.ckpt, device)
    cfg = SimConfig()
    d = np.load(args.data)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    indices = [int(i) for i in args.index.split(",")] if args.index else range(min(args.episodes, d["pos"].shape[0]))
    for e in indices:
        ep = {k: d[k][e] for k in ["pos", "pose", "action", "material"]}
        out_path = os.path.join(args.out, f"episode_{e:02d}.mp4")
        res = render_episode(model, mcfg, ep, cfg, device, out_path, fps=args.fps, frame_stride=args.frame_stride,
                             split=args.split, width=args.width, height=args.height)
        results[f"episode_{e:02d}"] = res
        print(f"{out_path}: mean err {res['mean_err_cm']:.2f} cm, err@h100 {res['err_h100_cm']:.2f} cm, "
              f"max {res['max_err_cm']:.2f} cm | {res['material']}", flush=True)
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

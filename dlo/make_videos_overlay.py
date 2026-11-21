"""Single-panel video of two cables: simulation, independent and coupled prediction on top of each other.

The camera follows both cables tightly, shadows are off, and the video plays slower than real time.
Simulation contacts come from the recorded forces and are for comparison only; the coupled
prediction sees its own force estimates, never simulator forces.

Usage:
    python -m dlo.make_videos_overlay --ckpt runs/drive1/best.pt --data data/test_drive.npz --out videos/drive_overlay --index 3,5
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont

from .config import SimConfig, material_dict, pad_material
from .two_cable_env import TwoCableSim
from .couple import CoupledPredictor, load_model
from .make_videos import compose, FONT, FONT_B
from .make_videos_two import TwoViz
from .contact_model import load_net, correct
from .contact_data import predict_batch
from .config import normalize_material

COL_SIM = [0.10, 0.10, 0.12, 0.35]
COL_INDEP = [0.16, 0.47, 0.84, 1.0]
COL_COUPLED = [0.95, 0.35, 0.10, 1.0]
COL_SIM_CONTACT = [0.10, 0.65, 0.35, 0.45]
COL_EST_CONTACT = [0.85, 0.05, 0.10, 0.55]
LEGEND = (("Simulation (dick, transparent)", COL_SIM), ("unabhängige Vorhersage", COL_INDEP), ("gekoppelte Vorhersage", COL_COUPLED),
          ("Kontakt Simulation", COL_SIM_CONTACT), ("Kontakt geschätzt", COL_EST_CONTACT))


LEGEND_CONTACT = (("Simulation (dick, transparent)", COL_SIM), ("DLO-Anything allein", COL_INDEP), ("mit Kontaktkorrektur", COL_COUPLED),
                  ("Kontakt Simulation", COL_SIM_CONTACT), ("Kontaktpunkt aus den Prognosen", COL_EST_CONTACT))


def predict_windows(cp: CoupledPredictor, pos, pose, action, material):
    """Consecutive open-loop windows with the algorithm's own forces in the context channel."""
    K, H = cp.mcfg.context_len, cp.mcfg.horizon
    T = pos.shape[1]
    pred_c = np.full(pos.shape, np.nan); pred_u = np.full(pos.shape, np.nan)
    force_est = np.zeros(pos.shape, np.float32)
    points = [None] * T
    start = np.full(T, -1)
    starts = list(range(K - 1, T - H, H))
    if starts[-1] + H < T - 1:
        starts.append(T - 1 - H)
    for t0 in starts:
        r = cp.predict(pos, pose, action, material, force_est, t0)
        sl = slice(t0 + 1, t0 + H + 1)
        new = start[sl] < 0
        pred_c[:, sl][:, new] = r["pred"][:, new]; pred_u[:, sl][:, new] = r["pred_uncoupled"][:, new]
        force_est[:, sl][:, new] = r["force"][:, new]
        for k in np.where(new)[0]:
            points[t0 + 1 + k] = r["points"][k][r["active"][k]]
        start[sl][new] = t0
    return pred_c, pred_u, points, start


def predict_windows_contact(model, mcfg, net, device, pos, pose, action, material):
    """wide4 prediction and the same prediction with contact correction, consecutive open-loop windows."""
    K, H = mcfg.context_len, mcfg.horizon
    T, N = pos.shape[1], pos.shape[2]
    mat = torch.from_numpy(np.stack([normalize_material(m) for m in material]).astype(np.float32))[None].to(device)
    pred_c = np.full(pos.shape, np.nan); pred_u = np.full(pos.shape, np.nan)
    points = [None] * T
    start = np.full(T, -1)
    starts = list(range(K - 1, T - H, H))
    if starts[-1] + H < T - 1:
        starts.append(T - 1 - H)
    tt = lambda x: torch.from_numpy(np.asarray(x, np.float32))[None].to(device)
    for t0 in starts:
        base = predict_batch(model, mcfg, device, [(pos[c], pose[c], action[c], material[c], t0) for c in (0, 1)]).reshape(2, H, N, 3)
        with torch.no_grad():
            p, geo, _ = correct(net, tt(base), tt(pos[:, t0]), tt(pose[:, t0]), mat, return_geo=True)
        p = p.float()[0].cpu().numpy(); act = geo["active"][0].cpu().numpy(); pt = geo["point"][0].cpu().numpy()
        sl = slice(t0 + 1, t0 + H + 1)
        new = start[sl] < 0
        pred_c[:, sl][:, new] = p[:, new]; pred_u[:, sl][:, new] = base[:, new]
        for k in np.where(new)[0]:
            points[t0 + 1 + k] = pt[k][None] if act[k] else np.zeros((0, 3))
        start[sl][new] = t0
    return pred_c, pred_u, points, start


def sim_contact_points(nodes, force):
    """One point per run of consecutive loaded nodes on cable 1, at the force-weighted centroid."""
    mag = np.linalg.norm(force, axis=-1)
    idx = np.where(mag > 0)[0]
    if len(idx) == 0:
        return np.zeros((0, 3))
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return np.array([(nodes[r] * mag[r, None]).sum(0) / mag[r].sum() for r in runs])


def merge_points(p, cell=0.01):
    """Many active segment pairs describe one contact: keep one point per `cell` metres."""
    if p is None or len(p) == 0:
        return np.zeros((0, 3))
    _, i = np.unique(np.round(np.asarray(p) / cell).astype(int), axis=0, return_index=True)
    return np.asarray(p)[i]


def camera_path(viz, pos, half_window=20, smooth=31, margin=0.10):
    """Lookat and distance per frame: tight around the true cables, then smoothed against jitter."""
    T = pos.shape[1]
    look, dist = np.zeros((T, 3)), np.zeros(T)
    for t in range(T):
        viz.frame_points(pos[:, max(0, t - half_window):t + half_window + 1].reshape(-1, 3), margin=margin)
        look[t], dist[t] = viz.cam.lookat, viz.cam.distance
    w = np.ones(smooth) / smooth
    pad = smooth // 2
    sm = lambda a: np.convolve(np.pad(a, pad, mode="edge"), w, mode="valid")
    return np.column_stack([sm(look[:, i]) for i in range(3)]), sm(dist)


def add_legend(img, font, legend=LEGEND):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    lh = font.size + 9
    wid = max(int(dr.textlength(t, font=font)) for t, _ in legend) + 48
    dr.rectangle((12, 12, 12 + wid, 18 + lh * len(legend)), fill=(255, 255, 255, 225), outline=(190, 190, 190, 255))
    for i, (t, c) in enumerate(legend):
        y = 18 + lh * i
        fill = tuple(int(v * 255) for v in c[:3]) + (max(int(c[3] * 255), 110),)
        dr.rectangle((20, y + 4, 44, y + font.size), fill=fill)
        dr.text((52, y), t, font=font, fill=(30, 30, 34, 255))
    return np.asarray(im.convert("RGB"))


def render_episode(cp, ep, cfg, out_path, fps, width, height, elevation, azimuth, contact=None):
    pos, pose, action = ep["pos"].astype(np.float64), ep["pose"], ep["action"]
    material = pad_material(ep["material"]); force_true = ep["force"].astype(np.float32)
    mats = [material_dict(material[c]) for c in range(2)]
    T = pos.shape[1]
    if contact is None:
        K, H = cp.mcfg.context_len, cp.mcfg.horizon
        pred_c, pred_u, points, start = predict_windows(cp, pos, pose, action, material)
    else:
        K, H = contact[1].context_len, contact[1].horizon
        pred_c, pred_u, points, start = predict_windows_contact(*contact, pos, pose, action, material)
    err_c = np.linalg.norm(pred_c - pos, axis=-1); err_u = np.linalg.norm(pred_u - pos, axis=-1)

    sim = TwoCableSim(cfg, mats, start_xy=(pose[0, 0, :2], pose[1, 0, :2]))
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, height)
    viz = TwoViz(sim, width, height)
    viz.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    viz.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    viz.cam.elevation, viz.cam.azimuth = elevation, azimuth
    look, dist = camera_path(viz, pos)
    hidden = viz.rgba_cable.copy(); hidden[:, 3] = 0.0
    sim.model.geom_rgba[viz.geoms] = hidden               # all cables are drawn from the recorded nodes

    fs = max(15, int(round(width / 72)))
    fonts = (ImageFont.truetype(FONT, fs), ImageFont.truetype(FONT_B, fs + 2))
    lfont = ImageFont.truetype(FONT, fs)
    mat_lines = [f"Kabel {c + 1}: bend {m['bend']:.1e} twist {m['twist']:.1e} damp {m['damping']:.3f} dens {m['density']:.0f} "
                 f"L {m['length']:.2f} m H {m['height']:.2f} m" for c, m in enumerate(mats)]
    frames = []
    for t in range(T):
        for c in range(2):
            sim.set_gripper(c, *pose[c, t])
        mujoco.mj_forward(sim.model, sim.data)
        viz.cam.lookat[:] = look[t]; viz.cam.distance = dist[t]
        viz.r.update_scene(sim.data, viz.cam)
        for c in range(2):
            viz._add_chain(pos[c, t], COL_SIM, radius=0.009)
        sim_pts = sim_contact_points(pos[0, t], force_true[0, t])
        for p in sim_pts:
            viz._add_sphere(p, COL_SIM_CONTACT, radius=0.022)
        n_est = 0
        if start[t] >= 0:
            for c in range(2):
                viz._add_chain(pred_u[c, t], COL_INDEP, radius=0.0035)
                viz._add_chain(pred_c[c, t], COL_COUPLED, radius=0.0045)
            est = merge_points(points[t])
            n_est = len(est)
            for p in est:
                viz._add_sphere(p, COL_EST_CONTACT, radius=0.022)
            k = t - start[t]
            status = f"Schritt {k:3d}/{H} der offenen Vorhersage   Kontakte: Simulation {len(sim_pts)} | geschätzt {n_est}"
            errors = f"Mittlerer Fehler: {'unabhängig' if contact is None else 'DLO-Anything'} {np.nanmean(err_u[:, t]) * 100:5.2f} cm | {'gekoppelt' if contact is None else 'mit Kontaktkorrektur'} {np.nanmean(err_c[:, t]) * 100:5.2f} cm"
        else:
            status = f"Kontext wird gesammelt ({t + 1}/{K} Schritte)"
            errors = "Mittlerer Fehler: noch keine Vorhersage"
        img = add_legend(viz.r.render(), lfont, LEGEND if contact is None else LEGEND_CONTACT)
        info = [f"t = {t * cfg.control_dt:5.2f} s   " + status, errors] + mat_lines
        frames.append(compose([img], ["Simulation, unabhängige und gekoppelte Vorhersage übereinander" if contact is None else "Simulation, DLO-Anything und Kontaktkorrektur übereinander"], info, fonts,
                              title_colors=((235, 235, 235),)))
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    return dict(mean_err_independent_cm=float(np.nanmean(err_u) * 100), mean_err_coupled_cm=float(np.nanmean(err_c) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/test_drive.npz")
    ap.add_argument("--out", default="videos/drive_overlay")
    ap.add_argument("--index", required=True, help="comma-separated episode indices")
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--kc", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=0.002)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--elevation", type=float, default=-50)
    ap.add_argument("--azimuth", type=float, default=140)
    ap.add_argument("--contact-net", default=None, help="contact correction checkpoint; --ckpt is then the frozen base model")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, mcfg = load_model(args.ckpt, device)
    cp = CoupledPredictor(model, mcfg, device, k_c=args.kc, n_iter=args.iters, margin=args.margin)
    contact = None
    if args.contact_net:
        net = load_net(args.contact_net, device)
        contact = (model, mcfg, net, device)
    d = np.load(args.data)
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for e in [int(i) for i in args.index.split(",")]:
        ep = {k: d[k][e] for k in ["pos", "pose", "action", "material", "force"]}
        path = os.path.join(args.out, f"episode_{e:03d}.mp4")
        results[f"episode_{e:03d}"] = render_episode(cp, ep, SimConfig(), path, args.fps, args.width, args.height, args.elevation, args.azimuth, contact)
        print(path, results[f"episode_{e:03d}"], flush=True)
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

"""Videos of the simulation and the model forecast, and a grid video of many scenes.

    python make_videos.py --models runs/F_12000/model_cable0.pt,runs/F_12000/model_cable1.pt --data data/test.npz --index 3 --out videos
    python make_videos.py --grid videos/tiles --rows 5 --cols 5 --out videos/grid.mp4

One scene per call. The forecast starts after 0.2 s of simulated context and is never reset to the simulation.
Dark band: simulation. Orange: model forecast. Green / yellow: cable and floor contact in the simulation (arrows: forces,
shown for orientation only, the model neither receives nor predicts forces). Magenta / cyan: where the predicted cables
touch each other / the floor, i.e. the contact the model itself works with.
For grid tiles use --plain --margin 0.12 (no legend, no arrows, scene fills the frame); --grid arranges all episode_*.mp4 of a folder."""
from __future__ import annotations
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import glob
import imageio
import mujoco
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from simulate import CableSim, SimConfig, TwoCableSim, material_dict
from model import SceneWindows, closest_on_cable, load, rollout


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
COL_PRED = ([0.95, 0.35, 0.1, 1.0], [0.95, 0.80, 0.1, 1.0])     # prediction cable 1 (orange), cable 2 (yellow)
COL_CONTACT = [0.9, 0.05, 0.05, 1.0]
COL_COUPLED = [0.95, 0.35, 0.10, 1.0]
# simulation as a dark band, forecasts thick and opaque
COL_SIM_SOLID = [0.10, 0.10, 0.14, 1.0]
SIM_ALPHA = 0.85
COL_SIM = COL_SIM_SOLID[:3] + [SIM_ALPHA]
COL_SIM_CONTACT = [0.10, 0.70, 0.35, 0.30]
COL_FLOOR_SIM = [0.98, 0.78, 0.05, 0.9]     # floor contact in the simulation: yellow disc on the floor
FLOOR_THR = 1e-3                            # [N] per node
# widths in pixels (half-widths), converted to metres per frame from the camera distance
PX_SIM, PX_INDEP, PX_COUPLED, PX_CONTACT, PX_FLOOR = 9.0, 2.6, 3.2, 13.0, 5.0
CABLE_AREA = 0.25   # share of the image the cables occupy (half width x half height)
ARROW_PX = 100.0    # arrow length in pixels per 100 mN (in the image plane), one arrow per node
ARROW_MAX = 0.3     # [N] longer arrows are capped (the label still shows the true sum)
ARROW_MIN = 0.002   # [N] no arrow below
A_SIM_CABLE, A_EST_CABLE = (10, 120, 60), (200, 20, 30)
A_SIM_FLOOR, A_EST_FLOOR = (170, 120, 0), (120, 40, 190)
COL_MODEL_CONTACT = [0.85, 0.10, 0.75, 0.95]      # where the predicted cables touch: this is the contact the model works with
MODEL_GAP = 0.01                                  # [m] centre distance below which the predicted cables count as touching (simulation: about 0.6-0.9 cm)
COL_MODEL_FLOOR = [0.05, 0.75, 0.95, 0.95]        # floor contact in the model: predicted node rests on the floor
MODEL_FLOOR = 0.008                               # [m] height of a predicted node centre below which it counts as lying on the floor (simulation: 4 mm)
LEGEND = (("Simulation (dark band)", COL_SIM), ("DLO-Anything forecast", COL_COUPLED),
          ("Cable contact in the simulation (arrow: force on the U)", [c / 255 for c in A_SIM_CABLE] + [1]),
          ("Floor contact in the simulation (arrow: floor force)", COL_FLOOR_SIM),
          (f"Cable contact in the model (predicted cables closer than {MODEL_GAP * 1000:.0f} mm)", COL_MODEL_CONTACT),
          (f"Floor contact in the model (predicted node lower than {MODEL_FLOOR * 1000:.0f} mm)", COL_MODEL_FLOOR))


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


def add_legend(img, font, legend):
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


def add_disc(viz, p, rgba, radius):
    scn = viz.r.scene
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, 0.0005]), np.array([p[0], p[1], 0.0006]),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _soft(m):
    """3x3 box blur of a mask, for anti-aliased edges."""
    p = np.pad(m, 1, mode="edge")
    return sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)) / 9.0


def force_runs(nodes, force, thr):
    """One arrow (point, force, label) per loaded node; the node with the largest force of every run of
    consecutive loaded nodes carries the label: the summed force of that contact site in mN."""
    mag = np.linalg.norm(force, axis=-1)
    idx = np.where(mag > thr)[0]
    if len(idx) == 0:
        return []
    out = []
    for r in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
        top = r[np.argmax(mag[r])]; tot = np.linalg.norm(force[r].sum(0)) * 1000
        out += [(nodes[i], force[i], (f"{tot:.0f} mN" if len(r) == 1 else f"total {tot:.0f} mN") if i == top else None) for i in r]
    return out


def projector(scn, width, height, tan_half):
    """World point -> pixel, from the camera of the scene just rendered."""
    c0, c1 = scn.camera[0], scn.camera[1]
    cp = (np.array(c0.pos) + np.array(c1.pos)) / 2
    fw = np.array(c0.forward, np.float64); up = np.array(c0.up, np.float64); rt = np.cross(fw, up)
    f = (height / 2) / tan_half

    def proj(q):
        v = np.asarray(q, np.float64) - cp
        z = max(float(v @ fw), 1e-6)
        return np.array([width / 2 + f * (v @ rt) / z, height / 2 - f * (v @ up) / z])
    return proj


def draw_arrows(img, arrows, proj, mpp, font):
    """arrows: (point, force [N], label or None, rgb). 2D arrows on top of the image: direction = projected force, length ~ force, label in mN."""
    im = Image.fromarray(img); dr = ImageDraw.Draw(im); boxes = []
    for q, F, lab, col in arrows:
        n = float(np.linalg.norm(F))
        if not np.isfinite(n) or n < ARROW_MIN or not np.all(np.isfinite(q)):
            continue
        a = proj(q); b = proj(q + F / n * min(n, ARROW_MAX) / 0.1 * ARROW_PX * mpp)
        v = b - a; L = float(np.linalg.norm(v))
        if L < 1e-6:
            continue
        u = v / L; w = np.array([-u[1], u[0]])
        if L < 3:
            b = a + u * 3; L = 3.0
        h = min(11.0, L * 0.6)
        dr.line([tuple(a), tuple(b - u * h * 0.8)], fill=(255, 255, 255), width=6)
        dr.line([tuple(a), tuple(b - u * h * 0.8)], fill=col, width=3)
        dr.polygon([tuple(b), tuple(b - u * h + w * h * 0.55), tuple(b - u * h - w * h * 0.55)], fill=col, outline=(255, 255, 255))
        if lab is None:
            continue
        txt = lab; tw = dr.textlength(txt, font=font)
        tx = b[0] + u[0] * 8 - (tw if u[0] < -0.3 else tw / 2 if abs(u[0]) <= 0.3 else 0); ty = b[1] + u[1] * 8 - font.size / 2 - (font.size / 2 if u[1] < -0.3 else -font.size / 2 if u[1] > 0.3 else 0)
        for _ in range(6):   # labels of overlapping arrows: stack instead of overprinting
            if not any(tx < x1 and tx + tw > x0 and ty < y1 and ty + font.size > y0 for x0, y0, x1, y1 in boxes):
                break
            ty += (font.size + 5) * (1 if u[1] > 0.3 else -1)
        boxes.append((tx - 4, ty - 3, tx + tw + 4, ty + font.size + 3))
        dr.text((tx, ty), txt, font=font, fill=col, stroke_width=3, stroke_fill=(255, 255, 255))
    return np.asarray(im)


def model_contacts(a, b, th=MODEL_GAP):
    """Points where the predicted cables a, b (N,3) are closer than th, and their smallest centre distance.
    The contact tokens are computed from exactly these predicted positions (closest point on the other cable per node)."""
    w = np.linspace(0, 1, 6, endpoint=False)[:, None, None]
    pts = np.concatenate([(a[None, :-1] * (1 - w) + a[None, 1:] * w).reshape(-1, 3), a[-1:]])
    q, _, gap, _, _ = closest_on_cable(torch.from_numpy(pts).float(), torch.from_numpy(b).float())
    gap, q = gap.numpy(), q.numpy()
    return merge_points((0.5 * (pts + q))[gap < th], cell=0.015), float(gap.min())


def add_floor_disc(viz, p, rgba, radius):
    """Disc on the floor, slightly above the simulation's floor-contact discs so that both stay visible."""
    scn = viz.r.scene
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, 0.0005]), np.array([p[0], p[1], 0.0018]),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def render(args):
    device = torch.device(args.device); models = [load(p, device)[0] for p in args.models.split(",")]
    ds = SceneWindows(args.data, models[0].cfg); d = np.load(args.data); cfg = SimConfig(); e = args.index
    os.makedirs(args.out, exist_ok=True)
    pos, pose, pf = ds.pos[e], ds.pose[e], rollout(models, ds, [e], device)[0]
    fs = max(15, int(round(args.width / 72))); lfont = ImageFont.truetype(FONT, fs); afont = ImageFont.truetype(FONT_B, max(13, fs - 4))
    ftrue = d["force_cable"][e].astype(np.float64)
    fl_sim = d["force_floor"][e].astype(np.float64) if "force_floor" in d else np.zeros_like(ftrue)
    single = not np.abs(ftrue).sum() > 0                        # no cable contact in this scene: only the pulled cable is shown
    cables = (0,) if single else (0, 1)
    mats = [material_dict(ds.material[e, c]) for c in range(2)]
    sim = TwoCableSim(cfg, mats, start_xy=(pose[0, 0, :2], pose[1, 0, :2]))
    sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, args.width)
    sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, args.height)
    viz = TwoViz(sim, args.width, args.height)
    viz.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
    viz.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
    ax = d["anchor_b"][e][:2] - pose[1, 0, :2]; ax = ax / np.linalg.norm(ax)
    viz.cam.azimuth, viz.cam.elevation = np.degrees(np.arctan2(ax[0], -ax[1])) + 35, -22
    look, dist = camera_path(viz, pos[:1] if single else pos, half_window=25, smooth=41, margin=1 - np.sqrt(CABLE_AREA) if args.margin is None else args.margin)
    dist = dist / args.zoom
    tan_half = np.tan(np.deg2rad(float(sim.model.vis.global_.fovy)) / 2)
    hidden = viz.rgba_cable.copy(); hidden[:, 3] = 0.0
    sim.model.geom_rgba[viz.geoms] = hidden
    frames = []
    for t in range(ds.T):
        for c in range(2):
            sim.set_gripper(c, *pose[c, t])
        mujoco.mj_forward(sim.model, sim.data)
        viz.cam.lookat[:] = look[t]; viz.cam.distance = dist[t]
        mpp = 2 * dist[t] * tan_half / args.height
        viz.r.enable_depth_rendering()
        viz.r.update_scene(sim.data, viz.cam); d_empty = viz.r.render()
        for c in cables:
            viz._add_chain(pos[c, t], COL_SIM_SOLID, radius=PX_SIM * mpp)
        d_sim = viz.r.render()
        viz.r.update_scene(sim.data, viz.cam)
        proj = projector(viz.r.scene, args.width, args.height, tan_half)
        sim_pts = [] if single else sim_contact_points(pos[1, t], ftrue[1, t])
        arrows = [] if single else [(q, F, lab, A_SIM_CABLE) for q, F, lab in force_runs(pos[1, t], ftrue[1, t], 0.0)]
        for c in cables:
            arrows += [(q, F, lab, A_SIM_FLOOR) for q, F, lab in force_runs(pos[c, t], fl_sim[c, t], FLOOR_THR)]
            for q in pos[c, t][np.linalg.norm(fl_sim[c, t], axis=-1) > FLOOR_THR]:
                add_disc(viz, q, COL_FLOOR_SIM, PX_FLOOR * mpp * 1.6)
        for q in sim_pts:
            viz._add_sphere(q, COL_SIM_CONTACT, radius=PX_CONTACT * mpp)
        started = np.isfinite(pf[0, t]).all()
        if started:
            for c in cables:
                viz._add_chain(pf[c, t], COL_COUPLED, radius=PX_COUPLED * mpp)
                for q in pf[c, t][pf[c, t][:, 2] < MODEL_FLOOR]:
                    add_floor_disc(viz, q, COL_MODEL_FLOOR, PX_FLOOR * mpp * 0.9)
            if not single:
                for q in model_contacts(pf[0, t], pf[1, t])[0]:
                    viz._add_sphere(q, COL_MODEL_CONTACT, radius=PX_CONTACT * mpp * 1.25)
        d_pred = viz.r.render()
        viz.r.disable_depth_rendering()
        img = viz.r.render().astype(np.float32)
        m = _soft((d_sim < d_empty - 1e-4).astype(np.float32)) * (1.0 - _soft((d_pred < d_empty - 1e-4).astype(np.float32)))
        a = (SIM_ALPHA * m)[..., None]
        img = ((1 - a) * img + a * (np.asarray(COL_SIM_SOLID[:3], np.float32) * 255)).clip(0, 255).astype(np.uint8)
        frames.append(img if args.plain else add_legend(draw_arrows(img, arrows, proj, mpp, afont), lfont, LEGEND))
    viz.r.close()
    path = os.path.join(args.out, f"episode_{e:03d}.mp4")
    imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
    print(path, f"{d['mode'][e]}, mean error {np.nanmean(np.linalg.norm(pf - pos, axis=-1)) * 100:.2f} cm", flush=True)


def grid(args, tile=(512, 288), gap=2):
    """All episode_*.mp4 of a folder (rendered with --plain) in rows x cols."""
    files = sorted(glob.glob(os.path.join(args.grid, "episode_*.mp4")))[:args.rows * args.cols]
    readers = [imageio.get_reader(f) for f in files]; fps = readers[0].get_meta_data()["fps"]; n = min(r.count_frames() for r in readers)
    W, H = args.cols * tile[0] + (args.cols + 1) * gap, args.rows * tile[1] + (args.rows + 1) * gap
    W += W % 2; H += H % 2
    with imageio.get_writer(args.out, fps=fps, codec="libx264", quality=7, macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]) as w:
        for t in range(n):
            canvas = Image.new("RGB", (W, H), (255, 255, 255))
            for k, r in enumerate(readers):
                canvas.paste(Image.fromarray(r.get_data(t)).resize(tile, Image.LANCZOS), (gap + (k % args.cols) * (tile[0] + gap), gap + (k // args.cols) * (tile[1] + gap)))
            w.append_data(np.asarray(canvas))
    print(args.out, f"{len(files)} tiles, {n} frames, {W}x{H}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="checkpoints 'pulled cable,fixed U'"); ap.add_argument("--data", help="scene file of simulate.py (with floor forces)")
    ap.add_argument("--index", type=int, default=0, help="scene in the file"); ap.add_argument("--out", default="videos", help="output folder (--grid: output file)")
    ap.add_argument("--fps", type=int, default=15); ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--plain", action="store_true", help="no legend and no force arrows (tiles of a grid video)")
    ap.add_argument("--zoom", type=float, default=1.0, help="camera closer by this factor")
    ap.add_argument("--margin", type=float, default=None, help="free border as share of the half image (default: room for the legend); 0.12 fills the frame")
    ap.add_argument("--grid", default=None, help="folder with tiles: build a grid video instead of rendering a scene")
    ap.add_argument("--rows", type=int, default=5); ap.add_argument("--cols", type=int, default=5)
    args = ap.parse_args()
    grid(args) if args.grid else render(args)


if __name__ == "__main__":
    main()

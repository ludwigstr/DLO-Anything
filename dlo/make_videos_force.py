"""Single-panel videos of the U scenario: simulation, DLO-Anything without force, and the loop result.

    python -m dlo.make_videos_force --move runs/u_move/best.pt --fixed runs/u_fixed/best.pt --net runs/force_u/best.pt --data data/test_u.npz --out videos/force_u --episodes 6
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch, mujoco, imageio
from PIL import Image, ImageDraw, ImageFont

from .config import SimConfig, material_dict, pad_material
from .force_model import Instances, UWindows, force_world, load_force_net, run_loop
from .make_videos import compose, FONT, FONT_B
from .make_videos_overlay import COL_COUPLED, COL_INDEP, add_legend, camera_path, sim_contact_points
from .make_videos_two import TwoViz
from .two_cable_env import TwoCableSim

# eigene, kräftigere Darstellung: Simulation als blasser Schatten, Prognosen dick und deckend
COL_SIM_SOLID = [0.10, 0.10, 0.14, 1.0]
SIM_ALPHA = 0.85
COL_SIM = COL_SIM_SOLID[:3] + [SIM_ALPHA]
COL_SIM_CONTACT = [0.10, 0.70, 0.35, 0.30]
COL_EST_CONTACT = [0.90, 0.05, 0.10, 0.35]
COL_FLOOR_SIM = [0.98, 0.78, 0.05, 0.9]     # floor contact in the simulation: yellow disc on the floor
COL_FLOOR_EST = [0.55, 0.15, 0.85, 0.9]     # floor contact according to the force model: purple dot on the prediction
FLOOR_THR = 1e-3                            # [N] per node


def add_disc(viz, p, rgba, radius):
    scn = viz.r.scene
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, 0.0005]), np.array([p[0], p[1], 0.0006]),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def add_tube(viz, nodes, rgba, radius):
    """Transparent tube from cylinders: no overlapping end caps, so no dark dots at the joints."""
    scn = viz.r.scene
    for a, b in zip(nodes[:-1], nodes[1:]):
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, radius, np.asarray(a, np.float64), np.asarray(b, np.float64))
        scn.ngeom += 1


# widths in pixels (half-widths), converted to metres per frame from the camera distance
PX_SIM, PX_INDEP, PX_COUPLED, PX_CONTACT, PX_FLOOR = 9.0, 2.6, 3.2, 13.0, 5.0
CABLE_AREA = 0.25   # share of the image the cables occupy (half width x half height)


def _soft(m):
    """3x3 box blur of a mask, for anti-aliased edges."""
    p = np.pad(m, 1, mode="edge")
    return sum(p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)) / 9.0

ARROW_PX = 100.0    # arrow length in pixels per 100 mN (in the image plane), one arrow per node
ARROW_MAX = 0.3     # [N] longer arrows are capped (the label still shows the true sum)
ARROW_MIN = 0.002   # [N] no arrow below
A_SIM_CABLE, A_EST_CABLE = (10, 120, 60), (200, 20, 30)
A_SIM_FLOOR, A_EST_FLOOR = (170, 120, 0), (120, 40, 190)


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


LEGEND = (("Simulation (transparent)", COL_SIM), ("DLO-Anything ohne Kontaktkraft", COL_INDEP), ("nach Kraft-Rückkopplung", COL_COUPLED),
          ("Kabelkontakt Simulation (Pfeil: Kraft auf das U)", [c / 255 for c in A_SIM_CABLE] + [1]),
          ("Kabelkontakt laut Kraftmodell (Pfeil: Kraft auf das U)", [c / 255 for c in A_EST_CABLE] + [1]),
          ("Bodenkontakt Simulation (Pfeil: Bodenkraft)", COL_FLOOR_SIM), ("Bodenkontakt laut Kraftmodell (Pfeil: Bodenkraft)", COL_FLOOR_EST))


@torch.no_grad()
def windows(net, inst, ds, e, device, n_iter, chain=True, floor_out=None):
    """Predictions over the whole episode. With `chain`, every 100-step window starts from the model's
    own last predictions (the first one from the true start state): one continuous forecast without
    any reset to the simulation. The no-force track and the loop track are chained separately."""
    K, H, T = ds.K, ds.H, ds.T
    shape = ds.pos[e].shape
    p0, pf = np.full(shape, np.nan), np.full(shape, np.nan)
    fe = np.zeros(shape); fl = np.zeros(shape); start = np.full(T, -1)
    tr0, trf = ds.pos[e].astype(np.float64).copy(), ds.pos[e].astype(np.float64).copy()   # context tracks
    t0s = list(range(K - 1, T - H, H))
    if t0s[-1] + H < T - 1:
        t0s.append(T - 1 - H)
    for t0 in t0s:
        sl = slice(t0 + 1, t0 + H + 1); new = start[sl] < 0
        if chain:
            b0 = {k: v[None].to(device) for k, v in ds.item(e, t0, tr0).items()}
            bf = {k: v[None].to(device) for k, v in ds.item(e, t0, trf).items()}
            q0 = run_loop(net, inst, b0, 0)[0][0][0].cpu().numpy()
            parts = []
            preds, forces, _ = run_loop(net, inst, bf, n_iter, parts_out=parts)
            qf = preds[-1][0].cpu().numpy()
            f = force_world(forces[-1], bf["pose0"])[0].cpu().numpy()
            ffl = force_world(parts[-1]["f_floor"].float(), bf["pose0"])[0].cpu().numpy() if parts else np.zeros_like(f)
            tr0[:, sl] = q0; trf[:, sl] = qf
        else:
            b = {k: v[None].to(device) for k, v in ds.item(e, t0).items()}
            parts = []
            preds, forces, _ = run_loop(net, inst, b, n_iter, parts_out=parts)
            q0, qf = preds[0][0].cpu().numpy(), preds[-1][0].cpu().numpy()
            f = force_world(forces[-1], b["pose0"])[0].cpu().numpy()
            ffl = force_world(parts[-1]["f_floor"].float(), b["pose0"])[0].cpu().numpy() if parts else np.zeros_like(f)
        p0[:, sl][:, new] = q0[:, new]; pf[:, sl][:, new] = qf[:, new]
        fe[:, sl][:, new] = f[:, new]; fl[:, sl][:, new] = ffl[:, new]; start[sl][new] = K - 1 if chain else t0
    if floor_out is not None:
        floor_out["f"] = fl
    return p0, pf, fe, start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", required=True); ap.add_argument("--fixed", required=True); ap.add_argument("--net", required=True)
    ap.add_argument("--data", default="data/test_u.npz"); ap.add_argument("--out", default="videos/force_u")
    ap.add_argument("--episodes", type=int, default=6); ap.add_argument("--index", default=None)
    ap.add_argument("--iters", type=int, default=2); ap.add_argument("--min-contact", type=int, default=60)
    ap.add_argument("--fps", type=int, default=15); ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    inst = Instances(args.move, args.fixed, device); net = load_force_net(args.net, device)
    ds = UWindows(args.data, inst.mcfg)
    d = np.load(args.data); cfg = SimConfig()
    os.makedirs(args.out, exist_ok=True)
    idx = [int(i) for i in args.index.split(",")] if args.index else [int(i) for i in np.where((d["n_contact"] > 0).sum(1) >= args.min_contact)[0][:args.episodes]]
    fs = max(15, int(round(args.width / 72)))
    fonts = (ImageFont.truetype(FONT, fs), ImageFont.truetype(FONT_B, fs + 2)); lfont = ImageFont.truetype(FONT, fs); afont = ImageFont.truetype(FONT_B, max(13, fs - 4))
    sp = os.path.join(args.out, "summary.json")
    results = json.load(open(sp)) if os.path.exists(sp) else {}
    for e in idx:
        pos = ds.pos[e]; pose = ds.pose[e]; ftrue = ds.force_cable[e].astype(np.float64)   # spheres: cable-cable contact only
        flo = {}
        p0, pf, fe, start = windows(net, inst, ds, e, device, args.iters, floor_out=flo)
        fl_est, fl_sim = flo["f"], ds.force_floor[e].astype(np.float64)
        single = not np.abs(ftrue).sum() > 0      # no cable-cable contact: the U plays no role, show only the pulled cable
        cables = (0,) if single else (0, 1)
        mats = [material_dict(ds.material[e, c]) for c in range(2)]
        err0, errf = np.linalg.norm(p0 - pos, axis=-1), np.linalg.norm(pf - pos, axis=-1)
        # a scene without cables: the cables are drawn from the node positions
        sim = TwoCableSim(cfg, mats, start_xy=(pose[0, 0, :2], pose[1, 0, :2]))
        sim.model.vis.global_.offwidth = max(sim.model.vis.global_.offwidth, args.width)
        sim.model.vis.global_.offheight = max(sim.model.vis.global_.offheight, args.height)
        viz = TwoViz(sim, args.width, args.height)
        viz.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        viz.r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        ax = d["anchor_b"][e][:2] - pose[1, 0, :2]; ax = ax / np.linalg.norm(ax)
        viz.cam.azimuth, viz.cam.elevation = np.degrees(np.arctan2(ax[0], -ax[1])) + 35, -22
        look, dist = camera_path(viz, pos[:1] if single else pos, half_window=25, smooth=41, margin=1 - np.sqrt(CABLE_AREA))
        tan_half = np.tan(np.deg2rad(float(sim.model.vis.global_.fovy)) / 2)
        hidden = viz.rgba_cable.copy(); hidden[:, 3] = 0.0
        sim.model.geom_rgba[viz.geoms] = hidden
        frames = []
        for t in range(ds.T):
            for c in range(2):
                sim.set_gripper(c, *pose[c, t])
            mujoco.mj_forward(sim.model, sim.data)
            viz.cam.lookat[:] = look[t]; viz.cam.distance = dist[t]
            mpp = 2 * dist[t] * tan_half / args.height          # metres per pixel at the look-at depth
            R_SIM, R_INDEP, R_COUPLED, R_CONTACT, R_FLOOR = PX_SIM * mpp, PX_INDEP * mpp, PX_COUPLED * mpp, PX_CONTACT * mpp, PX_FLOOR * mpp
            # pass 1/2: depth of the empty scene and of the simulated cables -> mask of the simulation
            viz.r.enable_depth_rendering()
            viz.r.update_scene(sim.data, viz.cam); d_empty = viz.r.render()
            for c in cables:
                viz._add_chain(pos[c, t], COL_SIM_SOLID, radius=R_SIM)
            d_sim = viz.r.render()
            # pass 3: predictions and contacts (opaque predictions, always fully visible)
            viz.r.update_scene(sim.data, viz.cam)
            ft = np.linalg.norm(ftrue[1, t], axis=-1); fm = np.linalg.norm(fe[1, t], axis=-1)
            sim_pts = [] if single else sim_contact_points(pos[1, t], ftrue[1, t])
            proj = projector(viz.r.scene, args.width, args.height, tan_half)
            arrows = [] if single else [(q, F, lab, A_SIM_CABLE) for q, F, lab in force_runs(pos[1, t], ftrue[1, t], 0.0)]
            for c in cables:
                arrows += [(q, F, lab, A_SIM_FLOOR) for q, F, lab in force_runs(pos[c, t], fl_sim[c, t], FLOOR_THR)]
            for q in sim_pts:
                viz._add_sphere(q, COL_SIM_CONTACT, radius=R_CONTACT)
            for c in cables:
                for q in pos[c, t][np.linalg.norm(fl_sim[c, t], axis=-1) > FLOOR_THR]:
                    add_disc(viz, q, COL_FLOOR_SIM, R_FLOOR * 1.6)
            if start[t] >= 0:
                for c in cables:
                    for q in pf[c, t][np.linalg.norm(fl_est[c, t], axis=-1) > FLOOR_THR]:
                        viz._add_sphere(q, COL_FLOOR_EST, radius=R_FLOOR)
                for c in cables:
                    viz._add_chain(p0[c, t], COL_INDEP, radius=R_INDEP)
                    viz._add_chain(pf[c, t], COL_COUPLED, radius=R_COUPLED)
                fc_est = fe[1, t] - fl_est[1, t]                     # cable part of the estimated force on the U
                fm = np.linalg.norm(fc_est, axis=-1)
                est_pts = [] if single else sim_contact_points(pf[1, t], fc_est * (fm > 0.003)[:, None])
                for q in est_pts:
                    viz._add_sphere(q, COL_EST_CONTACT, radius=R_CONTACT)
                if not single:
                    arrows += [(q, F, lab, A_EST_CABLE) for q, F, lab in force_runs(pf[1, t], fc_est, 0.003)]
                for c in cables:
                    arrows += [(q, F, lab, A_EST_FLOOR) for q, F, lab in force_runs(pf[c, t], fl_est[c, t], FLOOR_THR)]
            d_pred = viz.r.render()
            viz.r.disable_depth_rendering()
            img = viz.r.render().astype(np.float32)
            # simulation as a uniform translucent shadow underneath the predictions
            m = _soft((d_sim < d_empty - 1e-4).astype(np.float32)) * (1.0 - _soft((d_pred < d_empty - 1e-4).astype(np.float32)))
            a = (SIM_ALPHA * m)[..., None]
            img = (1 - a) * img + a * (np.asarray(COL_SIM_SOLID[:3], np.float32) * 255)
            img = draw_arrows(img.clip(0, 255).astype(np.uint8), arrows, proj, mpp, afont)
            if start[t] >= 0:
                st = f"Vorhersage seit {(t - start[t]) * cfg.control_dt:4.2f} s, ohne Rückgriff auf die Simulation"
                gs, ge = fl_sim[0, t, :, 2].sum(), fl_est[0, t, :, 2].sum()
                cab = "" if single else f"Kabelkraft auf das U: Simulation {ft.sum() * 1000:4.0f} mN | Modell {fm.sum() * 1000:4.0f} mN   Kontaktstellen: {len(sim_pts)} | {len(est_pts)}   "
                er = ((cab.rstrip() + "\n") if cab else "") + (f"Bodenkraft gezogenes Kabel: Simulation {gs * 1000:4.0f} mN | Modell {ge * 1000:4.0f} mN\n"
                      f"Mittlerer Fehler: ohne Kontaktkraft {np.nanmean(err0[list(cables), t]) * 100:5.2f} cm | nach Rückkopplung {np.nanmean(errf[list(cables), t]) * 100:5.2f} cm")
            else:
                st, er = f"Kontext wird gesammelt ({t + 1}/{ds.K} Schritte)", ("" if single else "Kabelkraft: noch keine Vorhersage\n") + "Bodenkraft: noch keine Vorhersage\nMittlerer Fehler: noch keine Vorhersage"
            info = [f"t = {t * cfg.control_dt:5.2f} s   Trajektorie {d['mode'][e]}   " + st] + er.split("\n") + [
                    (f"nur das gezogene Kabel (kein Kabelkontakt): L {mats[0]['length']:.2f} m, bend {mats[0]['bend']:.1e}, dens {mats[0]['density']:.0f}" if single else "") or f"U-Kabel: L {mats[1]['length']:.2f} m, bend {mats[1]['bend']:.1e}, dens {mats[1]['density']:.0f}   |   gezogenes Kabel: L {mats[0]['length']:.2f} m, bend {mats[0]['bend']:.1e}, dens {mats[0]['density']:.0f}"]
            frames.append(compose([add_legend(img, lfont, LEGEND)], ["Simulation, DLO-Anything und Kraft-Rückkopplung übereinander"], info, fonts, title_colors=((235, 235, 235),)))
        viz.r.close()   # a second live renderer in the same process corrupts the depth passes
        path = os.path.join(args.out, f"episode_{e:03d}.mp4")
        imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
        results[f"episode_{e:03d}"] = dict(no_force_cm=float(np.nanmean(err0) * 100), loop_cm=float(np.nanmean(errf) * 100), mode=str(d["mode"][e]))
        print(path, results[f"episode_{e:03d}"], flush=True)
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

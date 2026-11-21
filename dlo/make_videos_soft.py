"""Single-panel videos of the U scenario for the soft-contact tokens: simulation (dark band), the same
model without contact tokens, and with them. One continuous forecast, no reset to the simulation.
Arrows and discs show the contact forces of the simulation only; the model predicts no forces.

    One episode per call (the depth passes of the simulation band fail from the second episode of a process on):
    python -m dlo.make_videos_soft --token runs/soft/token_move/best.pt,runs/soft/token_fixed/best.pt --ohne runs/soft/ohne_move/best.pt,runs/soft/ohne_fixed/best.pt --index 82,41,55,98 --out videos/soft_token
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch, mujoco, imageio
from PIL import ImageFont

from .config import SimConfig, material_dict
from .make_videos import compose, FONT, FONT_B
from .make_videos_force import (A_SIM_CABLE, A_SIM_FLOOR, CABLE_AREA, COL_FLOOR_SIM, COL_SIM, COL_SIM_CONTACT, COL_SIM_SOLID, FLOOR_THR,
                                PX_CONTACT, PX_COUPLED, PX_FLOOR, PX_INDEP, PX_SIM, SIM_ALPHA, _soft, add_disc, draw_arrows, force_runs, projector)
from .make_videos_overlay import COL_COUPLED, COL_INDEP, add_legend, camera_path, merge_points, sim_contact_points
from .make_videos_two import TwoViz
from .soft_contact import SceneWindows, closest_on_cable, load, rollout
from .two_cable_env import TwoCableSim

LEGEND = (("Simulation (dark band)", COL_SIM), ("DLO-Anything without contact tokens", COL_INDEP), ("DLO-Anything with soft-contact tokens", COL_COUPLED),
          ("Cable contact in the simulation (arrow: force on the U)", [c / 255 for c in A_SIM_CABLE] + [1]),
          ("Floor contact in the simulation (arrow: floor force)", COL_FLOOR_SIM))


COL_MODEL_CONTACT = [0.85, 0.10, 0.75, 0.95]      # where the predicted cables touch: this is the contact the model works with
MODEL_GAP = 0.01                                  # [m] centre distance below which the predicted cables count as touching (simulation: about 0.6-0.9 cm)


def model_contacts(a, b, th=MODEL_GAP):
    """Points where the predicted cables a, b (N,3) are closer than th, and their smallest centre distance.
    The contact tokens are computed from exactly these predicted positions (closest point on the other cable per node)."""
    w = np.linspace(0, 1, 6, endpoint=False)[:, None, None]
    pts = np.concatenate([(a[None, :-1] * (1 - w) + a[None, 1:] * w).reshape(-1, 3), a[-1:]])
    q, _, gap, _, _ = closest_on_cable(torch.from_numpy(pts).float(), torch.from_numpy(b).float())
    gap, q = gap.numpy(), q.numpy()
    return merge_points((0.5 * (pts + q))[gap < th], cell=0.015), float(gap.min())


COL_MODEL_FLOOR = [0.05, 0.75, 0.95, 0.95]        # floor contact in the model: predicted node rests on the floor
MODEL_FLOOR = 0.008                               # [m] height of a predicted node centre below which it counts as lying on the floor (simulation: 4 mm)


def add_floor_disc(viz, p, rgba, radius):
    """Disc on the floor, slightly above the simulation's floor-contact discs so that both stay visible."""
    scn = viz.r.scene
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, radius, 0.0005]), np.array([p[0], p[1], 0.0018]),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    scn.ngeom += 1


def mats_len(nodes):
    return float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", required=True); ap.add_argument("--ohne", required=True)
    ap.add_argument("--data", default="data/val_u2.npz"); ap.add_argument("--out", default="videos/soft_token")
    ap.add_argument("--index", required=True)
    ap.add_argument("--fps", type=int, default=15); ap.add_argument("--width", type=int, default=1600); ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--name-a", default="without contact tokens", help="label of the --ohne pair"); ap.add_argument("--name-b", default="with soft-contact tokens", help="label of the --token pair")
    ap.add_argument("--solo", action="store_true", help="show the --token pair only")
    args = ap.parse_args()
    legend = (LEGEND[0], (f"DLO-Anything {args.name_a}", COL_INDEP), (f"DLO-Anything {args.name_b}", COL_COUPLED)) + LEGEND[3:] + ((f"Cable contact in the model (predicted cables closer than {MODEL_GAP * 1000:.0f} mm)", COL_MODEL_CONTACT),
                                                                                                                 (f"Floor contact in the model (predicted node lower than {MODEL_FLOOR * 1000:.0f} mm)", COL_MODEL_FLOOR))
    if args.solo: legend = legend[:1] + legend[2:]
    device = torch.device(args.device)
    m_tok = [load(p, device)[0] for p in args.token.split(",")]; m_ohne = [load(p, device)[0] for p in args.ohne.split(",")]; mcfg = m_tok[0].cfg
    ds = SceneWindows(args.data, mcfg)
    d = np.load(args.data); cfg = SimConfig()
    os.makedirs(args.out, exist_ok=True)
    idx = [int(i) for i in args.index.split(",")]
    P0 = rollout(m_ohne, ds, idx, device); PF = rollout(m_tok, ds, idx, device)
    fs = max(15, int(round(args.width / 72)))
    fonts = (ImageFont.truetype(FONT, fs), ImageFont.truetype(FONT_B, fs + 2)); lfont = ImageFont.truetype(FONT, fs); afont = ImageFont.truetype(FONT_B, max(13, fs - 4))
    sp = os.path.join(args.out, "summary.json")
    results = json.load(open(sp)) if os.path.exists(sp) else {}      # one episode per process: later episodes of a process lose the depth passes
    for n, e in enumerate(idx):
        pos, pose, p0, pf = ds.pos[e], ds.pose[e], P0[n], PF[n]
        ftrue = d["force_cable"][e].astype(np.float64); fl_sim = d["force_floor"][e].astype(np.float64)
        single = not np.abs(ftrue).sum() > 0
        fm = np.linalg.norm(ftrue, axis=-1); on = fm[1].sum(-1) > 0; wander = 0.0      # how far the contact travels along both cables in the simulation
        for c in (0, 1):
            arc = (fm[c] * np.arange(fm.shape[-1])).sum(-1) / np.maximum(fm[c].sum(-1), 1e-9) / (fm.shape[-1] - 1) * mats_len(pos[c, 0])
            wander += float(np.abs(np.diff(arc))[on[1:] & on[:-1]].sum())
        cables = (0,) if single else (0, 1)
        mats = [material_dict(ds.material[e, c]) for c in range(2)]
        err0, errf = np.linalg.norm(p0 - pos, axis=-1), np.linalg.norm(pf - pos, axis=-1)
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
            mpp = 2 * dist[t] * tan_half / args.height
            viz.r.enable_depth_rendering()
            viz.r.update_scene(sim.data, viz.cam); d_empty = viz.r.render()
            for c in cables:
                viz._add_chain(pos[c, t], COL_SIM_SOLID, radius=PX_SIM * mpp)
            d_sim = viz.r.render()
            viz.r.update_scene(sim.data, viz.cam)
            proj = projector(viz.r.scene, args.width, args.height, tan_half)
            ft = np.linalg.norm(ftrue[1, t], axis=-1)
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
                    if not args.solo: viz._add_chain(p0[c, t], COL_INDEP, radius=PX_INDEP * mpp)
                    viz._add_chain(pf[c, t], COL_COUPLED, radius=PX_COUPLED * mpp)
            mpts, mgap = model_contacts(pf[0, t], pf[1, t]) if started and not single else (np.zeros((0, 3)), float("nan"))
            for q in mpts:
                viz._add_sphere(q, COL_MODEL_CONTACT, radius=PX_CONTACT * mpp * 1.25)
            if started:
                for c in cables:
                    for q in pf[c, t][pf[c, t][:, 2] < MODEL_FLOOR]:
                        add_floor_disc(viz, q, COL_MODEL_FLOOR, PX_FLOOR * mpp * 0.9)
            d_pred = viz.r.render()
            viz.r.disable_depth_rendering()
            img = viz.r.render().astype(np.float32)
            m = _soft((d_sim < d_empty - 1e-4).astype(np.float32)) * (1.0 - _soft((d_pred < d_empty - 1e-4).astype(np.float32)))
            a = (SIM_ALPHA * m)[..., None]
            img = (1 - a) * img + a * (np.asarray(COL_SIM_SOLID[:3], np.float32) * 255)
            img = draw_arrows(img.clip(0, 255).astype(np.uint8), arrows, proj, mpp, afont)
            if started:
                st = f"Vorhersage seit {(t - ds.K + 1) * cfg.control_dt:4.2f} s, ohne Rückgriff auf die Simulation"
                er = ("" if single else f"Kabelkraft auf das U in der Simulation: {ft.sum() * 1000:4.0f} mN   Kontaktstellen: {len(sim_pts)}   (Kontakt wandert in dieser Szene {wander:.2f} m)\n") + \
                     ("" if single else f"Kontakt im Modell: {'ja  ' if len(mpts) else 'nein'} (kleinster Abstand der vorhergesagten Kabel {mgap * 100:5.1f} cm)\n") + \
                     ("Mittlerer Fehler: " if args.solo else f"Mittlerer Fehler: {args.name_a} {np.nanmean(err0[list(cables), t]) * 100:5.2f} cm | ") + f"{args.name_b} {np.nanmean(errf[list(cables), t]) * 100:5.2f} cm"
            else:
                st, er = f"Kontext wird gesammelt ({t + 1}/{ds.K} Schritte)", ("" if single else "Kabelkraft auf das U: noch keine Vorhersage\nKontakt im Modell: noch keine Vorhersage\n") + "Mittlerer Fehler: noch keine Vorhersage"
            info = [f"t = {t * cfg.control_dt:5.2f} s   Trajektorie {d['mode'][e]}   " + st] + er.split("\n") + [
                    (f"nur das gezogene Kabel (kein Kabelkontakt): L {mats[0]['length']:.2f} m, bend {mats[0]['bend']:.1e}, dens {mats[0]['density']:.0f}" if single else "") or f"U-Kabel: L {mats[1]['length']:.2f} m, bend {mats[1]['bend']:.1e}, dens {mats[1]['density']:.0f}   |   gezogenes Kabel: L {mats[0]['length']:.2f} m, bend {mats[0]['bend']:.1e}, dens {mats[0]['density']:.0f}"]
            frames.append(add_legend(img, lfont, legend))      # no header and no status line: the image fills the whole frame
        viz.r.close()
        path = os.path.join(args.out, f"episode_{e:03d}.mp4")
        imageio.mimwrite(path, frames, fps=args.fps, codec="libx264", quality=8, macro_block_size=1)
        results[f"episode_{e:03d}"] = dict(ohne_cm=float(np.nanmean(err0) * 100), token_cm=float(np.nanmean(errf) * 100), mode=str(d["mode"][e]))
        print(path, results[f"episode_{e:03d}"], flush=True)
    json.dump(results, open(os.path.join(args.out, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()

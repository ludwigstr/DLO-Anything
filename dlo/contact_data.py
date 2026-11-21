"""Stores the frozen DLO-Anything predictions used to train the contact correction.

    python -m dlo.contact_data --ckpt runs/force0/best.pt --data data/train_drive.npz --out data/contact_wide4/train --stride 10
"""
from __future__ import annotations
import argparse, os
import numpy as np
import torch

from .config import pad_material
from .couple import load_model
from .dataset import make_sample, from_gripper_frame, POS_SCALE
from .contact_model import contact_reference

STORE_SCALE = 1e4          # int16 in 0.1 mm


def predict_batch(model, mcfg, device, items):
    """items: list of (pos, pose, action, material, t0) for one cable -> world predictions (n,H,N,3)."""
    K, H = mcfg.context_len, mcfg.horizon
    ss = [make_sample(p, po, a, m, t0, K, H) for p, po, a, m, t0 in items]
    b = {k: torch.stack([s[k] for s in ss]).to(device) for k in ss[0]}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        y = model(b["node"], b["act"], b["mat"], b["cur"], force_ctx=b["force_ctx"], force_fut=b["force_fut"]).float().cpu().numpy()
    return np.stack([from_gripper_frame(y[i] / POS_SCALE, it[1][it[4]]) for i, it in enumerate(items)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/force0/best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device)
    model, mcfg = load_model(args.ckpt, device)
    K, H = mcfg.context_len, mcfg.horizon
    d = np.load(args.data)
    pos, pose, action = d["pos"].astype(np.float64), d["pose"], d["action"]
    material = pad_material(d["material"])
    E, _, T, N, _ = pos.shape
    meta = [(e, t0) for e in range(E) for t0 in range(K - 1, T - H, args.stride)]
    keep_meta, keep_base = [], []
    total = 0
    for i in range(0, len(meta), args.batch // 2):
        chunk = meta[i:i + args.batch // 2]
        items = [(pos[e, c], pose[e, c], action[e, c], material[e, c], t0) for e, t0 in chunk for c in (0, 1)]
        y = predict_batch(model, mcfg, device, items).reshape(len(chunk), 2, H, N, 3)
        q0 = np.stack([pos[e, :, t0] for e, t0 in chunk])
        geo = contact_reference(torch.from_numpy(y.astype(np.float32)).to(device), torch.from_numpy(q0.astype(np.float32)).to(device))
        use = geo["has"].any(1).cpu().numpy()
        total += len(chunk)
        keep_meta += [m for m, u in zip(chunk, use) if u]
        keep_base.append(np.clip(np.round(y[use] * STORE_SCALE), -32767, 32767).astype(np.int16))
        if (i // (args.batch // 2)) % 200 == 0:
            print(f"{total}/{len(meta)} windows, kept {len(keep_meta)}", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out + "_base.npy", np.concatenate(keep_base))
    np.save(args.out + "_meta.npy", np.array(keep_meta, np.int32))
    print(f"done: {len(keep_meta)} of {len(meta)} windows have a contact", flush=True)


if __name__ == "__main__":
    main()

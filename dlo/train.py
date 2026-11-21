"""Train the physics transformer.

Usage:
    python -m dlo.train --train data/train.npz --val data/val.npz --out runs/base --epochs 60
    # continue training on more data (new warmup + cosine cycle, weights from the previous best):
    python -m dlo.train --train data/train.npz,data/train_extra.npz --init runs/base/best.pt \
        --prev-log runs/base/log.json --out runs/cont1 --epochs 80 --spe 12
"""
from __future__ import annotations
import argparse, json, os, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ModelConfig, asdict_safe
from torch.utils.data import ConcatDataset
from .dataset import WindowDataset, TwoCableWindowDataset, POS_SCALE
from .model import PhysicsTransformer


def fwd(model, b, use_force=True, **kw):
    """Model call with the force channel of the batch (zeros for single-cable windows)."""
    if model.cfg.force and use_force:
        kw.update(force_ctx=b["force_ctx"], force_fut=b["force_fut"])
    return model(b["node"], b["act"], b["mat"], b["cur"], **kw)


def evaluate(model, loader, device, use_force=True):
    model.eval()
    se, n, per_h = 0.0, 0, None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            pred = fwd(model, b, use_force).float()
            err = torch.linalg.norm(pred - b["target"], dim=-1) / POS_SCALE   # metres, (B,H,N)
            se += err.sum().item()
            n += err.numel()
            ph = err.mean(dim=(0, 2)).cpu().numpy() * pred.shape[0]
            per_h = ph if per_h is None else per_h + ph
    model.train()
    return se / n, per_h / len(loader.dataset)


@torch.no_grad()
def evaluate_two(model, loader, device):
    """Two-cable validation windows: mean node error [m] over all windows, over windows whose
    horizon contains contact, and the same contact windows with the force channel zeroed
    (= what an uncoupled single-cable model predicts)."""
    model.eval()
    acc = dict(all=[0.0, 0], contact=[0.0, 0], contact_noforce=[0.0, 0])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for b in loader:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            touch = (b["force_fut"].abs().sum((1, 2, 3)) > 0)
            err = (fwd(model, b).float() - b["target"]).norm(dim=-1).mean((1, 2)) / POS_SCALE   # (B,)
            acc["all"][0] += err.sum().item(); acc["all"][1] += len(err)
            if touch.any():
                acc["contact"][0] += err[touch].sum().item(); acc["contact"][1] += int(touch.sum())
                err0 = (fwd(model, b, use_force=False).float() - b["target"]).norm(dim=-1).mean((1, 2)) / POS_SCALE
                acc["contact_noforce"][0] += err0[touch].sum().item(); acc["contact_noforce"][1] += int(touch.sum())
    model.train()
    return {f"val2_err_{k}_m": v[0] / max(v[1], 1) for k, v in acc.items()}


@torch.no_grad()
def evaluate_inverse(model, loader, device):
    """Inverse head on the validation windows (material masked): MSE of the normalised estimate per
    parameter and the forward error without known material."""
    model.eval()
    se, err, n = 0.0, 0.0, 0
    for b in loader:
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        B = b["mat"].shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred, inv = model(b["node"], b["act"], b["mat"], b["cur"],
                              mat_mask=torch.ones(B, dtype=torch.bool, device=device), return_inv=True)
        se = se + ((inv.float() - b["mat"]) ** 2).sum(0)
        err += (pred.float() - b["target"]).norm(dim=-1).mean().item() * B / POS_SCALE
        n += B
    model.train()
    per = (se / n).tolist()
    return dict(val_inv_mse=float(np.mean(per)), val_inv_mse_per_param=per, val_err_masked_m=err / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.npz")
    ap.add_argument("--val", default="data/val.npz")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--spe", type=int, default=32, help="windows sampled per episode per epoch")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--context", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--init", default=None, help="checkpoint to initialise the weights from (continued training)")
    ap.add_argument("--prev-log", default=None, help="log.json of the previous run; copied in front of this run's log")
    ap.add_argument("--mask-prob", type=float, default=0.0,
                    help="share of samples whose material tokens are masked ('unknown'); on those the inverse head is trained")
    ap.add_argument("--inv-weight", type=float, default=0.0, help="weight of the inverse (material-estimate) MSE loss")
    ap.add_argument("--force", action="store_true", help="enable the external contact-force channel")
    ap.add_argument("--train-two", default=None, help="two-cable datasets (comma-separated) mixed into training")
    ap.add_argument("--val-two", default=None, help="two-cable validation set (contact / no-force error)")
    ap.add_argument("--spe-two", type=int, default=32, help="windows per cable record per epoch (two-cable data)")
    ap.add_argument("--contact-frac", type=float, default=0.5, help="share of two-cable windows drawn with contact in the horizon")
    ap.add_argument("--cables", default="0,1", help="which cables of the two-cable data to train and validate on")
    ap.add_argument("--no-single", action="store_true", help="train on two-cable data only; --val is ignored")
    ap.add_argument("--best-on-two", action="store_true", help="select best.pt by the two-cable validation windows")
    ap.add_argument("--ctx-drop", type=float, default=0.0, help="share of windows whose past forces are zeroed: past forces are often unknown in use")
    ap.add_argument("--subset", type=int, default=0, help="train on the first N episodes of --train only (scaling study)")
    ap.add_argument("--total-steps", type=int, default=0, help="fixed budget of gradient steps: sets --spe so that one epoch has about "
                    "--steps-per-epoch steps whatever the number of episodes, and --epochs accordingly")
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cables = tuple(int(c) for c in args.cables.split(","))

    torch.manual_seed(args.seed)
    mcfg = ModelConfig(context_len=args.context, horizon=args.horizon, d_model=args.d_model,
                       n_heads=args.heads, n_layers=args.layers, force=args.force)
    os.makedirs(args.out, exist_ok=True)
    json.dump(asdict_safe(mcfg), open(os.path.join(args.out, "model_config.json"), "w"), indent=2)

    train_ds = None if args.no_single else WindowDataset(args.train, mcfg, args.spe, seed=args.seed, subset=args.subset)
    if args.total_steps:
        train_ds.spe = max(1, round(args.steps_per_epoch * args.batch / train_ds.E))
        args.epochs = max(1, round(args.total_steps / (len(train_ds) // args.batch)))
        json.dump(dict(vars(args), episodes=train_ds.E), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    n_train_eps = 0 if args.no_single else train_ds.E
    if args.train_two:
        two_ds = TwoCableWindowDataset(args.train_two, mcfg, args.spe_two, seed=2, contact_frac=args.contact_frac, cables=cables)
        print(f"two-cable records {two_ds.R}, windows/epoch {len(two_ds)}, steps in contact {two_ds.frac_touch * 100:.1f} %")
        train_ds = two_ds if args.no_single else ConcatDataset([train_ds, two_ds])
    val_ds = TwoCableWindowDataset(args.val_two, mcfg, 8, seed=1, contact_frac=0.5, cables=cables) if args.no_single else WindowDataset(args.val, mcfg, 8, seed=1)
    train_dl = DataLoader(train_ds, args.batch, shuffle=True, num_workers=args.workers,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, args.batch, shuffle=False, num_workers=4, pin_memory=True)
    val2_dl = None
    if args.val_two:
        val2_ds = TwoCableWindowDataset(args.val_two, mcfg, 8, seed=3, contact_frac=0.5, cables=cables)
        val2_dl = DataLoader(val2_ds, args.batch, shuffle=False, num_workers=4, pin_memory=True)
    print(f"train episodes {n_train_eps}, windows/epoch {len(train_ds)}, val windows {len(val_ds)}")

    device = torch.device(args.device)
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(device)
    print(f"params: {PhysicsTransformer.n_params(model) / 1e6:.2f} M")
    if args.init:
        ck = torch.load(args.init, map_location=device)
        own = model.state_dict()
        loaded, partial = 0, []
        for k, v in ck["model"].items():
            if k in own and own[k].shape == v.shape:
                own[k] = v; loaded += 1
            elif k == "node_proj.weight" and k in own and own[k].shape[0] == v.shape[0] and own[k].shape[1] > v.shape[1]:
                own[k][:, -v.shape[1]:] = v; own[k][:, :-v.shape[1]] = 0   # longer context: old weights act on the newest steps
                partial.append(f"{k} {tuple(v.shape)}->{tuple(own[k].shape)} (right-aligned)")
            elif k in own and own[k].dim() == v.dim() and all(a >= b for a, b in zip(own[k].shape, v.shape)):
                sl = tuple(slice(0, n) for n in v.shape)
                own[k][sl] = v; partial.append(f"{k} {tuple(v.shape)}->{tuple(own[k].shape)}")
        model.load_state_dict(own)
        print(f"initialised from {args.init}: {loaded} tensors copied, partially: {partial}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    total = args.epochs * len(train_dl)
    warm = max(1, min(1000, total // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))

    log = []
    ep0 = 0
    if args.prev_log:
        log = json.load(open(args.prev_log))
        ep0 = log[-1]["epoch"] + 1
        for r in log:
            r.setdefault("run", "prev")
    best = float("inf")
    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        run_loss, run_inv, nb = 0.0, 0.0, 0
        for b in train_dl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            if args.ctx_drop > 0:
                keep = (torch.rand(len(b["force_ctx"]), device=device) >= args.ctx_drop).float()[:, None, None]
                b["force_ctx"] = b["force_ctx"] * keep
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.mask_prob > 0:
                    mask = torch.rand(b["mat"].shape[0], device=device) < args.mask_prob
                    pred, inv = fwd(model, b, mat_mask=mask, return_inv=True)
                    loss_fwd = nn.functional.mse_loss(pred.float(), b["target"])
                    inv_err = ((inv.float() - b["mat"]) ** 2).mean(1)
                    loss_inv = (inv_err * mask).sum() / mask.sum().clamp(min=1)
                    loss = loss_fwd + args.inv_weight * loss_inv
                    run_inv += loss_inv.item()
                else:
                    pred = fwd(model, b)
                    loss = nn.functional.mse_loss(pred.float(), b["target"])
                    loss_fwd = loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run_loss += loss_fwd.item(); nb += 1
        val_err, per_h = evaluate(model, val_dl, device)
        rec = dict(epoch=ep0 + epoch, run=os.path.basename(args.out), train_mse=run_loss / nb, val_err_m=val_err,
                   val_err_h1=float(per_h[0]), val_err_h50=float(per_h[49]), val_err_h100=float(per_h[-1]),
                   lr=sched.get_last_lr()[0], time=time.time() - t0)
        if args.mask_prob > 0:
            rec["train_inv_mse"] = run_inv / nb
            rec.update(evaluate_inverse(model, val_dl, device))
        if val2_dl is not None:
            rec.update(evaluate_two(model, val2_dl, device))
        log.append(rec)
        print(f"ep {ep0 + epoch:3d} train_mse {rec['train_mse']:.5f} | val mean err {val_err * 100:.2f} cm "
              f"(h=1: {per_h[0] * 100:.2f}, h=50: {per_h[49] * 100:.2f}, h=100: {per_h[-1] * 100:.2f}) "
              + (f"| inv mse {rec['train_inv_mse']:.4f} val_inv {rec['val_inv_mse']:.4f} masked-val {rec['val_err_masked_m'] * 100:.2f} cm " if args.mask_prob > 0 else "")
              + (f"| two: all {rec['val2_err_all_m'] * 100:.2f} contact {rec['val2_err_contact_m'] * 100:.2f} "
                 f"no-force {rec['val2_err_contact_noforce_m'] * 100:.2f} cm " if val2_dl is not None else "")
              + f"| {rec['time']:.0f}s", flush=True)
        json.dump(log, open(os.path.join(args.out, "log.json"), "w"), indent=1)
        np.save(os.path.join(args.out, "val_err_per_horizon.npy"), per_h)
        torch.save({"model": model.state_dict(), "cfg": asdict_safe(mcfg)}, os.path.join(args.out, "last.pt"))
        score = rec["val2_err_all_m"] if args.best_on_two else val_err
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "cfg": asdict_safe(mcfg)}, os.path.join(args.out, "best.pt"))
    print("best score, mean err (m):", best)


if __name__ == "__main__":
    main()

"""Train the two cable models of DLO-Anything (pulled cable and fixed U) in two stages.

    python train.py --train data/F_12000.npz --val data/val_designed.npz --out runs/F_12000

Stage 1: each model alone on single 0.5-s predictions that start from the simulated state (29,720 steps, batch 128).
Stage 2: both models together on chains of twelve predictions (6 s): every prediction starts from the models' own output of
         the one before, the contact tokens are rebuilt from the predicted cables, and the gradient passes through the
         whole chain (12,000 steps, batch 24).
Checkpoints are selected on --val, which must contain trained motion kinds only (never the held-out validation / test files).
Result: <out>/model_cable0.pt (pulled cable) and <out>/model_cable1.pt (fixed U)."""
from __future__ import annotations
import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import ModelConfig, POS_SCALE, PhysicsTransformer, SceneWindows, asdict_safe, load, own_in, predict, rigid_in


def chain(models, bb, W, detach):
    """-> list of W predictions (B,2,H,N,3); pass w+1 gets the last K predicted steps of pass w as context."""
    ctx, K, out = bb["ctx"], bb["ctx"].shape[2], []
    for w in range(W):
        b = {k: v[:, w] for k, v in bb.items() if k != "ctx"}
        p = predict(models, ctx, b); out.append(p)
        ctx = torch.cat([ctx, p.detach() if detach else p], 2)[:, :, -K:]          # horizons shorter than K keep older context steps
    return out


def optimizer(params, lr, steps, warm):
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))


def batches(ds, batch, steps, workers):
    """Endless stream of training batches (at least `steps` of them)."""
    ds.spe = math.ceil(steps * batch / ds.E) + 1
    while True:
        yield from DataLoader(ds, batch, shuffle=True, num_workers=workers, drop_last=True, pin_memory=True)


@torch.no_grad()
def validate(models, loader, dev, W, cables=(0, 1)):
    """Mean node error [cm] of the chained forecast over W predictions (W = 1: a single prediction from the simulated state)."""
    [m.eval() for m in models.values()]; tot, n = 0.0, 0
    for bb in loader:
        bb = {k: v.to(dev) for k, v in bb.items()}
        if W == 1:
            b = {k: v[:, 0] for k, v in bb.items() if k != "ctx"}
            e = (predict(models, bb["ctx"], b, cables=cables) - b["gt"][:, list(cables)]).norm(dim=-1).mean((1, 2, 3)) * 100
        else:
            ps = chain([models[0], models[1]], bb, W, True)
            e = torch.stack([(p - bb["gt"][:, w]).norm(dim=-1).mean((1, 2, 3)) for w, p in enumerate(ps)], 1).mean(1) * 100
        tot += e.sum().item(); n += len(e)
    [m.train() for m in models.values()]
    return tot / n


def stage1(args, dev, cable):
    """One cable model on single predictions from the simulated state."""
    path = os.path.join(args.out, f"stage1_cable{cable}.pt")
    if os.path.exists(path):
        return path
    torch.manual_seed(0)
    mcfg = ModelConfig(d_model=args.d_model, n_heads=args.heads, n_layers=args.layers, horizon=args.horizon)
    mcfg.contact_in, mcfg.rigid_in = own_in(mcfg), rigid_in(mcfg)
    model = PhysicsTransformer(mcfg, node_in=mcfg.context_len * 6).to(dev)
    if args.init:                                   # e.g. a model pretrained on single cables: tensors of matching shape are taken over
        own = model.state_dict(); src = torch.load(args.init, map_location=dev)["model"]
        hit = [k for k, v in src.items() if k in own and own[k].shape == v.shape]
        own.update({k: src[k] for k in hit}); model.load_state_dict(own); print(f"initialised from {args.init}: {len(hit)} of {len(own)} tensors", flush=True)
    tr = SceneWindows(args.train, mcfg); va = DataLoader(SceneWindows(args.val, mcfg, 8, seed=1), args.batch1, num_workers=4)
    opt, sched = optimizer(model.parameters(), args.lr1, args.steps1, max(1, min(1000, args.steps1 // 10)))
    print(f"stage 1, cable {cable}: {PhysicsTransformer.n_params(model) / 1e6:.2f} M parameters, {tr.E} scenes", flush=True)
    best, run, t0, every = float("inf"), 0.0, time.time(), max(1, args.steps1 // 40)
    for step, bb in enumerate(batches(tr, args.batch1, args.steps1, args.workers), 1):
        bb = {k: v.to(dev, non_blocking=True) for k, v in bb.items()}
        b = {k: v[:, 0] for k, v in bb.items() if k != "ctx"}
        loss = F.mse_loss(predict({cable: model}, bb["ctx"], b, cables=(cable,)) * POS_SCALE, b["gt"][:, [cable]] * POS_SCALE)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); run += loss.item()
        if step % every == 0 or step == args.steps1:
            v = validate({cable: model}, va, dev, 1, (cable,))
            print(f"  step {step} train_mse {run / every:.5f} | validation {v:.2f} cm | {time.time() - t0:.0f}s", flush=True); run = 0.0
            if v < best:
                best = v; torch.save({"model": model.state_dict(), "cfg": asdict_safe(mcfg)}, path)
        if step >= args.steps1:
            break
    return path


def stage2(args, dev, paths):
    """Both models together on the chained 6-s forecast."""
    models = [load(p, dev)[0] for p in paths]; mcfg = models[0].cfg; W = args.chain
    [m.train() for m in models]; params = [p for m in models for p in m.parameters()]
    tr = SceneWindows(args.train, mcfg, n_windows=W); va = DataLoader(SceneWindows(args.val, mcfg, 4, seed=1, n_windows=W), args.batch2, num_workers=4)
    opt, sched = optimizer(params, args.lr2, args.steps2, 200)

    def save():
        for c, m in enumerate(models):
            torch.save({"model": m.state_dict(), "cfg": asdict_safe(mcfg)}, os.path.join(args.out, f"model_cable{c}.pt"))

    best, run, t0, every = validate(dict(enumerate(models)), va, dev, W), 0.0, time.time(), max(1, min(500, args.steps2 // 4))
    print(f"stage 2: {W * mcfg.horizon * 0.02:.0f}-s chain, validation at the start {best:.2f} cm", flush=True); save()
    for step, bb in enumerate(batches(tr, args.batch2, args.steps2, args.workers), 1):
        bb = {k: v.to(dev, non_blocking=True) for k, v in bb.items()}
        ps = chain(models, bb, W, False)
        loss = sum(F.mse_loss(p * POS_SCALE, bb["gt"][:, w] * POS_SCALE) for w, p in enumerate(ps)) / W
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step(); run += loss.item()
        if step % every == 0 or step == args.steps2:
            v = validate(dict(enumerate(models)), va, dev, W)
            print(f"  step {step} train_mse {run / every:.5f} | chain validation {v:.2f} cm | {time.time() - t0:.0f}s", flush=True); run = 0.0
            if v < best:
                best = v; save()
        if step >= args.steps2:
            break
    print(f"best chain validation {best:.2f} cm", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True); ap.add_argument("--val", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="optional checkpoint to start stage 1 from")
    ap.add_argument("--steps1", type=int, default=29720); ap.add_argument("--batch1", type=int, default=128); ap.add_argument("--lr1", type=float, default=3e-4)
    ap.add_argument("--steps2", type=int, default=12000); ap.add_argument("--batch2", type=int, default=24); ap.add_argument("--lr2", type=float, default=1e-4)
    ap.add_argument("--chain", type=int, default=12, help="predictions per chain in stage 2 (12 x 0.5 s = 6 s)")
    ap.add_argument("--horizon", type=int, default=25, help="steps per prediction (25 = 0.5 s)")
    ap.add_argument("--d_model", type=int, default=256); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); dev = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True); json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    stage2(args, dev, [stage1(args, dev, c) for c in (0, 1)])


if __name__ == "__main__":
    main()

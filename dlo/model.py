"""Physics transformer (DLO-Anything, flat-surface variant).

Tokens:  N cable-node tokens (context positions + velocities, node-id embedding)
       + M material tokens   (one per material parameter, parameter-id embedding)
       + H action tokens     (future gripper motion, time embedding)
       + 1 global token      (scene constants)
A transformer encoder mixes them; a factorised decoder predicts, for every (future step k,
node i), the displacement of node i relative to its current position.

Force channel (cfg.force, 2026-11): external contact forces on the nodes enter in three places,
all through zero-initialised layers so that a model without forces is reproduced exactly when the
channel is switched on and the forces are zero:
  * node tokens   += node_force_proj(context forces of node i, K steps x 3)
  * action tokens += act_force_proj(future forces of all nodes at step k, N x 3)
  * decoder: the (k, i) interaction term gets + dec_force(future force on node i at step k)

Soft-contact tokens (cfg.contact_in > 0, 2026-11-19; replaces the force channel + force network): one
token per node of a contact partner (another cable), see soft_contact.py. The tokens sit between
the action tokens and the global token; contact is resolved by attention, no force is passed in.
The planned motion of the partner's hand is added to the action tokens (zero-initialised).
Rigid-contact tokens (cfg.rigid_in > 0): one token per patch of a rigid surface; in the prototype the
floor patch below every node. They follow the soft-contact tokens.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from .config import ModelConfig


class PhysicsTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, node_in: int, act_in: int = 7, global_in: int = 3):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.node_proj = nn.Linear(node_in, d)
        if cfg.force:
            self.node_force_proj = nn.Linear(cfg.context_len * 3, d)
            self.act_force_proj = nn.Linear(cfg.n_nodes * 3, d)
            self.dec_force = nn.Linear(3, 128)
            for lin in (self.node_force_proj, self.act_force_proj, self.dec_force):
                nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        if cfg.contact_in:
            self.contact_proj = nn.Linear(cfg.contact_in, d)
            self.contact_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)   # index of the partner node
            self.contact_type = nn.Parameter(torch.randn(d) * 0.02)
            self.act_partner_proj = nn.Linear(act_in, d)
            nn.init.zeros_(self.act_partner_proj.weight); nn.init.zeros_(self.act_partner_proj.bias)
        if getattr(cfg, "contact_fut_in", 0):
            # per-step contact over the horizon, built from the predicted motion of both cables (soft_contact_iter.py)
            self.contact_fut_proj = nn.Linear(cfg.contact_fut_in, d)
            nn.init.zeros_(self.contact_fut_proj.weight); nn.init.zeros_(self.contact_fut_proj.bias)
        if cfg.rigid_in:
            self.rigid_proj = nn.Linear(cfg.rigid_in, d)
            self.rigid_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)     # index of the node above the patch
            self.rigid_type = nn.Parameter(torch.randn(d) * 0.02)
        self.node_emb = nn.Parameter(torch.randn(cfg.n_nodes, d) * 0.02)
        self.mat_proj = nn.Linear(1, d)
        self.mat_emb = nn.Parameter(torch.randn(cfg.n_material, d) * 0.02)
        self.act_proj = nn.Linear(act_in, d)
        self.time_emb = nn.Parameter(torch.randn(cfg.horizon, d) * 0.02)
        self.global_proj = nn.Linear(global_in, d)
        self.type_emb = nn.Parameter(torch.randn(4, d) * 0.02)  # node / material / action / global
        # "material unknown": replaces the material tokens of masked samples (inverse-head training)
        self.mat_unknown = nn.Parameter(torch.randn(cfg.n_material, d) * 0.02)
        # inverse head: pooled node tokens + global token -> the 6 normalised material parameters
        self.inv_head = nn.Sequential(nn.Linear(2 * d, 256), nn.GELU(), nn.Linear(256, cfg.n_material))

        layer = nn.TransformerEncoderLayer(d, cfg.n_heads, 4 * d, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(d)

        # factorised decoder: (time token k, node token i) -> displacement of node i at step k
        dd = 128
        self.dec_t = nn.Linear(d, dd)
        self.dec_n = nn.Linear(d, dd)
        self.dec_mlp = nn.Sequential(nn.Linear(3 * dd, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                     nn.Linear(256, 3))
        nn.init.zeros_(self.dec_mlp[-1].weight)
        nn.init.zeros_(self.dec_mlp[-1].bias)

    def forward(self, node, act, mat, cur, glob=None, mat_mask=None, return_inv=False,
                force_ctx=None, force_fut=None, contact=None, act_partner=None, rigid=None, contact_fut=None):
        """node (B,N,F)  act (B,H,7)  mat (B,M)  cur (B,N,3) -> pred positions (B,H,N,3)

        mat_mask (B,) bool: samples whose material tokens are replaced by the "unknown" embedding.
        return_inv: also return the inverse head's material estimate (B,M) from the context.
        force_ctx (B,N,K*3): external force on every node over the context steps (hand frame, scaled)
        force_fut (B,H,N,3): external force on every node over the future steps (same units)
        Both are optional; None means no external force (the single-cable case).
        contact (B,P,cfg.contact_in): soft-contact tokens, one per partner node; None = no partner.
        act_partner (B,H,7): planned hand motion of the partner in the own frame.
        rigid (B,R,cfg.rigid_in): rigid-contact tokens, one per patch of a rigid surface (the floor).
        """
        B, N, _ = node.shape
        H = act.shape[1]
        if glob is None:
            glob = torch.zeros(B, 3, device=node.device, dtype=node.dtype)
        tn = self.node_proj(node) + self.node_emb[None] + self.type_emb[0]
        tm = self.mat_proj(mat[..., None]) + self.mat_emb[None] + self.type_emb[1]
        if mat_mask is not None:
            tm = torch.where(mat_mask[:, None, None], (self.mat_unknown + self.type_emb[1])[None].expand_as(tm), tm)
        ta = self.act_proj(act) + self.time_emb[None, :H] + self.type_emb[2]
        if self.cfg.force and force_ctx is not None:
            tn = tn + self.node_force_proj(force_ctx)
        if self.cfg.force and force_fut is not None:
            ta = ta + self.act_force_proj(force_fut.reshape(B, H, N * 3))
        tg = (self.global_proj(glob) + self.type_emb[3])[:, None]
        toks = [tn, tm, ta]
        if self.cfg.contact_in and contact is not None:
            if act_partner is not None:
                toks[2] = ta + self.act_partner_proj(act_partner)
            tc = self.contact_proj(contact) + self.contact_emb[None, :contact.shape[1]] + self.contact_type
            if contact_fut is not None and getattr(self.cfg, "contact_fut_in", 0):
                tc = tc + self.contact_fut_proj(contact_fut)
            toks.append(tc)
        if self.cfg.rigid_in and rigid is not None:
            toks.append(self.rigid_proj(rigid) + self.rigid_emb[None, :rigid.shape[1]] + self.rigid_type)
        x = torch.cat(toks + [tg], dim=1)
        x = self.norm(self.encoder(x))
        hn = x[:, :N]
        ha = x[:, N + mat.shape[1]:N + mat.shape[1] + H]
        a = self.dec_t(ha)[:, :, None, :].expand(B, H, N, -1)
        n = self.dec_n(hn)[:, None, :, :].expand(B, H, N, -1)
        an = a * n
        if self.cfg.force and force_fut is not None:
            an = an + self.dec_force(force_fut)
        disp = self.dec_mlp(torch.cat([an, a, n], dim=-1))
        pred = cur[:, None] + disp
        if return_inv:
            inv = self.inv_head(torch.cat([hn.mean(1), x[:, -1]], dim=-1))
            return pred, inv
        return pred

    @torch.no_grad()
    def estimate_material(self, node, act, cur):
        """Direct material estimate (B,M, normalised) from the context alone (material masked)."""
        B = node.shape[0]
        mat = torch.zeros(B, self.cfg.n_material, device=node.device, dtype=node.dtype)
        return self(node, act, mat, cur, mat_mask=torch.ones(B, dtype=torch.bool, device=node.device), return_inv=True)[1]

    @staticmethod
    def n_params(m):
        return sum(p.numel() for p in m.parameters())

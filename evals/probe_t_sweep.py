#!/usr/bin/env python3
"""探针噪声域扫描: v5模型在真实place窗上, pred_x0的探针AUC随t的曲线。
决定 T_PROBE=650 能否上提(监督带向部署操作点σ>=0.625靠拢)。
Usage: python probe_t_sweep.py <ckpt>"""
import os, sys, numpy as np, torch
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2")
from mg2_model import load_expanded_base, PLACE_IDX
from wan.vae.wanx_vae import get_wanx_vae_wrapper

ckpt = sys.argv[1]
DATA = "/root/autodl-tmp/vpt_mc_352"; F = 15; NUM_PIX = (F - 1) * 4 + 1
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
dev, dt = "cuda", torch.bfloat16
ATTACK_IDX = 5; CAM_SCALE = 1.0 / 15.0; CAM_DEG_CLIP = 20.0
TS = [400, 500, 600, 650, 700, 750, 800, 900]
NW = 24

# --- 采样窗口: 新段, held稳定, 窗内有place, 无GUI ---
wins = []
for s in sorted(os.listdir(DATA)):
    if s.startswith(("bumpy", "cheeky")): continue
    npz = f"{DATA}/{s}/seg_000.npz"; latp = f"{DATA}/{s}/seg_000_lat44.npy"
    if not (os.path.exists(npz) and os.path.exists(latp)): continue
    z = np.load(npz)
    if "place_fire" not in z.files: continue
    pf = z["place_fire"] > 0; h = z["held_id"]; g = np.asarray(z.get("gui", np.zeros(len(h))))
    ev = np.flatnonzero(pf)
    for e in ev:
        st = e - 7
        if st < 0 or st + F > len(h): continue
        if (h[st:st + F] != h[st]).any() or h[st] <= 0 or g[st:st + F].any(): continue
        wins.append((s, int(st)))
        break
    if len(wins) >= NW: break
print(f"窗口 {len(wins)}", flush=True)

model = load_expanded_base(dev, dt, ledger=True)
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
model.eval().requires_grad_(False)
vae = get_wanx_vae_wrapper("/root/autodl-tmp/mg2_weights", torch.float16).to(dev, dt)

class MCProbe(torch.nn.Module):
    def __init__(self, in_ch, dim=256):
        super().__init__(); nn = torch.nn
        self.enc = nn.Sequential(nn.Conv2d(in_ch, 64, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(128, dim, 3, 2, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1))
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2); self.pos = nn.Parameter(torch.zeros(1, F, dim))
        self.head_mine = nn.Linear(dim, 1); self.head_pick = nn.Linear(dim, 1); self.head_use = nn.Linear(dim, 1)
    def forward(self, x):
        B, Fr = x.shape[:2]
        h = self.enc(x.flatten(0, 1)).reshape(B, Fr, -1) + self.pos; h = self.tf(h)
        return self.head_mine(h)[..., 0], self.head_pick(h)[..., 0], self.head_use(h)[..., 0]
probe = MCProbe(32).to(dev).eval().requires_grad_(False)
probe.load_state_dict(torch.load("checkpoints/mc_probe44/best.pt", map_location="cpu")["model"])
HUD_R, HUD_C = slice(38, 44), slice(20, 60)

def place_logit(x_lat):                                  # [1,16,F,44,80] -> [F]
    x = x_lat.permute(0, 2, 1, 3, 4).float().clone(); x[:, :, :, HUD_R, HUD_C] = 0
    dxx = torch.diff(x, dim=1, prepend=x[:, :1])
    _, _, su = probe(torch.cat([x, dxx], dim=2))
    return su[0]

def auc(pos, neg):
    lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]; sc = np.r_[pos, neg]
    o = np.argsort(sc); r = np.empty_like(o, float); r[o] = np.arange(1, len(sc) + 1)
    return (r[lab == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg) + 1e-9)

scores = {t: ([], []) for t in TS}
leak = {t: ([], []) for t in TS}
with torch.no_grad():
    for wi, (s, st) in enumerate(wins):
        z = np.load(f"{DATA}/{s}/seg_000.npz")
        x0 = torch.from_numpy(np.asarray(np.load(f"{DATA}/{s}/seg_000_lat44.npy", mmap_mode="r")[st:st + F], np.float32)).to(dev)
        x0 = x0.permute(1, 0, 2, 3).unsqueeze(0)                    # [1,16,F,44,80]
        act = z["action"][st:st + F].astype(np.float32)
        kb = np.zeros((F, 7), np.float32)
        kb[:, 0] = act[:, 0]; kb[:, 1] = act[:, 2]; kb[:, 2] = act[:, 1]; kb[:, 3] = act[:, 3]
        kb[:, 4] = act[:, 4]; kb[:, ATTACK_IDX] = act[:, 7]; kb[:, PLACE_IDX] = (act[:, 8] > 0)
        ms = np.zeros((F, 2), np.float32)
        ms[:, 0] = -np.clip(act[:, 10], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
        ms[:, 1] = np.clip(act[:, 9], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
        kbt = torch.from_numpy(np.repeat(kb, 4, 0)[:NUM_PIX])[None].to(dev, dt)
        mst = torch.from_numpy(np.repeat(ms, 4, 0)[:NUM_PIX])[None].to(dev, dt)
        held = torch.from_numpy(z["held_id"][st:st + F].astype(np.int64))[None].to(dev)
        bag = np.asarray(z["bag_counts"][st:st + F], np.float32)
        cnt = torch.from_numpy(bag[np.arange(F), z["held_id"][st:st + F].clip(0, 39)])[None].to(dev, dt)
        placef = z["place_fire"][st:st + F] > 0
        # cond_concat/visual_context 与训练一致(首帧i2v)
        fp = vae.decode(x0[:, :, 0:1].to(dt), device=dev, **TK).float().clamp(-1, 1).to(dev)
        fp0 = (fp[:, :, 0:1] if fp.shape[1] == 3 else fp.permute(0, 2, 1, 3, 4)[:, :, 0:1]).to(dev)
        pad = torch.zeros(1, 3, NUM_PIX - 1, fp0.shape[-2], fp0.shape[-1], device=dev, dtype=dt)
        img_cond = vae.encode(torch.cat([fp0.to(dt), pad], 2), device=dev, **TK).to(dev, dt)
        if img_cond.shape[1] != 16: img_cond = img_cond.permute(0, 2, 1, 3, 4)
        mask = torch.zeros(1, 4, F, 44, 80, device=dev, dtype=dt); mask[:, :, 0] = 1.0
        y = torch.cat([mask, img_cond], 1)
        vc = vae.clip.encode_video(fp0.to(dt)).to(dt)
        g = torch.Generator(device=dev).manual_seed(1000 + wi)
        noise = torch.randn(x0.shape, generator=g, device=dev)
        for t in TS:
            sig = t / 1000.0
            xt = ((1 - sig) * x0 + sig * noise).to(dt)
            pred = model(xt, t=torch.full((1,), float(t), device=dev, dtype=dt), visual_context=vc,
                         cond_concat=y, keyboard_cond=kbt, mouse_cond=mst,
                         ledger_held=held, ledger_count=cnt)
            if isinstance(pred, (list, tuple)): pred = pred[0]
            px0 = xt.float() - sig * pred.float()
            lg = place_logit(px0).cpu().numpy()
            scores[t][0].extend(lg[placef].tolist()); scores[t][1].extend(lg[~placef].tolist())
            lgL = place_logit(xt.float()).cpu().numpy()          # 纯泄漏基线: 探针直接读xt, 无模型贡献
            leak[t][0].extend(lgL[placef].tolist()); leak[t][1].extend(lgL[~placef].tolist())
        if wi % 8 == 0: print(f"  {wi+1}/{len(wins)}", flush=True)
print("\nt      AUC(pred_x0)   AUC(xt纯泄漏基线)")
for t in TS:
    print(f"{t:>4}   {auc(np.array(scores[t][0]), np.array(scores[t][1])):.3f}        {auc(np.array(leak[t][0]), np.array(leak[t][1])):.3f}", flush=True)
print("PROBE-T-SWEEP-DONE", flush=True)

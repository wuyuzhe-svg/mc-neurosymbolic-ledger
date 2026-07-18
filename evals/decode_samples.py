#!/usr/bin/env python3
"""Decode a few place-event windows from NEW 6xx segs so labels can be eyeballed
against pixels. Red border = latent frame with place_fire>0; text row printed to
console gives held_id / action for each frame. Light job on one GPU."""
import os, sys, numpy as np, torch, imageio

sys.path.insert(0, "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2")
from wan.vae.wanx_vae import get_wanx_vae_wrapper

DATA = "/root/autodl-tmp/vpt_mc_352"
OUT = "/root/autodl-tmp/vrisingwm/sample_clips"
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
N_SEGS, W = 3, 16                      # 16 latent frames = 4s per clip
dev, dt = "cuda", torch.float16
os.makedirs(OUT, exist_ok=True)

# pick new segs with the most place events (skip old naming + quarantined)
cands = []
for s in sorted(os.listdir(DATA)):
    if s.startswith(("bumpy", "cheeky")): continue
    npz = f"{DATA}/{s}/seg_000.npz"; lat = f"{DATA}/{s}/seg_000_lat44.npy"
    if not (os.path.exists(npz) and os.path.exists(lat)): continue
    z = np.load(npz)
    if "place_fire" not in z.files: continue
    cands.append((int((z["place_fire"] > 0).sum()), s))
cands.sort(reverse=True)
picks = [s for _, s in cands[:N_SEGS]]
print("挑中:", picks)

vae = get_wanx_vae_wrapper("/root/autodl-tmp/mg2_weights", torch.float16).to(dev, dt)

for s in picks:
    z = np.load(f"{DATA}/{s}/seg_000.npz")
    pf = z["place_fire"] > 0; held = z["held_id"]; act = z["action"]
    lat = np.load(f"{DATA}/{s}/seg_000_lat44.npy", mmap_mode="r")
    ev = np.flatnonzero(pf)
    st = max(0, min(int(ev[len(ev) // 2]) - W // 2, lat.shape[0] - W))  # window around a mid-seg event
    x = torch.from_numpy(np.asarray(lat[st:st + W])).to(dev, dt)       # [W,16,44,80]
    with torch.no_grad():
        pix = vae.decode(x.permute(1, 0, 2, 3).unsqueeze(0), device=dev, **TK)[0]
        pix = pix.float().clamp(-1, 1)
        if pix.shape[0] == 3: pix = pix.permute(1, 0, 2, 3)            # [T,3,H,W]
    v = ((pix.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    Tpix = v.shape[0]
    print(f"\n{s}  latent[{st}:{st+W}]  解码{Tpix}像素帧")
    for i in range(W):
        mark = " <-- PLACE" if pf[st + i] else ""
        print(f"  lat{st+i}: held={held[st+i]:>3} rmb={act[st+i,8]:.1f} atk={act[st+i,7]:.1f}{mark}")
        if pf[st + i]:                                                  # red border on that latent frame's pixel frames
            p0 = max(0, (i * Tpix) // W); p1 = max(p0 + 1, ((i + 1) * Tpix) // W)
            v[p0:p1, :6] = [255, 0, 0]; v[p0:p1, -6:] = [255, 0, 0]
            v[p0:p1, :, :6] = [255, 0, 0]; v[p0:p1, :, -6:] = [255, 0, 0]
    imageio.mimwrite(f"{OUT}/{s[:40]}_lat{st}.mp4", v, fps=16, quality=8)
print(f"\n完成 → {OUT}/")

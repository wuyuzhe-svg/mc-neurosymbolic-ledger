#!/usr/bin/env python3
"""Contact sheet of candidate cf windows: bright scenes, steady held item, place
events inside the window. Decode frame-0 of each candidate for visual pick."""
import os, sys, numpy as np, torch, imageio
sys.path.insert(0, "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2")
from wan.vae.wanx_vae import get_wanx_vae_wrapper

DATA = "/root/autodl-tmp/vpt_mc_352"
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
F = 15
dev, dt = "cuda", torch.float16

cands = []
for s in sorted(os.listdir(DATA)):
    if s.startswith(("bumpy", "cheeky")): continue
    npz = f"{DATA}/{s}/seg_000.npz"; latp = f"{DATA}/{s}/seg_000_lat44.npy"
    if not (os.path.exists(npz) and os.path.exists(latp)): continue
    z = np.load(npz)
    if "place_fire" not in z.files: continue
    pf = z["place_fire"] > 0; h = z["held_id"]; g = np.asarray(z.get("gui", np.zeros(len(h))))
    ev = np.flatnonzero(pf)
    for e in ev:
        st = e - 8                              # place lands ~mid-window like the cf protocol
        if st < 0 or st + F > len(h): continue
        w_h = h[st:st + F]
        if (w_h != w_h[0]).any() or w_h[0] <= 0: continue   # steady, known held
        if g[st:st + F].any(): continue
        cands.append((s, int(st), int(w_h[0]), int(pf[st:st + F].sum())))
        break                                   # one per seg
print(f"候选 {len(cands)} 段")

# brightness from latent mean of channel 0 is unreliable -> decode frame 0 of 24 random cands and rank by pixel brightness
cands = [c for c in cands if c[2] == 13 and c[3] >= 2]
print(f"held=13且窗内≥2次place: {len(cands)}")
rng = np.random.RandomState(7)
pickidx = rng.choice(len(cands), min(24, len(cands)), replace=False)
vae = get_wanx_vae_wrapper("/root/autodl-tmp/mg2_weights", torch.float16).to(dev, dt)
rows = []
for i in pickidx:
    s, st, hid, npl = cands[i]
    lat = np.load(f"{DATA}/{s}/seg_000_lat44.npy", mmap_mode="r")
    x = torch.from_numpy(np.asarray(lat[st:st + 1], np.float32)).to(dev, dt)
    with torch.no_grad():
        pix = vae.decode(x.permute(1, 0, 2, 3).unsqueeze(0), device=dev, **TK)[0].float().clamp(-1, 1)
    if pix.shape[0] == 3: pix = pix.permute(1, 0, 2, 3)
    img = ((pix[0].permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)
    rows.append((float(img.mean()), s, st, hid, npl, img))
rows.sort(key=lambda r: -r[0])                  # brightest first
sheet = []
for j, (br, s, st, hid, npl, img) in enumerate(rows[:12]):
    print(f"[{j}] bright={br:.0f} held={hid} place_in_win={npl} st={st} {s}")
    sheet.append(img)
grid = np.concatenate([np.concatenate(sheet[r * 4:(r + 1) * 4], 1) for r in range(3)], 0)
imageio.imwrite("/root/autodl-tmp/cf_stage1/cf_window_candidates.png", grid)
print("网格图: /root/autodl-tmp/cf_stage1/cf_window_candidates.png  (行优先编号0-11)")

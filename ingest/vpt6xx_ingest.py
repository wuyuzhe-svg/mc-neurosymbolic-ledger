#!/usr/bin/env python3
"""One-shot ingest for the 6xx download: mp4+jsonl (flat in vpt_6xx_raw) ->
/root/autodl-tmp/vpt_mc_352/<seg>/{seg_000_lat44.npy, seg_000.npz}.
- video: extract 16fps 480x832 -> resizecrop 352x640 -> Wan-VAE encode (SINGLE pass, no 60x104 detour)
- base labels: identical logic to vpt_to_npz (action/gui/hotbar/mine/pickup/craft/use fires)
- enriched labels: held_id (anchor-U-slot propagation, offhand-torch purge), bag_counts[.,40], place_fire
Usage: python vpt6xx_ingest.py [--shard K --n_shards N] [--limit M]
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, "/root/autodl-tmp/vrisingwm")
import preprocess_video_segments as pvs
import preprocess_vrising_final as pvf
from utils.wan_wrapper import WanVAEWrapper
import torch.nn.functional as Fun

RAW = "/root/autodl-tmp/vpt_6xx_raw"
OUT = "/root/autodl-tmp/vpt_mc_352"
VOCAB40 = json.load(open("/root/autodl-tmp/vrisingwm/data/held_vocab.json"))
VID = {n: i for i, n in enumerate(VOCAB40)}
PLACEABLE = (set(range(12, 26)) | set(range(30, 40))) - {16}
MINE_VOCAB = ("oak_log", "birch_log", "spruce_log", "dirt", "grass_block", "stone",
              "cobblestone", "coal_ore", "iron_ore", "sand", "gravel", "oak_planks",
              "stick", "oak_sapling", "andesite", "granite")
HOLD_KEYS = ("key.keyboard.w", "key.keyboard.a", "key.keyboard.s",
             "key.keyboard.d", "key.keyboard.space",
             "key.keyboard.left.shift", "key.keyboard.left.control")
dev, dt = "cuda", torch.bfloat16
PH, PW = 352, 640
CHUNK = 60

def stat_delta(prev, cur, pref):
    out = {}
    for k in set(prev) | set(cur):
        if not k.startswith(pref): continue
        d = cur.get(k, 0) - prev.get(k, 0)
        if d > 0: out[k.split(":minecraft.")[-1]] = d
    return out

def resizecrop_pix(pix):
    T, C, H, W = pix.shape
    if H / W > PH / PW:
        nh = int(W * PH / PW); top = (H - nh) // 2; pix = pix[:, :, top:top + nh, :]
    else:
        nw = int(H * PW / PH); left = (W - nw) // 2; pix = pix[:, :, :, left:left + nw]
    return Fun.interpolate(pix, size=(PH, PW), mode="bilinear", align_corners=False)

@torch.no_grad()
def encode_video(mp4, n_model_frames, cfg, vae):
    frames = pvs.extract_frames(mp4, 0.0, n_model_frames, cfg, orig_fps=20.0,
                                resize_device=os.environ.get("RESIZE_DEV", "cuda"))  # [n,480,832,3] u8
    outs = []
    px = CHUNK * 4  # pixel frames per latent chunk... encode in pixel chunks divisible by 4
    n = frames.shape[0]
    # encode whole clip in chunks; Wan-VAE causal temporal 4x: encode contiguous chunks of 240 px frames
    for i in range(0, n, px):
        f = frames[i:i + px]
        if f.shape[0] < 8: break
        f = f[: (f.shape[0] // 4) * 4]
        tpix = torch.from_numpy(f).to(dev).float().permute(0, 3, 1, 2) / 127.5 - 1.0   # [t,3,480,832]
        tpix = resizecrop_pix(tpix).to(dt)
        tpix = tpix.permute(1, 0, 2, 3).unsqueeze(0)                                    # [1,3,t,352,640]
        z = vae.encode_to_latent(tpix)[0]                                               # [t/4,16,44,80]
        outs.append(z.to(torch.float16).cpu().numpy())
    return np.concatenate(outs, 0) if outs else None

def resolve_held(ticks):
    """per-tick held item NAME via anchor(use_item stats, torch purged) U slot(inv[hotbar]);
    anchors propagate within (hotbar unchanged, no GUI) intervals; nearest anchor wins."""
    n = len(ticks)
    hot = np.array([t.get("hotbar", 0) if t.get("hotbar") is not None else 0 for t in ticks])
    gui = np.array([bool(t.get("isGuiOpen")) for t in ticks])
    slotname = []
    for t in ticks:
        h = t.get("hotbar", 0) or 0; inv = t.get("inventory", [])
        slotname.append(inv[h].get("type") if (0 <= h < len(inv) and isinstance(inv[h], dict)) else None)
    # interval id: increments when hotbar changes or GUI tick
    iv = np.zeros(n, np.int64); cur = 0
    for i in range(1, n):
        if hot[i] != hot[i - 1] or gui[i] or gui[i - 1]: cur += 1
        iv[i] = cur
    # anchors
    anchors = []                                            # (tick, name)
    prev = {}
    for i, t in enumerate(ticks):
        st = t.get("stats", {})
        if i > 0:
            for name, d in stat_delta(prev, st, "minecraft.use_item").items():
                if name != "torch": anchors.append((i, name))
        prev = st
    held = list(slotname)                                   # default: slot estimate
    if anchors:
        A = {}
        for i, name in anchors: A.setdefault(iv[i], []).append((i, name))
        for ivid, alist in A.items():
            idxs = np.where(iv == ivid)[0]
            for j in idxs:
                # nearest anchor in this interval wins
                near = min(alist, key=lambda a: abs(a[0] - j))
                held[j] = near[1]
    return held, gui

def convert(name, cfg, vae):
    od = f"{OUT}/{name}"
    if os.path.exists(f"{od}/seg_000_lat44.npy") and os.path.exists(f"{od}/seg_000.npz"):
        return "skip"
    jsonl, mp4 = f"{RAW}/{name}.jsonl", f"{RAW}/{name}.mp4"
    if not (os.path.exists(jsonl) and os.path.exists(mp4)): return "missing"
    ticks = []
    for l in open(jsonl, encoding="utf-8", errors="ignore"):
        try: ticks.append(json.loads(l))
        except Exception: pass
    n_ticks = len(ticks)
    if n_ticks < 300: return "short"
    dur = n_ticks / 20.0
    lat = encode_video(mp4, int(dur * cfg.fps), cfg, vae)
    if lat is None or lat.shape[0] < 15: return "novideo"
    L = lat.shape[0]
    tick_to_lf = lambda i: min(L - 1, pvf.pixel_to_latent(int(i / 20.0 * cfg.fps), cfg))
    # ---- base labels (identical to vpt_to_npz) ----
    mine = np.zeros(L, np.int64); pick = np.zeros(L, np.int64); craft = np.zeros(L, np.int64)
    use = np.zeros(L, np.int64); gui_l = np.zeros(L, np.int64); act = np.zeros((L, 11), np.float32)
    hot_l = np.zeros(L, np.int64); mine_t = np.zeros((L, 16), np.int64); pick_t = np.zeros((L, 16), np.int64)
    vidx = {v: i for i, v in enumerate(MINE_VOCAB)}
    counts = np.zeros(L, np.int64)
    # ---- enriched ----
    heldname, gui_ticks = resolve_held(ticks)
    held_l = np.full(L, -2, np.int64)
    bag_l = np.zeros((L, 40), np.float32)
    place_l = np.zeros(L, np.int64)
    prev = {}
    for i, t in enumerate(ticks):
        lf = tick_to_lf(i)
        counts[lf] += 1
        if t.get("isGuiOpen"): gui_l[lf] = 1
        keys = set(t.get("keyboard", {}).get("keys", []))
        for j, k in enumerate(HOLD_KEYS): act[lf, j] += k in keys
        mb = t.get("mouse", {})
        act[lf, 7] += 0 in mb.get("buttons", [])
        act[lf, 8] += 1 in mb.get("buttons", [])
        act[lf, 9] += mb.get("dx", 0.0); act[lf, 10] += mb.get("dy", 0.0)
        st = t.get("stats", {})
        if i > 0:
            for nm, d in stat_delta(prev, st, "minecraft.mine_block").items():
                mine[lf] += d
                if nm in vidx: mine_t[lf, vidx[nm]] += d
            for nm, d in stat_delta(prev, st, "minecraft.pickup").items():
                pick[lf] += d
                if nm in vidx: pick_t[lf, vidx[nm]] += d
            for nm, d in stat_delta(prev, st, "minecraft.use_item").items():
                use[lf] += d
                if VID.get(nm, -1) in PLACEABLE: place_l[lf] += d
            craft[lf] += sum(stat_delta(prev, st, "minecraft.craft_item").values())
        prev = st
    nz = np.maximum(counts, 1)
    act[:, :9] /= nz[:, None]; act[:, 9:] /= nz[:, None]
    for lf in range(L):
        ft = min(int(lf * cfg.temporal_compress / cfg.fps * 20), n_ticks - 1)
        hot_l[lf] = ticks[ft].get("hotbar", 0) or 0
        nm = heldname[ft]
        held_l[lf] = 0 if nm is None else VID.get(nm, -1)
        for it in ticks[ft].get("inventory", []):
            vi = VID.get(it.get("type"), -1)
            if vi >= 0: bag_l[lf, vi] += it.get("quantity", 0)
    np.save(f"{od}_tmp_lat.npy", lat) if False else None
    os.makedirs(od, exist_ok=True)
    np.save(f"{od}/seg_000_lat44.npy", lat)
    np.savez(f"{od}/seg_000.npz",
             n_latent=np.array([L]), mine_fire=mine, pickup_fire=pick, craft_fire=craft,
             use_fire=use, gui=gui_l, action=act, hotbar=hot_l,
             mine_types=mine_t, pickup_types=pick_t, vocab=np.array(MINE_VOCAB),
             held_id=held_l, bag_counts=bag_l, place_fire=place_l)
    return f"L={L} place={place_l.sum()} held-1={float((held_l==-1).mean()):.0%}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--n_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = pvf.Config()
    vae = WanVAEWrapper().to(dev, dt)
    names = sorted(x[:-4] for x in os.listdir(RAW) if x.endswith(".mp4"))
    names = names[a.shard::a.n_shards]
    if a.limit: names = names[:a.limit]
    print(f"ingest shard {a.shard}/{a.n_shards}: {len(names)} segs", flush=True)
    for k, nm in enumerate(names):
        try:
            r = convert(nm, cfg, vae)
        except Exception as e:
            r = f"ERR {type(e).__name__}: {str(e)[:80]}"
        print(f"[{k+1}/{len(names)}] {nm[-28:]}: {r}", flush=True)
    print("INGEST-DONE", flush=True)

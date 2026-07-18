#!/usr/bin/env python3
"""Stage-1 combined CF: 5 generations, same seed/window/cond:
  1 place-ON   (place once @lat8, held=GT)     -> block should appear
  2 place-OFF  (no place, held=GT)             -> no block
  3 held=A GT  (= run1, reused)                -> item A in hand/placed
  4 held=B     (place ON, held=alt item)       -> item B in hand/placed
  5 held=none  (place ON but held=0 empty)     -> empty hand render check
Usage: python cf_stage1.py <ckpt> <seg> <st> <altheld>
"""
import os, sys, numpy as np, torch, imageio
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2")
from utils.scheduler import FlowMatchScheduler
from mg2_model import load_expanded_base, PLACE_IDX
from wan.vae.wanx_vae import get_wanx_vae_wrapper
dev, dt = "cuda", torch.bfloat16
OUT = "/root/autodl-tmp/cf_stage1"; os.makedirs(OUT, exist_ok=True)
ckpt, seg, st, ALT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
DATA = "/root/autodl-tmp/vpt_mc_352"; F = 15; NUM_PIX = (F - 1) * 4 + 1
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
PLACE_LAT = int(os.environ.get("PLACE_LAT", "8"))

lat = np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r")
z = np.load(f"{DATA}/{seg}/seg_000.npz")
latw = torch.from_numpy(np.asarray(lat[st:st + F], np.float32)).to(dev, dt)
held_gt = int(z["held_id"][st]); print(f"held_gt={held_gt} alt={ALT}", flush=True)

model = load_expanded_base(dev, dt, ledger=True)
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
model.eval().requires_grad_(False)
vae = get_wanx_vae_wrapper("/root/autodl-tmp/mg2_weights", torch.float16).to(dev, dt)

with torch.no_grad():
    x0 = latw.permute(1, 0, 2, 3).unsqueeze(0)
    fp = vae.decode(x0[:, :, 0:1], device=dev, **TK).float().clamp(-1, 1).to(dev)
    fp0 = fp[:, :, 0:1] if fp.shape[1] == 3 else fp.permute(0, 2, 1, 3, 4)[:, :, 0:1]
    pad = torch.zeros(1, 3, NUM_PIX - 1, fp0.shape[-2], fp0.shape[-1], device=dev, dtype=dt)
    img_cond = vae.encode(torch.cat([fp0.to(dt), pad], dim=2), device=dev, **TK).to(dev, dt)
    if img_cond.shape[1] != 16: img_cond = img_cond.permute(0, 2, 1, 3, 4)
    mask = torch.zeros(1, 4, F, 44, 80, device=dev, dtype=dt); mask[:, :, 0] = 1.0
    y = torch.cat([mask, img_cond], dim=1)
    vc = vae.clip.encode_video(fp0.to(dt)).to(dt)

ms = torch.zeros(1, NUM_PIX, 2, device=dev, dtype=dt)
sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True); sched.set_timesteps(50, training=False)
tsteps = sched.timesteps.to(dev); sigmas = sched.sigmas.to(dev).float()

SWITCH_LAT = 4                                          # mid-window switch frame (ledger-style)

@torch.no_grad()
def gen(place_on, held_id, count=5.0, switch_from=None):
    """switch_from: if set, frames 0..SWITCH_LAT-1 use switch_from (consistent with the
    first frame's visual), then switch to held_id — mimics the ledger flipping held
    mid-window. Avoids frame-0 conflict with the i2v first frame."""
    kb = torch.zeros(1, NUM_PIX, 7, device=dev, dtype=dt)
    if place_on:
        p = PLACE_LAT * 4; kb[:, p:p + 4, PLACE_IDX] = 1.0
    held = torch.full((1, F), held_id, device=dev, dtype=torch.long)
    cnt = torch.full((1, F), count, device=dev, dtype=dt)
    if switch_from is not None:
        held[:, :SWITCH_LAT] = switch_from
        cnt[:, :SWITCH_LAT] = 5.0
    torch.manual_seed(42); x = torch.randn(1, 16, F, 44, 80, device=dev, dtype=dt)
    for i, t in enumerate(tsteps):
        fl = model(x, t=t.reshape(1).to(dt), visual_context=vc, cond_concat=y,
                   keyboard_cond=kb, mouse_cond=ms_use, ledger_held=held, ledger_count=cnt)
        if isinstance(fl, (list, tuple)): fl = fl[0]
        sg = sigmas[i]; ns = sigmas[i + 1] if i + 1 < len(sigmas) else sigmas.new_zeros(())
        x = (x.float() + (ns - sg) * fl.float()).to(dt)
    pix = vae.decode(x, device=dev, **TK)[0].float().clamp(-1, 1)
    if pix.shape[0] == 3: pix = pix.permute(1, 0, 2, 3)
    return ((pix + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()

# frozen probe readout on generated latents: the EXACT quantity L_probe optimizes.
# rises before pixels become eye-visible; flat across ckpts = probe gradient not landing.
class MCProbe(torch.nn.Module):
    def __init__(self, in_ch, dim=256):
        super().__init__()
        nn = torch.nn
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
_probe = MCProbe(32).to(dev).eval().requires_grad_(False)
_pk = torch.load("checkpoints/mc_probe44/best.pt", map_location="cpu"); _probe.load_state_dict(_pk["model"])
HUD_R, HUD_C = slice(38, 44), slice(20, 60)
@torch.no_grad()
def probe_place(x_lat):                                # [1,16,F,44,80] -> [F] place prob
    x = x_lat.permute(0, 2, 1, 3, 4).float().clone(); x[:, :, :, HUD_R, HUD_C] = 0
    dxx = torch.diff(x, dim=1, prepend=x[:, :1])
    _, _, su = _probe(torch.cat([x, dxx], dim=2))
    return torch.sigmoid(su[0]).cpu().numpy()

LATS = {}
_orig_gen = gen
ATTACK_IDX = 5
REAL_ACT = os.environ.get("REAL_ACT", "0") == "1"
CAM_SCALE = 1.0 / 15.0; CAM_DEG_CLIP = 20.0
def real_kbms(place_zero=False, cut_after=None):
    """回放窗口真实动作轨(WASD/跳/attack/相机), 与train_mg2.map_actions逐字节一致。
    cut_after=k: k帧之后所有动作置零(切除行为回声, 隔离首次放置)"""
    act = z["action"][st:st + F].astype(np.float32)
    kb_ = np.zeros((F, 7), np.float32)
    kb_[:, 0] = act[:, 0]; kb_[:, 1] = act[:, 2]; kb_[:, 2] = act[:, 1]; kb_[:, 3] = act[:, 3]
    kb_[:, 4] = act[:, 4]; kb_[:, ATTACK_IDX] = act[:, 7]
    kb_[:, PLACE_IDX] = 0.0 if place_zero else (act[:, 8] > 0).astype(np.float32)
    ms_ = np.zeros((F, 2), np.float32)
    ms_[:, 0] = -np.clip(act[:, 10], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
    ms_[:, 1] = np.clip(act[:, 9], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
    if cut_after is not None:
        kb_[cut_after + 1:] = 0.0; ms_[cut_after + 1:] = 0.0
    kb_ = np.repeat(kb_, 4, axis=0)[:NUM_PIX]; ms_ = np.repeat(ms_, 4, axis=0)[:NUM_PIX]
    return (torch.from_numpy(kb_)[None].to(dev, dt), torch.from_numpy(ms_)[None].to(dev, dt))

def gen(place_on, held_id, count=5.0, switch_from=None, attack_on=False, real_act=False, cut_after=None, _tag=[0]):
    # wrap: also stash final latent for probe readout
    if real_act:
        kb, ms_use = real_kbms(place_zero=not place_on, cut_after=cut_after)
    else:
        kb = torch.zeros(1, NUM_PIX, 7, device=dev, dtype=dt); ms_use = ms
        if place_on:
            p = PLACE_LAT * 4; kb[:, p:p + 4, PLACE_IDX] = 1.0
        if attack_on:
            p = PLACE_LAT * 4; kb[:, p:p + 4, ATTACK_IDX] = 1.0
    held = torch.full((1, F), held_id, device=dev, dtype=torch.long)
    cnt = torch.full((1, F), count, device=dev, dtype=dt)
    if switch_from is not None:
        held[:, :SWITCH_LAT] = switch_from
        cnt[:, :SWITCH_LAT] = 5.0
    torch.manual_seed(42); x = torch.randn(1, 16, F, 44, 80, device=dev, dtype=dt)
    with torch.no_grad():
        for i, t in enumerate(tsteps):
            fl = model(x, t=t.reshape(1).to(dt), visual_context=vc, cond_concat=y,
                       keyboard_cond=kb, mouse_cond=ms_use, ledger_held=held, ledger_count=cnt)
            if isinstance(fl, (list, tuple)): fl = fl[0]
            sg = sigmas[i]; ns = sigmas[i + 1] if i + 1 < len(sigmas) else sigmas.new_zeros(())
            x = (x.float() + (ns - sg) * fl.float()).to(dt)
        LATS[len(LATS)] = probe_place(x)
        pix = vae.decode(x, device=dev, **TK)[0].float().clamp(-1, 1)
    if pix.shape[0] == 3: pix = pix.permute(1, 0, 2, 3)
    return ((pix + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()

SKIP_SYNTH = os.environ.get("SKIP_SYNTH", "0") == "1"   # 只跑realACT两行(需REAL_ACT=1)
runs = {}
SYNTH_MIN = os.environ.get("SYNTH_MIN", "0") == "1"
if SYNTH_MIN:
    runs["1_placeON_heldGT"] = gen(True,  held_gt)
    runs["2_placeOFF_heldGT"] = gen(False, held_gt)
    pon_, poff_ = LATS[0], LATS[1]
    print(f"SYNTH_MIN place@lat{PLACE_LAT}: 探针ON全线={[round(float(x),2) for x in pon_]}", flush=True)
    print(f"                OFF全线={[round(float(x),2) for x in poff_]}", flush=True)
if not SKIP_SYNTH:
    runs["1_placeON_heldGT"] = gen(True,  held_gt)
    runs["2_placeOFF_heldGT"] = gen(False, held_gt)
    runs["4_placeON_heldALT"] = gen(True,  ALT, switch_from=held_gt)     # GT->ALT switch @lat4, place @lat8
    runs["5_placeON_heldNONE"] = gen(True, 0, count=0.0, switch_from=held_gt)  # GT->none (ledger depletion style)
    runs["6_attackON_heldGT"] = gen(False, held_gt, attack_on=True)  # attack same frames: semantics-contamination check
if REAL_ACT:                                       # 分布内反事实: 回放真实动作轨(WASD/相机/attack全带)
    runs["7_realON"] = gen(True, held_gt, real_act=True)      # 原样, 含真实place
    runs["8_realOFF"] = gen(False, held_gt, real_act=True)    # 同轨, 仅place列清零
FIRST_ONLY = os.environ.get("FIRST_ONLY", "0") == "1"
if FIRST_ONLY:                                     # 隔离首次放置: 首发后动作全零(切除行为回声)
    f1 = int(np.flatnonzero(z["place_fire"][st:st + F] > 0)[0])
    print(f"FIRST_ONLY: 首次place@lat{f1}, 之后动作全零", flush=True)
    runs["9_firstON"] = gen(True, held_gt, real_act=True, cut_after=f1)   # 只留首发
    runs["10_firstOFF"] = gen(False, held_gt, real_act=True, cut_after=f1)  # 首发也不给
for k, v in runs.items():
    imageio.mimwrite(f"{OUT}/{k}.mp4", v, fps=16, quality=8)

idx = [0, 24, 32, 40, 48, 56]
row = lambda a: np.concatenate([a[i] for i in idx], 1)
rows_m = []; tag = ""
if not SKIP_SYNTH:
    on, off = runs["1_placeON_heldGT"], runs["2_placeOFF_heldGT"]
    alt, none = runs["4_placeON_heldALT"], runs["5_placeON_heldNONE"]
    d_onoff = [round(float(np.abs(on[i].astype(float) - off[i].astype(float)).mean()), 2) for i in [16, 28, 36, 48, 56]]
    d_alt = [round(float(np.abs(on[i].astype(float) - alt[i].astype(float)).mean()), 2) for i in [16, 36, 56]]
    d_none = [round(float(np.abs(on[i].astype(float) - none[i].astype(float)).mean()), 2) for i in [16, 36, 56]]
    print(f"place ON-OFF diff @[16,28,36,48,56]: {d_onoff}  (place@32; 后段>前段=因果分化)", flush=True)
    atk = runs["6_attackON_heldGT"]
    d_atk_on = [round(float(np.abs(on[i].astype(float) - atk[i].astype(float)).mean()), 2) for i in [16, 36, 56]]
    d_atk_off = [round(float(np.abs(atk[i].astype(float) - off[i].astype(float)).mean()), 2) for i in [16, 36, 56]]
    pon, poff, patk = LATS[0], LATS[1], LATS[4]
    print(f"探针读生成: ON place分@lat[7..10]={[round(float(x),3) for x in pon[7:11]]} "
          f"OFF同帧={[round(float(x),3) for x in poff[7:11]]}  (ON高OFF低=Lpon在起效)", flush=True)
    print(f"attackON 探针place分@lat[7..10]={[round(float(x),3) for x in patk[7:11]]}  (应≈0; 高=attack语义污染)", flush=True)
    print(f"attack-vs-placeON diff @[16,36,56]: {d_atk_on}  attack-vs-OFF diff: {d_atk_off}  "
          f"(atk-OFF大=attack有响应; atk-ON大=两动作分得开)", flush=True)
    print(f"held GT-ALT diff @[16,36,56]: {d_alt}   held GT-NONE diff: {d_none}", flush=True)
    rows_m = [row(on), row(off), row(alt), row(none), row(atk)]
    tag = "1=ON 2=OFF 3=heldALT 4=heldNONE 5=attackON"
if REAL_ACT:
    ron, roff = runs["7_realON"], runs["8_realOFF"]
    pfr = np.flatnonzero(z["place_fire"][st:st + F] > 0)
    d_real = [round(float(np.abs(ron[i].astype(float) - roff[i].astype(float)).mean()), 2) for i in [16, 36, 56]]
    pr_on, pr_off = LATS[len(LATS) - 2], LATS[len(LATS) - 1]
    print(f"realACT: 窗内真实place@lat{list(pfr)}; 探针ON@这些帧={[round(float(pr_on[i]),3) for i in pfr]} "
          f"OFF同帧={[round(float(pr_off[i]),3) for i in pfr]}  realON-OFF diff@[16,36,56]={d_real}", flush=True)
    rows_m += [row(ron), row(roff)]; tag += " 6=realON 7=realOFF"
if FIRST_ONLY:
    fon, foff = runs["9_firstON"], runs["10_firstOFF"]
    d_f = [round(float(np.abs(fon[i].astype(float) - foff[i].astype(float)).mean()), 2) for i in [16, 36, 56]]
    p_on, p_off = LATS[len(LATS) - 2], LATS[len(LATS) - 1]
    print(f"FIRST_ONLY 探针全时间线:\n  ON ={[round(float(x),2) for x in p_on]}\n  OFF={[round(float(x),2) for x in p_off]}", flush=True)
    print(f"  firstON-OFF diff@[16,36,56]={d_f}  (f1后若ON块持久+OFF全零=干净单发因果)", flush=True)
    rows_m += [row(fon), row(foff)]; tag += " +firstON +firstOFF"
mont = np.concatenate(rows_m, 0)
imageio.imwrite(f"{OUT}/cf_stage1_montage.png", mont)
print(f"montage rows: {tag}  帧{idx} (place@像素32)", flush=True)
print("CF-STAGE1-DONE", flush=True)

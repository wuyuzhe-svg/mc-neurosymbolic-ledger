#!/usr/bin/env python3
"""Fine-tune Matrix-Game 2.0 base_model to TEACH place (attack already in base @ kb5).
FULL fine-tune (a05 + Solaris + ForgeWM all full-FT). Conventions aligned to ForgeWM/MG2:
- keyboard 6->7: [W S A D jump ATTACK(base) | PLACE(new)]; drop sneak.
- mouse [pitch(-, up+), yaw(+, right+)] scaled from VPT degrees.
- i2v cond built the MG2 way: decode 1st latent frame -> [pixel, ZEROS(gray) x] -> encode
  -> [mask(4)|img(16)]=20ch; real CLIP visual_context (NOT zeros).  (black/-1 padding collapses!)
Anti-forgetting: frozen-base L_preserve on non-place frames.
"""
import os, sys, glob, random, numpy as np, torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2")
from utils.scheduler import FlowMatchScheduler
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from mg2_model import load_expanded_base, KB_NEW, ATTACK_IDX, PLACE_IDX

# DDP-aware: launched via `torchrun --nproc_per_node=N` -> multi-GPU; via `python` -> single.
WORLD = int(os.environ.get("WORLD_SIZE", 1)); RANK = int(os.environ.get("RANK", 0))
LOCAL = int(os.environ.get("LOCAL_RANK", 0)); DDP_ON = WORLD > 1
if DDP_ON:
    dist.init_process_group("nccl"); torch.cuda.set_device(LOCAL)
IS0 = (RANK == 0)
dev = f"cuda:{LOCAL}" if DDP_ON else "cuda"
dt = torch.bfloat16
DATA = "/root/autodl-tmp/vpt_mc_352"
OUT = os.environ.get("OUT", "checkpoints/mg2_ft"); os.makedirs(OUT, exist_ok=True)
F = 15; NUM_PIX = (F - 1) * 4 + 1
CAM_SCALE = 1.0 / 15.0; CAM_DEG_CLIP = 20.0      # Solaris-exact: clip deg to ±20 then ×1/15 (MG2: 1deg=1/15)
STEPS = int(os.environ.get("STEPS", 4000)); BS = int(os.environ.get("BS", 2))
LEDGER = int(os.environ.get("LEDGER", 0))         # 1: attach (held_id,count) injector (zero-init)
LPROBE = int(os.environ.get("LPROBE", 0))         # 1: L_probe (teach place via frozen probe + action-contrast)
LR = 5e-5; W_PRESERVE = float(os.environ.get("W_PRESERVE", "0.25"))   # 0 = drop base-preserve (recon+ctr suffice)
POS_FRAC = float(os.environ.get("POS_FRAC", "0.20"))   # oversample successful-place windows
NEG_FRAC = float(os.environ.get("NEG_FRAC", "0.20"))   # oversample RMB-but-no-place (sky/invalid) hard negatives
T_PROBE = float(os.environ.get("T_PROBE", "800"))   # 探针裁判域上限(t扫描: AUC@800=0.984, @900=0.955)
LAM_PROBE = 0.1; LAM_CTR_PR = 0.5  # L_probe gate/weights
T_HI_FRAC = float(os.environ.get("T_HI_FRAC", "0.5"))  # event样本偏高噪: 这比例的事件样本t压入[T_HI_LO,T_HI_HI]
T_HI_LO = float(os.environ.get("T_HI_LO", "500"))       # v3配方: 高噪逼迫, teacher零信号区只能从动作token取证
T_HI_HI = float(os.environ.get("T_HI_HI", "1000"))      # 上不封顶; t>T_PROBE时tg门自动关探针项, 聚光recon独扛
LAM_OUT = float(os.environ.get("LAM_OUT", "0.5"))       # ON-OFF差异的空间局部化: 允许区(准星框∪右下手臂区)之外抑制
W_POFF = float(os.environ.get("W_POFF", "0"))           # OFF绝对压制开关, 默认关(会学出涂抹作弊解, 见下方注释)
MARGIN_D = float(os.environ.get("MARGIN_D", "2.0")); LAM_MARGIN = float(os.environ.get("LAM_MARGIN", "0.25"))  # (v2, unused in v3)
V3 = int(os.environ.get("V3", "0"))                # v3: crosshair-weighted recon + counterfactual change-budget
WREC_K = float(os.environ.get("WREC_K", "30"))     # crosshair recon weight on/after place frames
W_CF = float(os.environ.get("W_CF", "1.0"))        # change-budget weight
CR_R = slice(14, 34); CR_C = slice(26, 54)          # crosshair region (latent 44x80 center)
HUD_R = slice(38, 44); HUD_C = slice(20, 60)      # probe HUD mask (44x80)
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
rng = random.Random(RANK); torch.manual_seed(RANK)                 # each rank samples different windows

segs = [d for d in sorted(os.listdir(DATA)) if os.path.exists(f"{DATA}/{d}/seg_000_lat44.npy")]
if IS0: print(f"segments={len(segs)} F={F} full-FT LR={LR} kb={KB_NEW}(attack@{ATTACK_IDX} place@{PLACE_IDX}) "
               f"WORLD={WORLD} BS={BS} eff_batch={BS*WORLD} LPROBE={LPROBE}", flush=True)
PLACEABLE = (set(list(range(12, 26)) + list(range(30, 40))) - {16})   # main-hand SOLID placeable blocks; excl 16=torch (global-light, diff visual)
place_windows = []                                 # (seg, st) on a MAIN-HAND placeable-block place (excl offhand torch)
neg_windows = []                                    # (seg, st) hard negatives: RMB pressed but NO placement (sky/invalid target)
if LPROBE or V3:
    n_all = n_torch = 0
    for s in segs:
        z = np.load(f"{DATA}/{s}/seg_000.npz")
        if "place_fire" not in z.files or "held_id" not in z.files: continue
        N = np.load(f"{DATA}/{s}/seg_000_lat44.npy", mmap_mode="r").shape[0]
        g = np.asarray(z.get("gui", np.zeros(N))); h = z["held_id"]; bag = z["bag_counts"]
        pf = z["place_fire"][:N] > 0; rmb = z["action"][:N, 8] > 0
        for t in np.where(pf)[0]:
            n_all += 1
            torch_place = t > 0 and bag[t, 16] < bag[t - 1, 16]      # torch count dropped -> torch placed (main/offhand)
            if int(h[t]) not in PLACEABLE and not torch_place: n_torch += 1; continue   # not a block/torch place -> skip (use-item etc)
            st = int(max(0, min(t - F // 2, N - F)))
            if N > F and not g[st:st + F].any(): place_windows.append((s, st))
        for t in np.where(rmb & ~pf)[0]:                             # hard neg: RMB down, no block placed (sky/invalid)
            st = int(max(0, min(t - F // 2, N - F)))
            if N > F and not g[st:st + F].any(): neg_windows.append((s, st))
    if IS0: print(f"place windows (block+torch): {len(place_windows)}  neg windows (RMB no-place): {len(neg_windows)}  (excluded {n_torch}/{n_all} use-items)", flush=True)

def map_actions(act):    # ours[11]=[w a s d space shift ctrl LMB RMB cam_x cam_y] -> kb[.,7], ms[.,2], place_ev[F]
    kb = np.zeros((act.shape[0], KB_NEW), np.float32)
    kb[:, 0] = act[:, 0]; kb[:, 1] = act[:, 2]; kb[:, 2] = act[:, 1]; kb[:, 3] = act[:, 3]   # W S A D
    kb[:, 4] = act[:, 4]                                                                     # jump (space)
    kb[:, ATTACK_IDX] = act[:, 7]                                                            # attack (LMB) -> base dim5
    kb[:, PLACE_IDX] = (act[:, 8] > 0).astype(np.float32)   # place BINARIZED: a tap is a tap (fractions were 0.2-0.8 = quantization noise; ledger/cf always send 1.0)
    ms = np.zeros((act.shape[0], 2), np.float32)
    ms[:, 0] = -np.clip(act[:, 10], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE   # pitch (flip: VPT +down -> MG2 +up)
    ms[:, 1] = np.clip(act[:, 9], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE      # yaw (+ = right, unchanged)
    place_ev = (act[:, 8] > 0).astype(np.float32)
    kb = np.repeat(kb, 4, axis=0)[:NUM_PIX]; ms = np.repeat(ms, 4, axis=0)[:NUM_PIX]
    return kb, ms, place_ev

def sample_window():
    for _ in range(20):
        r = rng.random()
        if (LPROBE or V3) and place_windows and r < POS_FRAC:                         # oversample successful place
            seg, st = place_windows[rng.randrange(len(place_windows))]
        elif (LPROBE or V3) and neg_windows and r < POS_FRAC + NEG_FRAC:              # oversample RMB-no-place hard negatives
            seg, st = neg_windows[rng.randrange(len(neg_windows))]
        else:
            seg = segs[rng.randrange(len(segs))]
            N = np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r").shape[0]
            if N <= F: continue
            st = rng.randrange(0, N - F)
        z = np.load(f"{DATA}/{seg}/seg_000.npz")
        if "held_id" not in z.files or "bag_counts" not in z.files: continue   # 2 old-version segs lack these
        if "gui" in z.files and np.asarray(z["gui"])[st:st + F].any(): continue  # skip GUI/menu windows (OOD)
        latw = np.asarray(np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r")[st:st + F], np.float32)
        kb, ms, ev = map_actions(z["action"][st:st + F].astype(np.float32))
        held = z["held_id"][st:st + F].astype(np.int64)                       # -1..39
        bag = z["bag_counts"][st:st + F].astype(np.float32)                   # [F,40]
        hcount = np.where(held >= 0, bag[np.arange(F), held.clip(0, 39)], 0.0).astype(np.float32)
        pf = (z["place_fire"][st:st + F] > 0).astype(np.float32) if "place_fire" in z.files else np.zeros(F, np.float32)
        return latw, kb, ms, ev, held, hcount, pf
    raise RuntimeError("no window")

def build_batch(bs):
    L, K, M, E, H, C, PF = [], [], [], [], [], [], []
    for _ in range(bs):
        latw, kb, ms, ev, held, hc, pf = sample_window()
        L.append(latw); K.append(kb); M.append(ms); E.append(ev); H.append(held); C.append(hc); PF.append(pf)
    latw = torch.from_numpy(np.stack(L)).to(dev, dt)                    # [B,F,16,44,80]
    x0 = latw.permute(0, 2, 1, 3, 4).contiguous()                      # [B,16,F,44,80]
    kb = torch.from_numpy(np.stack(K)).to(dev, dt); ms = torch.from_numpy(np.stack(M)).to(dev, dt)
    event = torch.from_numpy(np.stack(E)).to(dev, dt)                   # [B,F] 1=place frame (from action dim8)
    held = torch.from_numpy(np.stack(H)).to(dev); hcount = torch.from_numpy(np.stack(C)).to(dev, dt)  # [B,F]
    placef = torch.from_numpy(np.stack(PF)).to(dev)                     # [B,F] 1=place_fire GT; KEEP float32 (bf16 target downcasts BCE to bf16 -> negative-loss artifacts)
    return x0, latw, kb, ms, event, held, hcount, placef

@torch.no_grad()
def build_cond(x0):                                                    # MG2-way i2v cond + real clip
    B = x0.shape[0]
    first_pix = vae.decode(x0[:, :, 0:1], device=dev, **TK)            # decode 1st latent frame -> [B,3,1,H,W]
    fp = first_pix.float().clamp(-1, 1).to(dev)
    if fp.shape[1] == 3: fp0 = fp[:, :, 0:1]                            # [B,3,1,Hp,Wp]
    else: fp0 = fp.permute(0, 2, 1, 3, 4)[:, :, 0:1]
    Hp, Wp = fp0.shape[-2], fp0.shape[-1]
    pad = torch.zeros(B, 3, NUM_PIX - 1, Hp, Wp, device=dev, dtype=dt)  # ZEROS = gray (NOT -1 black)
    padded = torch.cat([fp0.to(dt), pad], dim=2)                       # [B,3,NUM_PIX,Hp,Wp]
    img_cond = vae.encode(padded, device=dev, **TK).to(dev, dt)        # [B,16,F,44,80]
    if img_cond.shape[1] != 16: img_cond = img_cond.permute(0, 2, 1, 3, 4)
    mask = torch.zeros(B, 4, F, 44, 80, device=dev, dtype=dt); mask[:, :, 0] = 1.0
    cond_concat = torch.cat([mask, img_cond], dim=1)                   # [B,20,F,44,80]
    vc = vae.clip.encode_video(fp0.to(dt)).to(dt)                      # real visual_context
    return cond_concat, vc

# --- models + vae ---
model = load_expanded_base(dev, dt, ledger=bool(LEDGER)); model.requires_grad_(True); model.train()
if hasattr(model, "enable_gradient_checkpointing"): model.enable_gradient_checkpointing()
ref = None
if W_PRESERVE > 0:                                    # only need frozen base if using base-preserve
    ref = load_expanded_base(dev, dt); ref.requires_grad_(False); ref.eval()
vae = get_wanx_vae_wrapper("/root/autodl-tmp/mg2_weights", torch.float16).to(dev, dt)
START = 0
RESUME = os.environ.get("RESUME", "")
if RESUME and os.path.exists(RESUME):
    rck = torch.load(RESUME, map_location="cpu", weights_only=False)
    model.load_state_dict(rck["model"], strict=False); START = int(rck.get("step", 0))  # strict=False: ledger_injector stays zero-init on non-ledger ckpt
    if IS0: print(f"RESUMED from {RESUME} @ step {START}", flush=True)
if int(os.environ.get("WARM_PLACE", "0")):
    with torch.no_grad():
        nw = 0
        for n_, p_ in model.named_parameters():
            if n_.endswith("keyboard_embed.0.weight") and p_.shape[1] == KB_NEW and p_[:, PLACE_IDX].abs().mean() < 0.01:
                p_[:, PLACE_IDX].copy_(p_[:, ATTACK_IDX]); nw += 1
    if IS0: print(f"[warm] place col <- attack col in {nw} blocks", flush=True)
if DDP_ON:
    model = DDP(model, device_ids=[LOCAL], find_unused_parameters=True)
core = model.module if DDP_ON else model
if IS0: print(f"trainable={sum(p.numel() for p in core.parameters() if p.requires_grad)/1e6:.0f}M (full) start={START}", flush=True)

# --- frozen event probe for L_probe (reads pred_x0 LATENT directly, no VAE decode) ---
if LPROBE:
    class MCProbe(torch.nn.Module):
        def __init__(self, in_ch, dim=256):
            super().__init__()
            self.enc = torch.nn.Sequential(torch.nn.Conv2d(in_ch, 64, 3, 2, 1), torch.nn.GELU(),
                torch.nn.Conv2d(64, 128, 3, 2, 1), torch.nn.GELU(),
                torch.nn.Conv2d(128, dim, 3, 2, 1), torch.nn.GELU(), torch.nn.AdaptiveAvgPool2d(1))
            layer = torch.nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True, norm_first=True)
            self.tf = torch.nn.TransformerEncoder(layer, 2); self.pos = torch.nn.Parameter(torch.zeros(1, F, dim))
            self.head_mine = torch.nn.Linear(dim, 1); self.head_pick = torch.nn.Linear(dim, 1); self.head_use = torch.nn.Linear(dim, 1)
        def forward(self, x):
            B, Fr = x.shape[:2]; h = self.enc(x.flatten(0, 1)).reshape(B, Fr, -1) + self.pos; h = self.tf(h)
            return self.head_use(h)[..., 0]                          # place head
    probe = MCProbe(32).to(dev).eval().requires_grad_(False)
    pk = torch.load("checkpoints/mc_probe44/best.pt", map_location="cpu"); probe.load_state_dict(pk["model"])
    if IS0: print(f"[L_probe] probe loaded (place AUC {pk['auc']['place']:.3f}) lam={LAM_PROBE} t<={T_PROBE} t_hi={T_HI_FRAC}@[{T_HI_LO},{T_HI_HI}] lam_out={LAM_OUT}", flush=True)
    # ON-OFF差异允许区: 放大版准星框(24×36≈24.5%) ∪ 右下手臂/手持区; 区外差异被L_out抑制
    ALLOW = torch.zeros(1, 1, 1, 44, 80, device=dev)
    ALLOW[..., 12:36, 22:58] = 1.0
    ALLOW[..., 26:44, 52:80] = 1.0
    def place_logit(px0):                                           # px0 [B,16,F,44,80] -> [B,F]
        x = px0.permute(0, 2, 1, 3, 4).float().clone(); x[:, :, :, HUD_R, HUD_C] = 0
        d = torch.diff(x, dim=1, prepend=x[:, :1])
        return probe(torch.cat([x, d], dim=2))
    bcelogit = torch.nn.BCEWithLogitsLoss()

sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
sched.set_timesteps(1000, training=True)
sigmas = sched.sigmas.to(dev).float(); tsteps = sched.timesteps.to(dev).float()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

bce_nored = torch.nn.BCEWithLogitsLoss(reduction="none") if LPROBE else None
for step in range(START, STEPS):
    x0, latw, kb, ms, event, held, hcount, placef = build_batch(BS); B = x0.shape[0]
    ledg = {"ledger_held": held, "ledger_count": hcount} if LEDGER else {}
    cond_concat, vc = build_cond(x0)
    if LPROBE or V3:                                # uniform sigma: t<=650 -> 65% (L_probe needs low-noise pred_x0)
        sig_v = torch.rand(B, device=dev)
        if LPROBE and T_HI_FRAC > 0:                # event样本偏高噪: 贴生成态起点且100%落在探针有效域内
            hi = (event.sum(dim=1) > 0) & (torch.rand(B, device=dev) < T_HI_FRAC)
            sig_v = torch.where(hi, (T_HI_LO + sig_v * (T_HI_HI - T_HI_LO)) / 1000.0, sig_v)
        sig = sig_v.view(B, 1, 1, 1, 1); t = (sig_v * 1000.0).to(dt)
    else:
        idx = torch.randint(0, len(tsteps), (B,), device=dev)
        sig = sigmas[idx].view(B, 1, 1, 1, 1); t = tsteps[idx]
    noise = torch.randn_like(x0)
    xt = ((1 - sig) * x0.float() + sig * noise.float()).to(dt)
    target = (noise.float() - x0.float())
    # BATCHED ON+OFF forward: the counterfactual (place-zeroed) samples ride in the same
    # model call as extra batch rows (identical math to two sequential calls; ~30% faster,
    # and removes the DDP straggler effect of conditional second forwards).
    jj = None; pred_off = None
    if LPROBE:
        sub = (event.sum(dim=1) > 0)                     # only samples where place col differs ON vs OFF
        if bool(sub.any()): jj = sub.nonzero(as_tuple=True)[0]
    if jj is not None:
        kb_off = kb[jj].clone(); kb_off[:, :, PLACE_IDX] = 0
        ledg_cat = {k_: torch.cat([v_, v_[jj]], 0) for k_, v_ in ledg.items()} if ledg else {}
        out = model(torch.cat([xt, xt[jj]], 0), t=torch.cat([t, t[jj]], 0),
                    visual_context=torch.cat([vc, vc[jj]], 0),
                    cond_concat=torch.cat([cond_concat, cond_concat[jj]], 0),
                    keyboard_cond=torch.cat([kb, kb_off], 0),
                    mouse_cond=torch.cat([ms, ms[jj]], 0), **ledg_cat)
        if isinstance(out, (list, tuple)): out = out[0]
        pred, pred_off = out[:B], out[B:]
    else:
        pred = model(xt, t=t, visual_context=vc, cond_concat=cond_concat, keyboard_cond=kb, mouse_cond=ms, **ledg)
        if isinstance(pred, (list, tuple)): pred = pred[0]
    err = (pred.float() - target) ** 2                                   # [B,16,F,44,80]
    if V3:
        # crosshair-emphasis: frames t..t+4 after each place, center region, weight WREC_K (real GT, just spotlighting)
        fidx = torch.arange(F, device=dev)
        # symmetric spotlight: place frames (block appears) AND RMB-no-place frames (nothing
        # appears) both get crosshair emphasis -> real-GT contrast pair, no context-shortcut bias
        spot = torch.maximum(placef, event)
        ext = spot.clone()
        for s_ in range(1, 5):
            ext = torch.maximum(ext, torch.roll(spot, s_, dims=1) * (fidx >= s_))
        wrec = torch.ones(B, 1, F, 44, 80, device=dev)
        wrec[:, :, :, CR_R, CR_C] += (WREC_K - 1.0) * ext.view(B, 1, F, 1, 1)
        L_recon = (err * wrec).sum() / (wrec.sum() * err.shape[1] + 1e-6)
    else:
        L_recon = err.mean()
    w = (1.0 - event).view(B, 1, F, 1, 1)                # non-place frames (used by preserve & ctr)
    L_preserve = torch.zeros((), device=dev)
    if W_PRESERVE > 0:                                   # base-preserve: conflicts with GT block on post-place frames -> default off
        with torch.no_grad():                            # anti-forgetting: ref = base w/ PLACE zeroed
            kb_base = kb.clone(); kb_base[:, :, PLACE_IDX] = 0
            ref_pred = ref(xt, t=t, visual_context=vc, cond_concat=cond_concat, keyboard_cond=kb_base, mouse_cond=ms)
            if isinstance(ref_pred, (list, tuple)): ref_pred = ref_pred[0]
        L_preserve = ((pred.float() - ref_pred.float()) ** 2 * w).sum() / (w.sum() * pred[0, :, 0].numel() + 1e-6)
    L_cf = torch.zeros((), device=dev)
    posm = (placef.sum(dim=1) > 0)
    if V3 and W_CF > 0 and bool(posm.any()):   # skip at W_CF=0: the metric-only double fwd/bwd costs ~40% step time
        ii = posm.nonzero(as_tuple=True)[0]
        kb_off = kb[ii].clone(); kb_off[:, :, PLACE_IDX] = 0                 # counterfactual: place OFF
        ledg_off = {k_: v_[ii] for k_, v_ in ledg.items()} if ledg else {}
        pred_off = model(xt[ii], t=t[ii], visual_context=vc[ii], cond_concat=cond_concat[ii],
                         keyboard_cond=kb_off, mouse_cond=ms[ii], **ledg_off)
        if isinstance(pred_off, (list, tuple)): pred_off = pred_off[0]
        x0off = noise[ii].float() - pred_off.float()                          # eps - v_hat: PURE model belief, zero GT leak
        dof = torch.diff(x0off[:, :, :, CR_R, CR_C], dim=2)                   # [b,16,F-1,.,.] transitions
        e_off = (dof ** 2).mean(dim=(1, 3, 4))                                # [b,F-1]
        # baseline from the OFF branch's OWN pre-place transitions (same estimator ->
        # same noise floor; a clean-GT baseline undercounts eps-vhat noise and misfires)
        e_gt = e_off.detach()
        pfi = placef[ii]
        def _shift(m, s_):
            sh = torch.roll(m, -s_, dims=1)
            if s_ > 0: sh[:, -s_:] = 0
            elif s_ < 0: sh[:, :(-s_)] = 0
            return sh
        band = torch.zeros_like(e_off)
        for s_ in (-1, 0, 1, 2): band = torch.maximum(band, _shift(pfi, s_)[:, :F - 1])
        excl = band.clone()
        for s_ in (-3, -2, 3): excl = torch.maximum(excl, _shift(pfi, s_)[:, :F - 1])
        basem = 1 - excl
        E_base = (e_gt * basem).sum(1) / (basem.sum(1) + 1e-6)
        E_base = torch.where(basem.sum(1) > 0.5, E_base, e_gt.mean(1))
        E_evt = (e_off * band + (band - 1) * 1e9).amax(dim=1)                 # max over event transitions only
        L_cf = torch.relu(E_evt - 1.5 * E_base).mean()
    loss = L_recon + W_PRESERVE * L_preserve + W_CF * L_cf
    Lmar = Lneg = Lpon = Lpoff = Lctr = L_out = torch.zeros((), device=dev)
    if LPROBE:
        # LEAKAGE FIX: pred_x0 = (1-sig)*x0_GT + sig*x0_model -> the GT block leaks into pred_x0
        # and satisfies an absolute BCE target for free (measured: leak-only scores .75-.97 at
        # sig .2-.6). Both branches share xt, so the leak CANCELS in the difference z_on-z_off.
        if jj is not None:                                # pred_off precomputed in the batched forward above
            tg = (t[jj] <= T_PROBE).float().view(-1, 1)                  # probe domain-valid region
            pred_x0 = xt[jj].float() - sig[jj] * pred[jj].float()
            z_on = place_logit(pred_x0)                                  # [b,F] place logit
            pred_x0_off = xt[jj].float() - sig[jj] * pred_off.float()
            z_off = place_logit(pred_x0_off)
            margin = z_on - z_off                                        # leak-free: pure model contribution
            pos = placef[jj] * tg                                        # GT place frames
            negm = event[jj] * (1 - placef[jj]) * tg                     # RMB pressed but no place (sky/invalid)
            Lmar = (torch.nn.functional.softplus(MARGIN_D - margin) * pos).sum() / (pos.sum() + 1e-6)
            Lneg = (torch.nn.functional.softplus(margin) * negm).sum() / (negm.sum() + 1e-6)
            # weak absolute elicit pressure ONLY where the model owns pred_x0 (sig^2 downweights leak zone)
            sw = (sig.view(B, 1)[jj] ** 2) * tg
            Lpon = (bce_nored(z_on, placef[jj]) * sw).sum() / (sw.sum() * F + 1e-6)
            # Lpoff: OFF分支绝对压制 —— margin只管差值, z_on饱和后z_off可以飘到中位
            # (实测: realOFF后续按键会漏0.57/0.38); 按了不算数的世界必须不像place
            swo = sw * placef[jj]
            Lpoff = (bce_nored(z_off, torch.zeros_like(z_off)) * swo).sum() / (swo.sum() + 1e-6)
            # Lctr ONLY on PRE-place frames: the two worlds are genuinely identical before the press
            # (post-place tether to blocky-ON was the fake-GT poison -> removed; post frames answer
            # to the probe holistically via margin, not per-pixel)
            after_pf = torch.cummax(placef[jj], dim=1).values
            wctr = ((1 - event[jj]) * (1 - after_pf)).view(-1, 1, F, 1, 1)
            Lctr = ((pred[jj].float() - pred_off.float()) ** 2 * wctr).sum() / (wctr.sum() * pred[0, :, 0].numel() + 1e-6)
            # L_out: place后帧的ON-OFF差异只许留在允许区内(方块@准星, 挥臂@右下);
            # 区外差异=全局雾/色偏类捷径, 压掉. 与Lmar形成夹逼: 必须有差异+差异必须在框内
            wout = after_pf.view(-1, 1, F, 1, 1) * (1.0 - ALLOW)
            L_out = ((pred[jj].float() - pred_off.float()) ** 2 * wout).sum() / (wout.sum() * pred.shape[1] + 1e-6)
            # W_POFF=0: Lpoff对无锚OFF分支索要绝对擦除 → 三千步学出涂抹作弊解
            # (@13000 OFF后段糊成手持纹理); 因果靠margin差分+外部屏蔽, 不靠OFF自证清白
            loss = loss + LAM_MARGIN * (Lmar + Lneg) + LAM_PROBE * (Lpon + W_POFF * Lpoff) + LAM_CTR_PR * Lctr + LAM_OUT * L_out
    loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); opt.zero_grad(set_to_none=True)
    if IS0 and step % 20 == 0:
        extra = (f" Lmar={float(Lmar):.3f} Lneg={float(Lneg):.3f} Lpon={float(Lpon):.3f} Lpoff={float(Lpoff):.3f} Lctr={float(Lctr):.4f} Lout={float(L_out):.4f}" if LPROBE else "") + (f" Lcf={float(L_cf):.4f}" if (V3 and W_CF > 0) else "")
        print(f"step {step} loss={loss.item():.4f} recon={L_recon.item():.4f} presv={L_preserve.item():.4f}{extra} gn={float(gn):.2f}", flush=True)
    if IS0 and step % 500 == 0 and step > 0:
        torch.save({"model": {k: v.detach().cpu() for k, v in core.state_dict().items()}, "step": step}, f"{OUT}/ft_{step:05d}.pt")
        for old in sorted(glob.glob(f"{OUT}/ft_[0-9]*.pt"))[:-2]: os.remove(old)
        print(f"saved ft_{step:05d}.pt", flush=True)
if IS0:
    torch.save({"model": {k: v.detach().cpu() for k, v in core.state_dict().items()}, "step": STEPS}, f"{OUT}/ft_final.pt")
    print("MG2-FT-DONE", flush=True)
if DDP_ON: dist.destroy_process_group()

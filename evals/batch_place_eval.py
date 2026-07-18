#!/usr/bin/env python3
"""批量放置成功率: N窗 × lat4零前奏协议(静止, 单发按键@lat4)。
成功判据(自动): ON-OFF 在准星区(像素)的差异在窗尾持久 > 阈值。
输出: 每窗 探针@lat5 / 瞬时diff / 持久diff / 判定, + 帧条(ON上OFF下, 帧24/36/56)。
Usage: CUDA_VISIBLE_DEVICES=3 python3 batch_place_eval.py <ckpt> <tag>"""
import os, sys, numpy as np, torch, imageio
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2")
from utils.scheduler import FlowMatchScheduler
from mg2_model import load_expanded_base, PLACE_IDX
from wan.vae.wanx_vae import get_wanx_vae_wrapper

ckpt, TAG = sys.argv[1], sys.argv[2]
DATA = "/root/autodl-tmp/vpt_mc_352"; F = 15; NUM_PIX = (F - 1) * 4 + 1
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
dev, dt = "cuda", torch.bfloat16
OUT = "/root/autodl-tmp/place_eval"; os.makedirs(OUT, exist_ok=True)
PLACE_LAT = 4
PROTO = os.environ.get("PROTO", "lat4")
def pick_first_windows(n=12):
    wins = []
    for s in sorted(os.listdir(DATA)):
        if s.startswith(("bumpy", "cheeky")): continue
        npz = f"{DATA}/{s}/seg_000.npz"
        if not (os.path.exists(npz) and os.path.exists(f"{DATA}/{s}/seg_000_lat44.npy")): continue
        z = np.load(npz)
        if "place_fire" not in z.files: continue
        pf = z["place_fire"] > 0; h = z["held_id"]; g = np.asarray(z.get("gui", np.zeros(len(h))))
        for e in np.flatnonzero(pf):
            st = e - 7
            if st < 0 or st + F > len(h): continue
            if (h[st:st + F] != h[st]).any() or h[st] <= 0 or g[st:st + F].any(): continue
            wins.append((s, int(st)))
            break
        if len(wins) >= n: break
    return wins
_UNUSED = [  # 场景多样的6窗(早前候选网格)
    ("Player588-f153ac423f61-20210912-171704", 222),   # 石廊(12000/15000分歧窗)
    ("Player148-11d51f62bd79-20210705-175408", 386),   # 草地圆石小径
    ("Player149-f153ac423f61-20211121-141912", 10),    # 挖掘坡补土
    ("Player83-0c463f302bdc-20210715-155042", 616),    # 草地木屋台阶
    ("Player403-65a5bbcab443-20210730-134649", 168),   # 明亮场景 held=12
    ("Player346-70ca541fa3cf-20211122-185319", 68),    # 木结构场景
]
# 准星区(像素): latent CR 12:36/22:58 ×8
CR_PR, CR_PC = slice(96, 288), slice(176, 464)

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

sched = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True); sched.set_timesteps(50, training=False)
tsteps = sched.timesteps.to(dev); sigmas = sched.sigmas.to(dev).float()

@torch.no_grad()
def run_window(seg, st):
    lat = np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r")
    z = np.load(f"{DATA}/{seg}/seg_000.npz")
    latw = torch.from_numpy(np.asarray(lat[st:st + F], np.float32)).to(dev, dt)
    held_gt = int(z["held_id"][st])
    x0 = latw.permute(1, 0, 2, 3).unsqueeze(0)
    fp = vae.decode(x0[:, :, 0:1], device=dev, **TK).float().clamp(-1, 1).to(dev)
    fp0 = (fp[:, :, 0:1] if fp.shape[1] == 3 else fp.permute(0, 2, 1, 3, 4)[:, :, 0:1]).to(dev)
    pad = torch.zeros(1, 3, NUM_PIX - 1, fp0.shape[-2], fp0.shape[-1], device=dev, dtype=dt)
    img_cond = vae.encode(torch.cat([fp0.to(dt), pad], 2), device=dev, **TK).to(dev, dt)
    if img_cond.shape[1] != 16: img_cond = img_cond.permute(0, 2, 1, 3, 4)
    mask = torch.zeros(1, 4, F, 44, 80, device=dev, dtype=dt); mask[:, :, 0] = 1.0
    y = torch.cat([mask, img_cond], 1)
    vc = vae.clip.encode_video(fp0.to(dt)).to(dt)
    ms = torch.zeros(1, NUM_PIX, 2, device=dev, dtype=dt)
    held = torch.full((1, F), held_gt, device=dev, dtype=torch.long)
    cnt = torch.full((1, F), 5.0, device=dev, dtype=dt)
    outs = {}
    f1 = 7
    act = z["action"][st:st + F].astype(np.float32)
    CFG_W = float(os.environ.get("CFG_W", "0"))
    for on in (True, False):
        kb = torch.zeros(1, NUM_PIX, 7, device=dev, dtype=dt)
        if PROTO == "first":
            kbn = np.zeros((F, 7), np.float32)
            kbn[:, 0] = act[:, 0]; kbn[:, 1] = act[:, 2]; kbn[:, 2] = act[:, 1]; kbn[:, 3] = act[:, 3]
            kbn[:, 4] = act[:, 4]; kbn[:, 5] = act[:, 7]
            kbn[:, PLACE_IDX] = (act[:, 8] > 0).astype(np.float32) if on else 0.0
            msn = np.zeros((F, 2), np.float32)
            msn[:, 0] = -np.clip(act[:, 10], -20.0, 20.0) / 15.0
            msn[:, 1] = np.clip(act[:, 9], -20.0, 20.0) / 15.0
            kbn[f1 + 1:] = 0.0; msn[f1 + 1:] = 0.0
            kb = torch.from_numpy(np.repeat(kbn, 4, 0)[:NUM_PIX])[None].to(dev, dt)
            ms = torch.from_numpy(np.repeat(msn, 4, 0)[:NUM_PIX])[None].to(dev, dt)
        elif on:
            p = PLACE_LAT * 4; kb[:, p:p + 4, PLACE_IDX] = 1.0
        kb_u = kb.clone(); kb_u[:, :, PLACE_IDX] = 0
        use_cfg = CFG_W > 0 and on and bool(kb[:, :, PLACE_IDX].max() > 0)
        torch.manual_seed(42); x = torch.randn(1, 16, F, 44, 80, device=dev, dtype=dt)
        for i, t in enumerate(tsteps):
            fl = model(x, t=t.reshape(1).to(dt), visual_context=vc, cond_concat=y,
                       keyboard_cond=kb, mouse_cond=ms, ledger_held=held, ledger_count=cnt)
            if isinstance(fl, (list, tuple)): fl = fl[0]
            if use_cfg:
                flu = model(x, t=t.reshape(1).to(dt), visual_context=vc, cond_concat=y,
                            keyboard_cond=kb_u, mouse_cond=ms, ledger_held=held, ledger_count=cnt)
                if isinstance(flu, (list, tuple)): flu = flu[0]
                fl = flu.float() + CFG_W * (fl.float() - flu.float())
            sg = sigmas[i]; ns = sigmas[i + 1] if i + 1 < len(sigmas) else sigmas.new_zeros(())
            x = (x.float() + (ns - sg) * fl.float()).to(dt)
        xl = x.permute(0, 2, 1, 3, 4).float().clone(); xl[:, :, :, HUD_R, HUD_C] = 0
        dxx = torch.diff(xl, dim=1, prepend=xl[:, :1])
        _, _, su = probe(torch.cat([xl, dxx], dim=2))
        pr = torch.sigmoid(su[0]).cpu().numpy()
        pix = vae.decode(x, device=dev, **TK)[0].float().clamp(-1, 1)
        if pix.shape[0] == 3: pix = pix.permute(1, 0, 2, 3)
        v = ((pix + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
        outs[on] = (pr, v)
    pr_on, v_on = outs[True]; pr_off, v_off = outs[False]
    d = np.abs(v_on.astype(float) - v_off.astype(float))[:, CR_PR, CR_PC].mean(axis=(1, 2, 3))  # [57] 准星区diff
    transient = float(d[28:36].mean()) if PROTO == 'first' else float(d[20:28].mean())          # 事件附近
    persist = float(d[48:57].mean())            # 窗尾
    ok = persist > 3.0                          # 持久判据
    strip = np.concatenate([np.concatenate([v_on[i] for i in (24, 36, 56)], 1),
                            np.concatenate([v_off[i] for i in (24, 36, 56)], 1)], 0)
    imageio.imwrite(f"{OUT}/{TAG}_{seg[:12]}_{st}.png", strip)
    imageio.mimwrite(f"{OUT}/{TAG}_{seg[:12]}_{st}_ON.mp4", v_on, fps=16, quality=8)
    imageio.mimwrite(f"{OUT}/{TAG}_{seg[:12]}_{st}_OFF.mp4", v_off, fps=16, quality=8)
    rd = (7 + 1) if PROTO == 'first' else (PLACE_LAT + 1)
    return held_gt, float(pr_on[rd]), float(pr_off[rd]), transient, persist, ok

if PROTO == "first":
    WINS = pick_first_windows(12)
print(f"== {TAG} ({ckpt}) PROTO={PROTO} {len(WINS)}窗 ==", flush=True)
n_ok = 0
for seg, st in WINS:
    h, pon, poff, tr, pe, ok = run_window(seg, st)
    n_ok += ok
    print(f"{seg[:16]}@{st} held={h:>2}  探针ON/OFF={pon:.2f}/{poff:.2f}  瞬时={tr:.1f} 持久={pe:.1f}  {'✓放上' if ok else '✗未放'}", flush=True)
print(f"成功 {n_ok}/{len(WINS)}", flush=True)
print("BATCH-EVAL-DONE", flush=True)

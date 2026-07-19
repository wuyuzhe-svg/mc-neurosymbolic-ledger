#!/usr/bin/env python3
"""最小账本demo(单块生命周期剧本):
手持不变(泥土), 两趟生成:
  趟A(真实前奏0-4): place@5(账本2→1) → attack@8
  趟B(首帧=趟A末帧): place@3(账本1→0) → place@9(账本=0→动作被掐, 画面应安静)
账本 = 旁路状态机: 探针当事件哨兵(fire才扣账), count=0时清零后续place位。
HUD: 账本合成计数角标(右上), 事件字幕(底部)。输出拼接mp4 + 帧条。
Usage: CUDA_VISIBLE_DEVICES=3 python3 demo_ledger_minimal.py <ckpt> <seg> <st>
"""
import os, sys, numpy as np, torch, imageio
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.environ.get("MG2_REPO", "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2"))
from utils.scheduler import FlowMatchScheduler
from mg2_model import load_expanded_base, PLACE_IDX
from wan.vae.wanx_vae import get_wanx_vae_wrapper

ckpt = sys.argv[1]
seg = sys.argv[2] if len(sys.argv) > 2 else "Player759-f153ac423f61-20210704-140425"
st = int(sys.argv[3]) if len(sys.argv) > 3 else 88
DATA = os.environ.get("MC_DATA", "/root/autodl-tmp/vpt_mc_352"); F = 15; NUM_PIX = (F - 1) * 4 + 1
TK = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
dev, dt = "cuda", torch.bfloat16
OUT = os.environ.get("OUT_DIR", "/root/autodl-tmp/demo_ledger"); os.makedirs(OUT, exist_ok=True)
ATTACK_IDX = 5; CAM_SCALE = 1.0 / 15.0; CAM_DEG_CLIP = 20.0

z = np.load(f"{DATA}/{seg}/seg_000.npz")
held_gt = int(z["held_id"][st])
model = load_expanded_base(dev, dt, ledger=True)
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
model.eval().requires_grad_(False)
vae = get_wanx_vae_wrapper(os.environ.get("MG2_WEIGHTS", "/root/autodl-tmp/mg2_weights"), torch.float16).to(dev, dt)

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
tst = sched.timesteps.to(dev); sig = sched.sigmas.to(dev).float()

def build_cond(first_latent_1f):
    """first_latent_1f: [1,16,1,44,80] -> (y, vc)"""
    with torch.no_grad():
        fp = vae.decode(first_latent_1f, device=dev, **TK).float().clamp(-1, 1).to(dev)
        fp0 = (fp[:, :, 0:1] if fp.shape[1] == 3 else fp.permute(0, 2, 1, 3, 4)[:, :, 0:1]).to(dev)
        pad = torch.zeros(1, 3, NUM_PIX - 1, fp0.shape[-2], fp0.shape[-1], device=dev, dtype=dt)
        ic = vae.encode(torch.cat([fp0.to(dt), pad], 2), device=dev, **TK).to(dev, dt)
        if ic.shape[1] != 16: ic = ic.permute(0, 2, 1, 3, 4)
        mask = torch.zeros(1, 4, F, 44, 80, device=dev, dtype=dt); mask[:, :, 0] = 1.0
        return torch.cat([mask, ic], 1), vae.clip.encode_video(fp0.to(dt)).to(dt)

CFG_W = float(os.environ.get("CFG_W", "2.5"))
@torch.no_grad()
def gen_pass(y, vc, kb_lat, ms_lat, seed, cfg=False):
    kb = torch.from_numpy(np.repeat(kb_lat, 4, 0)[:NUM_PIX])[None].to(dev, dt)
    ms = torch.from_numpy(np.repeat(ms_lat, 4, 0)[:NUM_PIX])[None].to(dev, dt)
    kb_u = kb.clone(); kb_u[:, :, PLACE_IDX] = 0            # uncond: 仅place位清零
    held = torch.full((1, F), held_gt, device=dev, dtype=torch.long)
    cnt = torch.full((1, F), 5.0, device=dev, dtype=dt)
    torch.manual_seed(seed); x = torch.randn(1, 16, F, 44, 80, device=dev, dtype=dt)
    use_cfg = cfg and bool(kb_lat[:, PLACE_IDX].max() > 0)
    for i, t in enumerate(tst):
        fl = model(x, t=t.reshape(1).to(dt), visual_context=vc, cond_concat=y,
                   keyboard_cond=kb, mouse_cond=ms, ledger_held=held, ledger_count=cnt)
        if isinstance(fl, (list, tuple)): fl = fl[0]
        if use_cfg:
            flu = model(x, t=t.reshape(1).to(dt), visual_context=vc, cond_concat=y,
                        keyboard_cond=kb_u, mouse_cond=ms, ledger_held=held, ledger_count=cnt)
            if isinstance(flu, (list, tuple)): flu = flu[0]
            fl = flu.float() + CFG_W * (fl.float() - flu.float())
        ns = sig[i + 1] if i + 1 < len(sig) else sig.new_zeros(())
        x = (x.float() + (ns - sig[i]) * fl.float()).to(dt)
    xl = x.permute(0, 2, 1, 3, 4).float().clone(); xl[:, :, :, HUD_R, HUD_C] = 0
    dxx = torch.diff(xl, dim=1, prepend=xl[:, :1])
    sm, _, su = probe(torch.cat([xl, dxx], dim=2))
    pr = torch.sigmoid(su[0]).cpu().numpy()
    pm = torch.sigmoid(sm[0]).cpu().numpy()
    pix = vae.decode(x, device=dev, **TK)[0].float().clamp(-1, 1)
    if pix.shape[0] == 3: pix = pix.permute(1, 0, 2, 3)
    v = ((pix + 1) * 127.5).clamp(0, 255).byte().cpu().permute(0, 2, 3, 1).numpy()
    return v, pr, pm, x

# ---------- 账本 ----------
class Ledger:
    def __init__(self, item, count): self.item, self.count, self.log = item, count, []
    def try_place(self, lat_frame):
        """返回该帧place位是否放行"""
        if self.count > 0:
            self.log.append(("PLACE_SENT", lat_frame, self.count)); return 1.0
        self.log.append(("PLACE_BLOCKED", lat_frame, 0)); return 0.0
    def settle_mine(self, pm, frames):
        hit = max(float(pm[f]) for f in frames)
        if hit > 0.5:
            self.count += 1; self.log.append(("MINE_CREDIT", frames[0], self.count))
        else:
            self.log.append(("NO_MINE", frames[0], self.count))
    def settle_break_visual(self, v, px_pre, px_placed, px_post, credit_frame):
        """像素观测: 敲击后准星区回到放置前的样子 → 方块消失 → +1"""
        CRr, CRc = slice(96, 288), slice(176, 464)
        pre = v[px_pre, CRr, CRc].astype(float); placed = v[px_placed, CRr, CRc].astype(float)
        post = v[px_post, CRr, CRc].astype(float)
        d_pre = float(np.abs(post - pre).mean()); d_placed = float(np.abs(post - placed).mean())
        if d_pre < d_placed:
            self.count += 1; self.log.append(("BREAK_CREDIT", credit_frame, self.count))
        else:
            self.log.append(("NO_BREAK", credit_frame, self.count))
        return d_pre, d_placed
    def settle(self, pr, sent_frames):
        """探针哨兵: 对已放行的按帧, fire(>0.5)才扣账"""
        for f in sent_frames:
            hit = pr[f:f + 2].max() if f + 2 <= len(pr) else pr[f]
            if hit > 0.5:
                self.count -= 1; self.log.append(("FIRE_CONFIRMED", f, self.count))
            else:
                self.log.append(("NO_FIRE", f, self.count))

ledger = Ledger(held_gt, 1)   # 单块: 放(−1)-拆回(+1)-再放(−1)-枯竭拦截

# ---------- 趟A: 真实前奏0-4 + place@5 + attack@8 ----------
lat = np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r")
first = torch.from_numpy(np.asarray(lat[st:st + 1], np.float32)).to(dev, dt).permute(1, 0, 2, 3).unsqueeze(0)
act = z["action"][st:st + F].astype(np.float32)
kbA = np.zeros((F, 7), np.float32); msA = np.zeros((F, 2), np.float32)
kbA[:8, 0] = act[:8, 0]; kbA[:8, 1] = act[:8, 2]; kbA[:8, 2] = act[:8, 1]; kbA[:8, 3] = act[:8, 3]
kbA[:8, 4] = act[:8, 4]
msA[:8, 0] = -np.clip(act[:8, 10], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
msA[:8, 1] = np.clip(act[:8, 9], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
kbA[8, PLACE_IDX] = ledger.try_place(8)
yA, vcA = build_cond(first)
vA, prA, pmA, xA = gen_pass(yA, vcA, kbA, msA, seed=42, cfg=True)
ledger.settle(prA, [8])
print(f"趟A: 探针@lat[8..11]={[round(float(p),2) for p in prA[8:12]]} 账本={ledger.count}", flush=True)

# ---------- 趟B: 专职敲碎(attack按住lat5-10), 视觉消失检测+1 ----------
lastA = xA[:, :, -1:].clone()
kbB = np.zeros((F, 7), np.float32); msB = np.zeros((F, 2), np.float32)
kbB[1:4, 0] = 1.0                       # 走近两步
kbB[5:12, ATTACK_IDX] = 1.0             # 贴脸连敲
yB, vcB = build_cond(lastA)
vB, prB, pmB, xB = gen_pass(yB, vcB, kbB, msB, seed=43)
# 参照: 趟A放置前(帧30, 无块) vs 趟B开场(帧4, 有块); 敲后=趟B帧54
CRr, CRc = slice(96, 288), slice(176, 464)
noblock = vA[30, CRr, CRc].astype(float); withblock = vB[4, CRr, CRc].astype(float)
after = vB[54, CRr, CRc].astype(float)
d_no = float(np.abs(after - noblock).mean()); d_with = float(np.abs(after - withblock).mean())
if d_no < d_with:
    ledger.count += 1; ledger.log.append(("BREAK_CREDIT", 13, ledger.count))
else:
    ledger.log.append(("NO_BREAK", 13, ledger.count))
print(f"趟B敲碎: 距无块={d_no:.1f} 距有块={d_with:.1f} → {'消失+1' if d_no<d_with else '未消失'} 账本={ledger.count}", flush=True)

# ---------- 趟C: 再放置(CFG) ----------
lastB = xB[:, :, -1:].clone()
kbC2 = np.zeros((F, 7), np.float32); msC2 = np.zeros((F, 2), np.float32)
msC2[1:5, 1] = np.array([0.05, -0.05, 0.04, -0.04])
kbC2[6, PLACE_IDX] = ledger.try_place(6)
yC2, vcC2 = build_cond(lastB)
vC2, prC2, pmC2, xC2 = gen_pass(yC2, vcC2, kbC2, msC2, seed=44, cfg=True)
ledger.settle(prC2, [6])
print(f"趟C(再放置): 探针@lat[6..9]={[round(float(p),2) for p in prC2[6:10]]} 账本={ledger.count}", flush=True)

# ---------- 趟D: 请求→拦截 ----------
lastC = xC2[:, :, -1:].clone()
kbD = np.zeros((F, 7), np.float32); msD = np.zeros((F, 2), np.float32)
msD[1:5, 1] = np.array([0.05, -0.05, 0.04, -0.04])
kbD[6, PLACE_IDX] = ledger.try_place(6)
yD, vcD = build_cond(lastC)
vD, prD, pmD, xD = gen_pass(yD, vcD, kbD, msD, seed=45, cfg=True)
print(f"趟D: place请求(kb位={kbD[6, PLACE_IDX]}), 探针@lat[6..9]={[round(float(p),2) for p in prD[6:10]]} 账本={ledger.count}", flush=True)
print("账本流水:", ledger.log, flush=True)

# ---------- HUD合成 + 导出 ----------
def mc_hotbar(count, item_rgb, blocked=False):
    S = 40
    bar = np.zeros((S+6, 9*S+6, 4), np.uint8)
    bar[..., :3] = 30; bar[..., 3] = 200
    rng = np.random.RandomState(3)
    for i in range(9):
        x0 = 3 + i*S
        bar[3:3+S, x0:x0+S, :3] = 55
        bar[3:5, x0:x0+S, :3] = 100; bar[1+S:3+S, x0:x0+S, :3] = 15
        bar[3:3+S, x0:x0+2, :3] = 100; bar[3:3+S, x0+S-2:x0+S, :3] = 15
    x0 = 3
    for t in range(3):
        bar[t, max(0,x0-3):x0+S+3, :3] = 230; bar[S+5-t, max(0,x0-3):x0+S+3, :3] = 230
        bar[0:S+6, x0-3+t if x0>=3 else t, :3] = 230; bar[0:S+6, x0+S+2-t, :3] = 230
    if count > 0:
        base = np.array(item_rgb)
        tex = (base[None,None] + rng.randint(-25, 25, (12,12,3))).clip(0,255).astype(np.uint8)
        tex = np.kron(tex, np.ones((2,2,1), np.uint8))
        y, x = 3+8, 3+8
        bar[y:y+24, x:x+24, :3] = tex
        digits = {0:[7,5,5,5,7],1:[2,6,2,2,7],2:[7,1,7,4,7],3:[7,1,7,1,7]}
        d = digits[count]
        for r in range(5):
            for c in range(3):
                if d[r] >> (2-c) & 1:
                    bar[y+16+r*2:y+18+r*2, x+18+c*2:x+20+c*2, :3] = 255
    if blocked:                                   # 拦截: 槽0画红叉
        y, x = 3+4, 3+4
        for k in range(32):
            bar[y+k, x+k:x+k+3, :3] = [220, 40, 40]
            bar[y+k, x+31-k:x+34-k, :3] = [220, 40, 40]
    return bar

ITEM_RGB = [134, 96, 67] if held_gt == 13 else ([127,127,127] if held_gt == 12 else [160,130,90])
def annotate(v, count_track, blocked_track=None):
    out = v.copy()
    for i in range(len(out)):
        c = count_track(i)
        blk = blocked_track(i) if blocked_track else False
        bar = mc_hotbar(c, ITEM_RGB, blocked=blk)
        h, w = bar.shape[:2]
        y0, x0 = out.shape[1]-h-6, (out.shape[2]-w)//2
        a = bar[...,3:4]/255.0
        out[i, y0:y0+h, x0:x0+w] = (out[i, y0:y0+h, x0:x0+w]*(1-a) + bar[...,:3]*a).astype(np.uint8)
    return out

def ledger_track(log, pass_frames):
    """从账本流水构造该趟的逐像素帧计数轨道: FIRE_CONFIRMED@latf → 像素f*4+2处扣减"""
    events = [(f*4+2, cnt_after) for (tag, f, cnt_after) in log if tag in ("FIRE_CONFIRMED", "MINE_CREDIT", "BREAK_CREDIT") and f in pass_frames]
    start = [cnt for (tag, f, cnt) in log if tag in ("PLACE_SENT","PLACE_BLOCKED") and f in pass_frames]
    c0 = start[0] if start else 0
    def track(i):
        c = c0
        for (px, after) in events:
            if i >= px: c = after
        return c
    return track
vA_ann = annotate(vA, ledger_track([l for l in ledger.log if l[1]==8], [8]))
vB_ann = annotate(vB, ledger_track([l for l in ledger.log if l[0] in ("BREAK_CREDIT","NO_BREAK")], [13]))
vC_ann = annotate(vC2, ledger_track([l for l in ledger.log if l[1]==6 and l[0] in ("PLACE_SENT","FIRE_CONFIRMED")], [6]))
vD_ann = annotate(vD, lambda i: 0, blocked_track=lambda i: 20 <= i < 36)
full = np.concatenate([vA_ann, vB_ann, vC_ann, vD_ann], 0)
imageio.mimwrite(f"{OUT}/ledger_demo_4act_{seg[:10]}.mp4", full, fps=16, quality=8)
idx = [0, 20, 24, 32, 40, 56]
strip = np.concatenate([np.concatenate([v_[i] for i in idx], 1) for v_ in (vA_ann, vB_ann, vC_ann, vD_ann)], 0)
imageio.imwrite(f"{OUT}/ledger_demo_4act_{seg[:10]}_strip.png", strip)
print(f"输出: {OUT}/ledger_demo_4act_{seg[:10]}.mp4", flush=True)
print("DEMO-DONE", flush=True)

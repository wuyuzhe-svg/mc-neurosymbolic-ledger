import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
#!/usr/bin/env python3
"""v3 (mc_inv step_005600) 的 place 动作反事实 — 与 mg2 realON/realOFF 同口径:
同窗、同种子、真实动作轨(WASD/相机/attack), 唯一分叉 = RMB(place)列清零与否。
bag 两个世界都用真实值(available)。探针 A/B 读生成画面的 place 分。
  realON >> realOFF -> v3 的放置确实读动作位
  realON ~= realOFF -> v3 靠上下文出块(纯正例recon的路B), 动作位被无视
Usage: python cf_place_v3.py [ckpt]
"""
import sys, json, os, numpy as np, torch
from PIL import Image
from train_mc_joint import ROOT, N_HELD, F, mulaw_cam
from train_mc_probe import MCProbe
from train_state_layer_m4d import load_prompt_context
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/mc_inv/step_005600.pt"
dev, dt = "cuda", torch.bfloat16
OUT = "/root/autodl-tmp/cf_place_v3"; os.makedirs(OUT, exist_ok=True)
WINS = [("indist", json.load(open("/root/autodl-tmp/place_train.json"))["seg"], json.load(open("/root/autodl-tmp/place_train.json"))["st"])]
for c in json.load(open("/root/autodl-tmp/r1b_positive_windows.json"))[:2]:
    WINS.append(("heldout", c["seg"], c["st"]))

wrap = WanDiffusionWrapper(model_name="Wan2.1-Fun-1.3B-InP", timestep_shift=5.0,
                           is_causal=False, enable_injectors=True, n_action_dims=11, n_slots=N_HELD)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
wrap.model.load_state_dict(ck["trainable"], strict=False); wrap = wrap.to(dev, dt).eval()
print(f"loaded {CKPT} (step {ck.get('step','?')})", flush=True)
vae = WanVAEWrapper().to(device=dev, dtype=dt)
def lp(t):
    p = MCProbe(32).to(dev).float().eval()
    k = torch.load(f"checkpoints/mc_probe_{t}/best.pt", map_location="cpu", weights_only=False)
    p.load_state_dict(k["model"] if "model" in k else k); return p
pA, pB = lp("v27_A"), lp("v27_B")
prompt = load_prompt_context("data/prompt_embed_mc.pt", 1, dev, dt)
sched = wrap.get_scheduler(); sched.set_timesteps(40)

def load(seg, st):
    z = np.load(f"{ROOT}/{seg}/seg_000.npz")
    act = z["action"][st:st+F].astype(np.float32)
    hid = torch.from_numpy(z["held_id"][st:st+F].astype(np.int64)).unsqueeze(0).to(dev)
    bag = torch.from_numpy(np.asarray(z["bag_counts"][st:st+F], np.float32)).unsqueeze(0).to(dev, dt)
    img = torch.from_numpy(np.asarray(np.load(f"{ROOT}/{seg}/seg_000_imglat3.npy", mmap_mode="r")[st//3], np.float32)).to(dev, dt)
    place = z["place_fire"][st:st+F] > 0
    hid0 = int(z["held_id"][st])
    H, W = img.shape[-2:]; y = torch.zeros(1, F, 20, H, W, device=dev, dtype=dt); y[:, 0, :4] = 1.0; y[:, 0, 4:] = img
    return act, hid, bag, y, place, hid0, H, W

def cond(actm, hid, occ, y):
    return {"prompt_embeds": prompt, "control": actm, "camera": actm[..., 9:11].contiguous(),
            "state_held": hid, "state_occupancy": occ, "y": y,
            "clip_fea": torch.zeros(1, 257, 1280, device=dev, dtype=dt)}

def place_score(x, mask):
    xf = x.float(); xf[:, :, :, 52:, 26:78] = 0
    d = torch.diff(xf, dim=1, prepend=xf[:, :1]); inp = torch.cat([xf, d], dim=2)
    def rd(pr):
        with torch.no_grad(): _, _, lu = pr(inp)
        return round(float(torch.sigmoid(lu[0][mask].max()) if mask.any() else torch.sigmoid(lu[0].max())), 3)
    return rd(pA), rd(pB)

@torch.no_grad()
def gen(actm, hid, occ, y, H, W):
    torch.manual_seed(100); x = torch.randn(1, F, 16, H, W, device=dev, dtype=dt)
    for tt in sched.timesteps:
        tv = tt.to(dev).reshape(1, 1).expand(1, F)
        fp, _ = wrap(noisy_image_or_video=x, conditional_dict=cond(actm, hid, occ, y), timestep=tv)
        x = sched.step(fp.flatten(0, 1), tv.flatten(0, 1), x.flatten(0, 1)).unflatten(0, (1, F)).to(dt)
    return x

@torch.no_grad()
def montage(x, name, idxs):
    pix = vae.decode_to_pixel(x); v = ((pix[0].float()+1)*127.5).clamp(0,255).byte().cpu().permute(0,2,3,1).numpy()
    Image.fromarray(np.concatenate([v[i] for i in idxs if i<len(v)], axis=1)).save(f"{OUT}/{name}.png")

torch.set_grad_enabled(False)
for tag, seg, st in WINS:
    act, hid, bag, y, place, hid0, H, W = load(seg, st)
    mask = torch.from_numpy(place).to(dev)
    occ = torch.log1p(bag)                                     # 两个世界都用真实bag
    act_off = act.copy(); act_off[:, 8] = 0.0                  # 唯一分叉: RMB列清零
    actm_on = torch.from_numpy(mulaw_cam(act)).unsqueeze(0).to(dev, dt)
    actm_off = torch.from_numpy(mulaw_cam(act_off)).unsqueeze(0).to(dev, dt)
    xo = gen(actm_on, hid, occ, y, H, W); so = place_score(xo, mask)
    xf = gen(actm_off, hid, occ, y, H, W); sf = place_score(xf, mask)
    dpx = float((xo.float() - xf.float()).abs().mean())
    print(f"\n=== {tag} {seg[-15:]}@{st} held={hid0} place_frames={int(place.sum())} ===", flush=True)
    print(f" realON  place A/B = {so}", flush=True)
    print(f" realOFF place A/B = {sf}", flush=True)
    print(f" ACTION delta A/B  = ({round(so[0]-sf[0],3)}, {round(so[1]-sf[1],3)})  latent|diff|={dpx:.4f}  (>0 = 动作位有因果)", flush=True)
    montage(xo, f"{tag}_{st}_realON", [2,5,8,11,14,17]); montage(xf, f"{tag}_{st}_realOFF", [2,5,8,11,14,17])
print("\nCF-PLACE-V3-DONE", flush=True)

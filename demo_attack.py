"""attack维护背包demo(单幕): 撸树 → 探针(when)+像素结算(gone)+what-head(what) → 账本+1。
首帧: Player758…112728 st131(白天村庄, 准星对橡树干, 臂展)。输出 attack_demo.mp4 + strip。
Usage: CUDA_VISIBLE_DEVICES=0 python3 demo_attack.py"""
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os as _os
_SEG = _os.environ.get("ASEG", "Player758-f153ac423f61-20210720-112728")
_AST = _os.environ.get("AST", "131"); _TAG = _os.environ.get("ATAG", "oak46")
_SEED = int(_os.environ.get("ASEED", "46")); _AEND = int(_os.environ.get("AEND", "14"))
sys.argv = ["x", "checkpoints/keep/ft_15000.pt", _SEG, _AST]
exec(open(__file__.rsplit("/",1)[0] + "/demo_ledger_minimal.py").read().split("ledger = Ledger")[0])
ledger = Ledger(held_gt, 0)          # 白手起家
lat = np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r")
first = torch.from_numpy(np.asarray(lat[st:st+1], np.float32)).to(dev, dt).permute(1,0,2,3).unsqueeze(0)

kb1 = np.zeros((F,7), np.float32); ms1 = np.zeros((F,2), np.float32)
if _os.environ.get("REAL_ACT", "0") == "1":
    _P = int(_os.environ.get("PAUSE", "0"))       # 开头静止的latent帧数, GT源前移1帧保破坏在窗内
    _src = st + (1 if _P > 0 else 0)
    _act = z["action"][_src:_src+F-_P].astype(np.float32)
    kb1[_P:, 0] = _act[:, 0]; kb1[_P:, 1] = _act[:, 2]; kb1[_P:, 2] = _act[:, 1]; kb1[_P:, 3] = _act[:, 3]
    kb1[_P:, 4] = _act[:, 4]; kb1[_P:, ATTACK_IDX] = _act[:, 7]  # place列保持0
    ms1[_P:, 0] = -np.clip(_act[:, 10], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
    ms1[_P:, 1] = np.clip(_act[:, 9], -CAM_DEG_CLIP, CAM_DEG_CLIP) * CAM_SCALE
else:
    kb1[2:_AEND, ATTACK_IDX] = 1.0
y1, vc1 = build_cond(first)
v1, pr1, pm1, x1 = gen_pass(y1, vc1, kb1, ms1, seed=_SEED)
CRr, CRc = slice(96,288), slice(176,464)
d_end = float(np.abs(v1[54,CRr,CRc].astype(float) - v1[4,CRr,CRc].astype(float)).mean())
mine_hit = float(pm1.max())
dd = [float(np.abs(v1[(i+1)*4].astype(float)-v1[i*4].astype(float))[CRr,CRc].mean()) for i in range(2,13)]
e_tr = int(np.argmax(dd)) + 3
import torch.nn as _nn
class MineID(_nn.Module):
    def __init__(s, n=16):
        super().__init__()
        s.net = _nn.Sequential(_nn.Conv2d(48,96,3,1,1), _nn.GELU(), _nn.MaxPool2d(2),
                               _nn.Conv2d(96,192,3,1,1), _nn.GELU(), _nn.MaxPool2d(2),
                               _nn.Conv2d(192,256,3,1,1), _nn.GELU(), _nn.AdaptiveAvgPool2d(1),
                               _nn.Flatten(), _nn.Linear(256,n))
    def forward(s,x): return s.net(x)
_mid = MineID().to(dev)
_mid.load_state_dict(torch.load("checkpoints/keep/mine_id_head.pt", map_location="cpu")["model"]); _mid.eval()
_xl1 = x1[0].float().cpu().numpy()
_crop = _xl1[:, e_tr-2:e_tr+1, 12:36, 22:58].transpose(1,0,2,3)
with torch.no_grad():
    _pw = torch.softmax(_mid(torch.tensor(_crop.reshape(1,48,24,36), dtype=torch.float32, device=dev)),1)[0].cpu().numpy()
MVOC = ["oak_log","birch_log","spruce_log","dirt","grass_block","stone","cobblestone",
        "coal_ore","iron_ore","sand","gravel","oak_planks","stick","oak_sapling","andesite","granite"]
what_name = MVOC[int(_pw.argmax())]; what_p = float(_pw.max())
print(f"跳变帧lat{e_tr} what-head: {what_name} {what_p:.2f} (top3: "
      f"{[(MVOC[i], round(float(_pw[i]),2)) for i in _pw.argsort()[-3:][::-1]]})", flush=True)
if mine_hit > 0.5 and d_end > 15 and what_p > 0.5:
    ledger.count += 1; ledger.log.append((f"MINE_CREDIT[{what_name}@{what_p:.2f}]", e_tr, ledger.count))
print(f"攻击挖掘: mine探针={mine_hit:.2f} 区域diff={d_end:.1f} 账本={ledger.count}", flush=True)

CLS_RGB = {"dirt":[134,96,67], "grass_block":[110,140,70], "stone":[125,125,125], "cobblestone":[110,110,110],
           "sand":[222,206,160], "gravel":[136,126,126], "oak_log":[102,81,50], "birch_log":[196,187,163],
           "spruce_log":[58,37,16], "coal_ore":[70,70,70], "iron_ore":[175,142,118], "granite":[149,103,86],
           "andesite":[130,131,131], "oak_planks":[162,130,78], "stick":[104,78,47], "oak_sapling":[64,110,40]}
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
    return bar
ITEM_RGB = CLS_RGB.get(what_name, [200,200,200])
def annotate(v, count_track):
    out = v.copy()
    for i in range(len(out)):
        bar = mc_hotbar(count_track(i), ITEM_RGB)
        h, w = bar.shape[:2]
        y0, x0 = out.shape[1]-h-6, (out.shape[2]-w)//2
        a = bar[...,3:4]/255.0
        out[i, y0:y0+h, x0:x0+w] = (out[i, y0:y0+h, x0:x0+w]*(1-a) + bar[...,:3]*a).astype(np.uint8)
    return out
credit_px = e_tr*4 + 2
va = annotate(v1, lambda i: 1 if (ledger.count > 0 and i >= credit_px) else 0)
imageio.mimwrite(f"{OUT}/attack_demo_{_TAG}.mp4", va, fps=16, quality=8)
idx=[0, max(0,credit_px-8), credit_px, min(56,credit_px+10), 56]
strip = np.concatenate([va[i] for i in idx], 1)
imageio.imwrite(f"{OUT}/attack_demo_{_TAG}_strip.png", strip)
print(f"credit_px={credit_px} 流水: {ledger.log}", flush=True)
print("DONE", flush=True)

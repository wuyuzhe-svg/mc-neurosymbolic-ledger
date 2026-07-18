import sys
sys.path.insert(0, "/root/autodl-tmp/vrisingwm")
sys.argv = ["x", "checkpoints/keep/ft_15000.pt", "Player1-f153ac423f61-20210905-123307", "65"]
exec(open(__file__.rsplit("/",1)[0] + "/demo_ledger_minimal.py").read().split("ledger = Ledger")[0])
ledger = Ledger(held_gt, 0)          # 白手起家
lat = np.load(f"{DATA}/{seg}/seg_000_lat44.npy", mmap_mode="r")
first = torch.from_numpy(np.asarray(lat[st:st+1], np.float32)).to(dev, dt).permute(1,0,2,3).unsqueeze(0)

# 幕1: 挖掘
kb1 = np.zeros((F,7), np.float32); ms1 = np.zeros((F,2), np.float32)
kb1[2:14, ATTACK_IDX] = 1.0
y1, vc1 = build_cond(first)
v1, pr1, pm1, x1 = gen_pass(y1, vc1, kb1, ms1, seed=46)
CRr, CRc = slice(96,288), slice(176,464)
d_end = float(np.abs(v1[54,CRr,CRc].astype(float) - v1[4,CRr,CRc].astype(float)).mean())
mine_hit = float(pm1.max())
# --- what-head: 跳变帧定位 + 身份识别 ---
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
print(f"跳变帧lat{e_tr} what-head: {what_name} {what_p:.2f}", flush=True)
if mine_hit > 0.5 and d_end > 15 and what_p > 0.5:
    ledger.count += 1; ledger.log.append((f"MINE_CREDIT[{what_name}@{what_p:.2f}]", 8, ledger.count))
print(f"幕1挖掘: mine探针={mine_hit:.2f} 区域diff={d_end:.1f} 账本={ledger.count}", flush=True)

# 幕2: 放置(CFG)
last1 = x1[:,:,-1:].clone()
kb2 = np.zeros((F,7), np.float32); ms2 = np.zeros((F,2), np.float32)
ms2[1:5,1] = np.array([0.05,-0.05,0.04,-0.04])
kb2[6, PLACE_IDX] = ledger.try_place(6)
y2, vc2 = build_cond(last1)
v2, pr2, pm2, x2 = gen_pass(y2, vc2, kb2, ms2, seed=47, cfg=True)
ledger.settle(pr2, [6])
print(f"幕2放置: 探针@lat[6..9]={[round(float(p),2) for p in pr2[6:10]]} 账本={ledger.count}", flush=True)

# 幕3: 拦截
last2 = x2[:,:,-1:].clone()
kb3 = np.zeros((F,7), np.float32); ms3 = np.zeros((F,2), np.float32)
ms3[1:5,1] = np.array([0.05,-0.05,0.04,-0.04])
kb3[6, PLACE_IDX] = ledger.try_place(6)
y3, vc3 = build_cond(last2)
v3, pr3, pm3, x3 = gen_pass(y3, vc3, kb3, ms3, seed=48, cfg=True)
print(f"幕3: kb位={kb3[6,PLACE_IDX]} 探针max={float(pr3.max()):.2f} 账本={ledger.count}", flush=True)
print("流水:", ledger.log, flush=True)

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

CLS_RGB = {"dirt":[134,96,67], "grass_block":[110,140,70], "stone":[125,125,125], "cobblestone":[110,110,110],
           "sand":[222,206,160], "gravel":[136,126,126], "oak_log":[102,81,50], "birch_log":[196,187,163],
           "spruce_log":[58,37,16], "coal_ore":[70,70,70], "iron_ore":[175,142,118], "granite":[149,103,86],
           "andesite":[130,131,131], "oak_planks":[162,130,78], "stick":[104,78,47], "oak_sapling":[64,110,40]}
ITEM_RGB = CLS_RGB.get(what_name, [200,200,200])
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
    events = [(f*4+2, cnt_after) for (tag, f, cnt_after) in log if (tag.startswith("MINE_CREDIT") or tag in ("FIRE_CONFIRMED", "BREAK_CREDIT")) and f in pass_frames]
    start = [cnt for (tag, f, cnt) in log if tag in ("PLACE_SENT","PLACE_BLOCKED") and f in pass_frames]
    c0 = start[0] if start else 0
    def track(i):
        c = c0
        for (px, after) in events:
            if i >= px: c = after
        return c
    return track
va = annotate(v1, ledger_track([l for l in ledger.log if l[0].startswith("MINE_CREDIT")], [8]))
vb = annotate(v2, ledger_track([l for l in ledger.log if l[1]==6 and l[0] in ("PLACE_SENT","FIRE_CONFIRMED")], [6]))
vc_ = annotate(v3, lambda i: 0, blocked_track=lambda i: 20<=i<36)
full = np.concatenate([va, vb, vc_], 0)
imageio.mimwrite("/root/autodl-tmp/demo_ledger/economy_demo.mp4", full, fps=16, quality=8)
idx=[0,20,28,40,56]
strip = np.concatenate([np.concatenate([v_[i] for i in idx],1) for v_ in (va,vb,vc_)],0)
imageio.imwrite("/root/autodl-tmp/demo_ledger/economy_demo_strip.png", strip)
print("DONE", flush=True)

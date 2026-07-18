#!/usr/bin/env python3
"""what-head: 挖掘对象身份分类器。
输入 = 跳变前后3帧准星区latent crop [3,16,24,36] → 48ch CNN → 16类(MINE_VOCAB)。
段级train/val划分(SEGID哈希), 类别加权CE。Usage: python train_mine_id.py"""
import numpy as np, torch, torch.nn as nn

d = np.load("/root/autodl-tmp/mine_id_ds.npz")
X, Y, SEG = d["X"], d["Y"], d["SEGID"]
val_mask = (SEG % 10 == 0)                       # ~10% 段做验证
Xtr, Ytr = X[~val_mask], Y[~val_mask]
Xva, Yva = X[val_mask], Y[val_mask]
print(f"train={len(Ytr)} val={len(Yva)}")

dev = "cuda"
cnt = np.bincount(Ytr, minlength=16).astype(np.float32)
w = 1.0 / np.clip(cnt, 1, None); w = w / w.sum() * 16
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=dev))

class MineID(nn.Module):
    def __init__(self, n_cls=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(48, 96, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, 1, 1), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(192, 256, 3, 1, 1), nn.GELU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(256, n_cls))
    def forward(self, x): return self.net(x)

m = MineID().to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
Xtr_t = torch.tensor(Xtr.reshape(len(Xtr), 48, 24, 36), dtype=torch.float32)
Ytr_t = torch.tensor(Ytr, dtype=torch.long)
Xva_t = torch.tensor(Xva.reshape(len(Xva), 48, 24, 36), dtype=torch.float32).to(dev)
Yva_t = torch.tensor(Yva, dtype=torch.long).to(dev)
BS = 256
best = 0.0
for ep in range(20):
    m.train(); perm = torch.randperm(len(Ytr_t))
    for i in range(0, len(perm), BS):
        idx = perm[i:i+BS]
        xb, yb = Xtr_t[idx].to(dev), Ytr_t[idx].to(dev)
        loss = crit(m(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        pred = m(Xva_t).argmax(1)
        acc = float((pred == Yva_t).float().mean())
    print(f"ep{ep} val_acc={acc:.3f}", flush=True)
    if acc > best:
        best = acc
        torch.save({"model": m.state_dict(), "acc": acc}, "/root/autodl-tmp/vrisingwm/checkpoints/keep/mine_id_head.pt")
# 逐类准确率
with torch.no_grad():
    pred = m(Xva_t).argmax(1).cpu().numpy()
VOCAB = ["oak_log","birch_log","spruce_log","dirt","grass_block","stone","cobblestone",
         "coal_ore","iron_ore","sand","gravel","oak_planks","stick","oak_sapling","andesite","granite"]
print("\n逐类(验证集):")
for c in range(16):
    mask = Yva == c
    if mask.sum() == 0: continue
    print(f"  {VOCAB[c]:>12s}: n={int(mask.sum()):>5} acc={float((pred[mask]==c).mean()):.3f}")
print(f"BEST={best:.3f}  → checkpoints/keep/mine_id_head.pt")

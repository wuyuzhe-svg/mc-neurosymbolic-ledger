import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minecraft probe v1 -- THE 7.14 gate. Reads mine/pickup FIRE events off
raw Wan-VAE latents of first-person footage.

Two arms (the first-person unknown):
  --diff 0 : raw latent window only (camera motion handled by the net)
  --diff 1 : + temporal difference channels (V4's inductive bias -- may be
             swamped by camera motion here; that's exactly what we test)

Gate metric: per-frame fire AUC on held-out sessions (target: mine AUC
comparable to V Rising gather 0.87; pickup was 0.98 there).

    python train_mc_probe.py --diff 0 --ckpt_dir checkpoints/mc_probe_raw
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

ROOT = "/root/autodl-tmp/vpt_minecraft/segments_mc"
F = 21


class MCWindowDataset:
    def __init__(self, segs, stride=3):
        self.items = []
        self.cache = {}
        for s in segs:
            z = np.load(os.path.join(ROOT, s, "seg_000.npz"))
            L = int(z["n_latent"][0])
            gui = z["gui"]
            for st in range(0, L - F + 1, stride):
                if gui[st:st + F].any():
                    continue                     # GUI windows excluded
                self.items.append((s, st))
            self.cache[s] = {
                "lat": np.load(os.path.join(ROOT, s, "seg_000_latent.npy"),
                               mmap_mode="r"),
                "mine": z["mine_fire"], "pick": z["pickup_fire"],
                "use": z["place_fire"] if "place_fire" in z.files else z["use_fire"] if "use_fire" in z.files else np.zeros(L, np.int64),
            }

    def __len__(self):
        return len(self.items)

    def get(self, i):
        s, st = self.items[i]
        c = self.cache[s]
        lat = torch.from_numpy(np.asarray(c["lat"][st:st + F], dtype=np.float32))
        mine = torch.from_numpy((c["mine"][st:st + F] > 0).astype(np.float32))
        pick = torch.from_numpy((c["pick"][st:st + F] > 0).astype(np.float32))
        use = torch.from_numpy((c["use"][st:st + F] > 0).astype(np.float32))
        return lat, mine, pick, use


class MCProbe(nn.Module):
    def __init__(self, in_ch, dim=256):
        super().__init__()
        self.enc = nn.Sequential(                 # per-frame [C,60,104] -> dim
            nn.Conv2d(in_ch, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, dim, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, batch_first=True,
                                           norm_first=True)
        self.tf = nn.TransformerEncoder(layer, 2)
        self.pos = nn.Parameter(torch.zeros(1, F, dim))
        self.head_mine = nn.Linear(dim, 1)
        self.head_pick = nn.Linear(dim, 1)
        self.head_use = nn.Linear(dim, 1)

    def forward(self, x):                         # [B,F,C,H,W]
        B, Fr = x.shape[:2]
        h = self.enc(x.flatten(0, 1)).reshape(B, Fr, -1) + self.pos
        h = self.tf(h)
        return (self.head_mine(h).squeeze(-1), self.head_pick(h).squeeze(-1),
                self.head_use(h).squeeze(-1))


def auc(scores, labels):
    s, l = np.asarray(scores), np.asarray(labels)
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = l.sum(), (1 - l).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return (ranks[l > 0].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diff", type=int, default=0)
    ap.add_argument("--mask_hud", type=int, default=0)
    ap.add_argument("--place", type=int, default=0)
    ap.add_argument("--ckpt_dir", default="checkpoints/mc_probe")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--n_val_segs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda"
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    segs = sorted(s for s in os.listdir(ROOT) if not s.startswith(".")
                  and os.path.exists(os.path.join(ROOT, s, "seg_000.npz")))
    # clean hash-disjoint split (2026-07-12 leakage fix): train on train_segs,
    # select/report on devval; final test held out entirely. Falls back to the
    # legacy leaked val5.json only if the clean split is absent.
    sp = os.path.join(os.path.dirname(ROOT), "splits")
    if os.path.exists(f"{sp}/train_segs.json"):
        tr_set = set(json.load(open(f"{sp}/train_segs.json")))
        dv_set = set(json.load(open(f"{sp}/devval_segs.json")))
        train_segs = [s for s in segs if s in tr_set]
        val_segs = [s for s in segs if s in dv_set]
        print("using CLEAN hash-disjoint split (train/devval)")
    else:
        val_manifest = os.path.join(os.path.dirname(ROOT), "val5.json")
        if os.path.exists(val_manifest):
            val_segs = [s for s in json.load(open(val_manifest)) if s in segs]
            train_segs = [s for s in segs if s not in set(val_segs)]
        else:
            val_segs, train_segs = segs[:args.n_val_segs], segs[args.n_val_segs:]
    ds_tr = MCWindowDataset(train_segs)
    ds_va = MCWindowDataset(val_segs, stride=21)
    print(f"train windows {len(ds_tr)}  val windows {len(ds_va)}")

    in_ch = 32 if args.diff else 16
    model = MCProbe(in_ch).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"probe params {n_par/1e6:.2f}M  diff={args.diff}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    bce = nn.BCEWithLogitsLoss()
    log = open(os.path.join(args.ckpt_dir, "train_log.jsonl"), "a")

    def prep(lat):
        if args.mask_hud:
            lat = lat.clone()
            lat[:, :, :, 52:, 26:78] = 0
        if args.diff:
            d = torch.diff(lat, dim=1, prepend=lat[:, :1])
            lat = torch.cat([lat, d], dim=2)
        return lat

    def batch(ds, idxs):
        lats, ms, ps, us = zip(*[ds.get(i) for i in idxs])
        return (prep(torch.stack(lats)).to(device),
                torch.stack(ms).to(device), torch.stack(ps).to(device),
                torch.stack(us).to(device))

    best = 0.0
    for step in range(args.steps):
        model.train()
        lat, m, p, u = batch(ds_tr, [rng.randrange(len(ds_tr))
                                     for _ in range(args.batch_size)])
        lm, lp, lu = model(lat)
        loss = bce(lm, m) + bce(lp, p) + (bce(lu, u) if args.place else 0.0)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.val_every == 0 and step > 0:
            model.eval()
            sm, sp, su, gm, gp, gu = [], [], [], [], [], []
            with torch.no_grad():
                for i0 in range(0, len(ds_va), args.batch_size):
                    idxs = list(range(i0, min(i0 + args.batch_size, len(ds_va))))
                    lat, m, p, u = batch(ds_va, idxs)
                    lm, lp, lu = model(lat)
                    sm += lm.flatten().tolist(); gm += m.flatten().tolist()
                    sp += lp.flatten().tolist(); gp += p.flatten().tolist()
                    su += lu.flatten().tolist(); gu += u.flatten().tolist()
            am, ap_, au_ = auc(sm, gm), auc(sp, gp), auc(su, gu)
            rec = {"step": step, "auc_mine": round(am, 4),
                   "auc_pick": round(ap_, 4), "auc_use": round(au_, 4),
                   "loss": round(loss.item(), 4)}
            print(rec, flush=True)
            log.write(json.dumps(rec) + "\n"); log.flush()
            if (am + ap_) / 2 > best:
                best = (am + ap_) / 2
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args), "auc": rec},
                           os.path.join(args.ckpt_dir, "best.pt"))


if __name__ == "__main__":
    main()

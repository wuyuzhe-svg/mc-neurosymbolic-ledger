#!/usr/bin/env python3
"""Full-corpus label QC before the final resume: every ingested 6xx segment gets
held-bag consistency, place-RMB coverage, -1 rate, place density. Fails loudly on
outliers so bad segments can be quarantined instead of fed to the decisive run."""
import numpy as np, os, json

DATA = "/root/autodl-tmp/vpt_mc_352"
OLD_PREFIXES = ("bumpy", "cheeky")            # 原290段的命名系
rows = []
bad = []
for s in sorted(os.listdir(DATA)):
    d = f"{DATA}/{s}"
    if not os.path.isdir(d): continue
    npz = f"{d}/seg_000.npz"; lat = f"{d}/seg_000_lat44.npy"
    if not (os.path.exists(npz) and os.path.exists(lat)): continue
    new = not s.startswith(OLD_PREFIXES)
    if not new: continue                       # 只查新入库的
    try:
        z = np.load(npz)
        h = z["held_id"]; bag = z["bag_counts"]; pf = z["place_fire"] > 0
        rmb = z["action"][:, 8] > 0
        L = len(h)
        latn = np.load(lat, mmap_mode="r")
        ok_shape = (latn.shape[0] == L and latn.shape[1:] == (16, 44, 80))
        m = h > 0
        con = float((bag[np.arange(L), h.clip(0, 39)] > 0)[m].mean()) if m.any() else 1.0
        r1 = np.roll(rmb, 1); r1[0] = 0
        cov = float((pf & (rmb | r1)).sum() / max(pf.sum(), 1)) if pf.any() else 1.0
        unk = float((h == -1).mean())
        row = dict(seg=s, L=int(L), con=con, cov=cov, unk=unk, place=int(pf.sum()), shape_ok=bool(ok_shape))
        rows.append(row)
        if (not ok_shape) or con < 0.7 or cov < 0.8 or unk > 0.9:
            bad.append(row)
    except Exception as e:
        bad.append(dict(seg=s, err=str(e)[:60]))
n = len(rows)
con = np.array([r["con"] for r in rows]); cov = np.array([r["cov"] for r in rows])
unk = np.array([r["unk"] for r in rows]); pl = np.array([r["place"] for r in rows])
print(f"新段质检 n={n}")
print(f"held-bag一致率: mean={con.mean():.1%} p10={np.percentile(con,10):.1%}  (原库基准96.7%)")
print(f"place-RMB覆盖:  mean={cov.mean():.1%} p10={np.percentile(cov,10):.1%}  (基准99.7%)")
print(f"-1率: mean={unk.mean():.1%} p90={np.percentile(unk,90):.1%}  (原库25.8%)")
print(f"place事件总数(新段): {pl.sum():,}  段均 {pl.mean():.1f}")
print(f"不合格段: {len(bad)}")
for b in bad[:10]: print("  BAD:", b)
json.dump([b["seg"] for b in bad if "seg" in b], open("/root/autodl-tmp/qc_bad_segs.json", "w"))
print("QC-DONE")

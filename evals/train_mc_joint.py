import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
MC joint: action-conditioned bidirectional world model on VPT latents.
v2 (2026-07-11): base switched to Wan2.1-Fun-1.3B-InP -- pretrained I2V.

Why (two-agent survey, 2026-07-11 night): scene identity comes from
PRETRAINED i2v weights, not from the concat mechanism -- Matrix-Game-2 runs
the exact Wan-I2V recipe (16 noisy + 4 mask + 16 image y channels + CLIP)
inherited from SkyReels-V2-I2V; our V Rising from-scratch ctx conv never
held the scene. Fun-InP is that same recipe on a 1.3B backbone, Apache-2.0,
backbone-isomorphic to T2V (only patch_embedding 16->36 and per-layer image
cross-attn differ). VAE identical, latents unchanged.

Recipe:
  - FULL backbone @ 1e-5 (frozen-backbone convicted 2026-07-09)
  - control injector 11-dim/frame [w a s d space shift ctrl, LMB, RMB,
    yaw, pitch] @ 1e-5, UNSHIFTED (deploy alignment); V Rising-era
    action/state/weapon injectors frozen (dead weight)
  - first-frame conditioning 100% of batches (user ruling 2026-07-11:
    deployment always has a first frame): y = [4ch mask(frame0=1),
    16ch image latent(frame0=imglat, rest zeros)], Fun channel order.
    imglat = IMAGE-mode VAE encode of the window's first pixel frame
    (seg_000_imglat3.npy, row st//3) -- continuous segment latents carry
    causal history a deploy-time PNG never has.
  - clip_fea = zeros (Fun's own no-image convention); CLIP branch is the
    next lever if scene drift persists
  - two-regime t: action-conditioned batches t >= 500 (injector zero-signal
    pathology gate); control-dropout batches full band. y rides in BOTH
    (pretrained pathway, no pathology risk -- matches Fun pretraining).
  - loss on ALL frames incl. frame 0 (Fun/I2V recipe; no zero-init copy
    circuit to protect against anymore)

老三查: load audit (missing == injectors only) / step-200 val finite &
falling / morning same-seed W-A-S-D-LMB demo.

    torchrun --nproc_per_node=3 train_mc_joint.py     # GPU 1 is squatted
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist

from train_state_layer_m4d import load_prompt_context
from utils.wan_wrapper import WanDiffusionWrapper

ROOT = "/root/autodl-tmp/vpt_minecraft/segments_mc"
F = 21

CAM_MAX, CAM_MU = 300.0, 20.0

# v2 state channel (2026-07-11): held-item identity from anchor-propagated
# labels (build_held_labels.py). Vocab index 0 is reserved "none"; the
# injector maps held==-1 (unknown/absent) to its own embedding row 0.
# QA: 42% frame coverage, 240 conflict intervals dropped, zero physical
# contradictions in the propagation assumption.
N_HELD = 40


def mulaw_cam(a):
    """action[:, 9:11] raw mean-deg-per-tick (±288 observed) -> [-1,1].
    Keys are 0-1 hold fractions; raw camera would out-scale them 300x into
    the injector MLP (audit 2026-07-11). mu-law keeps small-turn resolution.
    DEPLOY MUST APPLY THE SAME TRANSFORM to commanded camera velocities."""
    x = np.clip(a[:, 9:11] / CAM_MAX, -1, 1)
    a = a.copy()
    a[:, 9:11] = np.sign(x) * np.log1p(CAM_MU * np.abs(x)) / np.log1p(CAM_MU)
    return a


class MCJointDataset:
    """21-frame (latent, action11, imglat0) windows, GUI-overlap excluded.
    Window starts are multiples of 3 == imglat3 grid."""

    def __init__(self, segs, stride=3):
        assert stride % 3 == 0
        self.items, self.cache = [], {}
        for s in segs:
            imglat_p = os.path.join(ROOT, s, "seg_000_imglat3.npy")
            if not os.path.exists(imglat_p):
                continue                      # imglat precompute still running
            z = np.load(os.path.join(ROOT, s, "seg_000.npz"))
            L = int(z["n_latent"][0])
            gui = z["gui"]
            for st in range(0, L - F + 1, stride):
                if gui[st:st + F].any():
                    continue
                self.items.append((s, st))
            pf = (z["place_fire"] if "place_fire" in z.files
                  else np.zeros(L, np.int64))
            self.cache[s] = {
                "lat": np.load(os.path.join(ROOT, s, "seg_000_latent.npy"),
                               mmap_mode="r"),
                "img": np.load(imglat_p, mmap_mode="r"),
                "act": mulaw_cam(z["action"].astype(np.float32)),
                "held": z["held_id"].astype(np.int64)
                        if "held_id" in z.files else np.full(L, -1, np.int64),
                "bag": z["bag_counts"].astype(np.float32)
                       if "bag_counts" in z.files
                       else np.zeros((L, N_HELD), np.float32),
                "ev": np.stack([(z["mine_fire"] > 0), (z["pickup_fire"] > 0),
                                (pf > 0)], -1).astype(np.float32),
            }

    def __len__(self):
        return len(self.items)

    def get(self, i):
        s, st = self.items[i]
        c = self.cache[s]
        return {
            "latent": torch.from_numpy(
                np.asarray(c["lat"][st:st + F], dtype=np.float32)),
            "action": torch.from_numpy(c["act"][st:st + F]),
            "imglat0": torch.from_numpy(
                np.asarray(c["img"][st // 3], dtype=np.float32)),
            "held": torch.from_numpy(c["held"][st:st + F]),
            "bag0": torch.from_numpy(c["bag"][st]),          # [40] window-start
            "bagseq": torch.from_numpy(
                np.asarray(c["bag"][st:st + F], dtype=np.float32)),  # [F,40] per-frame running inventory
            "events": torch.from_numpy(c["ev"][st:st + F]),  # [F,3] GT fires
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Wan2.1-Fun-1.3B-InP")
    ap.add_argument("--prompt_embed", default="data/prompt_embed_mc.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints/mc_joint")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--timestep_shift", type=float, default=5.0)
    ap.add_argument("--control_dropout", type=float, default=0.15)
    ap.add_argument("--lowband_p", type=float, default=0.2,
                    help="conditioned batches: probability of sampling t<500 "
                         "(control still fed). Soft preference instead of the "
                         "V Rising hard gate (user ruling 2026-07-11): the "
                         "model must see conditioned low-noise refinement or "
                         "the sampler walks through a distribution hole; the "
                         "zero-signal pathology risk is bounded by 20% "
                         "exposure + injector lr 1e-5 + gnorm watch")
    ap.add_argument("--lmb_frac", type=float, default=0.4)
    ap.add_argument("--bright_frac", type=float, default=0.5,
                    help="sampling weight for bright-surface (9xx) windows; "
                         "counters the 62%%-cave exposure bias (luminance "
                         "overshoot 77->8 measured 2026-07-11) and puts "
                         "capacity where demos/claims live")
    ap.add_argument("--save_every", type=int, default=400)
    ap.add_argument("--keep_last_n", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--val_batches", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    # coupled training (2026-07-11 内化方案): frozen probe grades pred_x0
    # against GT events; gradient flows THROUGH the probe INTO the generator.
    # Domain validated: probe AUC on pred_x0 matches clean latents for t<=650
    # (mine .94-.96 / pickup .93-.95 / place .98-.99).
    ap.add_argument("--probe_loss_w", type=float, default=0.0,
                    help=">0 enables the probe-consistency loss at this weight")
    ap.add_argument("--probe_ckpt",
                    default="checkpoints/mc_probe_v25_A/best.pt")
    ap.add_argument("--probe_t_max", type=float, default=650.0)
    ap.add_argument("--bag_cond", type=int, default=1,
                    help="feed log1p(window-start bag counts) via the "
                         "state_occupancy channel (zero-init pathway)")
    ap.add_argument("--perframe_bag", type=int, default=0,
                    help="feed per-frame running inventory (bagseq) instead of "
                         "the broadcast window-start bag0 -- lets the model see "
                         "depletion within a window (needed for gating)")
    ap.add_argument("--depl_frac", type=float, default=0.15,
                    help="oversample depletion windows (an item count "
                         "crosses to 0 in-window) -- they carry the causal "
                         "signal for bag logic")
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--smoke_freeze_bb", action="store_true",
                    help="smoke only: freeze backbone so a full optimizer "
                         "step fits the GPU-1 memory niche")
    ap.add_argument("--save_full_state", type=int, default=0,
                    help="also save optimizer+RNG for exact resume (req 3)")
    ap.add_argument("--resume_full", default=None,
                    help="path to full_state.pt for EXACT resume")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    ddp = "LOCAL_RANK" in os.environ
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(rank)
        args.device = f"cuda:{rank}"
    device, dtype = args.device, torch.bfloat16
    rng = random.Random(args.seed + rank * 1000)
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    wrap = WanDiffusionWrapper(model_name=args.model_name,
                               timestep_shift=args.timestep_shift,
                               is_causal=False, enable_injectors=True,
                               n_action_dims=11, n_slots=N_HELD)
    if is_main:
        print(f"base {args.model_name} (pretrained i2v) + 11-dim control injectors")

    wrap.model.requires_grad_(True)
    # v2: state_injector is now LIVE (held-item coupling); only the V Rising
    # action cross-attn and weapon injectors stay frozen dead weight.
    for n, p in wrap.model.named_parameters():
        if any(k in n for k in ("action_injector", "weapon_injector")):
            p.requires_grad_(False)
        elif args.smoke_freeze_bb and "injector" not in n:
            p.requires_grad_(False)
    wrap.enable_gradient_checkpointing()
    wrap = wrap.to(device=device, dtype=dtype)
    wrap.model.train()
    if ddp:
        # find_unused=False: every trainable param (backbone + control
        # injectors) participates in every forward -- dropout zeroes control
        # VALUES, not the injector path. Frozen V Rising injectors are
        # excluded from DDP buckets by requires_grad=False. Graph traversal
        # off = measurable step-time win at 32.7k-token attention scale.
        wrap.model = torch.nn.parallel.DistributedDataParallel(
            wrap.model, device_ids=[rank], find_unused_parameters=False)
    scheduler = wrap.get_scheduler()
    bare = wrap.model.module if ddp else wrap.model

    inj_params = [p for n, p in bare.named_parameters()
                  if "injector" in n and p.requires_grad]
    inj_ids = {id(p) for p in inj_params}
    bb_params = [p for p in bare.parameters()
                 if p.requires_grad and id(p) not in inj_ids]
    optimizer = torch.optim.AdamW(
        [{"params": bb_params, "lr": args.lr},
         {"params": inj_params, "lr": args.lr}], weight_decay=0.01)
    params = bb_params + inj_params
    if is_main:
        print(f"param groups: backbone {sum(p.numel() for p in bb_params)/1e6:.0f}M"
              f" @{args.lr}, control injectors "
              f"{sum(p.numel() for p in inj_params)/1e6:.1f}M @{args.lr}")

    start_step = 0
    if args.resume:
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        # v1->v2 shim: v1 checkpoints carry the old 16-slot state_injector
        # (frozen, zero-init, never trained) -- drop shape mismatches so the
        # fresh 30-slot module keeps its zero-init.
        tgt = dict(bare.named_parameters())
        dropped = [k for k, v in rck["trainable"].items()
                   if k in tgt and tgt[k].shape != v.shape]
        for k in dropped:
            del rck["trainable"][k]
        if dropped and is_main:
            print(f"resume shim: dropped {len(dropped)} shape-mismatched "
                  f"tensors (old state_injector)")
        m, u = bare.load_state_dict(rck["trainable"], strict=False)
        assert not u, f"unexpected: {u[:5]}"
        start_step = rck["step"] + 1
        if is_main:
            print(f"resumed at step {start_step}")

    segs = sorted(s for s in os.listdir(ROOT) if not s.startswith(".")
                  and os.path.exists(os.path.join(ROOT, s, "seg_000.npz")))
    val_names = [s for s in json.load(
        open(os.path.join(os.path.dirname(ROOT), "val5.json"))) if s in segs]
    # bright demo-anchor holdout (2026-07-11): excluded from training so
    # demo first frames are honest; NOT part of the val metric (ruler pinned)
    dh = os.path.join(os.path.dirname(ROOT), "demo_holdout.json")
    demo_names = set(json.load(open(dh))) if os.path.exists(dh) else set()
    train_names = [s for s in segs
                   if s not in set(val_names) and s not in demo_names]
    if args.smoke:
        train_names, val_names = train_names[:6], val_names[:1]
    ds_tr, ds_va = MCJointDataset(train_names), MCJointDataset(val_names, stride=21)

    lmb_pool = [i for i, (s, st) in enumerate(ds_tr.items)
                if ds_tr.cache[s]["act"][st:st + F, 7].mean() > 0.2]
    # bright pool from the scene census (surface fraction >= 0.5) -- covers
    # every data batch uniformly; rerun scene_census.py after adding data
    sc = "/root/autodl-tmp/scene_census.json"
    census = json.load(open(sc)) if os.path.exists(sc) else {}
    bright_segs = {s for s, v in census.items() if v.get("surface", 0) >= 0.5}
    bright_pool = [i for i, (s, st) in enumerate(ds_tr.items)
                   if s in bright_segs]
    if is_main:
        print(f"train windows {len(ds_tr)} (LMB-active {len(lmb_pool)}), "
              f"val windows {len(ds_va)}")
    assert len(ds_tr) and len(ds_va) and lmb_pool

    # depletion windows: an in-window count crossing to zero is the causal
    # signal for bag logic -- rare, so oversampled via --depl_frac
    depl_pool = []
    for i, (s, st) in enumerate(ds_tr.items):
        bag = ds_tr.cache[s]["bag"][st:st + F]
        if ((bag[0] > 0) & (bag.min(0) == 0)).any():
            depl_pool.append(i)
    if is_main:
        print(f"depletion windows: {len(depl_pool)}")

    probe_net, bce_logits = None, None
    if args.probe_loss_w > 0:
        from train_mc_probe import MCProbe
        probe_net = MCProbe(32).to(device).float().eval()
        pk = torch.load(args.probe_ckpt, map_location="cpu",
                        weights_only=False)
        probe_net.load_state_dict(pk["model"] if "model" in pk else pk)
        probe_net.requires_grad_(False)          # frozen ruler
        bce_logits = torch.nn.BCEWithLogitsLoss()
        if is_main:
            print(f"coupled loss ON: w={args.probe_loss_w}, "
                  f"t<={args.probe_t_max}, probe={args.probe_ckpt}")

    prompt_embed = load_prompt_context(args.prompt_embed, args.batch_size,
                                       device, dtype)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.ckpt_dir, "train_log.jsonl")
    timesteps = scheduler.timesteps.to(device)

    def sample_batch(ds, bs, pool=None, bpool=None, dpool=None):
        idxs = []
        for _ in range(bs):
            if dpool and rng.random() < args.depl_frac:
                idxs.append(dpool[rng.randrange(len(dpool))])
            elif bpool and rng.random() < args.bright_frac:
                idxs.append(bpool[rng.randrange(len(bpool))])
            elif pool and rng.random() < args.lmb_frac:
                idxs.append(pool[rng.randrange(len(pool))])
            else:
                idxs.append(rng.randrange(len(ds)))
        items = [ds.get(i) for i in idxs]
        return {k: torch.stack([it[k] for it in items]) for k in items[0]}

    last_band = {"tag": "high"}

    def flow_step(batch, train=True, force_uncond=False):
        lat = batch["latent"].to(device=device, dtype=dtype)
        control = batch["action"].to(device=device, dtype=dtype)
        held = batch["held"].to(device)                     # [B,F] long, -1=unk
        B = lat.shape[0]
        dropped = force_uncond or (train and rng.random() < args.control_dropout)
        if dropped:
            control = torch.zeros_like(control)
            held = torch.full_like(held, -1)                # clean uncond arm
            band_idx = (torch.arange(len(timesteps), device=device)
                        if train else
                        (timesteps >= 500).nonzero().flatten())
            last_band["tag"] = "uncond"
        elif train and rng.random() < args.lowband_p:
            band_idx = (timesteps < 500).nonzero().flatten()
            last_band["tag"] = "low"
        else:
            band_idx = (timesteps >= 500).nonzero().flatten()
            last_band["tag"] = "high"
        tid = band_idx[torch.randint(0, len(band_idx), (B,), device=device)]
        t = timesteps[tid].unsqueeze(1).expand(B, F)
        noise = torch.randn_like(lat)
        noisy = scheduler.add_noise(lat.flatten(0, 1), noise.flatten(0, 1),
                                    t.flatten(0, 1)).unflatten(0, (B, F))
        # Fun-InP y: [4ch mask | 16ch image latent], frame 0 conditioned
        y = lat.new_zeros(B, F, 20, *lat.shape[-2:])
        y[:, 0, :4] = 1.0
        y[:, 0, 4:] = batch["imglat0"].to(device=device, dtype=dtype)
        cond = {"prompt_embeds": prompt_embed[:B], "control": control,
                "camera": control[..., 9:11].contiguous(),  # dedicated
                # GameFactory-style pathway (AdaLN camera convicted at step
                # 1600: keys delta 2-9, camera delta 0.5); zeroed along with
                # control on dropout batches since it's a slice of it
                "state_held": held,
                "state_occupancy": (
                    (torch.log1p(batch["bagseq"]).to(device=device, dtype=dtype)
                     if args.perframe_bag
                     else torch.log1p(batch["bag0"]).to(device=device, dtype=dtype)
                     .unsqueeze(1).expand(B, F, N_HELD))
                    if args.bag_cond and not dropped
                    else lat.new_zeros(B, F, N_HELD)),
                "y": y, "clip_fea": lat.new_zeros(B, 257, 1280)}
        flow_pred, pred_x0 = wrap(noisy_image_or_video=noisy,
                                  conditional_dict=cond, timestep=t)
        loss = ((flow_pred - (noise - lat)) ** 2).mean()
        # coupled loss: frozen probe grades the model's own x0 guess against
        # GT events -- gradient flows THROUGH the probe into the generator.
        # Only where the probe's readings are validated (t <= probe_t_max)
        # and never on the uncond arm.
        if args.probe_loss_w > 0 and train and not dropped:
            # per-sample gate (audit #1): each sample draws its own t; only
            # samples inside the probe's validated band contribute
            m = (t[:, 0] <= args.probe_t_max)
            if m.any():
                x = pred_x0[m].float()
                x = torch.cat([x, torch.diff(x, dim=1, prepend=x[:, :1])],
                              dim=2)
                x[:, :, :16][:, :, :, 52:, 26:78] = 0
                x[:, :, 16:][:, :, :, 52:, 26:78] = 0
                lm, lp, lu = probe_net(x)
                ev = batch["events"].to(device)[m]
                lp_loss = (bce_logits(lm, ev[..., 0])
                           + bce_logits(lp, ev[..., 1])
                           + bce_logits(lu, ev[..., 2])) / 3
                loss = loss + args.probe_loss_w * lp_loss
                last_band["probe_loss"] = float(lp_loss.detach())
        return loss

    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(args.grad_accum):
            loss = flow_step(sample_batch(ds_tr, args.batch_size, lmb_pool,
                                          bright_pool, depl_pool)) \
                / args.grad_accum
            loss.backward()
            tot += loss.item()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if is_main:
            with open(log_path, "a") as f:
                rec = {"step": step, "loss": tot, "gnorm": float(gn),
                       "band": last_band["tag"]}
                if "probe_loss" in last_band:
                    rec["probe_loss"] = round(last_band.pop("probe_loss"), 4)
                f.write(json.dumps(rec) + "\n")
            if step % 10 == 0:
                print(f"step {step:6d}  flow={tot:.4f}  gnorm={float(gn):.2f}",
                      flush=True)
        if args.smoke and step - start_step + 1 >= args.smoke:
            print(f"SMOKE-OK final loss {tot:.4f}")
            return

        if step % args.val_every == 0 and step > start_step and is_main:
            # DETERMINISTIC val (2026-07-11): freeze python rng (windows) AND
            # torch rng (noise + t draws) -- the old ±0.01 band was t/noise
            # resampling, which buried every real trend. cond_gap = uncond
            # minus conditioned loss on the SAME noise/t: growth means the
            # model is actually using action/held information.
            wrap.model.eval()
            st_ = rng.getstate()
            cpu_rng = torch.random.get_rng_state()
            gpu_rng = torch.cuda.get_rng_state_all()
            with torch.no_grad():
                rng.seed(1234); torch.manual_seed(4321)
                vl = sum(flow_step(sample_batch(ds_va, args.batch_size),
                                   train=False).item()
                         for _ in range(args.val_batches)) / args.val_batches
                rng.seed(1234); torch.manual_seed(4321)
                vu = sum(flow_step(sample_batch(ds_va, args.batch_size),
                                   train=False, force_uncond=True).item()
                         for _ in range(args.val_batches)) / args.val_batches
            rng.setstate(st_)
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(gpu_rng)
            wrap.model.train()
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "val": vl,
                                    "val_uncond": vu,
                                    "cond_gap": vu - vl}) + "\n")
            print(f"step {step:6d}  VAL flow={vl:.4f}  uncond={vu:.4f}  "
                  f"gap={vu - vl:+.4f}", flush=True)

        if step % args.save_every == 0 and step > start_step and is_main:
            # disk incident 2026-07-12: fp32 全参 7.4G 撑爆 250G 盘。改存
            # bf16(3.7G,模型本就 bf16 训练,无损)。写-then-删顺序保持不变
            # (中途失败也至少留住旧好档)。
            path = os.path.join(args.ckpt_dir, f"step_{step:06d}.pt")
            torch.save({"trainable": {n: p.detach().cpu().to(torch.bfloat16)
                                      for n, p in bare.named_parameters()},
                        "step": step, "args": vars(args)}, path)
            olds = sorted(f for f in os.listdir(args.ckpt_dir)
                          if f.startswith("step_") and f.endswith(".pt"))
            for f in olds[:-args.keep_last_n]:
                os.remove(os.path.join(args.ckpt_dir, f))
            print(f"saved {path}", flush=True)
            # req 3 (2026-07-12): full-state ckpt for EXACT resume on next
            # whole-machine rotation (optimizer moments + RNG). Separate file,
            # only keep the latest (fp32 optimizer ~14G — disk rule: keep 1).
            # Atomic: write .tmp then os.replace. DDP: rank0 only, barrier'd
            # by the is_main guard + save cadence.
            if args.save_full_state:
                fpath = os.path.join(args.ckpt_dir, "full_state.pt")
                tmp = fpath + ".tmp"
                torch.save({"model": bare.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step, "args": vars(args),
                            "py_rng": rng.getstate(),
                            "torch_rng": torch.random.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all()}, tmp)
                os.replace(tmp, fpath)
                print(f"saved full-state {fpath}", flush=True)


if __name__ == "__main__":
    main()

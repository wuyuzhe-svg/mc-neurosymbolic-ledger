import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M4-C (DESIGN.md SS7): real formal training run, full params, starting from the
real pretrained `longlive_base.pt` (DESIGN.md SS2.6 -- NOT ode_init.pt), not
the toy dim=128 / random-init config M1-M4B used for wiring smoke tests.

pickup axis is 方案 B (DESIGN.md SS4.1b): pickup_type only, no slot/dir; the
3-class pickup_color upgrade is deferred to a future checkpoint-resume run.

Session-level train/val split (leak-free -- windows from the same session
share nearby temporal context): one session held out entirely for validation,
never seen in a training batch.

Checkpointing: model + state_head + optimizer + step, resumable via --resume.

Usage (see the printed config block before training starts -- confirm before
launching for real):
    python train_state_layer_m4c.py --data_root data/segments_m4c \\
        --val_session 20260630_210027_051 --ckpt_dir checkpoints/m4c_run1
"""
import argparse
import itertools
import json
import os
import random
import time

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from utils.integrator import shift_state_for_injection
from utils.loss import FlowPredLoss
from utils.scheduler import FlowMatchScheduler
from utils.video_state_dataset import VideoStateClipDataset
from model.state_head import (
    StateHead, state_head_losses, RMSLossNormalizer, combine_state_and_diffusion_loss,
    TYPE_KEYS, MASKED_KEYS,
)
from wan.modules.causal_model import CausalWanModel

DENOISING_STEP_LIST = [1000, 750, 500, 250]  # configs/longlive_train_init.yaml


def label_specs(n_slots):
    # M4-C: pickup has no slot/dir, gather has no dir either (see model/state_head.py's
    # docstring / DESIGN.md SS4.1b -- gather_dir dropped 2026-07-02, real-data audit
    # found it 99.9% constant DIR_UP, same "zero information" pattern as pickup's dir).
    return {
        "gather_type": 2, "gather_slot": n_slots + 1,
        "pickup_type": 2,
        "equip_type": 2, "equip_slot": n_slots + 1,
    }


def sample_blockwise_timesteps(batch_size, window_frames, num_frame_per_block, device, rng):
    """CausalWanModel is NOT uniform_timestep (utils/wan_wrapper.py) -- each block
    gets its own step from the project's real denoising_step_list."""
    n_blocks = window_frames // num_frame_per_block
    t = torch.empty(batch_size, window_frames, device=device)
    for b in range(batch_size):
        for blk in range(n_blocks):
            step = rng.choice(DENOISING_STEP_LIST)
            t[b, blk * num_frame_per_block:(blk + 1) * num_frame_per_block] = step
    return t


def load_pretrained_generator(model, ckpt_path):
    """DESIGN.md SS2.6: from longlive_base.pt, not ode_init.pt. The checkpoint's
    "generator" state dict is a wrapper's (pipeline.generator's), prefixed
    "model." over every CausalWanModel param name -- strip it. strict=False
    because state_injector/weapon_injector (M2/M3) and the whole StateHead
    don't exist in this checkpoint; verify the "missing" set is EXACTLY those
    new zero-init modules, nothing else, so a genuine base-model loading bug
    doesn't silently pass as "expected new params"."""
    sd = torch.load(ckpt_path, map_location="cpu")["generator"]
    sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    new_modules = ("state_injector", "weapon_injector", "action_injector", "control_injector")
    bad_missing = [k for k in missing if not any(n in k for n in new_modules)]
    if bad_missing:
        raise RuntimeError(f"unexpected missing keys loading {ckpt_path} (not new M2/M3 params): {bad_missing[:10]}")
    if unexpected:
        raise RuntimeError(f"unexpected keys in {ckpt_path} that don't exist on the model: {unexpected[:10]}")
    print(f"loaded {ckpt_path}: all {len(sd)} pretrained tensors matched into the base model, "
          f"{len(missing)} new tensors left at their construction-time zero-init "
          f"(state_injector/weapon_injector/action_injector -- expected, not in this checkpoint)")


def save_checkpoint(path, model, state_head, optimizer, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "state_head": state_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }, path)


def build_batch_tensors(batch, device, dtype, scheduler, args, rng, train=True):
    B = batch["latent"].shape[0]
    clean_x = batch["context_latent"].to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
    target = batch["latent"].to(device=device, dtype=dtype)

    t = sample_blockwise_timesteps(B, args.window_frames, args.num_frame_per_block, device, rng)
    aug_t = torch.zeros_like(t)
    noise = torch.randn_like(target)
    noisy_x = scheduler.add_noise(
        target.flatten(0, 1), noise.flatten(0, 1), t.flatten(0, 1).long()
    ).unflatten(0, (B, args.window_frames)).permute(0, 2, 1, 3, 4)

    # M4-C: real text prompt/context is NOT wired up yet (DESIGN.md SS7 M4-C note) --
    # zero context, not random, so it's at least a stable, information-free signal
    # rather than noise the model has to learn to ignore.
    context = torch.zeros(B, args.text_len, 4096, device=device, dtype=dtype)

    held_full = torch.cat([batch["context_held"], batch["held"]], dim=1).to(device)
    weapon_full = torch.cat([batch["context_held_weapon_id"], batch["held_weapon_id"]], dim=1).to(device)
    count_full = torch.cat([batch["context_count"], batch["count"]], dim=1).to(device)
    state_held, weapon_id, state_occupancy = shift_state_for_injection(held_full, weapon_full, count_full)
    state_occupancy = state_occupancy.to(dtype)

    # 通路③ 方案X: 6-dim control vector = [W,A,S,D, offset_x/σ, offset_z/σ] --
    # WASD hold-state + std-normalized world-plane cursor offset (cursor label dims
    # 0:2 post the 2026-07-02 schema change, see segment_cursor; dim 2 = normalized
    # distance, deliberately NOT in the control vector yet). Clean+noisy concatenated
    # like the S_t tensors above but NOT shifted -- input t conditions frame t.
    # Control-CFG (Matrix-Game/GameFactory recipe): with prob args.action_cfg_dropout,
    # zero the whole control tensor for a sample, so inference-time CFG has a trained
    # unconditional branch. (WASD all-zero conflates with genuine "no keys held" --
    # accepted 2026-07-02; the cursor offset's zero vector is out-of-manifold, cleaner.)
    action_full = torch.cat([batch["context_action"], batch["action"]], dim=1)
    cursor_full = torch.cat([batch["context_cursor"], batch["cursor"]], dim=1)
    control_full = torch.cat([action_full, cursor_full[..., :2]], dim=-1).to(device=device, dtype=dtype)
    if train and args.action_cfg_dropout > 0:
        for b in range(B):
            if rng.random() < args.action_cfg_dropout:
                control_full[b] = 0.0

    return (clean_x, target, t, aug_t, noise, noisy_x, context,
            state_held, weapon_id, state_occupancy, control_full)


def run_step(model, state_head, batch, device, dtype, scheduler, diffusion_loss_fn,
             class_weights, normalizer, args, rng, seq_len, train=True):
    (clean_x, target, t, aug_t, noise, noisy_x, context,
     state_held, weapon_id, state_occupancy, control_full) = \
        build_batch_tensors(batch, device, dtype, scheduler, args, rng, train=train)

    # 方案X (adaln) vs 方案Y (xattn) switch: same 6-dim tensor, different injection arm.
    control_kwargs = ({"control": control_full} if args.control_inject_mode == "adaln"
                      else {"action": control_full})
    x_pred, hidden = model._forward_train(
        noisy_x, t=t, context=context, seq_len=seq_len, clean_x=clean_x, aug_t=aug_t,
        return_hidden=True, state_held=state_held, state_occupancy=state_occupancy, weapon_id=weapon_id,
        **control_kwargs,
    )
    x_pred = x_pred.permute(0, 2, 1, 3, 4)

    diffusion_loss = diffusion_loss_fn(
        x=target, x_pred=None, noise=noise, noise_pred=None,
        alphas_cumprod=None, timestep=t, flow_pred=x_pred,
    )
    labels = {k: batch[k].to(device) for k in TYPE_KEYS + MASKED_KEYS}
    logits = state_head(hidden)
    ce_losses = state_head_losses(logits, labels, class_weights=class_weights)
    total, normalized_ce, normalized_diff = combine_state_and_diffusion_loss(
        ce_losses, diffusion_loss, normalizer, lambda_ce=args.lambda_ce,
    )
    return total, diffusion_loss, ce_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/segments_m4c")
    ap.add_argument("--val_session", default="20260630_210027_051",
                     help="session id held out entirely for validation (default: smallest "
                          "session, ~26.4/341.6 min ~= 7.7%% of total data)")
    ap.add_argument("--ckpt_path", default="checkpoints/models/longlive_base.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints/m4c_run1")
    ap.add_argument("--resume", default=None, help="path to a checkpoint saved by this script to resume from")
    ap.add_argument("--window_frames", type=int, default=21)
    ap.add_argument("--num_frame_per_block", type=int, default=3)
    ap.add_argument("--local_attn_size", type=int, default=12)
    ap.add_argument("--sink_size", type=int, default=3)
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--n_weapons", type=int, default=16)
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr", type=float, default=2e-6)  # backbone, matches configs/longlive_train_init.yaml
    ap.add_argument("--lr_new_params", type=float, default=1e-4,
                     help="StateHead + state_injector/weapon_injector (fresh zero-init, no pretrained "
                          "signal to preserve) -- much higher than --lr or they barely move overnight")
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--action_cfg_dropout", type=float, default=0.1,
                    help="通路③ control-CFG: per-sample prob of zeroing the 6-dim control "
                         "conditioning during training (Matrix-Game/GameFactory recipe)")
    ap.add_argument("--control_inject_mode", choices=["adaln", "xattn"], default="adaln",
                    help="通路③ arm: adaln = 方案X ControlAdaLNInjector (主方案), "
                         "xattn = 方案Y ActionCrossAttnInjector (消融对照)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--keep_last_n", type=int, default=3,
                     help="rotate numbered checkpoints, keep only the last N (~27GB each -- "
                          "178GB free / 27GB ~= 6 unrotated saves before the disk fills)")
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--val_batches", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device
    dtype = torch.bfloat16

    print("=" * 100)
    print("M4-C TRAINING CONFIG")
    print("=" * 100)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:20s} {v}")
    print("=" * 100)

    train_ds = VideoStateClipDataset(
        args.data_root, window_frames=args.window_frames, num_frame_per_block=args.num_frame_per_block,
        context_frames=args.window_frames, exclude_sessions={args.val_session},
    )
    val_ds = VideoStateClipDataset(
        args.data_root, window_frames=args.window_frames, num_frame_per_block=args.num_frame_per_block,
        context_frames=args.window_frames, include_sessions={args.val_session},
    )
    print(f"train: {len(train_ds.clips)} clip(s), {len(train_ds)} windows "
          f"(sessions: {sorted(set(c['id'].split('/')[0] for c in train_ds.clips))})")
    print(f"val:   {len(val_ds.clips)} clip(s), {len(val_ds)} windows (session: {args.val_session})")

    static_weights = train_ds.compute_window_sample_weights()
    sampler = WeightedRandomSampler(static_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True)

    class_weights = train_ds.compute_inverse_freq_class_weights(label_specs(args.n_slots))
    print("inverse-freq class weights (SS5.2):")
    for k, w in class_weights.items():
        print(f"  {k:14s} {w.numpy().round(3)}")
    class_weights = {k: v.to(device=device, dtype=dtype) for k, v in class_weights.items()}

    model = CausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=args.text_len, in_dim=16,
        dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
        num_heads=12, num_layers=30, local_attn_size=args.local_attn_size, sink_size=args.sink_size,
        qk_norm=True, cross_attn_norm=True, n_slots=args.n_slots, n_weapons=args.n_weapons,
        n_action_dims=6,  # 通路③ 方案X: [W,A,S,D, offset_x/σ, offset_z/σ] (sizes BOTH arms)
    ).to(device=device, dtype=dtype)
    # Real full-size (2.27B param) forward+backward on window_frames=21 OOMs at ~94GB/95GB
    # without this (smoke-tested 2026-07-02) -- the 30-block loop in _forward_train supports
    # torch.utils.checkpoint per block (trades recompute for the block's activation memory);
    # flip it on for this single-GPU run.
    model.gradient_checkpointing = True
    model.num_frame_per_block = args.num_frame_per_block
    model.train()

    state_head = StateHead(dim=1536, n_slots=args.n_slots).to(device=device, dtype=dtype)
    state_head.train()

    start_step = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        state_head.load_state_dict(ck["state_head"])
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")
    else:
        load_pretrained_generator(model, args.ckpt_path)

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    diffusion_loss_fn = FlowPredLoss()
    normalizer = RMSLossNormalizer()

    # Differential LR: the pretrained backbone should only shift gently (args.lr,
    # configs/longlive_train_init.yaml's own value for fine-tuning this exact base
    # model) -- but StateHead + state_injector/weapon_injector are FRESH, zero-init
    # modules with no pretrained signal to preserve. At a single lr=2e-6 (Adam's
    # per-parameter normalization aside), they'd move too slowly to show any real
    # gather-axis signal over one night's steps, which defeats tonight's actual goal
    # (verify gather drift elimination). args.lr_new_params gives them a much larger,
    # still-conservative step size; --lr_new_params 0 (or equal to --lr) recovers the
    # original single-LR behavior if you'd rather not use this.
    new_param_names = ("state_injector", "weapon_injector", "action_injector", "control_injector")
    backbone_params, new_params = [], []
    for name, p in model.named_parameters():
        (new_params if any(n in name for n in new_param_names) else backbone_params).append(p)
    param_groups = [
        {"params": backbone_params, "lr": args.lr},
        {"params": new_params, "lr": args.lr_new_params},
        {"params": state_head.parameters(), "lr": args.lr_new_params},
    ]
    optimizer = torch.optim.AdamW(param_groups)
    if args.resume:
        optimizer.load_state_dict(ck["optimizer"])

    frame_seqlen = (60 // 2) * (104 // 2)
    seq_len = args.window_frames * frame_seqlen + 128

    log_path = os.path.join(args.ckpt_dir, "train_log.jsonl")
    os.makedirs(args.ckpt_dir, exist_ok=True)

    train_iter = itertools.cycle(train_loader)
    val_iter = itertools.cycle(val_loader)
    for step in range(start_step, args.steps):
        t0 = time.time()
        batch = next(train_iter)
        optimizer.zero_grad()
        total, diffusion_loss, ce_losses = run_step(
            model, state_head, batch, device, dtype, scheduler, diffusion_loss_fn,
            class_weights, normalizer, args, rng, seq_len, train=True,
        )
        total.backward()
        optimizer.step()
        dt = time.time() - t0

        if step % args.log_every == 0:
            gather_str = " ".join(f"{k}={v.item():.4f}" for k, v in ce_losses.items() if k.startswith("gather"))
            print(f"step {step:6d}  total={total.item():.4f}  diffusion={diffusion_loss.item():.4f}  "
                  f"[GATHER] {gather_str}  pickup_type={ce_losses['pickup_type'].item():.4f}  "
                  f"equip_type={ce_losses['equip_type'].item():.4f} equip_slot={ce_losses['equip_slot'].item():.4f}  "
                  f"({dt:.2f}s/step)")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "split": "train", "total": total.item(),
                    "diffusion": diffusion_loss.item(),
                    **{k: v.item() for k, v in ce_losses.items()},
                }) + "\n")

        if step % args.val_every == 0 and step > 0:
            model.eval()
            state_head.eval()
            val_totals = []
            val_ce = {k: [] for k in TYPE_KEYS + MASKED_KEYS}
            with torch.no_grad():
                for _ in range(args.val_batches):
                    vbatch = next(val_iter)
                    vtotal, vdiff, vce = run_step(
                        model, state_head, vbatch, device, dtype, scheduler, diffusion_loss_fn,
                        class_weights, normalizer, args, rng, seq_len, train=False,
                    )
                    val_totals.append(vtotal.item())
                    for k, v in vce.items():
                        val_ce[k].append(v.item())
            mean_total = sum(val_totals) / len(val_totals)
            gather_val_str = " ".join(f"{k}={sum(v)/len(v):.4f}" for k, v in val_ce.items() if k.startswith("gather"))
            print(f"  [VAL step {step}] total={mean_total:.4f}  [GATHER] {gather_val_str}")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "split": "val", "total": mean_total,
                    **{k: sum(v) / len(v) for k, v in val_ce.items()},
                }) + "\n")
            model.train()
            state_head.train()

        if step % args.save_every == 0 and step > 0:
            # ~27GB/checkpoint (2.27B params bf16 + AdamW fp32 moments) -- writing a
            # full second copy for "latest.pt" every save would double disk I/O for no
            # reason; symlink instead. Unattended overnight run WILL fill the disk
            # without rotation (178GB free / 27GB ~= 6 saves) -- keep only the last
            # --keep_last_n numbered checkpoints.
            ckpt_path = os.path.join(args.ckpt_dir, f"step_{step:06d}.pt")
            save_checkpoint(ckpt_path, model, state_head, optimizer, step, args)
            latest_path = os.path.join(args.ckpt_dir, "latest.pt")
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(ckpt_path), latest_path)
            print(f"  [CKPT] saved {ckpt_path} ({os.path.getsize(ckpt_path)/1e9:.1f} GB)")

            old_ckpts = sorted(
                p for p in os.listdir(args.ckpt_dir)
                if p.startswith("step_") and p.endswith(".pt")
            )
            for stale in old_ckpts[:-args.keep_last_n]:
                os.remove(os.path.join(args.ckpt_dir, stale))
                print(f"  [CKPT] rotated out {stale}")


if __name__ == "__main__":
    main()

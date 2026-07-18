import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
M4-D (方案B, 2026-07-03): FROZEN-backbone injector + count-regression training.

Why this exists (full diagnosis: demo_out/ + DESIGN.md SS7b): the M4-C run
fine-tuned all params with plain flow-matching loss and destroyed the base's
distilled few-step rollout (base generates coherently through the same
inference code; the fine-tuned weights produce texture mush at 4 AND 20
steps). Four fixes, all structural:

1. FREEZE the entire pretrained backbone -- trainable params are ONLY the
   three zero-init injection side-branches (①state adaLN, ②weapon x-attn,
   ③control adaLN) + StateHeadV2. Rollout capability is preserved by
   construction (GameFactory's phase-1 recipe). No lr knob can "gently"
   avoid the distillation loss -- the flow objective's optimum IS the
   blurry many-step model, so the backbone must not follow it at all.
2. WARPED denoising steps [1000, 937.5, 833.33, 625] -- the base's real
   distilled operating points (configs/*.yaml warp_denoising_step: true maps
   the literal [1000,750,500,250] through scheduler.timesteps[1000-step];
   M4-C trained at the unwarped literals, off the operating points).
3. REAL prompt embedding (data/prompt_embed.pt, fixed V Rising description)
   -- zero context is out-of-distribution for the base (verified: base +
   zero context renders black).
4. StateHeadV2: per-BLOCK event-count regression (Poisson) with attention-
   pooled readout + (1-sigma) noise-level loss weights, replacing per-frame
   instant classification (see model/state_head.py StateHeadV2 docstring for
   the evidence trail).

Checkpoints store ONLY trainable tensors (backbone is bit-identical to
longlive_base.pt forever) -- ~10GB with optimizer instead of 17.5GB.

Usage:
    python train_state_layer_m4d.py --data_root data/segments_m4c \\
        --val_session 20260630_210027_051 --ckpt_dir checkpoints/m4d_run1
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
    StateHeadV2, StateHeadV3, build_block_count_labels, state_head_v2_losses, RMSLossNormalizer,
)
from train_state_layer_m4c import load_pretrained_generator
from wan.modules.causal_model import CausalWanModel

# The base's DISTILLED operating points: configs/longlive_train_init.yaml's
# [1000,750,500,250] warped through scheduler.timesteps[1000-step] at shift=5
# (sigma = 1.0, 0.9375, 0.8333, 0.625). M4-C's literal list was the bug.
DENOISING_STEP_LIST = [1000.0, 937.5, 833.33, 625.0]
TRAINABLE_PATTERNS = ("state_injector", "weapon_injector", "action_injector", "control_injector")
COUNT_KEYS = ("gather_count", "pickup_count", "equip_count", "equip_slot")


def sample_blockwise_timesteps(batch_size, window_frames, num_frame_per_block, device, rng,
                               t0_p=0.0):
    """t0_p: per-block probability of t=0 (clean readout block, mirrors the
    deployment KV-recache pass). Diffusion loss is masked on those blocks --
    they exist so the count head sees frames it can actually read (SS7c)."""
    n_blocks = window_frames // num_frame_per_block
    t = torch.empty(batch_size, window_frames, device=device)
    for b in range(batch_size):
        for blk in range(n_blocks):
            step = 0.0 if (t0_p > 0 and rng.random() < t0_p) else rng.choice(DENOISING_STEP_LIST)
            t[b, blk * num_frame_per_block:(blk + 1) * num_frame_per_block] = step
    return t


def block_noise_weights(t, num_frame_per_block, scheduler, device):
    """[B, F] timesteps -> [B, nb] count-loss weights = (1-sigma), exact 1.0 at t=0.
    The head can only count events it can SEE: the deployment readout point is
    the clean t=0 recache pass (SS7c), so clean blocks carry full weight and
    sigma=0.625 blocks only 0.375 -- otherwise the 3:1 majority of unreadable
    noisy blocks keeps dragging the head back to the base-rate solution."""
    tb = t[:, ::num_frame_per_block]  # one step per block
    timesteps = scheduler.timesteps.to(device)
    sigmas = scheduler.sigmas.to(device)
    tid = torch.argmin((timesteps.unsqueeze(0) - tb.flatten().unsqueeze(1).float()).abs(), dim=1)
    sigma = sigmas[tid].reshape(tb.shape)
    sigma = torch.where(tb == 0.0, torch.zeros_like(sigma), sigma)  # exact-clean swap in build_batch_tensors
    # clean-only: noisy blocks' count-loss optimum IS the base rate, which
    # actively drags weak-signal axes (gather) away from event locking --
    # deployment reads at t=0 only, so train the head there only
    return (sigma == 0.0).to(sigma.dtype)


def freeze_backbone(model):
    """SS7b fix 1: requires_grad only on the injection side-branches. Returns
    (n_trainable, n_frozen) tensor counts for the config printout / assertions."""
    n_train = n_frozen = 0
    for name, p in model.named_parameters():
        trainable = any(pat in name for pat in TRAINABLE_PATTERNS)
        p.requires_grad_(trainable)
        n_train += trainable
        n_frozen += not trainable
    return n_train, n_frozen


def trainable_state_dict(model):
    return {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}


def save_checkpoint(path, model, state_head, optimizer, step, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "trainable": trainable_state_dict(model),  # injectors only; backbone == longlive_base.pt
        "state_head": state_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
    }, path)


def load_prompt_context(path, batch_size, device, dtype):
    ck = torch.load(path, map_location="cpu")
    emb = ck["embed"].to(device=device, dtype=dtype)  # [text_len, 4096]
    print(f"prompt context: {ck['prompt']!r}")
    return emb.unsqueeze(0).expand(batch_size, -1, -1)


def build_batch_tensors(batch, device, dtype, scheduler, args, rng, prompt_embed, train=True):
    B = batch["latent"].shape[0]
    clean_x = batch["context_latent"].to(device=device, dtype=dtype).permute(0, 2, 1, 3, 4)
    target = batch["latent"].to(device=device, dtype=dtype)

    t = sample_blockwise_timesteps(B, args.window_frames, args.num_frame_per_block, device, rng,
                                   t0_p=getattr(args, "readout_t0_p", 0.0))
    aug_t = torch.zeros_like(t)
    noise = torch.randn_like(target)
    noisy_x = scheduler.add_noise(
        target.flatten(0, 1), noise.flatten(0, 1), t.flatten(0, 1)
    ).unflatten(0, (B, args.window_frames)).permute(0, 2, 1, 3, 4)
    # t=0 blocks are the readout regime: feed exactly-clean frames (the scheduler's
    # nearest sigma is 0.005, close but not the deployment recache condition)
    t0_mask = (t == 0.0)
    if t0_mask.any():
        noisy_x = noisy_x.permute(0, 2, 1, 3, 4).clone()
        noisy_x[t0_mask] = target[t0_mask]
        noisy_x = noisy_x.permute(0, 2, 1, 3, 4)

    context = prompt_embed[:B]

    held_full = torch.cat([batch["context_held"], batch["held"]], dim=1).to(device)
    weapon_full = torch.cat([batch["context_held_weapon_id"], batch["held_weapon_id"]], dim=1).to(device)
    count_full = torch.cat([batch["context_count"], batch["count"]], dim=1).to(device)
    state_held, weapon_id, state_occupancy = shift_state_for_injection(held_full, weapon_full, count_full)
    if getattr(args, "drop_count_inject", False):
        # resource-slot counts are visually inert (HUD digits at most) and their
        # per-gather jitter is pure interference for the injectors + a weak leak
        # channel; zero them out (= constant no-information conditioning)
        state_occupancy = torch.zeros_like(state_occupancy)
    if getattr(args, "state_inject_granularity", "frame") == "block":
        # deployment contract: belief state updates only at block boundaries
        # (readout runs after a block completes), so inject each block's
        # start-of-block state for all its frames -- kills the within-block
        # transition leak that let the equip head read its own conditioning
        F = state_held.shape[1]
        fpb = args.num_frame_per_block
        idx = (torch.arange(F, device=state_held.device) // fpb) * fpb
        state_held = state_held[:, idx]
        weapon_id = weapon_id[:, idx]
        state_occupancy = state_occupancy[:, idx]
    state_occupancy = state_occupancy.to(dtype)

    action_full = torch.cat([batch["context_action"], batch["action"]], dim=1)
    cursor_full = torch.cat([batch["context_cursor"], batch["cursor"]], dim=1)
    control_full = torch.cat([action_full, cursor_full[..., :2]], dim=-1).to(device=device, dtype=dtype)
    if train and args.action_cfg_dropout > 0:
        for b in range(B):
            if rng.random() < args.action_cfg_dropout:
                control_full[b] = 0.0

    labels = build_block_count_labels(
        batch["count"].to(device), batch["pickup_count"].to(device),
        batch["equip_type"].to(device), batch["equip_slot"].to(device),
        args.num_frame_per_block,
        batch["context_count"][:, -1].to(device), batch["context_pickup_count"][:, -1].to(device),
    )
    return (clean_x, target, t, aug_t, noise, noisy_x, context,
            state_held, weapon_id, state_occupancy, control_full, labels)


def run_step(model, state_head, batch, device, dtype, scheduler, diffusion_loss_fn,
             normalizer, args, rng, seq_len, prompt_embed, train=True):
    (clean_x, target, t, aug_t, noise, noisy_x, context,
     state_held, weapon_id, state_occupancy, control_full, labels) = \
        build_batch_tensors(batch, device, dtype, scheduler, args, rng, prompt_embed, train=train)

    control_kwargs = ({"control": control_full} if args.control_inject_mode == "adaln"
                      else {"action": control_full})
    x_pred, hidden_tokens = model._forward_train(
        noisy_x, t=t, context=context, seq_len=seq_len, clean_x=clean_x, aug_t=aug_t,
        return_hidden="tokens", state_held=state_held, state_occupancy=state_occupancy,
        weapon_id=weapon_id, **control_kwargs,
    )
    x_pred = x_pred.permute(0, 2, 1, 3, 4)

    denoise_frames = t > 0.0  # [B, F]; flow target is undefined-for-training at t=0
    diffusion_loss = diffusion_loss_fn(
        x=target, x_pred=None, noise=noise, noise_pred=None,
        alphas_cumprod=None, timestep=t, flow_pred=x_pred,
        gradient_mask=None if denoise_frames.all() else denoise_frames,
    ) if denoise_frames.any() else x_pred.sum() * 0.0
    outputs = state_head(hidden_tokens)
    weights = block_noise_weights(t, args.num_frame_per_block, scheduler, device)
    count_losses = state_head_v2_losses(outputs, labels, block_weights=weights)

    normalized = {k: normalizer.normalize(k, v) for k, v in count_losses.items()}
    normalized_diff = normalizer.normalize("diffusion", diffusion_loss)
    total = normalized_diff + args.lambda_ce * sum(normalized.values())
    return total, diffusion_loss, count_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/segments_m4c")
    ap.add_argument("--val_session", default="20260630_210027_051")
    ap.add_argument("--ckpt_path", default="checkpoints/models/longlive_base.pt")
    ap.add_argument("--prompt_embed", default="data/prompt_embed.pt")
    ap.add_argument("--ckpt_dir", default="checkpoints/m4d_run1")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--window_frames", type=int, default=21)
    ap.add_argument("--num_frame_per_block", type=int, default=3)
    ap.add_argument("--local_attn_size", type=int, default=12)
    ap.add_argument("--sink_size", type=int, default=3)
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--n_weapons", type=int, default=16)
    ap.add_argument("--text_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=100000)
    ap.add_argument("--lr_new_params", type=float, default=1e-4,
                    help="injectors + StateHeadV2 (the ONLY trainable params -- "
                         "backbone frozen, no backbone lr exists in M4-D)")
    ap.add_argument("--lambda_ce", type=float, default=1.0)
    ap.add_argument("--action_cfg_dropout", type=float, default=0.1)
    ap.add_argument("--control_inject_mode", choices=["adaln", "xattn"], default="adaln")
    ap.add_argument("--head_version", type=int, choices=[2, 3], default=2,
                    help="3 = StateHeadV3 (feature-interaction adapter, SS7d fallback)")
    ap.add_argument("--drop_count_inject", action="store_true",
                    help="zero the resource-slot count injection (visually inert, "
                         "interference-only conditioning)")
    ap.add_argument("--state_inject_granularity", choices=["frame", "block"], default="frame",
                    help="block = inject start-of-block state for the whole block "
                         "(deployment contract; kills the equip conditioning leak)")
    ap.add_argument("--reinit_head", action="store_true",
                    help="on resume: load injectors but re-initialize StateHeadV2 and drop "
                         "optimizer state (escape hatch from the base-rate local minimum)")
    ap.add_argument("--readout_t0_p", type=float, default=0.25,
                    help="per-block probability of a clean (t=0) readout block: count-head "
                         "loss only, diffusion loss masked; mirrors the rollout KV-recache "
                         "pass the head reads at deployment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--keep_last_n", type=int, default=5)
    ap.add_argument("--val_every", type=int, default=200)
    ap.add_argument("--val_batches", type=int, default=32)
    ap.add_argument("--log_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = args.device
    dtype = torch.bfloat16

    print("=" * 100)
    print("M4-D TRAINING CONFIG (方案B: frozen backbone, warped steps, real prompt, count regression)")
    print("=" * 100)
    for k, v in sorted(vars(args).items()):
        print(f"  {k:20s} {v}")
    print(f"  {'DENOISING_STEPS':20s} {DENOISING_STEP_LIST}  (warped -- the base's operating points)")
    print("=" * 100)

    train_ds = VideoStateClipDataset(
        args.data_root, window_frames=args.window_frames, num_frame_per_block=args.num_frame_per_block,
        context_frames=args.window_frames, exclude_sessions={args.val_session},
    )
    val_ds = VideoStateClipDataset(
        args.data_root, window_frames=args.window_frames, num_frame_per_block=args.num_frame_per_block,
        context_frames=args.window_frames, include_sessions={args.val_session},
    )
    print(f"train: {len(train_ds.clips)} clip(s), {len(train_ds)} windows")
    print(f"val:   {len(val_ds.clips)} clip(s), {len(val_ds)} windows (session: {args.val_session})")

    static_weights = train_ds.compute_window_sample_weights()
    sampler = WeightedRandomSampler(static_weights, num_samples=len(train_ds), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = CausalWanModel(
        model_type="t2v", patch_size=(1, 2, 2), text_len=args.text_len, in_dim=16,
        dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
        num_heads=12, num_layers=30, local_attn_size=args.local_attn_size, sink_size=args.sink_size,
        qk_norm=True, cross_attn_norm=True, n_slots=args.n_slots, n_weapons=args.n_weapons,
        n_action_dims=6,
    ).to(device=device, dtype=dtype)
    model.gradient_checkpointing = True
    model.num_frame_per_block = args.num_frame_per_block

    load_pretrained_generator(model, args.ckpt_path)
    n_train_t, n_frozen_t = freeze_backbone(model)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"frozen backbone: {n_frozen/1e9:.3f}B params frozen ({n_frozen_t} tensors), "
          f"{n_trainable/1e6:.1f}M injector params trainable ({n_train_t} tensors)")
    model.train()

    head_cls = StateHeadV3 if args.head_version == 3 else StateHeadV2
    state_head = head_cls(
        dim=1536, n_slots=args.n_slots, num_frame_per_block=args.num_frame_per_block,
    ).to(device=device, dtype=dtype)
    state_head.train()
    n_head = sum(p.numel() for p in state_head.parameters())
    print(f"{head_cls.__name__}: {n_head/1e6:.1f}M params")

    start_step = 0
    optimizer = torch.optim.AdamW(
        [{"params": trainable_params, "lr": args.lr_new_params},
         {"params": state_head.parameters(), "lr": args.lr_new_params}],
    )
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        ck_hv = ck["args"].get("head_version", 2)
        assert args.reinit_head or ck_hv == args.head_version, \
            f"checkpoint was trained with head_version={ck_hv}, pass --head_version {ck_hv} (or --reinit_head)"
        missing, unexpected = model.load_state_dict(ck["trainable"], strict=False)
        assert not unexpected, f"unexpected keys in resume trainable dict: {unexpected[:5]}"
        loaded = set(ck["trainable"].keys())
        should = {n for n, p in model.named_parameters() if p.requires_grad}
        assert loaded == should, f"resume trainable set mismatch: {sorted(should ^ loaded)[:5]}"
        if args.reinit_head:
            # keep the injectors but restart the count head from its construction
            # init: the sigma>=0.625 regime taught it to ignore its input entirely
            # (attention pooling collapsed to a constant base rate), and that is a
            # local minimum not worth betting on escaping. Optimizer state is
            # dropped too (momentum refers to the old head params).
            print("reinit_head: state head left at fresh init, optimizer state dropped")
        else:
            state_head.load_state_dict(ck["state_head"])
            optimizer.load_state_dict(ck["optimizer"])
        start_step = ck["step"] + 1
        print(f"resumed from {args.resume} at step {start_step}")

    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(1000, training=True)
    diffusion_loss_fn = FlowPredLoss()
    normalizer = RMSLossNormalizer()
    prompt_embed = load_prompt_context(args.prompt_embed, args.batch_size, device, dtype)

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
        total, diffusion_loss, count_losses = run_step(
            model, state_head, batch, device, dtype, scheduler, diffusion_loss_fn,
            normalizer, args, rng, seq_len, prompt_embed, train=True,
        )
        total.backward()
        optimizer.step()
        dt = time.time() - t0

        if step % args.log_every == 0:
            loss_str = " ".join(f"{k}={v.item():.4f}" for k, v in count_losses.items())
            print(f"step {step:6d}  total={total.item():.4f}  diffusion={diffusion_loss.item():.4f}  "
                  f"{loss_str}  ({dt:.2f}s/step)")
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "split": "train", "total": total.item(),
                    "diffusion": diffusion_loss.item(),
                    **{k: v.item() for k, v in count_losses.items()},
                }) + "\n")

        if step % args.val_every == 0 and step > 0:
            model.eval()
            state_head.eval()
            val_totals, val_diff = [], []
            val_losses = {k: [] for k in COUNT_KEYS}
            with torch.no_grad():
                for _ in range(args.val_batches):
                    vbatch = next(val_iter)
                    vtotal, vdiffusion, vcount = run_step(
                        model, state_head, vbatch, device, dtype, scheduler, diffusion_loss_fn,
                        normalizer, args, rng, seq_len, prompt_embed, train=False,
                    )
                    val_totals.append(vtotal.item())
                    val_diff.append(vdiffusion.item())
                    for k, v in vcount.items():
                        val_losses[k].append(v.item())
            means = {k: sum(v) / len(v) for k, v in val_losses.items()}
            print(f"  [VAL step {step}] total={sum(val_totals)/len(val_totals):.4f} "
                  f"diffusion={sum(val_diff)/len(val_diff):.4f} "
                  + " ".join(f"{k}={v:.4f}" for k, v in means.items()))
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "split": "val",
                    "total": sum(val_totals) / len(val_totals),
                    "diffusion": sum(val_diff) / len(val_diff), **means,
                }) + "\n")
            model.train()
            state_head.train()

        if step % args.save_every == 0 and step > 0:
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

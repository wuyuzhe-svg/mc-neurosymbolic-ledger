#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate fake (latent, .npz label) clip pairs for developing the M0
dataloader (see DESIGN_1.md 3.2/3.3) without needing real V Rising footage yet.

There is no `test1_clip.npz` in this repo (checked -- no .npz files exist
anywhere in the tree). Instead of hand-rolling a separate fake-label format,
this script synthesizes plausible raw input/inventory logs and feeds them
through `preprocess_vrising_final.process_clip()` (the *real* pipeline, via
its `_injected=` test hook), so the resulting .npz has exactly the schema,
masking, and none_kind semantics real preprocessing would produce -- not an
approximation of it.

The paired "latent" is `randn(L, 16, 60, 104)`, except element [lf, 0, 0, 0]
is overwritten with `lf` itself (its own global latent-frame index). That's
the alignment self-check hook used by check_video_state_dataset.py: after
windowing, latent_window[i, 0, 0, 0] must equal (start + i) exactly. This
marker only exists in fake data -- real VAE-encoded latents obviously won't
have it, and the dataloader itself never looks at it.

Usage:
    python utils/make_fake_clips.py --out_dir data/fake_clips --n_clips 3
"""
import argparse
import os

import numpy as np

import preprocess_vrising_final as pvf


def _synth_logs(n_frames, cfg, n_events, rng):
    """Build (inputs, inv) logs that process_clip()'s _injected hook expects."""
    heartbeat_dt = 0.5
    duration = n_frames / cfg.fps
    heartbeat_times = list(np.arange(0.0, duration, heartbeat_dt))
    if heartbeat_times[-1] < duration - 1e-6:
        heartbeat_times.append(duration)

    margin = 2.0
    usable = duration - 2 * margin
    if usable <= 0:
        event_times = []
    else:
        event_times = sorted(rng.uniform(margin, margin + usable, size=n_events))

    events = []
    for t in event_times:
        axis = rng.choice(["gather", "pickup", "equip"])
        if axis == "equip":
            slot = int(rng.randint(0, min(4, cfg.n_slots)))
        else:
            slot = int(rng.randint(0, cfg.n_slots))
        events.append((t, axis, slot))

    counts = [0] * cfg.n_slots
    held = None
    held_g = None
    inv = []
    inputs = []
    ei = 0
    for hb_t in heartbeat_times:
        while ei < len(events) and events[ei][0] <= hb_t:
            t, axis, slot = events[ei]
            if axis == "gather":
                counts[slot] += int(rng.randint(1, 4))
                inputs.append(pvf.InputEvent(t=t, key="mouse0"))
            elif axis == "pickup":
                counts[slot] += int(rng.randint(1, 4))
                inputs.append(pvf.InputEvent(t=t, key="f"))
            else:  # equip
                held = slot
                # DESIGN.md v3 SS3.4: each hotbar slot keeps the same weapon (same `g`)
                # for the whole clip/dataset -- a pure function of slot, not randomized
                # per-event, so it's also consistent across clips for free.
                held_g = 1000 + slot
                inputs.append(pvf.InputEvent(t=t, key=cfg.equip_keys[slot % len(cfg.equip_keys)]))
            ei += 1
        inv.append(pvf.InventorySample(t=hb_t, counts=list(counts), held=held, held_g=held_g))

    return inputs, inv


def make_one_clip(out_dir, clip_id, n_latent_target, n_slots, n_events, seed, weapon_id_map):
    # DESIGN.md v3 SS3.2/SS3.4: weapon_id_map must be the SAME dict across every clip in
    # this dataset (same weapon -> same embedding index everywhere), not auto-numbered
    # per clip -- per-clip auto-numbering would assign different ids to the same g if
    # different clips happen to observe a different subset/order of weapons.
    cfg = pvf.Config(n_slots=n_slots, weapon_id_map=weapon_id_map)
    # invert num_latent_frames(n_frames) = ceil((n_frames-1)/temporal_compress) + 1
    n_frames = (n_latent_target - 1) * cfg.temporal_compress + 1

    rng = np.random.RandomState(seed)
    inputs, inv = _synth_logs(n_frames, cfg, n_events, rng)

    npz_path = os.path.join(out_dir, f"{clip_id}.npz")
    out = pvf.process_clip(
        None, None, None, None, npz_path, cfg,
        verbose=False, _injected=(inputs, inv, n_frames, 0.0),
    )
    n_latent = int(out["n_latent"][0])
    assert n_latent == n_latent_target, (n_latent, n_latent_target)

    latent = rng.randn(n_latent, 16, 60, 104).astype(np.float32)
    latent[:, 0, 0, 0] = np.arange(n_latent, dtype=np.float32)  # alignment marker, see module docstring
    np.save(os.path.join(out_dir, f"{clip_id}_latent.npy"), latent)

    n_events_fired = sum(int((out[f"{ax}_type"] == pvf.TYPE_FIRE).sum()) for ax in pvf.AXES)
    print(f"{clip_id}: n_latent={n_latent}  events_fired={n_events_fired}  "
          f"none_kind dist={np.bincount(out['none_kind'], minlength=3).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/fake_clips")
    ap.add_argument("--n_clips", type=int, default=3)
    ap.add_argument("--n_latent", type=int, default=63, help="latent frames per clip (must be >= window_frames, default a multiple of 21)")
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--n_events", type=int, default=10, help="approx number of gather/pickup/equip events per clip")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # slots 0..3 are the only ones _synth_logs ever equips -- pin g=1000+slot -> weapon_id=slot
    # once, shared by every clip (see make_one_clip's docstring note).
    weapon_id_map = {1000 + s: s for s in range(4)}
    for i in range(args.n_clips):
        make_one_clip(
            args.out_dir, f"clip_{i:03d}", args.n_latent, args.n_slots, args.n_events,
            seed=args.seed + i, weapon_id_map=weapon_id_map,
        )
    print(f"\nwrote {args.n_clips} fake clips to {args.out_dir}/")


if __name__ == "__main__":
    main()

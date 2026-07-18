# SPDX-License-Identifier: Apache-2.0
"""Network-external, deterministic state integrator (DESIGN.md SS6, M4-A).

Two independent pieces, both intentionally OUTSIDE the neural net
(DESIGN.md "不要把累加逻辑塞进神经网"):

1. `integrate_events`: batched-tensor re-implementation of
   `preprocess_vrising_final.integrate()`'s semantics (count_mode="events":
   each fire is +-1, clamped at 0; equip overwrites held/held_weapon_id) --
   but driven by discrete per-frame axis PREDICTIONS (argmaxed StateHead
   logits) instead of an offline `Event` list, so it can run online during
   autoregressive generation ("生成->状态头->积分器" in DESIGN.md SS5.4).
   Ground-truth training doesn't need this function -- the .npz/fake-data
   already carries pre-integrated `held`/`held_weapon_id`/`count` per frame
   (computed once at preprocessing time by the offline `integrate()`); only
   the self-generated (inference-style) path re-derives S_t from predictions.

2. `shift_state_for_injection`: the "S_t 右移一帧" causal shift (DESIGN.md
   SS5.4) shared by both paths -- frame t+1's injected S_t is frame t's
   absolute state, never frame t+1's own (that would leak the future).
   Frame 0 gets a caller-supplied initial S_0 (DESIGN.md SS5.6).
"""
import torch

from preprocess_vrising_final import TYPE_FIRE

NONE_SLOT = -1
NONE_WEAPON = -1


def integrate_events(
    gather_type, gather_slot,
    pickup_type,
    equip_type, equip_slot,
    n_slots, slot_to_weapon_id,
    init_held=None, init_held_weapon_id=None, init_count=None, init_pickup_count=None,
):
    """All `*_type/_slot`: [B, F] long, discrete per-frame predictions or
    labels (slot == n_slots means the "none" class, matching StateHead's
    slot_classes = n_slots + 1).
    `slot_to_weapon_id`: [n_slots] long, this session's FIXED slot->weapon-id
    binding (DESIGN.md SS3.4: bindings don't change mid-session; SS5.6: given
    externally, not learned/re-derived from vision each step).
    `init_*`: [B] / [B] / [B, n_slots] / [B], S_-1 the caller supplies
    (defaults to "nothing held, everything empty" -- overridden by real S_0
    at inference).

    pickup (M4-C, DESIGN.md SS3.2): no slot/dir, only `pickup_type` -- an
    8-session real-data audit found pickup touches 33 items over 251 events
    (unfittable per-item classifier regardless of visual disambiguation) and
    its `dir` is a hard constant across all 251 samples. It contributes to a
    separate, slot-independent `pickup_count` (occurrences, not magnitude),
    NOT to the per-slot `count` gather already owns -- no double counting is
    possible since they're different arrays entirely.

    gather's `dir` head was dropped the same way (M4-C, 2026-07-02 audit):
    3777/3781 real gather-fire frames are DIR_UP, the 4 DIR_DOWN outliers
    traced to consumption/crafting events misfiled into gather for lack of
    nearby F-key evidence, not real reverse-gather signal. Every fire here
    is treated as +1 (was `torch.where((dir==DIR_UP)|(dir==DIR_ADD), 1, -1)`
    reading a StateHead prediction that no longer exists) -- ground-truth
    TRAINING was never affected either way, since
    preprocess_vrising_final.integrate() (what .npz `count` is baked from)
    reads `Event.dir` straight from the log, not from this function.

    Returns held [B,F] long, held_weapon_id [B,F] long, count [B,F,n_slots]
    long, pickup_count [B,F] long -- same semantics as
    preprocess_vrising_final.integrate(), batched.
    """
    B, F = gather_type.shape
    device = gather_type.device

    held_out = torch.empty((B, F), dtype=torch.long, device=device)
    weapon_out = torch.empty((B, F), dtype=torch.long, device=device)
    count_out = torch.empty((B, F, n_slots), dtype=torch.long, device=device)
    pickup_count_out = torch.empty((B, F), dtype=torch.long, device=device)

    h = (init_held if init_held is not None
         else torch.full((B,), NONE_SLOT, dtype=torch.long, device=device)).clone()
    hw = (init_held_weapon_id if init_held_weapon_id is not None
          else torch.full((B,), NONE_WEAPON, dtype=torch.long, device=device)).clone()
    c = (init_count if init_count is not None
         else torch.zeros((B, n_slots), dtype=torch.long, device=device)).clone()
    pc = (init_pickup_count if init_pickup_count is not None
          else torch.zeros((B,), dtype=torch.long, device=device)).clone()

    for f in range(F):
        typ, slot = gather_type[:, f], gather_slot[:, f]
        fires = (typ == TYPE_FIRE) & (slot >= 0) & (slot < n_slots)
        idx = slot.clamp(min=0, max=n_slots - 1)
        delta = torch.zeros((B, n_slots), dtype=torch.long, device=device)
        delta.scatter_(1, idx.unsqueeze(1), torch.where(fires, 1, 0).unsqueeze(1))
        c = (c + delta).clamp(min=0)

        pc = pc + (pickup_type[:, f] == TYPE_FIRE).long()

        equip_fires = equip_type[:, f] == TYPE_FIRE
        equips = equip_fires & (equip_slot[:, f] >= 0) & (equip_slot[:, f] < n_slots)
        drops = equip_fires & ~equips  # slot == n_slots ("none" class) => took the weapon off
        new_slot = equip_slot[:, f].clamp(min=0, max=n_slots - 1)
        h = torch.where(equips, new_slot, torch.where(drops, torch.full_like(h, NONE_SLOT), h))
        hw = torch.where(
            equips, slot_to_weapon_id[new_slot],
            torch.where(drops, torch.full_like(hw, NONE_WEAPON), hw),
        )

        held_out[:, f] = h
        weapon_out[:, f] = hw
        count_out[:, f] = c
        pickup_count_out[:, f] = pc

    return held_out, weapon_out, count_out, pickup_count_out


def shift_state_for_injection(
    held, held_weapon_id, count,
    init_held=None, init_held_weapon_id=None, init_count=None,
):
    """DESIGN.md SS5.4: right-shift absolute per-frame state by one frame so
    frame t+1 is conditioned on frame t's S_t, never its own (t is never fed
    back to itself). `held`/`held_weapon_id`: [B, F] long; `count`: [B, F,
    n_slots] long or float. `init_*` fill frame 0 (DESIGN.md SS5.6's S_0);
    default is "nothing held, everything empty".
    """
    B, F = held.shape
    n_slots = count.shape[-1]
    device = held.device

    init_held = (init_held if init_held is not None
                 else torch.full((B,), NONE_SLOT, dtype=held.dtype, device=device))
    init_held_weapon_id = (init_held_weapon_id if init_held_weapon_id is not None
                            else torch.full((B,), NONE_WEAPON, dtype=held_weapon_id.dtype, device=device))
    init_count = (init_count if init_count is not None
                  else torch.zeros((B, n_slots), dtype=count.dtype, device=device))

    held_shifted = torch.cat([init_held.unsqueeze(1), held[:, :-1]], dim=1)
    weapon_shifted = torch.cat([init_held_weapon_id.unsqueeze(1), held_weapon_id[:, :-1]], dim=1)
    count_shifted = torch.cat([init_count.unsqueeze(1), count[:, :-1]], dim=1)
    return held_shifted, weapon_shifted, count_shifted

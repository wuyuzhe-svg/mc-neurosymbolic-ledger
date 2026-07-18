# SPDX-License-Identifier: Apache-2.0
"""
M0: video latent + .npz state-label dataloader for the symbolic-state-layer project
(see DESIGN_1.md). Slices a clip's latent + per-latent-frame labels into
`window_frames`-long blocks (default 21, matching the base model's training
clip length / `image_or_video_shape`), using the *same* global latent-frame
index for both, so latent[t] and every label[t] always refer to the same
frame by construction.

Switch to real data: point `data_root` at the directory produced by the real
preprocessing pipeline instead of `data/fake_clips`. Nothing else changes --
this loader only assumes the on-disk layout (see `_discover_clips`) and the
.npz schema produced by `preprocess_vrising_final.py`, not where the files
came from.
"""
import os
import glob
import math

import numpy as np
import torch
from torch.utils.data import Dataset

import preprocess_vrising_final as pvf

# Per-latent-frame fields from the .npz schema (preprocess_vrising_final.py::process_clip).
# 1-D int fields are [n_latent]; `count`/`count_true` are [n_latent, n_slots] int;
# `action` is [n_latent, len(hold_keys)] float (continuous hold-state, DESIGN_1.md v2 §3.2 -- NOT
# a state-layer label, it's an action-conditioning input for the injection paths added in M2/M3).
# M4-C: pickup has no `pickup_slot`/`pickup_dir` (real-data decision, see
# preprocess_vrising_final.py's AXES_WITH_SLOT/AXES_WITH_DIR) -- `pickup_count` is a
# slot-independent aggregate (occurrence count, not classified), hence 1-D like
# held/held_weapon_id, not 2-D like `count` (which stays gather/equip-only).
PER_FRAME_1D_KEYS = (
    "gather_type", "gather_slot", "gather_dir",
    "pickup_type", "pickup_count",
    "equip_type", "equip_slot",
    "none_kind", "held", "held_weapon_id",
)
PER_FRAME_2D_LONG_KEYS = ("count", "count_true")  # DESIGN_1.md 5.1: count_true is eval-only, never train on it.
PER_FRAME_2D_FLOAT_KEYS = ("action",)


def _pixel_to_latent(pixel_frame, temporal_compress):
    if pixel_frame <= 0:
        return 0
    return math.ceil(pixel_frame / temporal_compress)


def aggregate_cursor_to_latent(cursor_pixel, n_latent, fps, temporal_compress, mode="first"):
    """cursor_pixel: [n_frames, 3] (pixel-frame granularity) -> [n_latent, 3].

    OPEN QUESTION (DESIGN_1.md Q4): how to aggregate cursor into a latent frame
    is not decided yet. `mode="first"` (take the first pixel frame mapping to
    each latent frame) is a placeholder default, not a final answer -- swap
    the `mode` here once Q4 is resolved.
    """
    n_frames = cursor_pixel.shape[0]
    out = np.zeros((n_latent, 3), dtype=np.float32)
    # group pixel frames by the latent frame they map to
    buckets = [[] for _ in range(n_latent)]
    for f in range(n_frames):
        lf = _pixel_to_latent(f, temporal_compress)
        if lf < n_latent:
            buckets[lf].append(f)
    for lf, frames in enumerate(buckets):
        if not frames:
            continue
        if mode == "first":
            out[lf] = cursor_pixel[frames[0]]
        elif mode == "mean":
            out[lf] = cursor_pixel[frames].mean(axis=0)
        else:
            raise ValueError(f"unknown cursor aggregation mode: {mode}")
    return out


def _load_latent(latent_path):
    if latent_path.endswith(".npy"):
        # mmap, not a full read: real (non-fake) clips are hundreds of MB, and every
        # caller here only ever slices out one window before copying (M4-B: this is
        # now called from a real DataLoader, potentially many times per epoch).
        return np.load(latent_path, mmap_mode="r")
    return torch.load(latent_path, map_location="cpu", weights_only=True).numpy()


class VideoStateClipDataset(Dataset):
    """Pairs <clip>.npz (labels) with <clip>_latent.{npy,pt} (video latent),
    and exposes block-aligned `window_frames`-long windows over the shared
    latent-frame index.

    Directory layout expected under `data_root` (real or fake, identical):
        clip_000.npz
        clip_000_latent.npy
        clip_001.npz
        clip_001_latent.npy
        ...
    """

    def __init__(
        self,
        data_root,
        window_frames=21,
        num_frame_per_block=3,
        stride=None,
        cursor_agg_mode="first",
        fps=16,
        temporal_compress=4,
        context_frames=0,
        include_sessions=None,
        exclude_sessions=None,
    ):
        assert window_frames % num_frame_per_block == 0, (
            f"window_frames={window_frames} must be a multiple of "
            f"num_frame_per_block={num_frame_per_block} (DESIGN_1.md 3.3)"
        )
        self.data_root = data_root
        self.window_frames = window_frames
        self.num_frame_per_block = num_frame_per_block
        self.stride = stride or num_frame_per_block
        self.cursor_agg_mode = cursor_agg_mode
        self.fps = fps
        self.temporal_compress = temporal_compress
        # M4-B: teacher-forcing needs a preceding "clean_x" context window per item
        # (DESIGN.md SS5.1) -- when >0, __getitem__ also returns "context_*" keys for
        # the context_frames immediately before `start`, and windows without enough
        # preceding history are excluded from `self.index` up front (every sampled
        # item is guaranteed to have a valid context, no None-padding downstream).
        self.context_frames = context_frames

        self.clips = self._discover_clips(data_root)
        # M4-C: session-level train/val split (leak-free -- windows from the SAME
        # session share nearby temporal context, so splitting by window instead of
        # by whole session would leak). `clip["id"]` is "<session_id>/seg_NNN" for
        # M4-C's nested per-session output (flat data_root's ids have no "/", the
        # split(...)  [0] on those just returns the whole id -- harmless no-op there).
        if include_sessions is not None:
            self.clips = [c for c in self.clips if c["id"].split("/")[0] in include_sessions]
        if exclude_sessions is not None:
            self.clips = [c for c in self.clips if c["id"].split("/")[0] not in exclude_sessions]
        if not self.clips:
            raise RuntimeError(f"no clips left under {data_root} after include/exclude session filtering")
        self.index = []  # list of (clip_idx, start)
        for clip_idx, clip in enumerate(self.clips):
            with np.load(clip["npz"]) as npz:
                n_latent = int(npz["n_latent"][0])
            clip["n_latent"] = n_latent
            last_start = n_latent - window_frames
            if last_start < 0:
                continue
            first_start = context_frames  # 0 when context_frames=0, same range as before
            for start in range(first_start, last_start + 1, self.stride):
                self.index.append((clip_idx, start))

        if len(self.index) == 0:
            raise RuntimeError(
                f"No window of length {window_frames} fits inside any clip under {data_root}"
                + (f" with {context_frames} preceding context frames" if context_frames else "")
                + f". Either clips are too short or data_root has no clips."
            )

    @staticmethod
    def _discover_clips(data_root):
        clips = []
        # recursive: M4-C's per-session output (run_m4c_pipeline.py) nests segments one
        # level deep (data_root/<session_id>/seg_NNN.npz), unlike M0-M4B's flat
        # data_root/<clip>.npz -- ** matches both. `id` uses the path relative to
        # data_root (not just the basename) so seg_000 from two different sessions
        # doesn't collide into the same clip id.
        for npz_path in sorted(glob.glob(os.path.join(data_root, "**", "*.npz"), recursive=True)):
            stem = os.path.splitext(npz_path)[0]
            clip_id = os.path.splitext(os.path.relpath(npz_path, data_root))[0]
            latent_path = None
            for ext in (".npy", ".pt", ".pth"):
                cand = f"{stem}_latent{ext}"
                if os.path.exists(cand):
                    latent_path = cand
                    break
            if latent_path is None:
                raise FileNotFoundError(f"no matching <clip>_latent.{{npy,pt}} found for {npz_path}")
            clips.append({"id": clip_id, "npz": npz_path, "latent": latent_path})
        if not clips:
            raise FileNotFoundError(f"no .npz clips found under {data_root}")
        return clips

    def __len__(self):
        return len(self.index)

    def compute_window_sample_weights(self, static_downsample=0.1):
        """DESIGN.md SS5.3: downsample `none_kind==2` (static, nothing happening
        and no action key nearby) windows ~10:1; do NOT downsample
        `none_kind==1` (hard negative: an action key was pressed but nothing
        fired -- exactly the signal the gating heads need). Classifies a whole
        window by whether EVERY frame in it is static: a window with even one
        fire or hard-negative frame keeps full weight, since collapsing that
        signal would defeat the "don't downsample hard negatives" rule.
        Returns a `[len(self)]` float tensor, feed straight into
        `torch.utils.data.WeightedRandomSampler(weights, num_samples=len(self))`.
        """
        weights = torch.empty(len(self.index), dtype=torch.float)
        for i, (clip_idx, start) in enumerate(self.index):
            clip = self.clips[clip_idx]
            with np.load(clip["npz"]) as npz:
                none_kind = npz["none_kind"][start:start + self.window_frames]
            all_static = bool((none_kind == 2).all())
            weights[i] = static_downsample if all_static else 1.0
        return weights

    def compute_inverse_freq_class_weights(self, label_specs, eps=1.0):
        """DESIGN.md SS5.2: per-axis inverse-frequency class weights,
        sqrt(1/freq), for `model.state_head.state_head_losses`'s
        `class_weights`. `label_specs`: dict[label_key -> n_classes] (caller
        supplies this -- `model/state_head.py` owns that spec, and this
        module deliberately doesn't import `model.*`, see
        utils/cpu_dev_patches.py's docstring for why: `model/__init__.py`
        eagerly pulls in the full DMD trainer stack).

        Scans each DISTINCT clip's full label array ONCE (not per-window --
        windows overlap by `stride`, so counting per-window would over-count
        by however much consecutive windows overlap). `eps` Laplace-smooths
        classes that never appear in this dataset so they don't get an
        infinite weight; weights are re-normalized to mean 1 so a fully
        weighted CE stays on roughly the same scale as an unweighted one
        (RMS normalization downstream also absorbs any residual scale, but
        starting near 1 keeps intermediate values sane).
        """
        counts = {k: np.zeros(n, dtype=np.int64) for k, n in label_specs.items()}
        for clip in self.clips:
            with np.load(clip["npz"]) as npz:
                for k, n_classes in label_specs.items():
                    vals = npz[k]
                    valid = vals[vals >= 0]  # drop IGNORE_INDEX (-100) on masked keys
                    counts[k] += np.bincount(valid, minlength=n_classes)[:n_classes]
        weights = {}
        for k, c in counts.items():
            freq = c.astype(np.float64) + eps
            w = 1.0 / np.sqrt(freq)
            w = w / w.mean()
            weights[k] = torch.from_numpy(w).float()
        return weights

    def __getitem__(self, idx):
        clip_idx, start = self.index[idx]
        clip = self.clips[clip_idx]
        end = start + self.window_frames

        n_latent = clip["n_latent"]
        batch = {}
        with np.load(clip["npz"]) as npz:
            for key in PER_FRAME_1D_KEYS:
                batch[key] = torch.from_numpy(npz[key][start:end].copy()).long()
            for key in PER_FRAME_2D_LONG_KEYS:
                batch[key] = torch.from_numpy(npz[key][start:end].copy()).long()
            for key in PER_FRAME_2D_FLOAT_KEYS:
                batch[key] = torch.from_numpy(npz[key][start:end].copy()).float()

            # A0.5 / extended control (2026-07-08): 6-key track incl.
            # Mouse0(attack) + F(pickup), built by rebuild_action6.py
            for k in ("action6", "action5"):
                if k in npz.files:
                    batch[k] = torch.from_numpy(
                        npz[k][start:end].copy()).float()
                    break

            cursor_latent = aggregate_cursor_to_latent(
                npz["cursor"], n_latent, self.fps, self.temporal_compress, mode=self.cursor_agg_mode
            )
        batch["cursor"] = torch.from_numpy(cursor_latent[start:end].copy()).float()

        latent = _load_latent(clip["latent"])
        assert latent.shape[0] == n_latent, (
            f"{clip['id']}: latent has {latent.shape[0]} frames but npz says n_latent={n_latent}"
        )
        batch["latent"] = torch.from_numpy(latent[start:end].copy()).float()

        if self.context_frames:
            # M4-B teacher-forcing context (DESIGN.md SS5.1): the `context_frames`
            # immediately before `start` -- guaranteed to exist, `self.index` only
            # ever contains starts with enough preceding history (see __init__).
            c_start = start - self.context_frames
            batch["context_latent"] = torch.from_numpy(latent[c_start:start].copy()).float()
            with np.load(clip["npz"]) as npz:
                batch["context_held"] = torch.from_numpy(npz["held"][c_start:start].copy()).long()
                batch["context_held_weapon_id"] = torch.from_numpy(
                    npz["held_weapon_id"][c_start:start].copy()
                ).long()
                batch["context_count"] = torch.from_numpy(npz["count"][c_start:start].copy()).long()
                # M4-D block-count labels need the cumulative pickup counter at the
                # frame just before the window (same role as context_count's last frame).
                batch["context_pickup_count"] = torch.from_numpy(
                    npz["pickup_count"][c_start:start].copy()).long()
                # 通路③: the clean window needs its own control conditioning too --
                # teacher-forcing concatenates [clean, noisy] along the frame axis, and
                # control is per-frame (input t conditions frame t, NO causal shift,
                # unlike S_t). `action` = WASD hold-state; `cursor` (aggregated to
                # latent granularity above) = std-normalized aim offset, 方案X takes
                # its first 2 dims into the 6-dim control vector.
                batch["context_action"] = torch.from_numpy(npz["action"][c_start:start].copy()).float()
                batch["context_cursor"] = torch.from_numpy(cursor_latent[c_start:start].copy()).float()

        batch["clip_id"] = clip["id"]
        batch["start"] = start
        batch["n_latent"] = n_latent
        return batch

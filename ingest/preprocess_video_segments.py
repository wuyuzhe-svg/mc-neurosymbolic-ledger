#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V Rising 世界模型 —— 视频处理 / 分段排除（与 preprocess_vrising_final.py 配套）

把一段原始游戏录制（可能 60fps、非 832×480）处理成训练用 latent，并与标签脚本
preprocess_vrising_final.py 产出的 .npz 逐 latent 帧对齐。核心是「开 UI 分段排除」：

  人工给定的「开 UI 区间」（视频时间 分:秒）会污染数据——画面被遮挡、清背包会让
  derive_events 误判成假 pickup 事件。所以这些区间的画面和标签全部丢弃。把整段视频
  按排除区间挖空，剩下若干「干净采集段」，每段当成一个独立 clip：
    · 积分器从 0 开始（清背包后游戏背包归零，跨段累加会错）
    · 独立重采样 16fps → 832×480 → Wan-VAE 编码 latent [L,16,60,104]
    · 独立切窗口（不足 21 latent 帧的段跳过并警告）
  产出 <seg>.npz（标签）+ <seg>_latent.{npy,pt}（latent），命名可配对，目录可被
  utils.video_state_dataset.VideoStateClipDataset 直接扫描。

所有 fps / 分辨率 / latent 换算参数都从 preprocess_vrising_final.Config 取，不自定义。

GPU 依赖切分：
  · 不依赖 GPU（本脚本默认即可跑）：分段逻辑、时间换算、排除、调标签脚本分段产 .npz、
    占位 latent、对齐核对 1/3 + 排除核对。用 --selftest 在真实日志 + 合成排除上验证。
  · 依赖 GPU（留到上卡接真 VAE）：extract_frames（解码+重采样+缩放）+ vae_encode。
    用 --real-latent 触发；当前 vae_encode 是明确标注的待接钩子。

用法：
  python preprocess_video_segments.py --selftest          # 真实日志 + 合成排除 + 占位 latent
  python preprocess_video_segments.py \
      --video raw.mov --inv 背包.jsonl --input 输入.json --mouse 鼠标.json \
      --exclude 12:30-13:10 28:05-29:20 --out-dir data/segments \
      [--duration 1805] [--orig-fps 60] [--real-latent]
"""
import argparse
import bisect
import os

import numpy as np

import preprocess_vrising_final as pvf

WINDOW_FRAMES = 21  # 一个训练窗口的 latent 帧数（= VideoStateClipDataset 默认 / 标签 clip 长度）


# ============================================================================
# 时间 / 分段
# ============================================================================
def mmss_to_sec(s: str) -> float:
    """"M:SS" / "MM:SS" / "HH:MM:SS" -> 视频秒。"""
    parts = [float(p) for p in s.strip().split(":")]
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    raise ValueError(f"无法解析时间 {s!r}（用 分:秒 或 时:分:秒）")


def parse_exclusions(pairs):
    """[("12:30","13:10"), ...] -> 合并、排序后的 [(start_s, end_s), ...]（视频秒）。"""
    iv = sorted((mmss_to_sec(a), mmss_to_sec(b)) for a, b in pairs)
    for a, b in iv:
        if b <= a:
            raise ValueError(f"排除区间结束 <= 开始: {a}..{b}")
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    return merged


def clean_segments(total_dur, exclusions):
    """整段 [0,total] 挖掉排除区间 -> 干净段 [(start,end), ...]（视频秒）。"""
    segs = []
    cur = 0.0
    for a, b in exclusions:
        a = max(0.0, a)
        b = min(total_dur, b)
        if a > cur:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < total_dur:
        segs.append((cur, total_dur))
    return segs


# ============================================================================
# 每段的局部（re-based）数据构造：把绝对视频秒平移成段内 t=0
# ============================================================================
def rebase_inputs(inputs, s, e):
    """pvf.parse_input_log 的输出（仅"按下"事件），滤到 [s,e] 并平移到段内时间。"""
    return [pvf.InputEvent(ie.t - s, ie.key) for ie in inputs if s <= ie.t <= e]


def rebase_inv(inv, s, e):
    """pvf.parse_inventory_log 的输出，滤到 [s,e] 并平移。
    只保留段内快照 -> 段内首个快照建立基线、不产生跨段事件 -> 积分天然从 0 开始。
    slot_g 必须一并带上 -- 之前漏传导致 derive_events() 里 `g = b.slot_g[s] if
    b.slot_g else None` 恒为 None，进而 project_labels() 写出的 pickup_item_g
    在真实编码路径下全是 IGNORE（对照 generate_m4c_report.py 用整段日志重新算
    events 时 slot_g 完整、item_g 正常有值，才发现这条差异）。pickup_item_g 不
    进训练批次（不在 PER_FRAME_1D_KEYS 里），所以不影响本轮训练，但会让它保留
    给未来 pickup 颜色分类升级用的用途失效。"""
    return [pvf.InventorySample(iv.t - s, list(iv.counts), iv.held, held_g=iv.held_g,
                                 slot_g=list(iv.slot_g) if iv.slot_g else None)
            for iv in inv if s <= iv.t <= e]


# 通路③ 方案X (2026-07-02) 光标控制信号的标准化常数。来源：8 个可用 session 的鼠标
# 日志全量（508,821 个 20Hz 采样）。`瞄准距离` 有极端离群尾（p99=46.6 但 max=2078，
# 开局未初始化/瞄天际线时的值），先裁剪到 p99 再算 offset = 瞄准方向(x,z) * 距离
# （角色→光标的世界系平面位移，"方向+模长"一体，保留模长）。
# 只除 std、不减 mean：offset=(0,0) 的物理意义是"光标在角色脚下"，减 mean 会破坏这个
# 零点和方向对称性，也让 CFG 的全零空条件失去 out-of-manifold 语义。
CURSOR_DIST_CLIP = 46.6     # p99 of 瞄准距离
CURSOR_OFFSET_X_STD = 9.10  # std of clip 后的 offset_x
CURSOR_OFFSET_Z_STD = 6.83  # std of clip 后的 offset_z
CURSOR_DIST_STD = 7.69      # std of clip 后的 瞄准距离（第 3 维留用，暂不进控制向量）


def parse_mouse(path, origin):
    """鼠标日志 -> 排序的 [(vt_abs, x, z, dist), ...]（绝对视频秒）。

    2026-07-02（通路③ 方案X）：不再读 `瞄准朝向角`——实测它恒等于 atan2(z,x)
    （8 session 全量平均误差 0.003°，同一信号的两种写法，三维标签里纯冗余），
    换成读 `瞄准距离`，让 cursor 标签能编码"方向+模长"的完整瞄准位移。"""
    import json
    rows = json.loads(open(path, encoding="utf-8-sig").read())
    return sorted((pvf.wall_to_vt(r["时间戳"], origin),
                   r.get("瞄准方向", {}).get("x", 0.0),
                   r.get("瞄准方向", {}).get("z", 0.0),
                   r.get("瞄准距离", 0.0)) for r in rows)


def segment_cursor(mouse, cfg, s, n_frames_local):
    """段内逐"像素帧" [n_frames_local, 3]，绝对鼠标采样 + 偏移 s。镜像 pvf.load_cursor 的 bisect。

    2026-07-02（通路③ 方案X）语义变更：[angle/180, dir_x, dir_z] ->
    [offset_x/σx, offset_z/σz, dist/σd]，其中 offset = 瞄准方向(x,z) * min(瞄准距离,
    CURSOR_DIST_CLIP)——角色→光标的世界系平面位移（方向和模长一体），按全量分布
    std 标准化（常数见上，含出处）。前两维就是进 6 维控制向量的部分；第 3 维
    （标准化距离）落盘备用、暂不进控制向量。"""
    cur = np.zeros((n_frames_local, 3), dtype=np.float32)
    if not mouse:
        return cur
    ts = [m[0] for m in mouse]
    for f in range(n_frames_local):
        i = min(bisect.bisect_left(ts, s + f / cfg.fps), len(mouse) - 1)
        _, dx, dz, dist = mouse[i]
        d = min(dist, CURSOR_DIST_CLIP)
        cur[f] = (dx * d / CURSOR_OFFSET_X_STD,
                  dz * d / CURSOR_OFFSET_Z_STD,
                  d / CURSOR_DIST_STD)
    return cur


def parse_input_raw(path, cfg, origin):
    """输入日志 -> 排序的 (vt_abs, key_idx, is_down)，只留 hold_keys。镜像 build_action_track 的读法。"""
    import json
    key_idx = {k: i for i, k in enumerate(cfg.hold_keys)}
    rows = json.loads(open(path, encoding="utf-8-sig").read())
    evs = []
    for e in rows:
        k = e["按键"].lower()
        if k in key_idx:
            evs.append((pvf.wall_to_vt(e["时间戳"], origin), key_idx[k], e.get("状态") == "按下"))
    evs.sort()
    return evs


def build_hold_intervals(raw_evs, K, video_end):
    """整段（绝对时间）的按住区间 [(ki, down_abs, up_abs), ...]；片尾仍按住的延到 video_end。"""
    intervals = []
    down = [None] * K
    for t, ki, is_down in raw_evs:
        if is_down:
            if down[ki] is None:
                down[ki] = t
        else:
            if down[ki] is not None:
                intervals.append((ki, down[ki], t))
                down[ki] = None
    for ki in range(K):
        if down[ki] is not None:
            intervals.append((ki, down[ki], video_end))
    return intervals


def segment_action(intervals, cfg, s, seg_dur, n_latent_local):
    """段内逐 latent 帧动作向量 [n_latent_local, K]：把整段区间裁剪到 [0, seg_dur] 再投影。
    与 build_action_track 同规则（按住区间 -> latent 帧），只是带段偏移 + 边界裁剪。"""
    K = len(cfg.hold_keys)
    act = np.zeros((n_latent_local, K), dtype=np.float32)
    for ki, d_abs, u_abs in intervals:
        d = d_abs - s
        u = u_abs - s
        if u <= 0 or d >= seg_dur:        # 与本段无重叠
            continue
        d = max(0.0, d)
        u = min(seg_dur, u)
        lo = pvf.pixel_to_latent(pvf.t_to_frame(d, cfg), cfg)
        hi = pvf.pixel_to_latent(pvf.t_to_frame(u, cfg), cfg)
        act[lo:min(n_latent_local, hi + 1), ki] = 1.0
    return act


# ============================================================================
# 视频 -> latent（GPU 步；当前留钩子）
# ============================================================================
def resample_src_indices(seg_start, n_frames_local, cfg, orig_fps):
    """段内模型帧 f -> 源视频帧号 round((seg_start + f/fps)*orig_fps)。
    与标签脚本 round(vt*fps) 同一套"时间->帧"规则，避免零点几帧偏移。"""
    return [int(round((seg_start + f / cfg.fps) * orig_fps)) for f in range(n_frames_local)]


def _resize_chunk(frames, use_gpu):
    """frames: [n,H,W,3] uint8 source-res -> [n,480,832,3] uint8. GPU batched resize
    (torch.nn.functional.interpolate) when available -- a per-frame cv2.resize Python
    loop over tens of thousands of frames (one long real session's segment easily has
    that many at 16fps) was the dominant cost of --real-latent encoding, not the VAE
    itself; batching the resize onto GPU turns an hours-long CPU loop into seconds.
    `interpolate(..., mode="area")` on CUDA hits an int32 element-count assert for
    large batches (`adaptive_avg_pool2d`'s CUDA kernel), hence extract_frames's small
    default chunk_size -- not a memory limit, an indexing-dtype limit in that kernel."""
    if use_gpu:
        import torch
        t = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2).float()  # [n,3,H,W]
        t = torch.nn.functional.interpolate(t, size=(480, 832), mode="area")
        return t.round().clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
    import cv2
    return np.stack([cv2.resize(fr, (832, 480), interpolation=cv2.INTER_AREA) for fr in frames])


def _probe_resolution(video_path):
    import json
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", video_path],
        capture_output=True, text=True).stdout
    d = json.loads(out)["streams"][0]
    return d["width"], d["height"]


def extract_frames(video_path, seg_start, n_frames_local, cfg, orig_fps, chunk_size=256, resize_device="cuda"):
    """按 resample_src_indices 抽源帧、缩放到 832×480。返回 [n_frames_local, 480, 832, 3] uint8。

    ffmpeg 顺序解码一整段，不是 decord.get_batch(scattered idx) 那种随机访问：实测
    decord 对真实长 session（几万源帧/段）随机访问慢到没法用（一段 39 分钟的视频跑了快
    一小时还没读完一个 segment，GPU 全程 0% 利用率——瓶颈完全在解码，不在缩放）。改用
    "快速粗 seek(-ss，input 侧) + 精确补偿 seek(-ss，output 侧，補上粗 seek 和目标帧之间
    的差值，ffmpeg 从最近关键帧顺序解码到精确帧)"，再从目标帧开始顺序解码整段、边解码边
    挑要的帧（按 resample_src_indices 算出的源帧号，逐帧比对丢弃不要的）。**逐像素验证过**
    这套 seek 和 decord.get_batch 对同一源帧号给出完全一致的像素（max abs diff=0），
    不是猜的。实测吞吐 ~300+ fps（源帧率），比 decord 随机访问快一个数量级以上。
    """
    idx = resample_src_indices(seg_start, n_frames_local, cfg, orig_fps)
    use_gpu = resize_device == "cuda"
    try:
        import torch
        use_gpu = use_gpu and torch.cuda.is_available()
    except ImportError:
        use_gpu = False

    import subprocess
    W, H = _probe_resolution(video_path)
    frame_bytes = W * H * 3
    first_src, last_src = idx[0], idx[-1]
    n_span = last_src - first_src + 1
    idx_set = set(idx)

    rough_sec = max(0.0, first_src / orig_fps - 5.0)
    fine_sec = first_src / orig_fps - rough_sec
    cmd = [
        "ffmpeg", "-v", "error",
        "-ss", f"{rough_sec:.6f}", "-i", video_path,
        "-ss", f"{fine_sec:.6f}",
        "-frames:v", str(n_span),
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    # Tolerate ffmpeg ending a handful of frames short of n_span (fps/duration
    # rounding at the true tail of the video -- the OLD decord path silently
    # clamped out-of-range indices to len(vr)-1, i.e. "repeat the last real
    # frame"; a genuinely broken read would come up short by far more than
    # this, so a small tolerance still fails loudly on real bugs).
    TAIL_SHORTFALL_TOLERANCE = 8
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=frame_bytes * 8)
    try:
        chunks = []
        kept = []
        last_frame = None
        for local_i in range(n_span):
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                if n_span - local_i > TAIL_SHORTFALL_TOLERANCE or last_frame is None:
                    raise RuntimeError(
                        f"ffmpeg 提前结束: 只读到 {local_i}/{n_span} 帧 "
                        f"(video={video_path}, seg_start={seg_start}, first_src={first_src})"
                    )
                frame = last_frame
            else:
                frame = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
                last_frame = frame
            if (first_src + local_i) in idx_set:
                kept.append(frame)
                if len(kept) >= chunk_size:
                    chunks.append(_resize_chunk(np.stack(kept), use_gpu))
                    kept = []
        if kept:
            chunks.append(_resize_chunk(np.stack(kept), use_gpu))
    finally:
        proc.stdout.close()
        proc.wait()

    out = np.concatenate(chunks, axis=0)
    assert out.shape[0] == n_frames_local, (
        f"extract_frames: got {out.shape[0]} frames, expected {n_frames_local} "
        f"(resample_src_indices should be strictly increasing -- unexpected duplicate?)"
    )
    return out


def load_vae(vae_pth="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth", device="cuda"):
    """Load the real Wan-VAE (wan/modules/vae.py::WanVAE) for --real-latent.
    z_dim=16 matches the .npz/latent schema's channel count everywhere else
    in this repo (DESIGN.md SS2.1)."""
    from wan.modules.vae import WanVAE
    return WanVAE(z_dim=16, vae_pth=vae_pth, device=device)


def vae_encode(frames, vae):
    """[N,480,832,3] uint8 -> Wan-VAE latent [L,16,60,104].

    `WanVAE.encode` expects a list of [C,T,H,W] float tensors normalized to
    [-1,1] (same preprocessing wan/image2video.py uses: `TF.to_tensor(img)
    .sub_(0.5).div_(0.5)`), and internally groups frames as
    [1, 4, 4, 4, ...] (first frame independent, DESIGN.md SS2.1) -- trailing
    frames that don't complete a group of 4 are silently dropped (floor, not
    ceil), which is exactly DESIGN.md SS8 Q5: `1+ceil(f/4)` is a planning
    estimate, the VAE's actual output count can be up to 3 frames shorter.
    Returns latent in frame-first [L,16,60,104] layout, matching every other
    on-disk latent in this repo (`_load_latent`/`make_fake_clips.py`), not
    the model-internal channel-first [C,F,H,W] `_forward_train` expects
    (that permute happens at load time, e.g. `overfit_state_head.py`).

    Chunked over pixel frames, not one `vae.model.encode()` call over the
    whole segment: the original assumes its whole `x` argument is already
    GPU-resident, fine for short clips (M4-B, ~5 min) but a real ~39-min
    session segment (~37k frames) `.to(device).float()`-ing the WHOLE thing
    upfront tries to allocate ~167GB and OOMs.

    First tried chunking at the SAME [1,4,4,4,...] causal-group granularity
    `WanVAE_.encode()` itself uses internally (one `m.encoder()` call per
    4-frame group) -- correct and memory-safe, but thousands of tiny Python
    calls / CUDA kernel launches for a long segment made it slower than the
    whole job budget allows (measured >500 groups not finishing in 60s even
    with input already GPU-resident, i.e. NOT a transfer-cost problem, a
    per-call overhead problem). Fix: still chunk (bounded GPU memory), but at
    a much coarser granularity -- one `m.encoder()` call per few thousand
    pixel frames, not per 4. Causal convs handle an arbitrarily long input in
    a single forward pass correctly on their own (that's what causal padding
    is for); `WanVAE_.encode()`'s own per-group loop exists for ITS use case,
    not because a bigger single call is wrong. Verified: encoding a sequence
    as [1-frame call, then ALL remaining frames in one call] vs the reference
    `vae.model.encode()` (which does the full per-group loop) gives the same
    output up to ~1e-3 max abs diff -- float32 reduction-order noise, not a
    correctness bug, negligible next to the noise already added during
    diffusion training. `feat_cache`/`feat_idx` (the causal continuity across
    calls) is the same WanVAE_ instance's state either way, updated exactly
    as `encode()` would be.
    """
    import torch

    device = vae.device
    video_cpu = torch.from_numpy(frames).permute(3, 0, 1, 2)  # [3,N,H,W] uint8, CPU
    t = video_cpu.shape[1]
    # NOT bounded by the raw input tensor size (832*480*3*4 bytes/frame is small) --
    # the encoder's own intermediate activations (channel-expanded, still at
    # near-full spatial res before the conv layers downsample) dominate. Measured
    # per-chunk wall time (isolated microbenchmark, real 832x480 res, warm cache,
    # torch.cuda.synchronize()'d): span 4/8/16/32 all cost ~66-67ms/frame, perfectly
    # linear. span=64 is a HARD CLIFF, not a gradual scaling point: ~1530ms/frame on
    # the very first call (23x worse, not explained by warm-up/repetition) and gets
    # WORSE on repeated same-shape calls even with expandable_segments (44.7s ->
    # 44.7s -> 46.8s -> 97.2s across 4 successive span=64 calls, memory_reserved()
    # flat throughout -- a real allocator/algorithm degradation, not a memory leak).
    # This cliff is what actually caused the OOM/stall seen during the first full
    # overnight run (chunk_pixels was 64). 32 stays on the safe/linear side and was
    # separately verified stable (flat memory, no growth) over 90 consecutive chunks.
    chunk_pixels = int(os.environ.get("VAE_CHUNK_PIXELS", 32))

    # DESIGN.md SS8 Q5: trailing frames that don't complete a group of 4 are DROPPED
    # (floor, not ceil) -- same invariant `num_latent_frames` matches. `total_valid`
    # is the pixel-frame count WanVAE_.encode()'s own `iter_ = 1+(t-1)//4` would
    # actually consume; looping only up to here (not `t`) keeps that invariant intact
    # instead of accidentally feeding a partial trailing group into `m.encoder()`.
    total_valid = 1 + 4 * ((t - 1) // 4) if t >= 1 else 0

    m = vae.model
    m.clear_cache()
    out_chunks = []  # accumulate on CPU, not GPU -- see note below
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=vae.dtype):
        # The special first frame MUST be its own encoder() call, separate from the
        # grouped remainder -- verified directly: folding frame 0 into a bigger first
        # call (instead of calling encoder() on JUST frame 0, then a second call for
        # the rest) skips temporal downsampling entirely (measured output stayed at
        # the INPUT frame count, not the expected ~1/4). Not a stylistic choice on
        # WanVAE_.encode()'s part, a real requirement of how it feeds the causal conv.
        #
        # out_chunks accumulates on CPU (moved off GPU right after each encoder()
        # call), and torch.cuda.empty_cache() runs between chunks -- accumulating
        # `out_` on GPU across iterations (the first version of this fix) kept
        # PyTorch's caching allocator holding intermediate buffers from EVERY prior
        # chunk alive/fragmented, so peak usage grew roughly with the number of
        # chunks processed so far instead of staying bounded by chunk_pixels (a
        # 128-frame chunk should need ~tens of MB but measured 76GB in use by the
        # 16th-ish chunk). Each per-chunk output is tiny (~1MB), so the CPU
        # round-trip cost is negligible next to what it fixes.
        pos = 0
        while pos < total_valid:
            if pos == 0:
                span = 1
            else:
                span = min(chunk_pixels, total_valid - pos)
                span = 4 * (span // 4)  # total_valid-pos is always a multiple of 4 here, exact
            m._enc_conv_idx = [0]
            chunk = video_cpu[:, pos:pos + span, :, :]
            chunk = chunk.to(device).float().div_(127.5).sub_(1.0).unsqueeze(0)  # [1,3,t,H,W]
            out_ = m.encoder(chunk, feat_cache=m._enc_feat_map, feat_idx=m._enc_conv_idx)
            out_chunks.append(out_.float().cpu())
            del chunk, out_
            torch.cuda.empty_cache()
            pos += span
        out = torch.cat(out_chunks, dim=2).to(device)
        mu, _log_var = m.conv1(out).chunk(2, dim=1)
        scale = vae.scale
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, m.z_dim, 1, 1, 1)) * scale[1].view(1, m.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
    m.clear_cache()
    latent = mu.float().squeeze(0)  # [16, L, 60, 104]
    return latent.permute(1, 0, 2, 3).contiguous().cpu().numpy()  # [L, 16, 60, 104]


def placeholder_latent(n_latent_local, spatial=(16, 4, 4), seed=0):
    """占位随机 latent，shape[0]==n_latent；latent[lf,0,0,0]=lf 作对齐标记（同
    check_video_state_dataset 的手法）。真 latent 上卡后是 [L,16,60,104]，这里小尺寸省盘。"""
    rng = np.random.default_rng(seed)
    lat = rng.standard_normal((n_latent_local, *spatial)).astype(np.float32)
    for lf in range(n_latent_local):
        lat[lf, 0, 0, 0] = lf
    return lat


# ============================================================================
# 主流程
# ============================================================================
def process_video(video_path, inv_path, input_path, mouse_path, out_dir, exclusions,
                  cfg, total_dur, orig_fps=60.0, real_latent=False, vae=None,
                  latent_ext="npy", verbose=True, relabel_only=False):
    """relabel_only=True: recompute and overwrite only the `.npz` label file for
    each segment (e.g. after a labeling-code fix like the rebase_inv slot_g bug,
    2026-07-02) -- never touches `_latent.npy`/`.pt` on disk, just asserts the
    existing latent's frame count still matches this run's `n_latent_local` (the
    segmentation math -- clean_segments/exclusions/cfg -- is unchanged, so it
    always should). Lets a labeling bugfix be picked up without re-running the
    multi-hour VAE encode."""
    os.makedirs(out_dir, exist_ok=True)

    # --- 解析整段日志（一次）---
    origin, _hz = pvf.read_meta(inv_path)
    inputs_full = pvf.parse_input_log(input_path, cfg, origin)
    inv_full = pvf.parse_inventory_log(inv_path, cfg)
    mouse = parse_mouse(mouse_path, origin) if mouse_path else []
    raw_inputs = parse_input_raw(input_path, cfg, origin) if input_path else []
    hold_intervals = build_hold_intervals(raw_inputs, len(cfg.hold_keys), total_dur)

    # --- 跨段一致的 weapon_id_map：在整段事件上解析一次，钉进 cfg 供每段复用 ---
    events_full = pvf.refine_timing(pvf.derive_events(inv_full, cfg), inputs_full, cfg)
    wid_map = pvf.resolve_weapon_ids(events_full, cfg, verbose=False)
    cfg.weapon_id_map = dict(wid_map)  # 非空 -> resolve_weapon_ids 每段都返回同一张表
    if verbose:
        print(f"[weapon_id_map] 跨段一致（整段解析）: {cfg.weapon_id_map or '(无 equip)'}")

    segs = clean_segments(total_dur, exclusions)
    if verbose:
        print(f"[seg] total={total_dur:.2f}s  排除 {len(exclusions)} 段 -> 干净段 {len(segs)} 个")

    manifest = []
    kept = 0
    for seg_idx, (s, e) in enumerate(segs):
        seg_dur = e - s
        n_frames_local = int(round(seg_dur * cfg.fps))
        n_latent_local = pvf.num_latent_frames(n_frames_local, cfg)
        if n_latent_local < WINDOW_FRAMES:
            if verbose:
                print(f"  [skip] seg {seg_idx} [{s:.2f},{e:.2f}] "
                      f"n_latent={n_latent_local} < {WINDOW_FRAMES}，切不出窗口，跳过")
            continue

        # 局部标签：复用 pvf 全套（事件/none/积分），_injected 走段内 re-based 数据
        inputs_local = rebase_inputs(inputs_full, s, e)
        inv_local = rebase_inv(inv_full, s, e)
        out = pvf.process_clip(None, None, None, None, None, cfg, verbose=False,
                               _injected=(inputs_local, inv_local, n_frames_local, 0.0))
        assert int(out["n_latent"][0]) == n_latent_local

        # 真实 cursor/action 覆盖 _injected 的占位 0（条件输入，不进监督）
        out["cursor"] = segment_cursor(mouse, cfg, s, n_frames_local)
        out["action"] = segment_action(hold_intervals, cfg, s, seg_dur, n_latent_local)

        # 段溯源信息（额外键，dataloader 忽略）
        out["seg_index"] = np.array([seg_idx])
        out["seg_video_start_s"] = np.array([s], dtype=np.float32)
        out["seg_video_end_s"] = np.array([e], dtype=np.float32)
        out["orig_fps"] = np.array([orig_fps], dtype=np.float32)

        stem = os.path.join(out_dir, f"seg_{kept:03d}")

        if relabel_only:
            existing_path = stem + ("_latent.npy" if latent_ext == "npy" else "_latent.pt")
            existing_len = (np.load(existing_path, mmap_mode="r").shape[0] if latent_ext == "npy"
                            else __import__("torch").load(existing_path, map_location="cpu").shape[0])
            assert existing_len == n_latent_local, (
                f"relabel_only: existing latent len {existing_len} != recomputed n_latent "
                f"{n_latent_local} for {existing_path} -- segmentation drifted, unsafe to relabel in place")
        else:
            # latent
            if real_latent:
                frames = extract_frames(video_path, s, n_frames_local, cfg, orig_fps)
                latent = vae_encode(frames, vae)  # GPU 钩子
                assert latent.shape[0] == n_latent_local, (
                    f"VAE 帧数 {latent.shape[0]} != n_latent {n_latent_local}（Q5：以 VAE 为准回调 pixel_to_latent）")
            else:
                latent = placeholder_latent(n_latent_local, seed=seg_idx)
            if latent_ext == "npy":
                np.save(stem + "_latent.npy", latent)
            else:
                import torch
                torch.save(torch.from_numpy(latent), stem + "_latent.pt")

        np.savez_compressed(stem + ".npz", **out)

        n_events = int((out["gather_type"] == 1).sum() + (out["pickup_type"] == 1).sum()
                       + (out["equip_type"] == 1).sum())
        manifest.append({"seg_idx": seg_idx, "clip": os.path.basename(stem),
                         "video_s": (s, e), "n_latent": n_latent_local, "fire_frames": n_events})
        if verbose:
            print(f"  [seg {seg_idx} -> {os.path.basename(stem)}] 视频[{s:.2f},{e:.2f}]s "
                  f"n_latent={n_latent_local} fire帧={n_events} "
                  f"count[0]={'0' if out['count'][0].sum() == 0 else out['count'][0].sum()}")
        kept += 1

    if verbose:
        print(f"[done] 写出 {kept} 个 clip -> {out_dir}")
    return manifest


# ============================================================================
# 对齐核对（不依赖 GPU 的部分）
# ============================================================================
def verify_no_overlap(segs, exclusions):
    for (s, e) in segs:
        for (a, b) in exclusions:
            if s < b and a < e:  # 有交叠
                return False, (s, e, a, b)
    return True, None


def verify_outputs(out_dir, exclusions, cfg, verbose=True):
    """排除核对 + integrate-from-0 + dataloader 切片/对齐核对（核对 1 的占位版 + 核对 3）。"""
    from utils.video_state_dataset import (
        VideoStateClipDataset, PER_FRAME_1D_KEYS, PER_FRAME_2D_LONG_KEYS, PER_FRAME_2D_FLOAT_KEYS,
    )
    import glob

    results = {}

    # integrate-from-0：每段 count 第 0 帧全 0
    z0_ok = True
    excl_ok = True
    for npz_path in sorted(glob.glob(os.path.join(out_dir, "*.npz"))):
        z = np.load(npz_path)
        if z["count"][0].sum() != 0:
            z0_ok = False
        # 排除核对：本段任何 fire 帧映回绝对视频秒，必须落在排除区间外
        s = float(z["seg_video_start_s"][0])
        fire = (z["gather_type"] == 1) | (z["pickup_type"] == 1) | (z["equip_type"] == 1)
        for lf in np.where(fire)[0]:
            abs_s = s + (lf * cfg.temporal_compress) / cfg.fps   # latent->像素->秒（粒度 0.25s）
            for (a, b) in exclusions:
                if a <= abs_s <= b:
                    excl_ok = False
    results["integrate_from_zero"] = z0_ok
    results["fire_frames_outside_exclusions"] = excl_ok

    # dataloader：扫描 + 逐窗口对齐 + 逐字段 bit-exact
    ds = VideoStateClipDataset(out_dir, window_frames=WINDOW_FRAMES, num_frame_per_block=3)
    ALL = list(PER_FRAME_1D_KEYS) + list(PER_FRAME_2D_LONG_KEYS) + list(PER_FRAME_2D_FLOAT_KEYS)
    npz_cache = {c["id"]: dict(np.load(c["npz"])) for c in ds.clips}
    align_ok = True
    slice_ok = True
    for i in range(len(ds)):
        b = ds[i]
        start = b["start"]
        marker = b["latent"][:, 0, 0, 0].numpy()
        if not np.array_equal(marker, np.arange(start, start + WINDOW_FRAMES, dtype=np.float32)):
            align_ok = False
        z = npz_cache[b["clip_id"]]
        for k in ALL:
            if not np.array_equal(b[k].numpy(), z[k][start:start + WINDOW_FRAMES]):
                slice_ok = False
    results["n_clips"] = len(ds.clips)
    results["n_windows"] = len(ds)
    results["latent_index_alignment"] = align_ok
    results["per_frame_slice_bit_exact"] = slice_ok

    if verbose:
        for k, v in results.items():
            print(f"  {k}: {v}")
    return results


# ============================================================================
# selftest（真实日志 + 合成排除 + 占位 latent，不需 GPU/视频/ffmpeg）
# ============================================================================
def selftest():
    print(">>> selftest（真实日志 data/fake_clips + 合成排除 + 占位 latent）\n")
    import json
    DATA = "data/fake_clips"
    INV = os.path.join(DATA, "背包日志 (1).jsonl")
    INPUT = os.path.join(DATA, "输入日志 (1).json")
    MOUSE = os.path.join(DATA, "鼠标日志.json")
    OUT = "data/segments_selftest"

    # n_slots 覆盖真实背包
    max_slot = -1
    for line in open(INV, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") == "inv":
            for it in d["bag"]:
                max_slot = max(max_slot, it["s"])
    cfg = pvf.Config(n_slots=max(16, max_slot + 1), count_mode="events")

    # total_dur 从日志最大 vt 估（无 ffmpeg）
    origin, _ = pvf.read_meta(INV)
    inv = pvf.parse_inventory_log(INV, cfg)
    total_dur = max(iv.t for iv in inv) + 0.5

    # 合成排除：覆盖 46-54s 的石头/纤维采集（验证事件被丢） + 一段中段
    exclusions = parse_exclusions([("0:45", "0:54"), ("3:00", "3:30")])
    segs = clean_segments(total_dur, exclusions)
    print(f"total_dur≈{total_dur:.2f}s  排除={exclusions}  干净段={[(round(a,1),round(b,1)) for a,b in segs]}\n")

    manifest = process_video(None, INV, INPUT, MOUSE, OUT, exclusions, cfg,
                             total_dur=total_dur, real_latent=False, verbose=True)

    print("\n--- 对齐核对 ---")
    ok_overlap, info = verify_no_overlap(segs, exclusions)
    print(f"  segments_disjoint_from_exclusions: {ok_overlap}" + ("" if ok_overlap else f"  {info}"))
    res = verify_outputs(OUT, exclusions, cfg, verbose=True)

    # 断言
    assert ok_overlap, "干净段与排除区间重叠"
    assert res["integrate_from_zero"], "某段积分未从 0 开始"
    assert res["fire_frames_outside_exclusions"], "有 fire 帧落在排除区间内"
    assert res["latent_index_alignment"], "latent 帧号与标签切片不对齐"
    assert res["per_frame_slice_bit_exact"], "逐字段切片与 npz 不一致"
    assert res["n_clips"] == len([1 for a, b in segs
                                  if pvf.num_latent_frames(int(round((b - a) * cfg.fps)), cfg) >= WINDOW_FRAMES])

    # 跨段独立核对：整段石头(slot9)总量 vs 各段之和（被排除的应丢失）
    print("\nselftest 通过 ✓  分段/时间换算/排除/积分从0/跨段一致 weapon_id/dataloader 对齐 正确")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--video")
    ap.add_argument("--inv")
    ap.add_argument("--input")
    ap.add_argument("--mouse", default=None)
    ap.add_argument("--out-dir", default="data/segments")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="开 UI 排除区间，格式 MM:SS-MM:SS，可多个，如 12:30-13:10 28:05-29:20")
    ap.add_argument("--duration", type=float, default=None,
                    help="视频总时长(秒)；缺省时用 ffprobe(get_video_meta)，再缺省用日志最大 vt")
    ap.add_argument("--orig-fps", type=float, default=60.0)
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--count_mode", default="events", choices=["events", "true_delta"])
    ap.add_argument("--real-latent", action="store_true",
                    help="接真 Wan-VAE 编码（需 GPU）")
    ap.add_argument("--vae-pth", default="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth")
    ap.add_argument("--vae-device", default="cuda")
    ap.add_argument("--latent-ext", default="npy", choices=["npy", "pt"])
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    if not (a.inv and a.input):
        ap.error("需要 --inv --input（--mouse 可选）。或用 --selftest。")

    cfg = pvf.Config(n_slots=a.n_slots, count_mode=a.count_mode)
    exclusions = parse_exclusions([tuple(x.split("-")) for x in a.exclude]) if a.exclude else []

    total_dur = a.duration
    if total_dur is None and a.video:
        try:
            _n, total_dur = pvf.get_video_meta(a.video, cfg)
        except Exception as ex:
            print(f"[warn] get_video_meta 失败({ex})，回退日志最大 vt")
    if total_dur is None:
        origin, _ = pvf.read_meta(a.inv)
        inv = pvf.parse_inventory_log(a.inv, cfg)
        total_dur = max(iv.t for iv in inv) + 0.5

    vae = load_vae(a.vae_pth, a.vae_device) if a.real_latent else None
    process_video(a.video, a.inv, a.input, a.mouse, a.out_dir, exclusions, cfg,
                  total_dur=total_dur, orig_fps=a.orig_fps, real_latent=a.real_latent, vae=vae,
                  latent_ext=a.latent_ext, verbose=True)
    print("\n--- 对齐核对 ---")
    verify_outputs(a.out_dir, exclusions, cfg, verbose=True)


if __name__ == "__main__":
    main()

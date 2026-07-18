#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V Rising 世界模型 —— 数据预处理（最终版）

把 (输入日志, 背包日志, 鼠标日志, 视频) 对齐 -> 逐 latent 帧的三轴标签
{type, slot, dir} + 每帧绝对状态 S_t(held / count) + 光标 + none 分类。

落地的关键决策（都在真实 5 分钟数据上验证过）：
  1. 统一时钟：背包日志 meta 的 videoOriginMs 把键鼠/鼠标的 wall 时间戳换算成“视频秒”，
     背包日志本身有 vt。三份日志落到同一条时间轴。
  2. 从背包快照推事件：相邻快照差异 -> gather/pickup（数量变）/ equip（手持变）。
  3. 事件定位（反推）：背包“变了才记”，故事件时刻≈跳变记录那行(b.t)，不是相邻两行中点；
     跨心跳大间隔时，前一行(心跳)只证明“之前没变”、不参与定位。再用按键精定位：
     gather snap 到最近 mouse0、pickup snap 到 F、equip snap 到 Alpha。
  4. latent 帧映射：单帧定位 pixel_to_latent 用 ceil；但 num_latent_frames（一段视频总
     latent 帧数）是 floor（1+(F-1)//4）——真 Wan-VAE 按 [1,4,4,4,...] 分组，凑不满一组
     的尾帧直接丢弃、不会向上取整凑数，M4-B 在真实数据上实测确认（DESIGN.md SS8 Q5）。
  5. 三轴并行，一帧可多事件；equip 无方向，slot=新手持；pickup 无 slot 也无方向（M4-C，
     真实 8-session 日志审计：掉落物画面无区别 + 33 类里大多数样本数 <10，练不出分类器，
     dir 更是实测的硬常数），只用独立聚合计数器 pickup_count 记发生次数，不进 gather 的
     按 slot count。
  6. 标签：门控 type 全帧都有；none 帧 slot/dir 填 IGNORE(-100)，训练 mask。
  7. 两种 none：附近有动作键但无事件=难负例(保留)；都没有=静止(可降采样)。
  8. 记账在网络外：积分器累加 S_t。默认数“事件次数”(任务自定义语义、彻底脱离游戏数值、
     无漂移)；可切 true_delta 复刻游戏真值，仅作 eval 对照。
  9. 配置表外置：n_slots / 键位 / 物品名 / 哪些槽 renderable，换游戏只改这里。

用法：
  python preprocess_vrising.py --selftest                 # 合成数据验证逻辑，不需真数据
  python preprocess_vrising.py --inv 背包.jsonl --input 输入.json \
         --mouse 鼠标.json --video test1.mov --out clip.npz
  python preprocess_vrising.py ... --count_mode true_delta # 改用真实总量(eval对照)
"""

import argparse, json, math, bisect
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np

IGNORE = -100
# 方向 / 门控编码
DIR_UP, DIR_DOWN, DIR_ADD, DIR_REMOVE = 0, 1, 2, 3
TYPE_NONE, TYPE_FIRE = 0, 1
AXES = ("gather", "pickup", "equip")


# ============================================================================
# 配置表（游戏专属，换游戏改这里）
# ============================================================================
@dataclass
class Config:
    fps: int = 16                 # 模型工作帧率（喂 VAE 的帧率；视频若 60fps 先重采样到这个）
    temporal_compress: int = 4    # Wan-VAE T4
    n_slots: int = 16             # 槽位总数 N（覆盖背包所有槽 index）

    # 动作键 -> 轴（小写匹配；按你的键位改）
    attack_keys: Tuple[str, ...] = ("mouse0",)
    pickup_keys: Tuple[str, ...] = ("f",)
    equip_keys:  Tuple[str, ...] = ("alpha1", "alpha2", "alpha3", "alpha4")

    # 动作条件键（连续控制，进动作注入、不进状态层）。按住生效，要重建“按住区间”。
    # 顺序固定 = action 向量的通道顺序；按你的键位/需要增减。
    hold_keys: Tuple[str, ...] = ("w", "a", "s", "d")

    log_lag_s: float = 0.20       # 背包记录相对真实命中的滞后（从对齐分析估）
    snap_back_s: float = 0.50     # 用按键精定位时向前(更早)搜索的窗口
    snap_fwd_s: float = 0.15      # 向后搜索的窗口

    count_mode: str = "events"    # "events"(事件次数,主用) | "true_delta"(真值,eval对照)

    # 哪些槽 renderable（手持/穿戴，需回画面渲染）。供训练侧决定 cross-attn 取哪些 token。
    renderable_is_held: bool = True  # v1：仅“当前手持槽”可渲染

    # 物品名（仅打印好看；不影响逻辑）
    slot_names: Dict[int, str] = field(default_factory=lambda: {
        0: "Sword", 1: "Ring", 2: "Mace", 3: "Axe", 7: "Waterskin",
        8: "Wood", 9: "Stone", 10: "PlantFiber", 11: "BloodRose",
        12: "FireBlossom", 13: "T01"})

    # 武器 g(物品id) -> weapon_id(embedding索引)。**跨片段必须一致**：所有 clip 共用
    # 同一张 embedding 表，同一把武器在不同 clip 必须拿到同一个 weapon_id。
    # 正式多片段训练前，把游戏里所有手持武器的 g 在这里钉死（空时按本片段出现顺序自动编号，
    #   只适合单片段调试，会打印出来让你拷进 config）。-1 = 无手持。
    weapon_id_map: Dict[int, int] = field(default_factory=dict)


# ============================================================================
# 规范中间表示
# ============================================================================
@dataclass
class InputEvent:
    t: float            # 视频秒
    key: str            # 归一化小写键名

@dataclass
class InventorySample:
    t: float            # 视频秒（背包 vt）
    counts: List[int]
    held: Optional[int]
    held_g: Optional[int] = None   # 当前手持武器的 g(物品id)，用于 weapon-id embedding
    slot_g: Optional[List[Optional[int]]] = None  # 逐槽 g(物品id)，pickup_item_g 溯源用（M4-C）

@dataclass
class Event:
    axis: str
    slot: int
    dir: Optional[int]
    t: float
    pixel_frame: int
    delta: int = 0
    weapon_g: Optional[int] = None  # equip 事件携带：切到的武器 g
    item_g: Optional[int] = None    # gather/pickup 事件携带：变化那个槽的物品 g（M4-C，见 project_labels）


# ============================================================================
# 时钟
# ============================================================================
def read_meta(inv_path: str) -> Tuple[float, int]:
    meta = json.loads(open(inv_path, encoding="utf-8-sig").readline())
    return meta["videoOriginMs"], meta.get("hz", 2)

def wall_to_vt(s: str, origin_ms: float) -> float:
    dt = datetime.fromisoformat(s.replace(" ", "T", 1).replace(" +", "+"))
    return (dt.timestamp() * 1000 - origin_ms) / 1000.0


# ============================================================================
# 解析器（按真实日志格式；换数据源只改这三个）
# ============================================================================
def parse_input_log(path: str, cfg: Config, origin: float) -> List[InputEvent]:
    rows = json.loads(open(path, encoding="utf-8-sig").read())
    out = []
    for e in rows:
        if e.get("状态") != "按下":          # 只取按下时刻
            continue
        out.append(InputEvent(t=wall_to_vt(e["时间戳"], origin), key=e["按键"].lower()))
    return out

def parse_inventory_log(path: str, cfg: Config) -> List[InventorySample]:
    samples = []
    for line in open(path, encoding="utf-8-sig"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if d.get("type") != "inv":
            continue
        counts = [0] * cfg.n_slots
        slot_g = [None] * cfg.n_slots
        gmap = {}
        for it in d["bag"]:
            s = it["s"]
            if 0 <= s < cfg.n_slots:
                counts[s] = it["c"]; gmap[it["g"]] = s; slot_g[s] = it["g"]
        wg = (d.get("equip", {}).get("weapon") or {}).get("g")
        samples.append(InventorySample(t=d["vt"], counts=counts, held=gmap.get(wg), held_g=wg, slot_g=slot_g))
    return samples

def load_cursor(path: Optional[str], cfg: Config, origin: float, n_frames: int) -> np.ndarray:
    """每帧 [angle/180, dir_x, dir_z]；path=None 时占位(全0)，等真光标到了再传。"""
    if not path:
        return np.zeros((n_frames, 3), dtype=np.float32)
    rows = json.loads(open(path, encoding="utf-8-sig").read())
    samp = sorted(((wall_to_vt(r["时间戳"], origin),
                    r.get("瞄准朝向角", 0.0),
                    r.get("瞄准方向", {}).get("x", 0.0),
                    r.get("瞄准方向", {}).get("z", 0.0)) for r in rows), key=lambda x: x[0])
    ts = [s[0] for s in samp]
    cur = np.zeros((n_frames, 3), dtype=np.float32)
    for f in range(n_frames):
        i = min(bisect.bisect_left(ts, f / cfg.fps), len(samp) - 1)
        cur[f] = (samp[i][1] / 180.0, samp[i][2], samp[i][3])
    return cur


def build_action_track(input_path: str, cfg: Config, origin: float, n_latent: int) -> np.ndarray:
    """
    逐 latent 帧的动作条件向量 [n_latent, len(hold_keys)]，每维=该 latent 帧该键是否“按住”。
    与状态层无关 —— 这是喂动作注入的连续控制信号（WASD 等）。
    要点：移动键是按住持续生效，必须用“按下…松开”区间重建，不能像事件那样只取按下瞬间。
    （path=None / 缺文件 -> 占位全 0。）
    """
    K = len(cfg.hold_keys)
    act = np.zeros((n_latent, K), dtype=np.float32)
    if not input_path:
        return act
    key_idx = {k: i for i, k in enumerate(cfg.hold_keys)}
    rows = json.loads(open(input_path, encoding="utf-8-sig").read())
    # 收集每个 hold 键的“按下/松开”时刻（视频秒）
    evs = []  # (t, key_idx, is_down)
    for e in rows:
        k = e["按键"].lower()
        if k in key_idx:
            evs.append((wall_to_vt(e["时间戳"], origin), key_idx[k], e.get("状态") == "按下"))
    evs.sort()
    # 重建每个键的按住区间，投影到 latent 帧
    down_t = [None] * K  # 各键当前未闭合的按下时刻
    for t, ki, is_down in evs:
        if is_down:
            if down_t[ki] is None:
                down_t[ki] = t
        else:  # 松开 -> 闭合一个区间 [down, t]
            if down_t[ki] is not None:
                lo = pixel_to_latent(t_to_frame(down_t[ki], cfg), cfg)
                hi = pixel_to_latent(t_to_frame(t, cfg), cfg)
                act[max(0, lo):min(n_latent, hi + 1), ki] = 1.0
                down_t[ki] = None
    # 片段结束时仍按住的键，持续到末尾
    for ki in range(K):
        if down_t[ki] is not None:
            lo = pixel_to_latent(t_to_frame(down_t[ki], cfg), cfg)
            act[max(0, lo):n_latent, ki] = 1.0
    return act

def get_video_meta(video_path: str, cfg: Config) -> Tuple[int, float]:
    """返回 (模型帧数, 时长秒)。用 ffprobe 读时长，按 cfg.fps 折算模型帧数。"""
    import subprocess
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=duration", "-of", "json", video_path],
        capture_output=True, text=True).stdout
    dur = float(json.loads(out)["streams"][0]["duration"])
    return int(round(dur * cfg.fps)), dur


# ============================================================================
# 核心逻辑（与格式无关）
# ============================================================================
def pixel_to_latent(f: int, cfg: Config) -> int:
    """Which latent-frame chunk a real, IN-RANGE pixel frame index falls into
    (event localization). Still ceil: for f within a video's actual frame
    count, this correctly names the (1, 4, 4, 4, ...)-grouped chunk index the
    real Wan-VAE would put it in -- ceil and floor only diverge exactly at
    the total-count boundary, which is num_latent_frames's job below, not
    this function's."""
    if f <= 0:
        return 0
    return math.ceil(f / cfg.temporal_compress)

def num_latent_frames(n_frames: int, cfg: Config) -> int:
    """Total latent frames the real Wan-VAE emits for n_frames real frames
    (DESIGN.md SS8 Q5, resolved on real data in M4-B). FLOOR, not
    pixel_to_latent(n_frames-1)+1's ceil: WanVAE_.encode groups frames as
    [1, 4, 4, 4, ...] (`iter_ = 1+(t-1)//4`) and SILENTLY DROPS trailing
    frames that don't complete a group of `temporal_compress` -- it does not
    round up and pad. Measured on real 5-min test data: for several probed
    n_frames the ceil formula predicted 7 latent frames, the VAE actually
    returned 6. Do not revert this to ceil; that's the mismatch this fixes."""
    if n_frames <= 0:
        return 0
    return 1 + (n_frames - 1) // cfg.temporal_compress

def t_to_frame(t: float, cfg: Config) -> int:
    return max(0, int(round(t * cfg.fps)))


def derive_events(inv: List[InventorySample], cfg: Config) -> List[Event]:
    """相邻快照差异 -> 事件。事件时刻=跳变记录那行 b.t（反推：变了才记）。"""
    raw = []
    for a, b in zip(inv[:-1], inv[1:]):
        et = b.t
        for s in range(min(len(a.counts), len(b.counts))):
            d = b.counts[s] - a.counts[s]
            # M4-C: item_g resolved from the AFTER snapshot's slot content -- same slot
            # that changed still holds that item post-change (matches held_g's own logic
            # for equip). Not trained on; kept so a future pickup_item_g label doesn't
            # need to re-derive events from the raw logs (see project_labels).
            g = b.slot_g[s] if b.slot_g and s < len(b.slot_g) else None
            if d > 0:
                raw.append(Event("gather", s, DIR_UP, et, t_to_frame(et, cfg), delta=d, item_g=g))
            elif d < 0:
                raw.append(Event("pickup", s, DIR_DOWN, et, t_to_frame(et, cfg), delta=d, item_g=g))
        if a.held != b.held:
            if b.held is not None:
                # 切到某把武器：slot=真实槽位，weapon_g 带上
                raw.append(Event("equip", b.held, None, et, t_to_frame(et, cfg), delta=0,
                                 weapon_g=b.held_g))
            elif a.held is not None:
                # 收起武器变空手（held: 某把 -> 无）：也是一次 equip 事件。
                # slot 用 none-slot 类(索引 n_slots，即 N+1 softmax 的"无"类)；weapon_g=None -> held_weapon_id=-1
                raw.append(Event("equip", cfg.n_slots, None, et, t_to_frame(et, cfg), delta=0,
                                 weapon_g=None))
    return raw


def refine_timing(events: List[Event], inputs: List[InputEvent], cfg: Config) -> List[Event]:
    """用按键把事件精定位到具体帧：gather->mouse0, pickup->F, equip->Alpha。"""
    inputs = sorted(inputs, key=lambda e: e.t)
    its = [e.t for e in inputs]

    def nearest(center, keyset, back, fwd):
        best, bdt = None, 1e9
        i = bisect.bisect_left(its, center - back)
        while i < len(inputs) and inputs[i].t <= center + fwd:
            if inputs[i].key in keyset:
                dt = abs(inputs[i].t - center)
                if dt < bdt:
                    best, bdt = inputs[i], dt
            i += 1
        return best

    for ev in events:
        if ev.axis == "equip":
            hit = nearest(ev.t, set(cfg.equip_keys), cfg.snap_back_s + 0.3, cfg.snap_fwd_s)
            if hit:
                ev.t = hit.t
        else:
            # F 在窗口内 -> 是 pickup，否则 gather
            fhit = nearest(ev.t, set(cfg.pickup_keys), cfg.snap_back_s + 0.3, cfg.snap_fwd_s + 0.15)
            if fhit is not None:
                ev.axis = "pickup"
                ev.dir = DIR_ADD if ev.delta > 0 else DIR_REMOVE
                ev.t = fhit.t
            else:
                ev.axis = "gather"
                mhit = nearest(ev.t, set(cfg.attack_keys), cfg.log_lag_s + cfg.snap_back_s, cfg.snap_fwd_s)
                ev.t = mhit.t if mhit is not None else ev.t - cfg.log_lag_s
        ev.pixel_frame = t_to_frame(ev.t, cfg)
    return events


# M4-C real-data decision (DESIGN.md SS3.2): pickup gets neither slot nor dir.
# 8-session log audit: pickup touches 33 distinct items over only 251 events (most
# classes single-digit samples -- unfittable regardless of visual disambiguation,
# and dropped loot renders as an indistinguishable blob anyway), and pickup's `dir`
# measured as a hard constant (always DIR_ADD, 251/251) -- zero information content.
AXES_WITH_SLOT = ("gather", "equip")
AXES_WITH_DIR = ("gather",)


def project_labels(events: List[Event], n_latent: int, cfg: Config) -> Tuple[Dict[str, np.ndarray], int]:
    out = {}
    for ax in AXES:
        out[f"{ax}_type"] = np.full(n_latent, TYPE_NONE, dtype=np.int64)
    for ax in AXES_WITH_SLOT:
        out[f"{ax}_slot"] = np.full(n_latent, IGNORE, dtype=np.int64)
    for ax in AXES_WITH_DIR:
        out[f"{ax}_dir"] = np.full(n_latent, IGNORE, dtype=np.int64)
    # M4-C (deferred pickup_color upgrade, see DESIGN.md SS4.3b): raw item g at each
    # pickup fire frame, untrained metadata -- NOT in PER_FRAME_1D_KEYS, so it never
    # reaches a training batch, but it's on disk so upgrading pickup to a color
    # classifier later is a relabel-from-.npz pass, not a re-run of the whole
    # event-derivation pipeline against the raw logs.
    out["pickup_item_g"] = np.full(n_latent, IGNORE, dtype=np.int64)
    collisions = 0
    for ev in sorted(events, key=lambda e: e.pixel_frame):
        lf = pixel_to_latent(ev.pixel_frame, cfg)
        if lf >= n_latent:
            continue
        if out[f"{ev.axis}_type"][lf] == TYPE_FIRE:
            collisions += 1
        out[f"{ev.axis}_type"][lf] = TYPE_FIRE
        if ev.axis in AXES_WITH_SLOT:
            out[f"{ev.axis}_slot"][lf] = ev.slot
        if ev.axis in AXES_WITH_DIR:
            out[f"{ev.axis}_dir"][lf] = ev.dir
        if ev.axis == "pickup" and ev.item_g is not None:
            out["pickup_item_g"][lf] = ev.item_g
    return out, collisions


def mark_none_kind(labels: Dict[str, np.ndarray], inputs: List[InputEvent],
                   n_latent: int, cfg: Config) -> np.ndarray:
    action = set(cfg.attack_keys) | set(cfg.pickup_keys) | set(cfg.equip_keys)
    has_action = np.zeros(n_latent, dtype=bool)
    half = cfg.snap_back_s
    for e in inputs:
        if e.key in action:
            for f in (t_to_frame(e.t - half, cfg), t_to_frame(e.t, cfg), t_to_frame(e.t + half, cfg)):
                lf = pixel_to_latent(f, cfg)
                if 0 <= lf < n_latent:
                    has_action[lf] = True
    any_fire = np.zeros(n_latent, dtype=bool)
    for ax in AXES:
        any_fire |= (labels[f"{ax}_type"] == TYPE_FIRE)
    return np.where(any_fire, 0, np.where(has_action, 1, 2)).astype(np.int64)


def resolve_weapon_ids(events: List[Event], cfg: Config, verbose=True) -> Dict[int, int]:
    """g(物品id) -> weapon_id(embedding索引)。优先用 cfg.weapon_id_map（跨片段一致）；
    为空则按本片段 equip 出现的 g 排序自动编号（仅单片段调试用，会打印让你拷进 config）。"""
    if cfg.weapon_id_map:
        return dict(cfg.weapon_id_map)
    gs = sorted({ev.weapon_g for ev in events if ev.axis == "equip" and ev.weapon_g is not None})
    wid = {g: i for i, g in enumerate(gs)}
    if verbose and wid:
        print(f"[weapon_id] 本片段自动编号(单片段调试用；多片段训练前请钉进 Config.weapon_id_map):")
        for g, i in wid.items():
            print(f"           g={g} -> weapon_id={i}  ({cfg.slot_names.get('?', '')})")
    return wid


def integrate(events: List[Event], n_latent: int, cfg: Config, count_mode: str,
              wid_map: Optional[Dict[int, int]] = None):
    """逐 latent 帧 S_t。count_mode: events(事件次数) | true_delta(真值)。
    同时输出 held(槽位) 和 held_weapon_id(武器id索引, 供 weapon-id embedding)。

    M4-C: gather 和 pickup 不再共用同一个按 slot 分类的 count 数组——pickup 不再
    按物品分类（见 project_labels 上面的注释），只用一个与 slot 无关的聚合计数器
    pickup_count 记"拾取发生了几次"（8-session 审计：pickup 恒为 ADD 方向，故只
    数次数，不看 count_mode 的 events/true_delta 区分，两者语义在这里重合）。
    gather 自己的按 slot count 完全不受影响，两条线各数各的，不做去重（这两个数
    组本就分开维护，不存在重复计数的可能）。"""
    wid_map = wid_map or {}
    by = {}
    for ev in events:
        by.setdefault(pixel_to_latent(ev.pixel_frame, cfg), []).append(ev)
    held = np.full(n_latent, -1, dtype=np.int64)
    held_wid = np.full(n_latent, -1, dtype=np.int64)
    count = np.zeros((n_latent, cfg.n_slots), dtype=np.int64)
    pickup_count = np.zeros(n_latent, dtype=np.int64)
    c = [0] * cfg.n_slots; pc = 0; h = -1; hw = -1
    for lf in range(n_latent):
        for ev in by.get(lf, []):
            if ev.axis == "gather":
                if count_mode == "events":
                    step = 1 if (ev.dir in (DIR_UP, DIR_ADD) or ev.delta > 0) else -1
                else:
                    step = ev.delta
                c[ev.slot] = max(0, c[ev.slot] + step)
            elif ev.axis == "pickup":
                pc += 1
            elif ev.axis == "equip":
                if ev.slot >= cfg.n_slots:      # 收起武器变空手（none-slot 类）
                    h = -1; hw = -1
                else:
                    h = ev.slot
                    hw = wid_map.get(ev.weapon_g, -1) if ev.weapon_g is not None else -1
        held[lf] = h; held_wid[lf] = hw; count[lf] = c; pickup_count[lf] = pc
    return held, held_wid, count, pickup_count



# ============================================================================
# 串起来
# ============================================================================
def process_clip(inv_path, input_path, mouse_path, video_path, out_path, cfg: Config,
                 verbose=True, _injected=None):
    if _injected is not None:
        inputs, inv, n_frames, origin = _injected
    else:
        origin, _ = read_meta(inv_path)
        inputs = parse_input_log(input_path, cfg, origin)
        inv = parse_inventory_log(inv_path, cfg)
        n_frames, _dur = get_video_meta(video_path, cfg)

    n_latent = num_latent_frames(n_frames, cfg)
    events = refine_timing(derive_events(inv, cfg), inputs, cfg)
    labels, collisions = project_labels(events, n_latent, cfg)
    none_kind = mark_none_kind(labels, inputs, n_latent, cfg)
    wid_map = resolve_weapon_ids(events, cfg, verbose=verbose)
    held, held_wid, count, pickup_count = integrate(events, n_latent, cfg, cfg.count_mode, wid_map)
    _, _, count_true, _ = integrate(events, n_latent, cfg, "true_delta", wid_map)
    cursor = load_cursor(mouse_path, cfg, (origin if _injected is None else origin), n_frames) \
        if _injected is None else np.zeros((n_frames, 3), np.float32)
    # 动作条件（WASD 等按住状态，逐 latent 帧）；selftest 无真实 input 文件时给占位全 0
    action = build_action_track(input_path, cfg, origin, n_latent) \
        if _injected is None else np.zeros((n_latent, len(cfg.hold_keys)), np.float32)

    out = {**labels, "none_kind": none_kind, "held": held, "held_weapon_id": held_wid,
           "count": count, "count_true": count_true, "pickup_count": pickup_count,
           "n_latent": np.array([n_latent]), "n_frames": np.array([n_frames]),
           "cursor": cursor, "action": action, "collisions": np.array([collisions])}
    if out_path:
        np.savez_compressed(out_path, **out)

    if verbose:
        from collections import Counter
        nm = cfg.slot_names
        print(f"模型帧 {n_frames}({cfg.fps}fps) -> latent 帧 {n_latent}；事件 {len(events)}，同帧碰撞 {collisions}")
        cnt = Counter(e.axis for e in events)
        for ev in sorted(events, key=lambda e: e.t):
            d = f"{ev.delta:+d}" if ev.axis != "equip" else "手持"
            print(f"  vt={ev.t:6.2f} latent={pixel_to_latent(ev.pixel_frame,cfg):4d} "
                  f"{ev.axis:7s} slot{ev.slot:2d}({nm.get(ev.slot,'?')}) {d}")
        nk = np.bincount(none_kind, minlength=3)
        print(f"轴分布 {dict(cnt)}；none_kind fire={nk[0]} 难负例={nk[1]} 静止={nk[2]}")
        print("S_t.count[事件次数] ", {nm.get(i,i): int(count[-1][i]) for i in range(cfg.n_slots) if count[-1][i]})
        print("S_t.count[真值对照] ", {nm.get(i,i): int(count_true[-1][i]) for i in range(cfg.n_slots) if count_true[-1][i]})
        print("S_t.pickup_count[累计次数, 不分类] ", int(pickup_count[-1]))
        print("held =", int(held[-1]), nm.get(int(held[-1]), ""), "  cursor", cursor.shape)
        if out_path:
            print("已写出", out_path)
    return out


# ============================================================================
# selftest（合成数据，不需真文件）
# ============================================================================
def selftest():
    print(">>> selftest\n")
    cfg = Config(fps=16, temporal_compress=4, n_slots=4,
                 equip_keys=("3",), pickup_keys=("f",), attack_keys=("mouse0",))
    n_frames = 160
    inputs = [InputEvent(1.00, "mouse0"), InputEvent(2.00, "f"),
              InputEvent(3.00, "3"), InputEvent(5.00, "mouse0")]
    inv = []; counts = [0,0,0,0]; held = None; held_g = None
    slot_g = [5555, 8888, None, None]  # slot0=gather 目标(木头之类)，slot1=pickup 目标；固定绑定，同真实数据 SS3.4
    for k in range(21):
        t = 0.5 * k
        if abs(t-1.5) < 1e-6: counts[0] += 2
        if abs(t-2.5) < 1e-6: counts[1] += 1
        if abs(t-3.5) < 1e-6: held = 2; held_g = 7777   # 切到武器(slot2)，g=7777
        if abs(t-4.5) < 1e-6: held = None; held_g = None # 收起武器变空手
        inv.append(InventorySample(t, list(counts), held, held_g=held_g, slot_g=list(slot_g)))
    out = process_clip(None, None, None, None, None, cfg, verbose=False,
                       _injected=(inputs, inv, n_frames, 0.0))
    n_latent = int(out["n_latent"][0])
    assert n_latent == 40  # floor((160-1)/4)+1 -- DESIGN.md SS8 Q5, matches real VAE's `iter_`
    lf = pixel_to_latent(t_to_frame(1.0, cfg), cfg)
    assert out["gather_type"][lf] == TYPE_FIRE and out["gather_slot"][lf] == 0
    assert out["pickup_type"][pixel_to_latent(t_to_frame(2.0,cfg),cfg)] == TYPE_FIRE
    le = pixel_to_latent(t_to_frame(3.0, cfg), cfg)        # 切到武器(slot2)
    assert out["equip_type"][le] == TYPE_FIRE and out["equip_slot"][le] == 2
    assert out["none_kind"][pixel_to_latent(t_to_frame(5.0,cfg),cfg)] == 1   # 砍空=难负例
    assert out["count"][-1][0] == 1                                          # 事件计数不变
    # M4-C: pickup 不再分类（无 slot/dir），只有独立的聚合计数器
    assert "pickup_slot" not in out and "pickup_dir" not in out
    assert out["pickup_count"][-1] == 1                                      # 一次 pickup，不进 gather 的按 slot count
    assert out["count"][-1][1] == 0  # pickup 不写 count[slot]，slot1(f 拾取那次)保持 0
    # M4-C: pickup_item_g 元信息（不训练，供未来 pickup_color 升级复用，见 DESIGN.md SS4.3b）
    pu_lf = pixel_to_latent(t_to_frame(2.0, cfg), cfg)
    assert out["pickup_item_g"][pu_lf] == 8888          # 那一帧真实存了 slot1 的 g
    assert out["pickup_item_g"][pu_lf - 1] == IGNORE     # 非 fire 帧仍是 IGNORE 占位
    # held / held_weapon_id：equip 前 -1，切到武器后 slot=2 / weapon_id=0
    assert out["held"][le] == 2 and out["held_weapon_id"][le] == 0
    assert out["held_weapon_id"][le-1] == -1
    # 收起武器变空手：equip 事件 slot=none-slot 类(=n_slots)，之后 held 回到 -1
    lu = pixel_to_latent(t_to_frame(4.5, cfg), cfg)
    assert out["equip_type"][lu] == TYPE_FIRE and out["equip_slot"][lu] == cfg.n_slots
    assert out["held"][-1] == -1 and out["held_weapon_id"][-1] == -1
    print("selftest 通过 ✓  对齐/n_latent(floor)/三轴/难负例/积分器/weapon_id/收武器 正确")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--inv"); ap.add_argument("--input")
    ap.add_argument("--mouse", default=None); ap.add_argument("--video")
    ap.add_argument("--out", default="clip.npz")
    ap.add_argument("--n_slots", type=int, default=16)
    ap.add_argument("--count_mode", default="events", choices=["events", "true_delta"])
    a = ap.parse_args()
    if a.selftest:
        selftest(); return
    if not (a.inv and a.input and a.video):
        ap.error("需要 --inv --input --video（--mouse 可选；无光标先省略）。或用 --selftest。")
    cfg = Config(n_slots=a.n_slots, count_mode=a.count_mode)
    process_clip(a.inv, a.input, a.mouse, a.video, a.out, cfg)


if __name__ == "__main__":
    main()

# SPDX-License-Identifier: Apache-2.0
"""StateHead: 5 independent classification heads over 3 axes (DESIGN_1.md v2 §4.1).

Reads one hidden-state vector per latent frame (from the DiT's last block,
before CausalHead -- see `return_hidden` on CausalWanModel._forward_train)
and predicts, per axis, whether an event fired (`type`); gather additionally
predicts which inventory slot it touched (`slot`, N+1 classes: N real slots +
1 "none"); equip additionally predicts `slot`, no dir.

pickup has NEITHER slot nor dir (M4-C, real-data decision, DESIGN.md SS3.2):
dropped loot renders as a visually-indistinguishable blob regardless of
item, and the real 8-session log audit backs this up two ways independent
of the visual argument -- (1) pickup touches 33 distinct items across only
251 events, so most classes have single-digit sample counts, nowhere near
enough to fit a reliable classifier even if the pixels did disambiguate it;
(2) pickup's `dir` was measured to be a literal constant (always DIR_ADD,
zero events of the other value in 251 samples) -- an entire CE head with
provably zero information content. Only `pickup_type` (did a pickup fire)
survives; its magnitude is tracked by a separate, non-classified aggregate
counter (`preprocess_vrising_final.integrate`'s `pickup_count`), not by this
per-slot head structure at all -- see DESIGN.md SS6.

gather's `dir` head was dropped the same way (M4-C, real 8-session audit,
2026-07-02): 3777/3781 real gather-fire frames are DIR_UP, the 4 DIR_DOWN
outliers are isolated single-frame events traced to consumption/crafting
draws that fell into the gather bucket for lack of nearby F-key evidence
(refine_timing's default-to-gather branch), not genuine reverse-gather
signal -- same "hard constant, zero information content" pattern as
pickup's dir. `count`'s own accumulation is unaffected either way: the
TRAINING target (`preprocess_vrising_final.integrate()`, offline, reads
`Event.dir` straight from the log) never went through this head to begin
with. Only the INFERENCE-time closed-loop reconstruction
(`utils.integrator.integrate_events`, used by `overfit_closed_loop_m4a.py`)
consumed a predicted `gather_dir`; that function now hardcodes DIR_UP
instead of reading a prediction, matching how pickup_count already has no
direction concept at all.

M4-B additions (DESIGN.md SS5.2): `state_head_losses` takes optional
per-axis inverse-frequency class weights, and `RMSLossNormalizer` +
`combine_state_and_diffusion_loss` implement the "RMS normalize everything
against its own running scale, then sum" rule for combining the CE terms
with the diffusion loss -- so a loss whose natural scale happens to start
larger (or smaller) doesn't dominate (or vanish) just from that scale
difference, regardless of how important it actually is.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

AXES = ("gather", "pickup", "equip")

# (label_key -> num_classes), `slot_classes` filled in at construction time.
_HEAD_SPEC_TEMPLATE = (
    ("gather_type", 2), ("gather_slot", None),
    ("pickup_type", 2),
    ("equip_type", 2), ("equip_slot", None),
)

# DESIGN_1.md §5.2 masking rule.
TYPE_KEYS = ("gather_type", "pickup_type", "equip_type")          # full-frame CE, never masked
MASKED_KEYS = ("gather_slot", "equip_slot")  # ignore_index=-100 on non-fire frames
IGNORE_INDEX = -100


class StateHead(nn.Module):
    def __init__(self, dim, n_slots=16, use_trunk=True):
        super().__init__()
        slot_classes = n_slots + 1  # N real slots + 1 "none" class
        spec = {key: (slot_classes if n is None else n) for key, n in _HEAD_SPEC_TEMPLATE}

        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.GELU()) if use_trunk else nn.Identity()
        self.heads = nn.ModuleDict()
        for key, n_classes in spec.items():
            head = nn.Linear(dim, n_classes)
            nn.init.zeros_(head.weight)  # zero-init (DESIGN_1.md §0 rule 2): logits start at 0 for every class
            nn.init.zeros_(head.bias)
            self.heads[key] = head
        self.label_keys = tuple(spec.keys())

    def forward(self, hidden_states):
        """hidden_states: [B, L, dim] (one vector per latent frame) -> dict[label_key] = logits [B, L, num_classes]."""
        h = self.trunk(hidden_states)
        return {key: head(h) for key, head in self.heads.items()}


class StateHeadV2(nn.Module):
    """M4-D (DESIGN.md SS4.1c, 2026-07-03): per-BLOCK event-COUNT regression with
    attention-pooled readout, replacing V1's per-frame instant classification.

    Why the reformulation (real-data diagnosis, diag_002800 analysis):
    - V1's fire evidence is DIFFUSE (a plateau over the event's neighborhood,
      no frame-0 peak: chopping looks the same on the inventory-tick frame and
      its neighbors) -> "which exact frame" is unanswerable, but "how many
      events in this 0.75s block" is exactly what the diffuse evidence supports
      -- and it is directly the quantity the count integrator needs.
    - V1's threshold-counting was uncalibratable (2.4x over-fire on a trained
      session, 0.6x under on val); a Poisson rate is calibrated BY the loss.
    - V1 read a MEAN-POOL over each frame's 1560 spatial tokens, diluting
      small-footprint evidence (loot sparkle, inventory tick) by 1/1560; V2
      uses per-axis learned queries with multi-head attention over the block's
      tokens so each axis can look where its evidence lives.

    Outputs per block of `num_frame_per_block` latent frames:
      gather_rate [B, nb, n_slots]  expected gather events per slot (softplus)
      pickup_rate [B, nb]           expected pickup events
      equip_rate  [B, nb]           expected equip transitions
      equip_slot  [B, nb, n_slots+1] aux CE logits (new held slot; masked to
                                     blocks that contain a transition)

    gather's per-slot rates subsume V1's gather_slot classification: the count
    trajectory is just the cumulative sum of predicted per-slot rates.

    Init: rate-head weights zero; rate-head bias -4.5 (softplus ~= 0.011
    events/block, the empirical base-rate ballpark: gather 0.138/block over
    16 slots, pickup 0.009, equip 0.016). This is the count-regression analog
    of SS0 rule 2's "logits start 0": start at the PRIOR, not at softplus(0)
    = 0.69 events/block, which would be a 50x over-prediction pretraining shock.
    """

    def __init__(self, dim, n_slots=16, num_frame_per_block=3, pool_heads=8):
        super().__init__()
        self.n_slots = n_slots
        self.num_frame_per_block = num_frame_per_block
        # one learned query per axis; shared MHA over the block's tokens
        self.queries = nn.Parameter(torch.randn(3, dim) * 0.02)  # gather/pickup/equip
        self.pool = nn.MultiheadAttention(dim, pool_heads, batch_first=True)
        self.trunk = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

        self.gather_rate = nn.Linear(dim, n_slots)
        self.pickup_rate = nn.Linear(dim, 1)
        self.equip_rate = nn.Linear(dim, 1)
        self.equip_slot = nn.Linear(dim, n_slots + 1)
        for head in (self.gather_rate, self.pickup_rate, self.equip_rate):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -4.5)
        nn.init.zeros_(self.equip_slot.weight)
        nn.init.zeros_(self.equip_slot.bias)

    def forward(self, hidden_tokens):
        """hidden_tokens: [B, F, S, dim] token-level hidden (return_hidden="tokens").
        F must be a multiple of num_frame_per_block. Returns dict of per-block outputs."""
        B, F_, S, D = hidden_tokens.shape
        fpb = self.num_frame_per_block
        nb = F_ // fpb
        toks = hidden_tokens.reshape(B * nb, fpb * S, D)
        q = self.queries.unsqueeze(0).expand(B * nb, -1, -1).to(toks.dtype)
        pooled, _ = self.pool(q, toks, toks, need_weights=False)  # [B*nb, 3, D]
        pooled = self.trunk(pooled).unflatten(0, (B, nb))         # [B, nb, 3, D]
        g, p, e = pooled[:, :, 0], pooled[:, :, 1], pooled[:, :, 2]
        return {
            "gather_rate": F.softplus(self.gather_rate(g)),        # [B, nb, n_slots]
            "pickup_rate": F.softplus(self.pickup_rate(p)).squeeze(-1),  # [B, nb]
            "equip_rate": F.softplus(self.equip_rate(e)).squeeze(-1),    # [B, nb]
            "equip_slot": self.equip_slot(e),                      # [B, nb, n_slots+1]
        }


def build_block_count_labels(count_w, pickup_count_w, equip_type_w, equip_slot_w,
                             num_frame_per_block, count_prev, pickup_prev):
    """Per-block count labels from the window's per-frame label streams (M4-D).

    Inputs (window = the noisy target frames only, length F):
      count_w        [B, F, n_slots] cumulative per-slot gather count (events mode --
                     collision-safe: simultaneous events all appear in the delta)
      pickup_count_w [B, F] cumulative pickup count
      equip_type_w   [B, F] per-frame equip fire (0/1)
      equip_slot_w   [B, F] new held slot on fire frames, IGNORE elsewhere
      count_prev     [B, n_slots] cumulative count at the frame BEFORE the window
      pickup_prev    [B]
    Returns dict:
      gather_count [B, nb, n_slots] events per block per slot (long)
      pickup_count [B, nb] (long)
      equip_count  [B, nb] (long)
      equip_slot   [B, nb] new-held slot of the LAST transition in the block,
                   IGNORE_INDEX where the block has no transition
    """
    B, F_ = pickup_count_w.shape
    fpb = num_frame_per_block
    nb = F_ // fpb
    # cumulative streams -> per-block deltas: value at block end minus value at block start-1
    ends = count_w[:, fpb - 1::fpb]                                   # [B, nb, n_slots]
    starts = torch.cat([count_prev.unsqueeze(1), ends[:, :-1]], dim=1)
    gather_count = (ends - starts).long()
    p_ends = pickup_count_w[:, fpb - 1::fpb]
    p_starts = torch.cat([pickup_prev.unsqueeze(1), p_ends[:, :-1]], dim=1)
    pickup_count = (p_ends - p_starts).long()

    eq = equip_type_w.unflatten(1, (nb, fpb))                         # [B, nb, fpb]
    equip_count = eq.sum(dim=2).long()
    slot = torch.full((B, nb), IGNORE_INDEX, dtype=torch.long, device=equip_slot_w.device)
    es = equip_slot_w.unflatten(1, (nb, fpb))
    for f in range(fpb):  # later fires overwrite -> LAST transition's slot wins
        fired = eq[:, :, f] == 1
        slot = torch.where(fired, es[:, :, f], slot)
    return {"gather_count": gather_count, "pickup_count": pickup_count,
            "equip_count": equip_count, "equip_slot": slot}


def state_head_v2_losses(outputs, labels, block_weights=None):
    """M4-D losses. Poisson NLL for the three count heads, masked CE for the
    equip-slot aux head. `block_weights` [B, nb] (e.g. (1-sigma) noise-level
    weights, DESIGN.md SS5.2c): high where the block's frames are visible,
    ~0 where they're noise -- the model can only count events it can SEE.
    Weights are applied per block and renormalized so a batch that happened to
    sample mostly-noisy blocks doesn't read as 'tiny loss' for the wrong reason."""
    def wmean(per_block):  # per_block: [B, nb]
        if block_weights is None:
            return per_block.mean()
        w = block_weights.to(per_block.dtype)
        return (per_block * w).sum() / w.sum().clamp_min(1e-6)

    losses = {}
    losses["gather_count"] = wmean(F.poisson_nll_loss(
        outputs["gather_rate"], labels["gather_count"].to(outputs["gather_rate"].dtype),
        log_input=False, full=False, eps=1e-6, reduction="none").mean(dim=-1))
    losses["pickup_count"] = wmean(F.poisson_nll_loss(
        outputs["pickup_rate"], labels["pickup_count"].to(outputs["pickup_rate"].dtype),
        log_input=False, full=False, eps=1e-6, reduction="none"))
    losses["equip_count"] = wmean(F.poisson_nll_loss(
        outputs["equip_rate"], labels["equip_count"].to(outputs["equip_rate"].dtype),
        log_input=False, full=False, eps=1e-6, reduction="none"))

    target = labels["equip_slot"]
    valid = target != IGNORE_INDEX
    if valid.any():
        ce = F.cross_entropy(
            outputs["equip_slot"].flatten(0, 1), target.flatten(0, 1),
            ignore_index=IGNORE_INDEX, reduction="none").unflatten(0, target.shape)
        # same block weights, restricted to valid blocks
        w = torch.ones_like(ce) if block_weights is None else block_weights.to(ce.dtype)
        w = w * valid.to(ce.dtype)
        losses["equip_slot"] = (ce * w).sum() / w.sum().clamp_min(1e-6)
    else:
        losses["equip_slot"] = outputs["equip_slot"].sum() * 0.0
    return losses


def state_head_losses(logits, batch, class_weights=None):
    """Per-key CE losses, DESIGN_1.md §5.2:
    - *_type (3 axes): full-frame CE, NEVER masked -- includes none_kind==1 hard
      negatives ("pressed the key, nothing fired"), which is exactly the signal
      the model must learn to look at state before firing.
    - *_slot / *_dir: masked CE, ignore_index=-100 on non-fire frames (already
      baked into the .npz by preprocess_vrising_final.py -- no extra mask needed).
    `class_weights` (DESIGN.md SS5.2, M4-B): optional dict[label_key] ->
    [n_classes] float tensor, sqrt(1/freq) per class (see
    utils.video_state_dataset.compute_inverse_freq_class_weights) -- passed
    straight to F.cross_entropy's `weight`. None (default) reproduces M1's
    unweighted behavior exactly.
    Returns dict[label_key] -> scalar loss. A masked key with zero valid targets
    in the batch returns an in-graph, exactly-zero loss (not nan from a 0/0 mean).
    """
    class_weights = class_weights or {}
    losses = {}
    for key in TYPE_KEYS:
        target = batch[key].reshape(-1)
        losses[key] = F.cross_entropy(
            logits[key].reshape(-1, logits[key].shape[-1]), target, weight=class_weights.get(key)
        )
    for key in MASKED_KEYS:
        target = batch[key].reshape(-1)
        valid = target != IGNORE_INDEX
        if not valid.any():
            losses[key] = logits[key].sum() * 0.0
            continue
        losses[key] = F.cross_entropy(
            logits[key].reshape(-1, logits[key].shape[-1]), target,
            ignore_index=IGNORE_INDEX, weight=class_weights.get(key),
        )
    return losses


class RMSLossNormalizer:
    """DESIGN.md SS5.2: normalize every loss term against an EMA of its OWN
    running RMS magnitude before summing -- a per-batch RMS would be too
    noisy (one batch with zero fire frames for an axis reads as "loss is
    tiny" for the wrong reason); an EMA tracks the term's typical scale
    across many steps instead. `value / EMA_rms(value)` keeps every
    normalized term hovering near unit scale regardless of the term's
    natural magnitude, so `lambda_ce` (DESIGN.md's `loss = diffusion_loss +
    lambda*sum(8 CE)`) means the same thing regardless of which axes happen
    to be firing this batch.
    """

    def __init__(self, momentum=0.99, eps=1e-6):
        self.momentum = momentum
        self.eps = eps
        self._mean_sq = {}

    def normalize(self, name, value):
        with torch.no_grad():
            mean_sq = value.detach().float() ** 2
            prev = self._mean_sq.get(name)
            self._mean_sq[name] = mean_sq if prev is None else (
                self.momentum * prev + (1 - self.momentum) * mean_sq
            )
            rms = self._mean_sq[name].clamp_min(self.eps).sqrt()
        return value / rms


def combine_state_and_diffusion_loss(ce_losses, diffusion_loss, normalizer, lambda_ce=1.0):
    """DESIGN.md SS5.2: `loss = diffusion_loss + lambda * sum(8 CE)`, with
    every term (the 8 CE losses AND diffusion_loss) RMS-normalized against
    its own running scale first (`normalizer`, an `RMSLossNormalizer`
    shared/persisted across training steps -- a fresh one each step defeats
    the point). Returns (total_loss, normalized_ce_dict, normalized_diffusion)
    for logging."""
    normalized_ce = {k: normalizer.normalize(k, v) for k, v in ce_losses.items()}
    normalized_diffusion = normalizer.normalize("diffusion", diffusion_loss)
    total = normalized_diffusion + lambda_ce * sum(normalized_ce.values())
    return total, normalized_ce, normalized_diffusion


class StateHeadV3(StateHeadV2):
    """V2 + a trainable feature-interaction adapter (SS7d fallback: if the frozen
    backbone's t=0 features hold event evidence only in RELATIONAL form -- "this
    token changed between frame 1 and frame 3" -- V2's single pooling pass cannot
    extract it; a small self-attention stack over the block's tokens can).

    2x2 spatial merge (1560 -> 390 tokens/frame) + down-projection to adapter_dim,
    learned spatio-temporal position embedding, then TransformerEncoder layers over
    the block's merged tokens; V2's query-pooling + count heads run at adapter_dim.
    ~10M params on top of the (replaced) V2 pooling. Same outputs / labels / losses.
    """

    def __init__(self, dim, n_slots=16, num_frame_per_block=3, pool_heads=8,
                 adapter_dim=512, adapter_layers=2, spatial_hw=(30, 52)):
        super().__init__(adapter_dim, n_slots, num_frame_per_block, pool_heads)
        self.spatial_hw = spatial_hw
        h2, w2 = spatial_hw[0] // 2, spatial_hw[1] // 2
        self.merge = nn.Linear(dim * 4, adapter_dim)
        self.pos = nn.Parameter(
            torch.randn(1, num_frame_per_block * h2 * w2, adapter_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            adapter_dim, 8, dim_feedforward=4 * adapter_dim, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.adapter = nn.TransformerEncoder(layer, adapter_layers)

    def forward(self, hidden_tokens):
        B, F_, S, D = hidden_tokens.shape
        H, W = self.spatial_hw
        assert S == H * W, f"expected {H}x{W}={H*W} tokens, got {S}"
        fpb = self.num_frame_per_block
        nb = F_ // fpb
        x = hidden_tokens.reshape(B, F_, H // 2, 2, W // 2, 2, D)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, F_, (H // 2) * (W // 2), 2 * 2 * D)
        x = self.merge(x)                                   # [B, F, 390, a]
        x = x.reshape(B * nb, fpb * x.shape[2], -1) + self.pos.to(x.dtype)
        x = self.adapter(x)                                 # token interaction
        q = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1).to(x.dtype)
        pooled, _ = self.pool(q, x, x, need_weights=False)
        pooled = self.trunk(pooled).unflatten(0, (B, nb))
        g, p, e = pooled[:, :, 0], pooled[:, :, 1], pooled[:, :, 2]
        return {
            "gather_rate": F.softplus(self.gather_rate(g)),
            "pickup_rate": F.softplus(self.pickup_rate(p)).squeeze(-1),
            "equip_rate": F.softplus(self.equip_rate(e)).squeeze(-1),
            "equip_slot": self.equip_slot(e),
        }

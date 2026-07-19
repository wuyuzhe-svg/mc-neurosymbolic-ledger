#!/usr/bin/env python3
"""Migration core: load Matrix-Game 2.0 base_model (bidirectional WanModel) and
EXPAND its keyboard action space 6 -> 7 to teach PLACE (RMB); ATTACK is native.

Keyboard layout after expansion (per-block keyboard_embed.0: [128, 7]):
  0 forward, 1 back, 2 left, 3 right, 4 jump, 5 ATTACK(LMB)  (base, loaded from ckpt)
  6 PLACE(RMB)                                               (NEW, zero-init -> no-op at start)
Mouse stays 2-dim (camera pitch/yaw). Everything else loads verbatim from base_model
(missing=0/unexpected=0 at 6-dim; the only shimmed tensor is keyboard_embed.0.weight).
"""
import os, sys, json, torch
sys.path.insert(0, os.environ.get("MG2_REPO", "/root/autodl-tmp/MatrixGame_repo/Matrix-Game-2"))
from safetensors.torch import load_file
from wan.modules.model import WanModel

MG2 = os.environ.get("MG2_WEIGHTS", "/root/autodl-tmp/mg2_weights")
N_HELD = 40                            # held-item vocab (held_id in {-1(none),0..39}; aligns bag_counts)

class LedgerInjector(torch.nn.Module):
    """Per-frame (held_id, held_count) -> [B,T,dim], zero-init output (no-op at start).
    Learned via L_recon: anchors the held item (fixes drift) + natural depletion."""
    def __init__(self, dim, n_held=N_HELD):
        super().__init__()
        self.held_emb = torch.nn.Embedding(n_held + 1, 128)          # +1: none(-1)->idx0
        self.count_mlp = torch.nn.Sequential(torch.nn.Linear(1, 128), torch.nn.SiLU(), torch.nn.Linear(128, 128))
        self.proj = torch.nn.Linear(256, dim)
        torch.nn.init.zeros_(self.proj.weight); torch.nn.init.zeros_(self.proj.bias)  # zero-init
    def forward(self, held_id, count):                               # held_id[B,T] long (-1=UNKNOWN,0=none,1-39=item)
        h = self.held_emb((held_id + 1).clamp(0, self.held_emb.num_embeddings - 1))
        c = self.count_mlp(torch.log1p(count.clamp(min=0)).unsqueeze(-1))
        out = self.proj(torch.cat([h, c], dim=-1))                   # [B,T,dim]
        valid = (held_id >= 0).unsqueeze(-1).to(out.dtype)           # UNKNOWN(-1)->0 injection (25.8%, unlabeled)
        return out * valid
# base 6-dim keyboard = [W(fwd) S(back) A(left) D(right) jump ATTACK] (verified: kb5=attack,
# confirmed by ForgeWM pad_keyboard 4->6 + our motion probe arm/scene=4.3). attack ALREADY
# exists at index 5 -> we only ADD place (index 6). KB 6->7.
KB_BASE, KB_NEW = 6, 7
ATTACK_IDX, PLACE_IDX = 5, 6           # attack = base's existing dim 5; place = new dim 6

def load_expanded_base(device="cuda", dtype=torch.bfloat16, keyboard_dim=KB_NEW, verbose=True, ledger=False):
    """Build WanModel with keyboard_dim_in=keyboard_dim, load base_model weights,
    shimming keyboard_embed.0.weight from 6->keyboard_dim (copy base cols, zero new)."""
    cfg = json.load(open(f"{MG2}/base_model/base_config.json"))
    cfg.pop("_class_name", None); cfg.pop("_diffusers_version", None)
    cfg["action_config"]["keyboard_dim_in"] = keyboard_dim
    model = WanModel.from_config(cfg)

    sd = load_file(f"{MG2}/base_model/diffusion_pytorch_model.safetensors")
    # shim every block's keyboard_embed.0.weight [128,6] -> [128,keyboard_dim]
    n_shim = 0
    for k in list(sd.keys()):
        if k.endswith("keyboard_embed.0.weight") and sd[k].shape[1] == KB_BASE:
            w = sd[k]                                  # [128, 6]
            new = torch.zeros(w.shape[0], keyboard_dim, dtype=w.dtype)
            new[:, :KB_BASE] = w                       # copy base cols; new cols stay 0 (no-op)
            sd[k] = new; n_shim += 1
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if verbose:
        print(f"[mg2] keyboard {KB_BASE}->{keyboard_dim}, shimmed {n_shim} keyboard_embed.0; "
              f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        assert len(unexpected) == 0, f"unexpected keys: {unexpected[:5]}"
    if ledger:
        model.ledger_injector = LedgerInjector(cfg["dim"])           # zero-init, trained via L_recon
        if verbose: print(f"[mg2] ledger injector attached (dim={cfg['dim']}, n_held={N_HELD})", flush=True)
    return model.to(device, dtype)


if __name__ == "__main__":
    dev, dt = "cuda", torch.bfloat16
    m = load_expanded_base(dev, dt).eval().requires_grad_(False)
    # smoke forward at native 44x80, 15 latent frames, with attack+place actions set
    LH, LW, F = 44, 80, 15
    NUM_PIX = (F - 1) * 4 + 1
    x = torch.randn(1, 16, F, LH, LW, device=dev, dtype=dt)
    cond_concat = torch.zeros(1, 20, F, LH, LW, device=dev, dtype=dt); cond_concat[:, :4, 0] = 1.0
    vc = torch.zeros(1, 257, 1280, device=dev, dtype=dt)
    kb = torch.zeros(1, NUM_PIX, KB_NEW, device=dev, dtype=dt); kb[:, :, 0] = 1.0   # forward
    kb[:, 20:, PLACE_IDX] = 1.0                                                      # place in 2nd half
    ms = torch.zeros(1, NUM_PIX, 2, device=dev, dtype=dt)
    t = torch.tensor([500.0], device=dev, dtype=dt)
    with torch.no_grad():
        out = m(x, t=t, visual_context=vc, cond_concat=cond_concat, keyboard_cond=kb, mouse_cond=ms)
    print(f"[mg2] smoke forward OK: out {tuple(out.shape)} (expect [1,16,{F},{LH},{LW}])", flush=True)
    print("MG2-MODEL-SMOKE-DONE", flush=True)

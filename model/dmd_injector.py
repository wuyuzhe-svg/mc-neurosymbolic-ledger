# SPDX-License-Identifier: Apache-2.0
"""
M5-A (方案A): DMD with our injector-equipped causal generator (M5A_PLAN.md D2).

Overrides only model construction:
  - generator: frozen LongLive backbone + trainable injectors, init from an
    M4-D checkpoint (load_m4d_checkpoint), StateHeadV2 kept for t=0 readout
  - real_score: bidirectional Wan, frozen; loads the Stage A0 V-Rising-adapted
    weights when args.a0_ckpt is set (raw Wan otherwise -- smoke tests only)
  - fake_score: same init as real_score, trainable (standard DMD)
  - text encoder skipped entirely: we train with the fixed precomputed prompt
    embedding (data/prompt_embed.pt), never raw text (saves 22GB umt5)

Conditioning: control/state streams ride in conditional_dict; the per-chunk
slicing lives in WanDiffusionWrapper.forward (D3), so DMD/pipeline internals
stay untouched.
"""
import torch

from model.dmd import DMD
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper


class InjectorDMD(DMD):
    def __init__(self, args, device):
        super().__init__(args, device)
        # activation checkpointing is incompatible with the cached causal
        # rollout: backward recompute re-reads the KV cache AFTER later blocks
        # advanced it (saved 11-block K/V vs recomputed 8-block -> shape
        # mismatch). DMD.__init__ enables it on both nets; keep it ONLY on the
        # stateless bidirectional critic. The generator grad step is a single
        # 3-frame block -- cheap without checkpointing.
        self.generator.model.gradient_checkpointing = False

    def _run_generator(self, image_or_video_shape, conditional_dict,
                       initial_latent=None, slice_last_frames=21):
        """Multi-frame-prime rollout: pipeline returns prime+gen concatenated;
        the 21-frame bidirectional critics must score ONLY the generated
        window (the prime is real data -- nothing to judge). Base's slicing is
        unusable here: its slice_last_frames doubles as the pipeline's
        GRADIENT window (start_gradient_frame_index = total - slice), so -1
        kills all grads, and 21 triggers a per-step VAE decode/re-encode.
        Call the pipeline directly: gradient window = all gen frames, then
        drop the prime. Fixed-length rollout (min==max==gen_frames)."""
        assert initial_latent is not None, "M5-A always primes with real latents"
        conditional_dict["initial_latent"] = initial_latent
        # Gradient window = the LAST block only (Self-Forcing style). Wider
        # windows OOM at 96GB without activation checkpointing (which the
        # cached rollout cannot use, see __init__), and the last block is the
        # right training signal anyway: it sits on the most self-generated
        # context -- exactly the distribution the injectors must calibrate to.
        pred, t_from, t_to = self._consistency_backward_simulation(
            noise=torch.randn(image_or_video_shape,
                              device=self.device, dtype=self.dtype),
            slice_last_frames=self.num_frame_per_block,
            **conditional_dict)
        pred = pred[:, initial_latent.shape[1]:].to(self.dtype)
        return pred, None, t_from, t_to

    def _initialize_models(self, args, device):
        from evaluate_m4d import load_m4d_checkpoint  # deferred: heavy import chain

        model_kwargs = getattr(args, "model_kwargs", {})
        local_attn_size = model_kwargs.get("local_attn_size", 12)
        sink_size = model_kwargs.get("sink_size", 3)
        self.local_attn_size = local_attn_size

        # ---- generator: our causal model, frozen backbone + trainable injectors
        self.generator = WanDiffusionWrapper(
            is_causal=True, local_attn_size=local_attn_size, sink_size=sink_size,
            timestep_shift=getattr(args, "timestep_shift", 5.0))
        gen_model, self.state_head, self.m4d_args, m4d_step, _ = load_m4d_checkpoint(
            args.m4d_ckpt, device, getattr(args, "base_ckpt", "checkpoints/models/longlive_base.pt"))
        self.generator.model = gen_model  # frozen backbone, injectors requires_grad=True
        # state_head stays trainable: D5/D8 TF anchor batches update it (it never
        # participates in the DMD losses, so no gradient reaches it from those)
        n_train = sum(p.numel() for p in gen_model.parameters() if p.requires_grad) / 1e6
        print(f"InjectorDMD generator: M4-D step {m4d_step}, {n_train:.0f}M trainable injector params")

        # ---- real / fake score: bidirectional many-step Wan
        self.real_score = WanDiffusionWrapper(
            is_causal=False, timestep_shift=getattr(args, "timestep_shift", 5.0))
        a0 = getattr(args, "a0_ckpt", None)
        if a0:
            ck = torch.load(a0, map_location="cpu")
            self.real_score.model.load_state_dict(ck["model"])
            print(f"real_score: A0-adapted teacher from {a0} (step {ck['step']})")
        else:
            print("real_score: RAW Wan (no A0 adaptation) -- smoke-test configuration only")
        self.real_score.model.requires_grad_(False)

        self.fake_score = WanDiffusionWrapper(
            is_causal=False, timestep_shift=getattr(args, "timestep_shift", 5.0))
        if a0:
            self.fake_score.model.load_state_dict(ck["model"])
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = None  # fixed precomputed prompt embedding only
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

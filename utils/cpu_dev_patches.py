# SPDX-License-Identifier: Apache-2.0
"""
CPU-only development scaffolding for the symbolic-state-layer project
(DESIGN_1.md §2.5 / §7 M-1). The base model assumes a CUDA box: `utils/memory.py`
grabs a `cuda:0` device at import time, `wan/__init__.py` eagerly pulls in
T5/CLIP/VAE/ftfy/image2video (irrelevant for model-only dev), and both
`flex_attention` (self-attn) and `flash_attention` (cross-attn, hardcoded
`assert q.device.type=='cuda'`) require an actual CUDA+Triton stack.

`load_causal_wan_model()` works around all four, purely so the
teacher-forcing wiring (clean_x, masks, StateHead) can be developed and unit
tested on a CPU dev machine. It is NOT a substitute for verifying the real
fused Triton kernels on GPU -- that is still pending (DESIGN_1.md §2.5
"未在 GPU 上验证的部分"). Do not import this module from anything meant to
run on the real training box; import `wan.modules.causal_model` directly there.
"""
import importlib.util
import os
import sys
import types

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _stub_utils_memory():
    if "utils.memory" in sys.modules:
        return
    fake = types.ModuleType("utils.memory")
    fake.cpu = None
    fake.gpu = None
    fake.gpu_complete_modules = []
    fake.DynamicSwapInstaller = object
    fake.get_cuda_free_memory_gb = lambda *a, **k: 0
    fake.log_gpu_memory = lambda *a, **k: None
    fake.move_model_to_device_with_memory_preservation = lambda *a, **k: None
    sys.modules["utils.memory"] = fake


def _stub_pkg(name, path):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    m.__path__ = [path]
    sys.modules[name] = m


def _load_submodule(dotted_name, filename):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(
        dotted_name, os.path.join(_REPO_ROOT, "wan", "modules", filename)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_file_as_module(dotted_name, abs_path):
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    spec = importlib.util.spec_from_file_location(dotted_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _sdpa_flash_attention_shim(q, k, v, q_lens=None, k_lens=None, **kwargs):
    causal = kwargs.get("causal", False)
    qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)
    return out.transpose(1, 2)


_mask_dense_cache = {}


def _sdpa_flex_attention_shim(query, key, value, block_mask=None, **kwargs):
    key_id = id(block_mask)
    dense = _mask_dense_cache.get(key_id)
    if dense is None:
        Lq, Lk = query.shape[-2], key.shape[-2]
        q_idx = torch.arange(Lq, device=query.device).view(-1, 1)
        kv_idx = torch.arange(Lk, device=query.device).view(1, -1)
        dense = block_mask.mask_mod(0, 0, q_idx, kv_idx)
        _mask_dense_cache[key_id] = dense
    return torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=dense)


def load_causal_wan_model():
    """Import wan.modules.causal_model with CUDA-only bits patched to CPU/SDPA
    equivalents. Returns the module (use `.CausalWanModel` from it)."""
    _stub_utils_memory()
    if "wan.modules.causal_model" in sys.modules:
        return sys.modules["wan.modules.causal_model"]

    _stub_pkg("wan", os.path.join(_REPO_ROOT, "wan"))
    _stub_pkg("wan.modules", os.path.join(_REPO_ROOT, "wan", "modules"))

    _load_submodule("wan.modules.attention", "attention.py")
    wan_model_mod = _load_submodule("wan.modules.model", "model.py")
    causal_model = _load_submodule("wan.modules.causal_model", "causal_model.py")

    wan_model_mod.flash_attention = _sdpa_flash_attention_shim  # cross-attn, model.py:189
    causal_model.flex_attention = _sdpa_flex_attention_shim  # self-attn, no-kv_cache path
    return causal_model


def load_state_head_module():
    """Import model/state_head.py without going through `model/__init__.py`,
    which eagerly does `from .dmd import DMD` -> pulls in the full DMD trainer
    stack (pipeline/, utils.wan_wrapper, wan.* incl. T5/CLIP/ftfy) for no
    reason -- state_head.py itself only imports torch."""
    return _load_file_as_module("model.state_head", os.path.join(_REPO_ROOT, "model", "state_head.py"))

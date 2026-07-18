# SPDX-License-Identifier: Apache-2.0
"""
GPU counterpart to utils/cpu_dev_patches.py: imports wan.modules.causal_model
the same way -- bypassing wan/__init__.py's eager T5/CLIP/VAE/ftfy/image2video
imports, irrelevant for model-only verification -- but WITHOUT shimming
flex_attention/flash_attention to SDPA. Use this on an actual CUDA box to
exercise the real fused Triton flex_attention kernel and the real flash_attn
library; use cpu_dev_patches.py on a CPU dev machine instead.
"""
import importlib.util
import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


def load_causal_wan_model():
    """Import wan.modules.causal_model with the real CUDA flex_attention/
    flash_attention paths intact. Returns the module (use `.CausalWanModel`)."""
    if "wan.modules.causal_model" in sys.modules:
        return sys.modules["wan.modules.causal_model"]

    _stub_pkg("wan", os.path.join(_REPO_ROOT, "wan"))
    _stub_pkg("wan.modules", os.path.join(_REPO_ROOT, "wan", "modules"))

    _load_submodule("wan.modules.attention", "attention.py")
    _load_submodule("wan.modules.model", "model.py")
    causal_model = _load_submodule("wan.modules.causal_model", "causal_model.py")
    return causal_model

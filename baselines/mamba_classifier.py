"""Sequence classifier around Mamba-1 / Mamba-2 / Mamba-3 mixers.

    Linear(d_input -> d_model)
    -> num_layers x [ h + Dropout(mixer(RMSNorm(h))) ]      (pre-norm residual)
    -> RMSNorm -> mean over time -> Linear(d_model -> num_classes)

Backends
    "mamba_ssm": the official CUDA modules (mamba_ssm.modules.mamba_simple.Mamba,
                 mamba_ssm.modules.mamba2.Mamba2).  Needs a GPU even to import.
    "port":      the PyTorch ports in baselines/mamba_blocks.py.  Same parameter
                 shapes, runs anywhere, but Mamba-1/2 loop over time in Python,
                 so they are only practical for short sequences (CPU tests,
                 parameter counting).
    "auto":      "mamba_ssm" when CUDA and mamba_ssm are available, else "port".

Mamba-3 always uses the port with the chunked scan: its official SISO kernel
needs a Hopper GPU (TMA tensor descriptors), see baselines/mamba_blocks.py.
"""

import importlib.util
import warnings

import torch
import torch.nn as nn

from .mamba_blocks import Mamba1Block, Mamba2Block, Mamba3Block

VARIANTS = ("mamba", "mamba2", "mamba3")


def _official_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except ImportError as e:
        if importlib.util.find_spec("mamba_ssm") is not None:
            warnings.warn(f"mamba_ssm is installed but does not import ({e}); "
                          "using the PyTorch ports")
        return False
    return True


def resolve_backend(backend: str = "auto") -> str:
    """'auto' -> 'mamba_ssm' when CUDA and mamba_ssm are available, else 'port'."""
    if backend == "auto":
        return "mamba_ssm" if _official_available() else "port"
    return backend


def build_mixer(variant: str, d_model: int, d_state: int, expand: int,
                d_conv: int = 4, headdim: int = 64, rope_fraction: float = 0.5,
                backend: str = "auto", chunk_size: int = 64) -> nn.Module:
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    if variant == "mamba3":
        return Mamba3Block(d_model=d_model, d_state=d_state, expand=expand,
                           headdim=headdim, rope_fraction=rope_fraction,
                           scan="chunked", chunk_size=chunk_size)
    backend = resolve_backend(backend)
    if backend == "port":
        if variant == "mamba":
            return Mamba1Block(d_model=d_model, d_state=d_state, d_conv=d_conv,
                               expand=expand)
        return Mamba2Block(d_model=d_model, d_state=d_state, d_conv=d_conv,
                           expand=expand, headdim=headdim)
    if backend != "mamba_ssm":
        raise ValueError(f"backend must be 'auto', 'mamba_ssm' or 'port', got {backend!r}")
    if variant == "mamba":
        from mamba_ssm.modules.mamba_simple import Mamba
        return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    from mamba_ssm.modules.mamba2 import Mamba2
    return Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                  headdim=headdim)


class _Layer(nn.Module):
    def __init__(self, mixer: nn.Module, d_model: int, drop_rate: float):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.mixer = mixer
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.dropout(self.mixer(self.norm(h)))


class MambaClassifier(nn.Module):
    """Mean-pooled sequence classifier over Mamba-1/2/3 mixers.

    Args:
        variant:       "mamba", "mamba2" or "mamba3"
        d_input:       input channels (count a time channel here if you add one)
        num_classes:   output classes
        d_model:       residual width
        d_state:       SSM state size
        num_layers:    residual mixer layers
        expand:        mixer inner width = expand * d_model
        d_conv:        causal conv width (Mamba-1/2; Mamba-3 has no conv)
        headdim:       head width (Mamba-2/3)
        rope_fraction: fraction of d_state rotated by RoPE (Mamba-3)
        drop_rate:     dropout on each layer's residual branch
        backend:       "auto", "mamba_ssm" or "port" (Mamba-1/2 only)
    """

    def __init__(self, variant: str, d_input: int, num_classes: int,
                 d_model: int = 16, d_state: int = 16, num_layers: int = 1,
                 expand: int = 2, d_conv: int = 4, headdim: int = 64,
                 rope_fraction: float = 0.5, drop_rate: float = 0.0,
                 backend: str = "auto", chunk_size: int = 64):
        super().__init__()
        self.variant = variant
        self.embed = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            _Layer(build_mixer(variant, d_model, d_state, expand, d_conv, headdim,
                               rope_fraction, backend, chunk_size),
                   d_model, drop_rate)
            for _ in range(num_layers)
        ])
        self.norm_f = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_input) -> logits (B, num_classes)."""
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_f(h).mean(dim=1))

#!/usr/bin/env python3
"""Check the PyTorch Mamba ports (baselines/mamba_blocks.py) against mamba_ssm.

Run on a GPU node with mamba_ssm installed (see "Mamba baselines" in the README):

    python baselines/check_mamba_ports.py

Method: build the official module, copy its weights into the port (the gated
RMSNorms are the only name difference: submodule -> flat parameter), feed both
the same input and compare outputs.  Checked at the Fading Flash toy size and at
the EigenWorms drop-rate size.

Mamba-3's official SISO kernel only compiles on Hopper (sm_90: TMA tensor
descriptors plus a tl.dot with N=1).  Elsewhere the check stops at the kernel
boundary: the kernel call inside the official module is intercepted, and the
tensors the module prepared for it are fed to both PyTorch scans (pre-kernel
parity).  That verifies the projections, activations, norms, biases and RoPE
angles; the recurrence itself is transcribed from the official single-step
kernel (mamba3_siso_step.py) and the chunked scan is checked against it.

Reference results, RTX 4090, mamba_ssm 2.3.2.post1:
    Mamba-1 vs official CUDA scan, fp32        ~1e-6 (max abs)
    Mamba-2 vs official SSD kernel, fp32        3.7e-05
    Mamba-3 pre-kernel parity, both scans       ~1e-8
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from baselines import mamba_blocks as P  # noqa: E402


def _stub_missing_optional_deps():
    """mamba_ssm/__init__ imports its LM head (-> transformers) and mamba2.py a hub
    mixin (-> huggingface_hub), neither needed for the bare blocks.  Stub them only
    if they are not installed."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        tr = types.ModuleType('transformers')
        gen = types.ModuleType('transformers.generation')
        gen.GenerateDecoderOnlyOutput = object
        gen.TextStreamer = object
        utils = types.ModuleType('transformers.utils')
        utils.WEIGHTS_NAME, utils.CONFIG_NAME = 'pytorch_model.bin', 'config.json'
        hub = types.ModuleType('transformers.utils.hub')
        hub.cached_file = None
        utils.hub = hub
        tr.generation, tr.utils = gen, utils
        sys.modules.update({'transformers': tr, 'transformers.generation': gen,
                            'transformers.utils': utils, 'transformers.utils.hub': hub})
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        hh = types.ModuleType('huggingface_hub')

        class _Mixin:
            def __init_subclass__(cls, **kw):
                pass

        hh.PyTorchModelHubMixin = _Mixin
        sys.modules['huggingface_hub'] = hh


def report(tag, ref, got):
    ref, got = ref.float(), got.float()
    err = (ref - got).abs()
    print(f'  {tag:<44} max_abs {err.max().item():.3e}   '
          f'rel {(err.mean() / ref.abs().mean().clamp_min(1e-12)).item():.3e}')
    return err.max().item()


def remap(sd, renames):
    sd = dict(sd)
    for src, dst in renames.items():
        if src in sd:
            sd[dst] = sd.pop(src)
    return sd


def check_mamba1(device, d_model, d_state, expand, d_conv, seqlen, tag):
    from mamba_ssm.modules.mamba_simple import Mamba
    torch.manual_seed(0)
    official = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                     use_fast_path=False).to(device)
    port = P.Mamba1Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
    port.load_state_dict(official.state_dict(), strict=True)
    port.to(device)
    u = torch.randn(2, seqlen, d_model, device=device)
    with torch.no_grad():
        return report(f'Mamba-1 {tag}', official(u), port(u))


def check_mamba2(device, d_model, d_state, expand, d_conv, headdim, seqlen, tag):
    import mamba_ssm.modules.mamba2 as m2
    torch.manual_seed(0)
    official = m2.Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         headdim=headdim, ngroups=1, use_mem_eff_path=False).to(device)
    port = P.Mamba2Block(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
                         headdim=headdim)
    missing, unexpected = port.load_state_dict(
        remap(official.state_dict(), {'norm.weight': 'norm_weight'}), strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    port.to(device)
    u = torch.randn(2, seqlen, d_model, device=device)
    # causal_conv1d only takes a multiple of 8 channels here; below that, use the
    # module's own nn.Conv1d path.  The SSD kernel runs either way.
    conv_fn = m2.causal_conv1d_fn
    if official.conv1d.in_channels % 8:
        m2.causal_conv1d_fn = None
        tag += ', nn.Conv1d path'
    try:
        with torch.no_grad():
            return report(f'Mamba-2 {tag}', official(u), port(u))
    finally:
        m2.causal_conv1d_fn = conv_fn


def installed_a_activation():
    """Which A activation the installed Mamba3 uses: softplus up to the 2.3.2.post1
    release, heavy_tail on upstream main since #962."""
    import mamba_ssm.modules.mamba3 as m3
    return 'heavy_tail' if hasattr(m3, 'heavy_tail_activation') else 'softplus'


def check_mamba3_prekernel(device, d_model, d_state, expand, headdim, seqlen, tag):
    import mamba_ssm.modules.mamba3 as m3
    torch.manual_seed(0)
    official = m3.Mamba3(d_model=d_model, d_state=d_state, expand=expand,
                         headdim=headdim, ngroups=1).to(device)
    ports = {}
    for scan in ('sequential', 'chunked'):
        port = P.Mamba3Block(d_model=d_model, d_state=d_state, expand=expand,
                             headdim=headdim, scan=scan, a_activation=installed_a_activation())
        missing, unexpected = port.load_state_dict(
            remap(official.state_dict(), {'B_norm.weight': 'B_norm_weight',
                                          'C_norm.weight': 'C_norm_weight'}), strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        ports[scan] = port.to(device)

    captured = {}

    def capture(**kw):
        captured.update(kw)
        b, l, h, p = kw['V'].shape
        return torch.zeros(b, l, h, p, dtype=kw['V'].dtype, device=kw['V'].device)

    orig = m3.mamba3_siso_combined
    m3.mamba3_siso_combined = capture
    u = torch.randn(2, seqlen, d_model, device=device)
    try:
        with torch.no_grad():
            official(u)
    finally:
        m3.mamba3_siso_combined = orig
    if not captured:
        raise RuntimeError('mamba3_siso_combined was never called')

    worst = 0.0
    kw = {k: captured[k] for k in ('Q', 'K', 'V', 'ADT', 'DT', 'Trap', 'Q_bias', 'K_bias',
                                   'Angles', 'D', 'Z')}
    with torch.no_grad():
        y = P.mamba3_siso_recurrence(**kw, nheads=ports['sequential'].nheads)
        ref = official.out_proj(y.reshape(2, seqlen, -1))
        for scan, port in ports.items():
            worst = max(worst, report(f'Mamba-3 {tag}, {scan} scan', ref, port(u)))
    return worst


def check_mamba3_end_to_end(device, d_model, d_state, expand, headdim, seqlen, tag):
    from mamba_ssm.modules.mamba3 import Mamba3
    torch.manual_seed(0)
    official = Mamba3(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                      ngroups=1).to(device)
    port = P.Mamba3Block(d_model=d_model, d_state=d_state, expand=expand, headdim=headdim,
                         a_activation=installed_a_activation())
    port.load_state_dict(remap(official.state_dict(), {'B_norm.weight': 'B_norm_weight',
                                                        'C_norm.weight': 'C_norm_weight'}),
                         strict=False)
    port.to(device)
    u = torch.randn(2, seqlen, d_model, device=device)
    with torch.no_grad():
        return report(f'Mamba-3 {tag}, end to end', official(u), port(u))


def main():
    if not torch.cuda.is_available():
        sys.exit('needs a GPU: mamba_ssm imports its Triton kernels at module level')
    device = torch.device('cuda')
    _stub_missing_optional_deps()
    import mamba_ssm
    cap = torch.cuda.get_device_capability()
    print(f'torch {torch.__version__}  mamba_ssm {mamba_ssm.__version__}  '
          f'{torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})  '
          f'Mamba-3 A activation: {installed_a_activation()}\n')

    results = {
        'Mamba-1 toy': check_mamba1(device, 4, 3, 1, 3, 40, 'toy (d_model 4, d_state 3)'),
        'Mamba-1 EigenWorms': check_mamba1(device, 16, 16, 16, 4, 300,
                                           'EigenWorms (d_model 16, expand 16)'),
        'Mamba-2 toy': check_mamba2(device, 4, 3, 1, 3, 4, 40, 'toy (d_model 4, 1 head)'),
        'Mamba-2 EigenWorms': check_mamba2(device, 16, 16, 32, 4, 64, 300,
                                           'EigenWorms (expand 32, headdim 64)'),
        'Mamba-3 pre-kernel toy-ish': check_mamba3_prekernel(device, 4, 4, 1, 4, 40,
                                                             'pre-kernel, d_model 4'),
        'Mamba-3 pre-kernel EigenWorms': check_mamba3_prekernel(
            device, 16, 16, 32, 64, 300, 'pre-kernel, EigenWorms size'),
    }
    if cap[0] >= 9:
        results['Mamba-3 end to end'] = check_mamba3_end_to_end(
            device, 16, 16, 32, 64, 300, 'EigenWorms size')
    else:
        print('  Mamba-3 end to end: skipped, the official kernel needs Hopper (sm_90)')

    tol = {'Mamba-2': 1e-3}
    bad = [k for k, v in results.items() if v > tol.get(k.split()[0], 1e-4)]
    print('\nALL CHECKS PASS' if not bad else f'\nFAILED: {bad}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()

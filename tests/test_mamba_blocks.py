"""Mamba ports: the chunked Mamba-3 scan equals the sequential one."""
import pytest
import torch

from baselines.mamba_blocks import Mamba3Block, mamba3_siso_chunked, mamba3_siso_recurrence
from baselines.mamba_classifier import MambaClassifier


def _inputs(b, L, h, n, p, n_angles, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g, dtype=dtype)
    DT = torch.nn.functional.softplus(r(b, h, L))
    A = -(torch.rand(b, h, L, generator=g, dtype=dtype) * 2 + 1e-4)
    return dict(Q=r(b, L, 1, n), K=r(b, L, 1, n), V=r(b, L, h, p), ADT=A * DT, DT=DT,
                Trap=r(b, h, L), Q_bias=r(h, n), K_bias=r(h, n), Angles=r(b, L, h, n_angles),
                D=r(h), Z=r(b, L, h, p), nheads=h)


@pytest.mark.parametrize("L", [1, 15, 16, 17, 100])
@pytest.mark.parametrize("chunk", [16, 64])
def test_chunked_equals_sequential(L, chunk):
    kw = _inputs(2, L, 3, 8, 4, 2)
    ref = mamba3_siso_recurrence(**kw)
    got = mamba3_siso_chunked(**kw, chunk_size=chunk)
    assert (ref - got).abs().max() <= 1e-12 * ref.abs().max()


def test_chunked_gradients_equal_sequential():
    kw = _inputs(2, 40, 3, 8, 4, 2)
    leaves = ["Q", "K", "V", "ADT", "Trap", "Angles", "D", "Z"]
    for k in leaves:
        kw[k].requires_grad_(True)
    g_ref = torch.autograd.grad(mamba3_siso_recurrence(**kw).square().sum(), [kw[k] for k in leaves])
    g_got = torch.autograd.grad(mamba3_siso_chunked(**kw, chunk_size=16).square().sum(),
                                [kw[k] for k in leaves])
    for a, b in zip(g_ref, g_got):
        assert (a - b).abs().max() <= 1e-11 * a.abs().max()


def test_block_scans_agree():
    torch.manual_seed(0)
    seq = Mamba3Block(d_model=8, d_state=8, expand=2, headdim=8, scan="sequential")
    chk = Mamba3Block(d_model=8, d_state=8, expand=2, headdim=8, scan="chunked", chunk_size=16)
    chk.load_state_dict(seq.state_dict())
    u = torch.randn(2, 50, 8)
    with torch.no_grad():
        torch.testing.assert_close(seq(u), chk(u), rtol=1e-5, atol=1e-6)


def test_a_activation():
    """softplus (the 2.3.2.post1 release) and heavy_tail (upstream main) are different models."""
    torch.manual_seed(0)
    sp = Mamba3Block(d_model=8, d_state=8, expand=2, headdim=8)
    ht = Mamba3Block(d_model=8, d_state=8, expand=2, headdim=8, a_activation="heavy_tail")
    ht.load_state_dict(sp.state_dict())
    u = torch.randn(2, 20, 8)
    with torch.no_grad():
        assert (sp(u) - ht(u)).abs().max() > 1e-4
    with pytest.raises(ValueError):
        Mamba3Block(d_model=8, d_state=8, a_activation="relu")


@pytest.mark.parametrize("variant", ["mamba", "mamba2", "mamba3"])
def test_classifier_port_backend(variant):
    torch.manual_seed(0)
    m = MambaClassifier(variant, d_input=7, num_classes=5, d_model=8, d_state=8,
                        expand=2, headdim=8, backend="port")
    out = m(torch.randn(3, 20, 7))
    assert out.shape == (3, 5) and torch.isfinite(out).all()

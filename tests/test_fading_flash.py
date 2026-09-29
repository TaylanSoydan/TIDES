"""Fading Flash: the generator is the exact ZOH recurrence; model sizes as in the paper."""
import numpy as np
import torch

import task


def test_glow_is_exact_zoh_recurrence():
    np.random.seed(0); torch.manual_seed(0)
    x, dt, y = task.sample_batch(64)
    lam = task.RATE_LEVELS[x[..., 1:].argmax(-1)]
    alpha = torch.exp(-lam * dt[:, None])
    beta = (1 - alpha) / lam
    h = torch.zeros(64)
    for j in range(task.L):
        h = alpha[:, j] * h + beta[:, j] * x[:, j, 0]
        torch.testing.assert_close(h, y[:, j, 0])
    assert ((dt >= 0.5) & (dt <= 1.5)).all()


def test_seeded_stream_is_reproducible():
    outs = []
    for _ in range(2):
        np.random.seed(7); torch.manual_seed(7)
        outs.append(task.sample_batch(16))
    for a, b in zip(*outs):
        assert torch.equal(a, b)


def test_parameter_counts():
    import fading_flash as ff
    counts = {n: sum(p.numel() for p in make().parameters()) for n, make in ff.MODEL_FACTORIES.items()}
    assert counts == {'S5': 144, 'Mamba surrogate': 144, 'TIDES': 144, 'TIDES (Λ-only)': 144,
                      'Mamba-1': 153, 'Mamba-2': 160, 'Mamba-3': 151}

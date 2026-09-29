"""The Fading Flash task: generator and constants.

A sequence of L = 40 steps is split into 2-3 zones, each with its own decay
rate λ ∈ {1.0, 1.5, 2.0}; 2-4 unit "flashes" occur at random positions.  The
target is the glow

    h_j = α_j h_{j-1} + β_j x_j,   α_j = exp(-λ_j Δ),   β_j = (1 - α_j) / λ_j

i.e. the exact zero-order-hold discretisation of dh/dt = -λ h + x at step size
Δ, which is shared by the whole sequence.  Models see, per step, the flash
x_j and a one-hot of the zone's rate (D_IN = 4), plus Δ.  Training draws
Δ ~ U[0.5, 1.5]; testing sweeps Δ over DT_GRID, most of it out of range.

Both functions draw from NumPy's and PyTorch's global RNGs, so seeding those
(np.random.seed, torch.manual_seed) fixes the whole stream.
"""

import numpy as np
import torch
import torch.nn.functional as F

RATE_LEVELS = torch.tensor([1.0, 1.5, 2.0])
N_RATES = 3
L = 40
D_IN, D_OUT = 4, 1
TRAIN_DT_RANGE = (0.5, 1.5)
DT_GRID = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0])


def make_one_sequence(n_flashes=3, dt_val=1.0, n_zones=3, seed=None):
    """One sequence with n_flashes flashes and n_zones zones -> (pixels, rate_idx, y)."""
    if seed is not None:
        np.random.seed(seed); torch.manual_seed(seed)
    pixels = torch.zeros(L)
    positions = np.random.choice(L, size=n_flashes, replace=False)
    pixels[positions] = 1.0
    if n_zones > 1:
        boundaries = sorted(np.random.choice(range(4, L - 4), size=n_zones - 1, replace=False))
    else:
        boundaries = []
    zone_spans = [0] + list(boundaries) + [L]
    rate_idx = torch.zeros(L, dtype=torch.long)
    prev = -1
    for i in range(n_zones):
        r = np.random.randint(N_RATES)
        while r == prev:
            r = np.random.randint(N_RATES)
        rate_idx[zone_spans[i]:zone_spans[i+1]] = r
        prev = r
    rates = RATE_LEVELS[rate_idx]
    alpha = torch.exp(-rates * dt_val)
    beta  = (1 - alpha) / rates
    y, h = torch.zeros(L), torch.tensor(0.0)
    for j in range(L):
        h = alpha[j] * h + beta[j] * pixels[j]
        y[j] = h
    return pixels, rate_idx, y


def sample_batch(batch_size, dt_range=TRAIN_DT_RANGE):
    """A batch of training-distribution sequences.

    Returns x (B, L, D_IN) = [flash, one-hot rate], dt (B,) ~ U[dt_range],
    y (B, L, 1).  Pass dt_range=(d, d) for a fixed step size d.
    """
    pixels = torch.zeros(batch_size, L)
    rate_idx = torch.zeros(batch_size, L, dtype=torch.long)
    for b in range(batch_size):
        n_px = np.random.randint(2, 5)
        pos = np.random.choice(L, size=n_px, replace=False)
        pixels[b, pos] = 1.0
        n_s = np.random.randint(2, 4)
        sw = sorted(np.random.choice(range(4, L-4), size=n_s-1, replace=False)) if n_s > 1 else []
        bnd = [0] + list(sw) + [L]
        prev = -1
        for s in range(n_s):
            r = np.random.randint(N_RATES)
            while r == prev: r = np.random.randint(N_RATES)
            rate_idx[b, bnd[s]:bnd[s+1]] = r
            prev = r
    rate_oh = F.one_hot(rate_idx, num_classes=N_RATES).float()
    x = torch.cat([pixels.unsqueeze(-1), rate_oh], dim=-1)
    dt = torch.empty(batch_size).uniform_(*dt_range)
    rates = RATE_LEVELS[rate_idx]
    alpha = torch.exp(-rates * dt.unsqueeze(-1))
    beta  = (1 - alpha) / rates
    y = torch.zeros(batch_size, L); h = torch.zeros(batch_size)
    for j in range(L):
        h = alpha[:, j] * h + beta[:, j] * pixels[:, j]
        y[:, j] = h
    return x, dt, y.unsqueeze(-1)

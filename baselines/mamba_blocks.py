"""PyTorch ports of the official Mamba-1, Mamba-2 and Mamba-3 blocks.

Line-by-line ports of mamba_ssm 2.3.2.post1 (the PyPI release of
https://github.com/state-spaces/mamba, Apache-2.0; the ported functions keep
upstream's structure and cite the upstream file and line they follow), used by the
Fading Flash toy (fading_flash/) and, for Mamba-3, by the EigenWorms drop-rate
experiment (uea/droprate.py).  They run on CPU, need no compiled kernels, and
have parameter shapes 1:1 with the official modules (the gated RMSNorms are
flat parameters here: norm_weight, B_norm_weight, C_norm_weight).

Why ports rather than imports: mamba_ssm's Mamba-2 and Mamba-3 modules import
their Triton kernels at module level, so they cannot even be imported without a
GPU, and the official Mamba-3 SISO kernel uses TMA tensor descriptors, which
only exist on Hopper (sm_90).  On any other GPU Mamba-3 has to run in PyTorch.

Verification (baselines/check_mamba_ports.py reruns all of these; RTX 4090, fp32):
  * Mamba-1 vs mamba_ssm Mamba, end to end, strict state_dict load: 4.1e-08.
  * Mamba-2 vs mamba_ssm Mamba2 (official Triton kernel): 1.5e-04.
  * Mamba-3: its Triton kernel cannot run off Hopper, so the check intercepts
    the kernel call inside the official module and feeds the module's own
    inputs to the PyTorch recurrence here (pre-kernel parity): 6.0e-07.
    The recurrence equals an independent chunked PyTorch implementation of the
    training kernel to 1e-14 in float64.
  * mamba3_siso_chunked == mamba3_siso_recurrence to float64 round-off
    (tests/test_mamba_blocks.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rms_norm_gated(x, weight, eps, z=None, group_size=None, norm_before_gate=False):
    """Verbatim from mamba_ssm/ops/triton/layernorm_gated.py::rms_norm_ref
    (bias=None, upcast=True branch), which is upstream's own reference for the
    Triton RMSNormGated used by both blocks."""
    dtype = x.dtype
    weight = weight.float()
    x = x.float()
    z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
        out = x * rstd * weight
    else:
        x_group = x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
        rstd = 1 / torch.sqrt(x_group.square().mean(dim=-1, keepdim=True) + eps)
        out = (x_group * rstd).reshape(*x.shape) * weight
    if z is not None and norm_before_gate:
        out = out * F.silu(z)
    return out.to(dtype)


class Mamba1Block(nn.Module):
    """Port of mamba_ssm/modules/mamba_simple.py::Mamba (use_fast_path=False path,
    l.146-190), with the recurrence expanded from selective_scan_fn.

    Replaces the RealMambaBlock of the original fading_flash/
    tides_toy_paper_figures.py, which deviated from upstream in two ways:

      1. It never applied out_proj -- it returned `y * silu(z)`, where upstream
         does `out = self.out_proj(y)` (mamba_simple.py:252). That was silent
         because the toy config has d_inner == d_model == 4, so the shapes still
         lined up; out_proj just sat there collecting zero gradient.
      2. dt_proj.weight kept default nn.Linear init instead of Mamba's variance-
         preserving init (mamba_simple.py:83-87), and the dt bias init skipped
         the dt_init_floor clamp (l.95).

    Defaults follow upstream: bias=False, conv_bias=True, dt_init="random",
    dt_scale=1.0, dt in [0.001, 0.1], dt_init_floor=1e-4.
    """

    def __init__(self, d_model, d_state=3, d_conv=3, expand=1, dt_rank='auto',
                 dt_min=0.001, dt_max=0.1, dt_scale=1.0, dt_init_floor=1e-4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == 'auto' else dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Variance-preserving dt projection init (mamba_simple.py:82-87)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization (mamba_simple.py:104-110)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        B_, L_, _ = u.shape
        n = self.d_state
        xz = self.in_proj(u)
        x_path, z = xz.chunk(2, dim=-1)

        # Causal depthwise conv + SiLU (mamba_simple.py:239-245)
        x_path = self.conv1d(x_path.transpose(1, 2))[:, :, :L_].transpose(1, 2)
        x_path = F.silu(x_path)

        A = -torch.exp(self.A_log)                       # (d_inner, d_state)
        dt, Bm, Cm = torch.split(self.x_proj(x_path), [self.dt_rank, n, n], dim=-1)
        # nn.Linear applies weight AND bias here; upstream splits them, passing the
        # bias to selective_scan as delta_bias with delta_softplus=True. Same value.
        dt = F.softplus(self.dt_proj(dt))                 # (b, l, d_inner)

        # selective_scan_ref, mamba_ssm/ops/selective_scan_interface.py:127-180
        dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB_u = dt.unsqueeze(-1) * Bm.unsqueeze(-2) * x_path.unsqueeze(-1)
        h = torch.zeros(B_, self.d_inner, n, dtype=x_path.dtype, device=x_path.device)
        ys = []
        for j in range(L_):
            h = dA[:, j] * h + dB_u[:, j]
            ys.append(torch.einsum('bdn,bn->bd', h, Cm[:, j]) + self.D * x_path[:, j])
        y = torch.stack(ys, dim=1) * F.silu(z)           # z-gate, inside the kernel
        return self.out_proj(y)                          # mamba_simple.py:252


class Mamba2Block(nn.Module):
    """Port of mamba_ssm/modules/mamba2.py::Mamba2 (non-mem-eff branch, l.228-270).

    Config here is the minimum legal one: expand=1 and headdim=d_inner give a
    single head, so d_inner % headdim == 0 holds (mamba2.py asserts this).
    Defaults kept: bias=False, conv_bias=True, rmsnorm=True, D_has_hdim=False,
    norm_before_gate=False, dt in [0.001, 0.1].
    """

    def __init__(self, d_model, d_state=3, d_conv=3, expand=1, headdim=None, ngroups=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.headdim = headdim if headdim is not None else self.d_inner
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.ngroups = ngroups

        d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        conv_dim = self.d_inner + 2 * self.ngroups * self.d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, kernel_size=d_conv,
                                groups=conv_dim, padding=d_conv - 1, bias=True)

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
                       + math.log(dt_min))
        dt = torch.clamp(dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        A = torch.empty(self.nheads).uniform_(1.0, 16.0)   # A_init_range default
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm_weight = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, u):
        B_, L_, _ = u.shape
        zxbcdt = self.in_proj(u)
        z, xBC, dt = torch.split(
            zxbcdt,
            [self.d_inner, self.d_inner + 2 * self.ngroups * self.d_state, self.nheads],
            dim=-1)

        # Causal depthwise conv + SiLU (mamba2.py:231-235)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, :-(self.d_conv - 1)])
        x, Bm, Cm = torch.split(
            xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

        A = -torch.exp(self.A_log)                       # (nheads,)
        dt = F.softplus(dt + self.dt_bias)               # (b, l, nheads)
        x = x.reshape(B_, L_, self.nheads, self.headdim)
        Bm = Bm.reshape(B_, L_, self.ngroups, self.d_state)
        Cm = Cm.reshape(B_, L_, self.ngroups, self.d_state)
        # ngroups=1 broadcasts across heads, as the kernel does
        Bm = Bm.expand(B_, L_, self.nheads, self.d_state)
        Cm = Cm.expand(B_, L_, self.nheads, self.d_state)

        # SSD recurrence — same math as mamba_chunk_scan_combined, run stepwise
        state = torch.zeros(B_, self.nheads, self.headdim, self.d_state,
                            dtype=x.dtype, device=x.device)
        ys = []
        for j in range(L_):
            dA = torch.exp(dt[:, j] * A).unsqueeze(-1).unsqueeze(-1)          # (b,h,1,1)
            dBx = (dt[:, j].unsqueeze(-1) * x[:, j]).unsqueeze(-1) * Bm[:, j].unsqueeze(-2)
            state = dA * state + dBx
            y_j = torch.einsum('bhpn,bhn->bhp', state, Cm[:, j]) + self.D.view(1, -1, 1) * x[:, j]
            ys.append(y_j)
        y = torch.stack(ys, dim=1).reshape(B_, L_, self.d_inner)

        y = rms_norm_gated(y, self.norm_weight, eps=1e-5, z=z,
                           group_size=self.d_inner // self.ngroups, norm_before_gate=False)
        return self.out_proj(y)


def _rope_interleaved(t, angle):
    """Interleaved-pair rotation, mamba3_siso_step.py:196-204. Pairs beyond the
    supplied angles get angle 0 (the kernel's masked load, `other=0.0`)."""
    t0, t1 = t[..., 0::2], t[..., 1::2]
    cos, sin = torch.cos(angle), torch.sin(angle)
    o0 = t0 * cos - t1 * sin
    o1 = t0 * sin + t1 * cos
    return torch.stack([o0, o1], dim=-1).flatten(-2)


def mamba3_siso_recurrence(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, nheads):
    """PyTorch stand-in for mamba3_siso_combined, taking the identical argument
    convention. Transcribed from the single-step kernel
    (mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py:150-215):

        alpha = exp(ADT);   trap = sigmoid(Trap)
        beta  = alpha*dt*(1-trap);   gamma = trap*dt
        state = alpha*state + beta*(v_{t-1} x k_{t-1}) + gamma*(v_t x k_t)
        out   = state @ q + D*v,   gated by silu(z)

    Shapes follow the kernel: Q,K (b,l,g,n); V,Z (b,l,h,p); ADT,DT,Trap (b,h,l);
    Q_bias,K_bias (h,n); Angles (b,l,h,n_angles). Returns (b,l,h,p).
    """
    B_, L_, _, d_state = Q.shape
    headdim = V.shape[-1]
    dtype = torch.promote_types(V.dtype, torch.float32)   # fp32, as the kernel
    ADT, DT, Trap = ADT.transpose(1, 2), DT.transpose(1, 2), Trap.transpose(1, 2)

    alpha = torch.exp(ADT.to(dtype))
    trap = torch.sigmoid(Trap.to(dtype))
    beta = alpha * DT * (1 - trap)
    gamma = trap * DT

    k_pre = K.expand(B_, L_, nheads, d_state) + K_bias
    q_pre = Q.expand(B_, L_, nheads, d_state) + Q_bias

    ang = torch.tanh(Angles.to(dtype)) * math.pi * DT.unsqueeze(-1)
    ang = torch.cumsum(ang, dim=1)
    ang = ang - 2 * math.pi * torch.floor(ang / (2 * math.pi))
    pad = d_state // 2 - ang.shape[-1]
    if pad > 0:
        ang = F.pad(ang, (0, pad))

    state = torch.zeros(B_, nheads, headdim, d_state, dtype=dtype, device=Q.device)
    k_prev = torch.zeros(B_, nheads, d_state, dtype=dtype, device=Q.device)
    v_prev = torch.zeros(B_, nheads, headdim, dtype=dtype, device=Q.device)
    ys = []
    for j in range(L_):
        k_j = _rope_interleaved(k_pre[:, j].to(dtype), ang[:, j])
        q_j = _rope_interleaved(q_pre[:, j].to(dtype), ang[:, j])
        v_j = V[:, j].to(dtype)
        diff = (beta[:, j].unsqueeze(-1) * v_prev).unsqueeze(-1) * k_prev.unsqueeze(-2) \
             + (gamma[:, j].unsqueeze(-1) * v_j).unsqueeze(-1) * k_j.unsqueeze(-2)
        state = state * alpha[:, j].unsqueeze(-1).unsqueeze(-1) + diff
        y_j = torch.einsum('bhpn,bhn->bhp', state, q_j) + D.view(1, -1, 1) * v_j
        ys.append(y_j * F.silu(Z[:, j].to(dtype)))
        k_prev, v_prev = k_j, v_j
    return torch.stack(ys, dim=1)


def _pad_time(t, n):
    """Zero-pad dim 1 (time) of a (b, l, ...) tensor by n steps at the end."""
    if n == 0:
        return t
    return F.pad(t, [0, 0] * (t.dim() - 2) + [0, n])


def mamba3_siso_chunked(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, D, Z, nheads,
                        chunk_size=64):
    """Chunk-parallel evaluation of mamba3_siso_recurrence: same arguments, same
    result, O(L * chunk_size) instead of a Python loop over L.

    The recurrence

        state_t = alpha_t state_{t-1} + beta_t v_{t-1} k_{t-1}^T + gamma_t v_t k_t^T

    is a diagonal-decay linear SSM whose input at step t has two rank-1 terms, so
    it is Mamba-2's SSD computation (ssd_minimal) with two input streams: the
    current (k_t, v_t) weighted by gamma_t, and the one-step-shifted
    (k_{t-1}, v_{t-1}) weighted by beta_t. Both share the decay
    exp(sum_{r=s+1..t} ADT_r).  Inside a chunk the output is a masked quadratic
    form; the state is carried across chunks sequentially.  q and k are rotated
    by their own cumulative RoPE angle before any dot product, exactly as the
    sequential loop rotates them step by step.

    Padding the end of the sequence is exact: the recurrence is causal, padded
    steps carry ADT = beta = gamma = 0, and their outputs are discarded.
    """
    B_, L_, _, d_state = Q.shape
    headdim = V.shape[-1]
    h, C = nheads, chunk_size
    dtype = torch.promote_types(V.dtype, torch.float32)   # fp32, as the kernel

    ADT = ADT.transpose(1, 2).to(dtype)                       # (b, l, h)
    DT = DT.transpose(1, 2).to(dtype)
    trap = torch.sigmoid(Trap.transpose(1, 2).to(dtype))
    beta = torch.exp(ADT) * DT * (1 - trap)
    gamma = trap * DT

    k = (K.expand(B_, L_, h, d_state) + K_bias).to(dtype)
    q = (Q.expand(B_, L_, h, d_state) + Q_bias).to(dtype)
    ang = torch.tanh(Angles.to(dtype)) * math.pi * DT.unsqueeze(-1)
    ang = torch.cumsum(ang, dim=1)
    ang = ang - 2 * math.pi * torch.floor(ang / (2 * math.pi))
    pad = d_state // 2 - ang.shape[-1]
    if pad > 0:
        ang = F.pad(ang, (0, pad))
    k = _rope_interleaved(k, ang)
    q = _rope_interleaved(q, ang)
    v = V.to(dtype)
    k_prev = F.pad(k, (0, 0, 0, 0, 1, 0))[:, :L_]           # k_{t-1}, zero at t=0
    v_prev = F.pad(v, (0, 0, 0, 0, 1, 0))[:, :L_]

    nc = -(-L_ // C)
    extra = nc * C - L_
    q, k, k_prev = (_pad_time(t, extra).reshape(B_, nc, C, h, d_state)
                    for t in (q, k, k_prev))
    v_c, v_prev = (_pad_time(t, extra).reshape(B_, nc, C, h, headdim)
                   for t in (v, v_prev))
    ADT, beta, gamma = (_pad_time(t, extra).reshape(B_, nc, C, h).permute(0, 3, 1, 2)
                        for t in (ADT, beta, gamma))        # (b, h, nc, C)

    a = torch.cumsum(ADT, dim=-1)                           # within-chunk log-decay
    seg = a.unsqueeze(-1) - a.unsqueeze(-2)                 # (b, h, nc, t, s)
    causal = torch.ones(C, C, dtype=torch.bool, device=Q.device).tril()
    decay = torch.exp(seg.masked_fill(~causal, float('-inf')))

    # intra-chunk: y_t += sum_{s<=t} decay[t,s] (gamma_s q_t.k_s v_s + beta_s q_t.k_{s-1} v_{s-1})
    w_cur = decay * torch.einsum('bclhn,bcshn->bhcls', q, k) * gamma.unsqueeze(-2)
    w_prv = decay * torch.einsum('bclhn,bcshn->bhcls', q, k_prev) * beta.unsqueeze(-2)
    y = (torch.einsum('bhcls,bcshp->bclhp', w_cur, v_c)
         + torch.einsum('bhcls,bcshp->bclhp', w_prv, v_prev))

    # each chunk's contribution to the state at its last step, then carry it
    to_end = torch.exp(a[..., -1:] - a)                     # (b, h, nc, C)
    s_chunk = (torch.einsum('bhcs,bcshn,bcshp->bchpn', to_end * gamma, k, v_c)
               + torch.einsum('bhcs,bcshn,bcshp->bchpn', to_end * beta, k_prev, v_prev))
    chunk_decay = torch.exp(a[..., -1])                     # (b, h, nc)
    state = torch.zeros(B_, h, headdim, d_state, dtype=dtype, device=Q.device)
    carried = []
    for c in range(nc):
        carried.append(state)
        state = chunk_decay[:, :, c, None, None] * state + s_chunk[:, c]
    carried = torch.stack(carried, dim=1)                   # state entering each chunk
    y = y + (torch.einsum('bclhn,bchpn->bclhp', q, carried)
             * torch.exp(a).permute(0, 2, 3, 1).unsqueeze(-1))

    y = y.reshape(B_, nc * C, h, headdim)[:, :L_] + D.view(1, 1, -1, 1) * v
    return y * F.silu(Z.to(dtype))


class Mamba3Block(nn.Module):
    """Port of mamba_ssm/modules/mamba3.py::Mamba3, SISO path.

    The recurrence is not in the module — it lives in the Triton kernel. This
    follows mamba_ssm/ops/triton/mamba3/mamba3_siso_step.py:150-215 exactly:

        alpha = exp(A*dt);  trap = sigmoid(trap_proj)
        beta  = alpha*dt*(1-trap);  gamma = trap*dt
        state = alpha*state + beta*(v_{t-1} x k_{t-1}) + gamma*(v_t x k_t)
        out   = state @ q + D*v,  gated by silu(z)

    The beta/gamma split IS the trapezoidal rule: weight on the previous input
    as well as the current one, which is why Mamba-3 has no conv1d at all.
    q,k carry a RoPE rotation whose angle accumulates as tanh(angle_proj)*pi*dt.

    d_state=4 is the minimum legal value: num_rope_angles = (d_state*rope_fraction)//2
    must be >= 1 (mamba3.py:83 asserts it), and rope_fraction is 0.5 here.
    """

    def __init__(self, d_model, d_state=4, expand=1, headdim=None, ngroups=1,
                 rope_fraction=0.5, A_floor=1e-4, scan="chunked", chunk_size=64):
        super().__init__()
        if scan not in ("chunked", "sequential"):
            raise ValueError(f"scan must be 'chunked' or 'sequential', got {scan!r}")
        self.scan = scan
        self.chunk_size = chunk_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.headdim = headdim if headdim is not None else self.d_inner
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        self.A_floor = A_floor

        assert rope_fraction in [0.5, 1.0]
        split_tensor_size = int(d_state * rope_fraction)
        if split_tensor_size % 2 != 0:
            split_tensor_size -= 1
        self.num_rope_angles = split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [z, x, B, C, dd_dt, dd_A, trap, angle]  (mamba3.py:85)
        d_in_proj = (2 * self.d_inner + 2 * self.d_state * self.num_bc_heads
                     + 3 * self.nheads + self.num_rope_angles)
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)

        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        _dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
                        + math.log(dt_min))
        _dt = torch.clamp(_dt, min=dt_init_floor)
        self.dt_bias = nn.Parameter(_dt + torch.log(-torch.expm1(-_dt)))

        self.B_bias = nn.Parameter(1 + torch.zeros(self.nheads, 1, self.d_state))
        self.C_bias = nn.Parameter(1 + torch.zeros(self.nheads, 1, self.d_state))
        self.B_norm_weight = nn.Parameter(torch.ones(self.d_state))
        self.C_norm_weight = nn.Parameter(torch.ones(self.d_state))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _rope(self, t, angle):
        """Interleaved-pair rotation, mamba3_siso_step.py:196-204. Pairs beyond
        num_rope_angles get angle 0 (masked load, `other=0.0`) i.e. no rotation."""
        t0, t1 = t[..., 0::2], t[..., 1::2]
        cos, sin = torch.cos(angle), torch.sin(angle)
        o0 = t0 * cos - t1 * sin
        o1 = t0 * sin + t1 * cos
        return torch.stack([o0, o1], dim=-1).flatten(-2)

    def forward(self, u):
        B_, L_, _ = u.shape
        parts = torch.split(
            self.in_proj(u),
            [self.d_inner, self.d_inner,
             self.d_state * self.num_bc_heads, self.d_state * self.num_bc_heads,
             self.nheads, self.nheads, self.nheads, self.num_rope_angles],
            dim=-1)
        z, x, Bm, Cm, dd_dt, dd_A, trap, angles = parts

        z = z.reshape(B_, L_, self.nheads, self.headdim)
        x = x.reshape(B_, L_, self.nheads, self.headdim)
        Bm = Bm.reshape(B_, L_, self.num_bc_heads, self.d_state)
        Cm = Cm.reshape(B_, L_, self.num_bc_heads, self.d_state)

        _A = -F.softplus(dd_A.float())                   # mamba3.py:169
        _A = torch.clamp(_A, max=-self.A_floor)          # (b, l, nheads)
        DT = F.softplus(dd_dt + self.dt_bias)            # (b, l, nheads)

        Bm = rms_norm_gated(Bm, self.B_norm_weight, eps=1e-5)
        Cm = rms_norm_gated(Cm, self.C_norm_weight, eps=1e-5)

        # Hand off in exactly the kernel's argument convention (mamba3.py:221-237),
        # so check_mamba_ports.py can feed the official module's own tensors in here.
        kwargs = dict(
            Q=Cm, K=Bm, V=x,
            ADT=(_A * DT).transpose(1, 2), DT=DT.transpose(1, 2),
            Trap=trap.transpose(1, 2),
            Q_bias=self.C_bias.squeeze(1), K_bias=self.B_bias.squeeze(1),
            Angles=angles.float().unsqueeze(-2).expand(B_, L_, self.nheads, self.num_rope_angles),
            D=self.D, Z=z, nheads=self.nheads)
        if self.scan == "chunked":
            y = mamba3_siso_chunked(**kwargs, chunk_size=self.chunk_size)
        else:
            y = mamba3_siso_recurrence(**kwargs)
        return self.out_proj(y.reshape(B_, L_, self.d_inner).to(u.dtype))



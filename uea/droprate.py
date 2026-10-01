#!/usr/bin/env python3
"""Drop-rate generalisation on EigenWorms (paper: Figure 6 and the drop-rate table).

Every model is trained with a fraction r_train = 0.5 of the time steps dropped
(one fixed random mask per training seed, shared by train and validation), the
best-validation checkpoint is kept, and it is then tested with fresh random
masks at r_test in {0.1, 0.3, 0.5, 0.7, 0.9}.  The test masks are fixed per
r_test and shared by every model.

Models (names as in the results CSV):
    S5, TIDES_Lambda, TIDES_BC, TIDES, TIDES_full
        TIDES package with the Λ / B,C components switched between LTI and
        input-dependent.  They receive the gaps between kept steps as the SSM
        step (step_scale).
    Mamba_S
        TIDES package with an input-dependent step, step = softplus(W [x, Δ] + b):
        the Mamba-style treatment of Δ inside an otherwise identical model.
    RFormer
        Signature transformer with the published EigenWorms configuration
        (10 local windows, signature level 2, n_embd 20, 1 head, 2 layers).
    Mamba, Mamba2, Mamba3
        MambaClassifier (baselines/).  These have no step input, so a normalised
        time channel t/L is prepended on the FULL grid before dropping: each kept
        step carries its true timestamp.  Mamba/Mamba2 run on the official
        mamba_ssm CUDA kernels when they are installed (else on the PyTorch
        ports, see --mamba_backend); Mamba3 always uses the PyTorch port (SISO),
        because the official Mamba-3 kernel needs a Hopper GPU.

Shared settings: 400 epochs, Adam, 3 seeds (42-44).  The SSM columns use lr 1e-3,
batch 10, weight decay 0.1, L=1, P=16, bidirectional, ZOH; RFormer its published
lr 6.73e-3 and batch 5.  The Mamba columns use lr 1e-3 and weight decay 0 (batch
10, Mamba-2 32): with Adam's coupled decay at 0.1, Mamba-1 and Mamba-2 collapse
to constant predictors on this task.

Usage:
    python uea/droprate.py                    # all models -> results/droprate/
    python uea/droprate.py --models TIDES Mamba_S --seeds 42
    python uea/droprate.py --smoke            # wiring check (12 short sequences), CPU ok
    python uea/plot_droprate.py results/droprate/results.csv
"""

import argparse
import csv
import os
import random
import sys
import time
import types

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from utils import load_uea  # noqa: E402

from tides import TIDESClassifier, step_scale_from_indices  # noqa: E402
from baselines.mamba_classifier import MambaClassifier, resolve_backend  # noqa: E402

DATASET = "EigenWorms"
NUM_CLASSES = 5
NUM_CHANNELS = 6

R_TRAIN = 0.5
R_TEST_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]
EVAL_SEEDS = {r: 200 + i for i, r in enumerate(R_TEST_LIST)}   # one mask per r_test
TRAIN_SEEDS = [42, 43, 44]
SPLIT_SEED = 0
TEST_FRAC = 0.3       # of all sequences, held out for val + test
VAL_FRAC = 0.5        # of the held-out part, used for validation

# ── model configurations ──────────────────────────────────────────────────────
SSM_BASE = dict(
    num_blocks=1, ssm_size=16, ssm_blocks=2,
    discretization="zoh", learn_lambda="stable",
    bidir=True, encoder_depth=0, lambda_encoder_depth=0,
    bc_rank=16, drop_rate=0.0, proj_norm="rmsnorm",
    clip_eigs=True,
    dt_min=0.001, dt_max=0.1, ff_mult=1.0,
)
SSM_TRAIN = dict(lr=1e-3, weight_decay=0.1, batch_size=10)
MAMBA_BASE = dict(d_model=16, d_state=16, num_layers=1)

LTI, ID = "lti", "input_dependent"
CONFIGS = {
    # d_hidden sets the width; the LTI-B,C models get more width to match params.
    "S5":           dict(kind="ssm", d_hidden=82, lambda_re=LTI, lambda_im=LTI, bc=LTI, step=LTI),
    "TIDES_Lambda": dict(kind="ssm", d_hidden=80, lambda_re=ID,  lambda_im=LTI, bc=LTI, step=LTI),
    "Mamba_S":      dict(kind="ssm", d_hidden=16, lambda_re=LTI, lambda_im=LTI, bc=ID,  step=ID),
    "TIDES_BC":     dict(kind="ssm", d_hidden=16, lambda_re=LTI, lambda_im=LTI, bc=ID,  step=LTI),
    "TIDES":        dict(kind="ssm", d_hidden=16, lambda_re=ID,  lambda_im=LTI, bc=ID,  step=LTI),
    "TIDES_full":   dict(kind="ssm", d_hidden=16, lambda_re=ID,  lambda_im=ID,  bc=ID,  step=LTI),
    "RFormer":      dict(kind="rformer", n_head=1, n_layers=2, n_embd=20, num_windows=10,
                         sig_level=2, local_tight=True,
                         lr=6.73e-3, weight_decay=0.1, batch_size=5),
    "Mamba":        dict(kind="mamba", variant="mamba", expand=16, d_conv=4, drop_rate=0.0,
                         lr=1e-3, weight_decay=0.0, batch_size=10),
    "Mamba2":       dict(kind="mamba", variant="mamba2", expand=32, headdim=64, d_conv=4,
                         drop_rate=0.1, lr=1e-3, weight_decay=0.0, batch_size=32),
    "Mamba3":       dict(kind="mamba", variant="mamba3", expand=32, headdim=64,
                         rope_fraction=0.5, drop_rate=0.0,
                         lr=1e-3, weight_decay=0.0, batch_size=10),
}


def train_settings(cfg: dict) -> dict:
    if cfg["kind"] == "ssm":
        return SSM_TRAIN
    return {k: cfg[k] for k in ("lr", "weight_decay", "batch_size")}


def build_model(name: str, device, sig_dim: int = None, backend: str = "auto") -> nn.Module:
    cfg = CONFIGS[name]
    if cfg["kind"] == "ssm":
        model = TIDESClassifier(
            d_input=NUM_CHANNELS, num_classes=NUM_CLASSES, d_hidden=cfg["d_hidden"],
            lambda_re_mode=cfg["lambda_re"], lambda_im_mode=cfg["lambda_im"],
            bc_mode=cfg["bc"], step_mode=cfg["step"], **SSM_BASE)
    elif cfg["kind"] == "rformer":
        from model_classification import DecoderTransformer
        inner = types.SimpleNamespace(embd_pdrop=0.0, attn_pdrop=0.0, resid_pdrop=0.0,
                                      sparse=False, sub_len=1, scale_att=True, q_len=1)
        model = DecoderTransformer(
            config=inner, input_dim=sig_dim, n_head=cfg["n_head"], seq_num=1,
            layer=cfg["n_layers"], n_embd=cfg["n_embd"], win_len=cfg["num_windows"],
            num_classes=NUM_CLASSES)
    else:
        model = MambaClassifier(
            cfg["variant"], d_input=NUM_CHANNELS + 1, num_classes=NUM_CLASSES,
            expand=cfg["expand"], d_conv=cfg.get("d_conv", 4),
            headdim=cfg.get("headdim", 64), rope_fraction=cfg.get("rope_fraction", 0.5),
            drop_rate=cfg["drop_rate"], backend=backend, **MAMBA_BASE)
    return model.to(device)


# ── data ──────────────────────────────────────────────────────────────────────

def load_data(data_dir):
    """EigenWorms, duplicates removed, fixed stratified 70/15/15 split (165/36/35)."""
    X_raw, Y_raw = load_uea(DATASET, data_dir)                           # (N, C, L)
    n, c, L = X_raw.shape
    _, first = np.unique(X_raw.reshape(n, c * L), axis=0, return_index=True)
    X_raw, Y_raw = X_raw[np.sort(first)], Y_raw[np.sort(first)]
    _, Y = np.unique(Y_raw, return_inverse=True)
    Y = Y.astype(np.int64)
    X = torch.tensor(X_raw.transpose(0, 2, 1), dtype=torch.float32)       # (N, L, C)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=TEST_FRAC, random_state=SPLIT_SEED, stratify=Y)
    X_te, X_va, Y_te, Y_va = train_test_split(
        X_te, Y_te, test_size=VAL_FRAC, random_state=SPLIT_SEED, stratify=Y_te)
    return (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te)


def drop_mask(L: int, r: float, seed: int) -> list:
    """Sorted indices of the kept steps after dropping a fraction r (t=0 always kept)."""
    n_keep = max(2, int(round((1.0 - r) * L)))
    keep = sorted(random.Random(seed).sample(range(L), n_keep))
    if keep[0] != 0:
        keep[0] = 0
        keep = sorted(set(keep))
    return keep


def add_time_channel(X: torch.Tensor) -> torch.Tensor:
    """Prepend t/L on the full grid; call BEFORE dropping so kept steps keep their time."""
    L = X.shape[1]
    t = (torch.linspace(0, L, L) / L).reshape(1, L, 1).expand(X.shape[0], L, 1)
    return torch.cat([t, X], dim=2)


def signature_features(X: torch.Tensor, keep: list, cfg: dict) -> torch.Tensor:
    """(N, L, C) + kept indices -> (N, num_windows, sig_dim), computed on CPU."""
    from sig_utils import ComputeSignatures
    sig_cfg = types.SimpleNamespace(
        global_backward=False, global_forward=False, local_tight=cfg["local_tight"],
        local_wide=False, num_windows=cfg["num_windows"], sig_level=cfg["sig_level"],
        univariate=False, local_width=0)
    N, L, _ = X.shape
    t = torch.linspace(0, 1, L).view(1, L, 1).expand(N, -1, -1)
    selected = torch.cat([t, X.cpu()], dim=2)[:, keep, :]
    x_coords = np.linspace(0, L, L)[keep]
    parts = [ComputeSignatures(selected[i:i + 32], x_coords, sig_cfg, torch.device("cpu"))
             for i in range(0, N, 32)]
    return torch.cat(parts, dim=0).float()


def signature_dim(cfg: dict) -> int:
    import iisignature
    return iisignature.siglength(NUM_CHANNELS + 1, cfg["sig_level"])


def make_inputs(name, X, keep):
    """What model `name` sees for full-grid sequences X under the kept steps `keep`.

    Returns (inputs, step_scale); step_scale is None except for the SSM columns.
    """
    cfg = CONFIGS[name]
    if cfg["kind"] == "ssm":
        return X[:, keep, :], step_scale_from_indices(keep)
    if cfg["kind"] == "rformer":
        return signature_features(X, keep, cfg), None
    return add_time_channel(X)[:, keep, :], None


# ── training / evaluation ─────────────────────────────────────────────────────

def accuracy(model, X, Y, step_scale, batch_size, device):
    model.eval()
    loader = DataLoader(TensorDataset(X, torch.tensor(Y)), batch_size=batch_size, shuffle=False)
    ss = None if step_scale is None else step_scale.to(device)
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb) if ss is None else model(xb, step_scale=ss)
            correct += (out.argmax(1) == yb).sum().item()
    return correct / len(Y)


def train(model, X_tr, Y_tr, X_va, Y_va, step_scale, epochs, lr, weight_decay,
          batch_size, device):
    """Adam for a fixed number of epochs; keeps the best-validation-accuracy weights."""
    loader = DataLoader(TensorDataset(X_tr.to(device), torch.tensor(Y_tr).to(device)),
                        batch_size=batch_size, shuffle=True)
    ss = None if step_scale is None else step_scale.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_val, best_state = 0.0, None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb) if ss is None else model(xb, step_scale=ss)
            criterion(out, yb).backward()
            optimizer.step()
        val = accuracy(model, X_va, Y_va, step_scale, batch_size, device)
        if val > best_val:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"      epoch {epoch + 1:3d}/{epochs}  val={val:.4f}  best={best_val:.4f}"
                  f"  ({(time.time() - t0) / (epoch + 1):.1f}s/epoch)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarise(rows, path):
    """Mean and std over seeds per (model, r_test), plus the ratio to r_test = r_train."""
    out = []
    for name in dict.fromkeys(r["config"] for r in rows):
        accs = {r_test: [float(r["test_acc"]) for r in rows
                         if r["config"] == name and float(r["r_test"]) == r_test]
                for r_test in R_TEST_LIST}
        ref = np.mean(accs[R_TRAIN]) if accs.get(R_TRAIN) else float("nan")
        for r_test in R_TEST_LIST:
            if not accs[r_test]:
                continue
            m = float(np.mean(accs[r_test]))
            out.append({"config": name, "r_test": r_test, "mean_acc": f"{m:.4f}",
                        "std_acc": f"{np.std(accs[r_test]):.4f}", "n_seeds": len(accs[r_test]),
                        "ratio_to_r_train": f"{m / ref:.4f}" if ref == ref and ref > 0 else ""})
    write_csv(out, path)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=os.path.join(_HERE, "..", "data", "UEA_datasets"),
                   help="Where aeon stores / downloads EigenWorms")
    p.add_argument("--out_dir", default="results/droprate")
    p.add_argument("--models", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=TRAIN_SEEDS)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    p.add_argument("--mamba_backend", default="auto", choices=["auto", "mamba_ssm", "port"],
                   help="Mamba-1/2 implementation (Mamba-3 always uses the PyTorch port)")
    p.add_argument("--smoke", action="store_true",
                   help="Wiring check: 1 epoch, 1 seed, r_test=0.5, 12 sequences of 512 steps")
    args = p.parse_args()

    if args.smoke:
        args.epochs, args.seeds = 1, args.seeds[:1]
    r_tests = [R_TRAIN] if args.smoke else R_TEST_LIST
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}   models: {args.models}   seeds: {args.seeds}")

    (X_tr, Y_tr), (X_va, Y_va), (X_te, Y_te) = load_data(args.data_dir)
    if args.smoke:   # 12 sequences per split, cropped to 512 steps
        X_tr, Y_tr, X_va, Y_va, X_te, Y_te = (X_tr[:12, :512], Y_tr[:12], X_va[:12, :512],
                                               Y_va[:12], X_te[:12, :512], Y_te[:12])
    L = X_tr.shape[1]
    print(f"EigenWorms: train={len(Y_tr)} val={len(Y_va)} test={len(Y_te)} length={L}")

    rows, param_rows = [], []
    for name in args.models:
        cfg = CONFIGS[name]
        hp = train_settings(cfg)
        sig_dim = signature_dim(cfg) if cfg["kind"] == "rformer" else None
        backend = ""
        if cfg["kind"] == "mamba":
            backend = ("  backend=port (chunked scan)" if cfg["variant"] == "mamba3"
                       else f"  backend={resolve_backend(args.mamba_backend)}")
        print(f"\n{'=' * 60}\n{name}  lr={hp['lr']} wd={hp['weight_decay']} "
              f"batch={hp['batch_size']}{backend}\n{'=' * 60}")
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = build_model(name, device, sig_dim, backend=args.mamba_backend)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if seed == args.seeds[0]:
                param_rows.append({"config": name, "n_params": n_params})
            keep = drop_mask(L, R_TRAIN, seed)
            X_tr_in, ss = make_inputs(name, X_tr, keep)
            X_va_in, _ = make_inputs(name, X_va, keep)
            print(f"\n  seed {seed}: {n_params:,} params, r_train={R_TRAIN} keeps "
                  f"{len(keep)}/{L} steps", flush=True)
            best_val = train(model, X_tr_in, Y_tr, X_va_in, Y_va, ss, args.epochs,
                             hp["lr"], hp["weight_decay"], hp["batch_size"], device)
            for r_test in r_tests:
                keep_te = drop_mask(L, r_test, EVAL_SEEDS[r_test])
                X_te_in, ss_te = make_inputs(name, X_te, keep_te)
                acc = accuracy(model, X_te_in, Y_te, ss_te, hp["batch_size"], device)
                print(f"    r_test={r_test:.1f}  acc={acc:.4f}")
                rows.append({"config": name, "seed": seed, "r_test": r_test,
                             "best_val": f"{best_val:.4f}", "test_acc": f"{acc:.4f}"})
            write_csv(rows, os.path.join(args.out_dir, "results.csv"))   # after every seed

    write_csv(param_rows, os.path.join(args.out_dir, "params.csv"))
    summary = summarise(rows, os.path.join(args.out_dir, "summary.csv"))
    print(f"\n{'model':14s}" + "".join(f"  r={r:<5}" for r in r_tests))
    for name in args.models:
        cells = {s["r_test"]: s["mean_acc"] for s in summary if s["config"] == name}
        print(f"{name:14s}" + "".join(f"  {cells.get(r, '-'):7s}" for r in r_tests))
    print(f"\nResults in {args.out_dir}/ (results.csv, summary.csv, params.csv)")


if __name__ == "__main__":
    main()

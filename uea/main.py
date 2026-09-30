#!/usr/bin/env python3
"""Train TIDES on one UEA dataset over several seeds.

    python uea/main.py --config uea/configs/tides/EW.yaml

Each seed re-splits the data 70/15/15 at random.  The script reports test
accuracy at the epoch with the lowest validation loss ("test@val") and at the
last epoch, mean +- std over seeds.
"""
import argparse
import os
import random
import sys
import time
import pprint
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# uea/ uses flat imports; put it (and the repo root, for `tides`) on the path so
# both `python uea/main.py` and `python -m uea.main` work from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from utils import get_dataset_preprocess  # noqa: E402
from tides import TIDESClassifier, step_scale_from_indices  # noqa: E402



def parse_args():
    """Parses command-line arguments and loads defaults from a YAML config file."""
    prelim = argparse.ArgumentParser(add_help=False)
    prelim.add_argument(
        '--config', '-c', type=str,
        help='Path to a YAML config file with default argument values'
    )
    args, remaining_argv = prelim.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train TIDES on a UEA dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[prelim]
    )
    # Data and seeds
    parser.add_argument("--dataset", type=str, default="TSC_SelfRegulationSCP1", help="Dataset to use")
    parser.add_argument("--data_dir", type=str, default=os.path.join(_HERE, "..", "data", "UEA_datasets"),
                        help="Directory aeon reads / downloads the UEA datasets into")
    parser.add_argument("--n_seeds", type=int, default=4, help="Number of random seeds (42, 43, ...)")

    # Training
    parser.add_argument("--epoch", type=int, default=300, help="Max training epochs")
    parser.add_argument("--early_stop_patience", type=int, default=15, help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--batch_size", type=int, default=20, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.00040788, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for optimizer")
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer to use")

    # Random drop
    parser.add_argument("--use_random_drop", action="store_true", help="Enable random dropping of time points")
    parser.add_argument("--random_percentage", type=float, default=0.7, help="Fraction of points to keep when randomly dropping")

    # TIDES hyperparameters
    parser.add_argument("--tides_hidden", type=int, default=64, help="TIDES hidden dimension")
    parser.add_argument("--tides_ssm_size", type=int, default=64, help="TIDES total SSM state size")
    parser.add_argument("--tides_ssm_blocks", type=int, default=2, help="TIDES HiPPO diagonal blocks")
    parser.add_argument("--tides_num_blocks", type=int, default=2, help="TIDES number of SSM blocks")
    parser.add_argument("--tides_lambda_re_mode", choices=["lti", "input_dependent"], default="lti", help="TIDES Lambda real mode")
    parser.add_argument("--tides_lambda_im_mode", choices=["lti", "input_dependent"], default="lti", help="TIDES Lambda imaginary mode")
    parser.add_argument("--tides_bc_mode", choices=["lti", "input_dependent"], default="lti", help="TIDES B/C mode")
    parser.add_argument("--tides_bc_rank", type=int, default=8, help="TIDES rank of the low-rank B/C projectors (>= 1)")
    parser.add_argument("--tides_drop_rate", type=float, default=0.05, help="TIDES dropout rate inside blocks")
    parser.add_argument("--tides_learn_lambda", choices=["standard", "exp", "stable", "softplus"], default="standard", help="TIDES Lambda reparameterization")
    parser.add_argument("--tides_encoder_depth", type=int, default=1, help="TIDES GLU layers in input encoder")
    parser.add_argument("--tides_ff_mult", type=float, default=1.0, help="TIDES feed-forward expansion in GLU")
    parser.add_argument("--tides_dt_min", type=float, default=0.001, help="TIDES minimum log-step init value")
    parser.add_argument("--tides_dt_max", type=float, default=0.1, help="TIDES maximum log-step init value")
    parser.add_argument("--tides_discretization", choices=["zoh", "bilinear"], default="zoh", help="TIDES discretization method")
    parser.add_argument("--tides_clip_eigs", action="store_true", help="TIDES clip eigenvalues to left half-plane")
    parser.add_argument("--tides_bidir", action="store_true", help="TIDES bidirectional SSM")
    parser.add_argument("--tides_lambda_encoder_depth", type=int, default=0, help="TIDES GLU layers in lambda encoder")
    parser.add_argument("--tides_proj_norm", type=str, default="rmsnorm", help="TIDES projection norm (rmsnorm or none)")

    # Data splits
    parser.add_argument("--test_size", type=float, default=0.3, help="Fraction of data for test set")
    parser.add_argument("--val_size", type=float, default=0.5, help="Fraction of test set for validation")

    # Load YAML defaults if provided
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)

    # Final parse (remaining_argv allows CLI overrides of YAML)
    return parser.parse_args(remaining_argv)


def calculate_accuracy(model, loader, indices_keep, step_scale, device):
    """Accuracy and mean cross-entropy loss of the model on a data loader."""
    model.eval()
    correct = total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            inputs = batch['input'].to(device)[:, indices_keep, :]
            labels = batch['label'].to(device)
            outputs = model(inputs, step_scale=step_scale)
            total_loss += criterion(outputs, labels).item()
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total, total_loss / len(loader)


def create_model(config, num_features, num_classes, device):
    """The TIDES classifier described by config."""
    return TIDESClassifier(
            d_input=num_features,
            num_classes=num_classes,
            d_hidden=config.tides_hidden,
            ssm_size=config.tides_ssm_size,
            ssm_blocks=config.tides_ssm_blocks,
            num_blocks=config.tides_num_blocks,
            lambda_re_mode=config.tides_lambda_re_mode,
            lambda_im_mode=config.tides_lambda_im_mode,
            bc_mode=config.tides_bc_mode,
            bc_rank=config.tides_bc_rank,
            drop_rate=config.tides_drop_rate,
            learn_lambda=config.tides_learn_lambda,
            encoder_depth=config.tides_encoder_depth,
            lambda_encoder_depth=config.tides_lambda_encoder_depth,
            ff_mult=config.tides_ff_mult,
            dt_min=config.tides_dt_min,
            dt_max=config.tides_dt_max,
            discretization=config.tides_discretization,
            clip_eigs=config.tides_clip_eigs,
            bidir=config.tides_bidir,
            proj_norm=config.tides_proj_norm if config.tides_proj_norm != "none" else None,
        ).to(device)


def train_one_seed(config, seed, device):
    """Main training and evaluation loop for a single random seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    start_time = time.time()
    print("\n" + "="*50)
    print(f"Starting Training for Seed: {seed}")
    print("="*50)
    print("Configuration:")
    pprint.pprint(vars(config))

    train_loader, val_loader, test_loader, seq_len_orig, num_classes, num_samples, num_features = \
        get_dataset_preprocess(config, seed)

    print(f"\nDataset Info: Classes={num_classes}, Samples={num_samples}, Features={num_features}, SeqLen={seq_len_orig}")

    indices = list(range(seq_len_orig))
    if config.use_random_drop:
        keep_indices = sorted(random.sample(indices, int(config.random_percentage * seq_len_orig)))
        if 0 not in keep_indices:
            keep_indices.insert(0, 0)
    else:
        keep_indices = indices

    # Per-timestep step sizes from the kept indices: step_scale[i] = gap between
    # kept index i and i-1 on the original grid.  With no drop every gap is 1.
    step_scale = step_scale_from_indices(keep_indices, device=device)

    model = create_model(config, num_features, num_classes, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config.optimizer)(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_val_acc = -1.0
    best_test_at_val = -1.0
    no_improve = 0
    patience = getattr(config, 'early_stop_patience', 15)
    for epoch in range(config.epoch):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        for batch in train_loader:
            inputs, labels = batch['input'].to(device), batch['label'].to(device)
            inputs = inputs[:, keep_indices, :]
            outputs = model(inputs, step_scale=step_scale)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_correct += (outputs.argmax(dim=1) == labels).sum().item()
            epoch_total += labels.size(0)

        avg_loss = epoch_loss / len(train_loader)
        train_acc = epoch_correct / epoch_total
        val_acc, val_loss = calculate_accuracy(model, val_loader, keep_indices, step_scale, device)
        test_acc, test_loss = calculate_accuracy(model, test_loader, keep_indices, step_scale, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_test_at_val = test_acc
            no_improve = 0
        else:
            no_improve += 1

        print(f"Epoch {epoch+1:03}/{config.epoch} | Loss: {avg_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
              f"Val Acc: {val_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}% | "
              f"Best Val Loss: {best_val_loss:.4f} | No improve: {no_improve}/{patience}")

        if patience > 0 and no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    print(f"\nTotal training time: {time.time() - start_time:.2f}s")
    final_acc, _ = calculate_accuracy(model, test_loader, keep_indices, step_scale, device)
    return best_val_acc, best_test_at_val, final_acc


def main():
    """Main execution function."""
    config = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

        
    seeds = [42 + i for i in range(config.n_seeds)]
    at_val, final = [], []

    # The paper reports test accuracy at the epoch with the lowest validation
    # loss ("test@val"), averaged over seeds; the last-epoch accuracy is shown too.
    for s in seeds:
        _, test_at_val, final_acc = train_one_seed(config, s, device)
        print(f"\nSeed {s} -> test@val: {test_at_val*100:.2f}%   last epoch: {final_acc*100:.2f}%\n")
        at_val.append(test_at_val)
        final.append(final_acc)

    print(f"\n{config.dataset} over {len(seeds)} seed(s) {seeds}:")
    print(f"  test@val   : {np.mean(at_val)*100:.2f}% ± {np.std(at_val)*100:.2f}%")
    print(f"  last epoch : {np.mean(final)*100:.2f}% ± {np.std(final)*100:.2f}%")


if __name__ == "__main__":
    main()
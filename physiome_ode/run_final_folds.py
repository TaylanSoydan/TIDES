"""
Train and test TIDES on all 5 folds of one Physiome-ODE dataset.
Reports mean ± std test MSE, directly comparable to the Physiome-ODE leaderboard.

Reproduce a paper number (configs in physiome_ode/configs/winners.csv, one per
dataset; see docs/reproducibility.md):

    python physiome_ode/run_final_folds.py --winner HOD01
    python physiome_ode/run_final_folds.py --winner hodgkin_huxley_1952_variant01

From an Optuna study written by hypersearch_physio.py:

    python physiome_ode/run_final_folds.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --storage sqlite:///results/hypersearch.db \\
        --study_name tides_physio_hodgkin_huxley_1952_variant01_f0

Or with hyperparameters given by hand (any flag also overrides --winner):

    python physiome_ode/run_final_folds.py \\
        --dataset hodgkin_huxley_1952_variant01 \\
        --hidden_size 32 --ssm_size 8 --ssm_blocks 2 --num_blocks 3 \\
        --lr 5e-4 --weight_decay 1e-4 --batch_size 32

Folds use seed = fold index; training runs up to 200 epochs with early stopping
on validation MSE (patience 30), as in the paper.
"""

import argparse
import csv
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from tides_train_fn import train_tides  # noqa: E402
from utils import IMTS_dataset  # noqa: E402,F401 — pickle deserialization

WINNERS_CSV = os.path.join(_HERE, "configs", "winners.csv")
WANDB_PROJECT = "tides"

# train_tides keyword -> type, for every hyperparameter a config can set
HP_TYPES = {
    "hidden_size": int, "ssm_size": int, "ssm_blocks": int, "num_blocks": int,
    "encoder_depth": int, "lambda_encoder_depth": int, "bc_rank": int,
    "lambda_re_mode": str, "lambda_im_mode": str, "bc_mode": str,
    "learn_lambda": str, "discretization": str, "proj_init_method": str,
    "proj_norm": str, "drop_rate": float, "dt_min": float, "ff_mult": float,
    "bidir": bool, "clip_eigs": bool, "conj_sym": bool,
    "lr": float, "lr_factor": int, "warmup_epochs": int,
    "weight_decay": float, "batch_size": int,
}
DEFAULTS = {
    "encoder_depth": 1, "lambda_encoder_depth": 0, "bc_rank": 8,
    "lambda_re_mode": "input_dependent", "lambda_im_mode": "lti",
    "bc_mode": "input_dependent", "learn_lambda": "standard",
    "discretization": "zoh", "proj_init_method": "zeros", "proj_norm": "rmsnorm",
    "drop_rate": 0.0, "dt_min": 0.001, "ff_mult": 1.0, "bidir": False,
    "clip_eigs": False, "conj_sym": False, "lr": 1e-3, "lr_factor": 1,
    "warmup_epochs": 0, "weight_decay": 1e-3, "batch_size": 32,
}


def _cast(key, value):
    kind = HP_TYPES[key]
    if kind is bool:
        return value if isinstance(value, bool) else str(value).lower() in ("true", "1", "yes")
    if key == "proj_norm" and str(value).lower() in ("", "none"):
        return None
    return kind(float(value)) if kind is int else kind(value)


def load_winner(name: str) -> tuple:
    """(dataset, hyperparameters) for a dataset code (HOD01) or name from winners.csv."""
    with open(WINNERS_CSV) as f:
        for row in csv.DictReader(f):
            if name in (row["code"], row["dataset"]):
                return row["dataset"], {k: _cast(k, row[k]) for k in HP_TYPES if k in row}
    raise SystemExit(f"{name!r} is not a code or dataset name in {WINNERS_CSV}")


def load_best_params_from_optuna(storage: str, study_name: str) -> dict:
    import optuna
    p = optuna.load_study(study_name=study_name, storage=storage).best_params
    parts = p["mode_combo"].split("/")
    if len(parts) == 3:
        lambda_re_mode, lambda_im_mode, bc_mode = parts
    else:
        lambda_re_mode, bc_mode = parts
        lambda_im_mode = lambda_re_mode
    hp = {k: _cast(k, p[k]) for k in HP_TYPES if k in p}
    hp.update(lambda_re_mode=lambda_re_mode, lambda_im_mode=lambda_im_mode, bc_mode=bc_mode)
    if "ssm_size" not in p:
        hp["ssm_size"] = 2 * p["ssm_blocks"] * p["ssm_dim_mult"]
    return hp


def main():
    parser = argparse.ArgumentParser(
        description="Run 5 folds of TIDES on one Physiome-ODE dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--winner", default=None,
                        help="Dataset code (e.g. HOD01) or name: use its row of configs/winners.csv")
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--data_base_path", default=os.path.join(_HERE, "..", "data", "physiome_ode"),
                        help="Directory holding <dataset>/<fold>/ (the Zenodo 'final/' folder)")
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--early_stop", default=30, type=int)
    parser.add_argument("--folds", default=[0, 1, 2, 3, 4], type=int, nargs="+")
    parser.add_argument("--saved_models_dir", default="saved_models", type=str)
    parser.add_argument("--wandb_mode", default="disabled", choices=["disabled", "offline", "online"],
                        help="Weights & Biases logging (off unless asked for)")
    # Optuna study
    parser.add_argument("--storage", default=None, type=str, help="Optuna storage URL")
    parser.add_argument("--study_name", default=None, type=str, help="Optuna study name")
    # Manual hyperparameters; any flag given overrides --winner / the study
    for key, kind in HP_TYPES.items():
        if kind is bool:
            parser.add_argument(f"--{key}", default=None, type=lambda v: _cast("bidir", v),
                                metavar="{true,false}")
        else:
            parser.add_argument(f"--{key}", default=None, type=kind if kind is not int else float)
    parser.add_argument("--ssm_dim_mult", default=None, type=int,
                        help="Alternative to --ssm_size: ssm_size = 2 * ssm_blocks * ssm_dim_mult")
    parser.add_argument("--mode_combo", default=None, type=str,
                        help="lambda_re/lambda_im/bc modes, e.g. input_dependent/lti/input_dependent")
    args = parser.parse_args()

    # ── Collect hyperparameters: defaults < winner / study < explicit flags ──
    hp = dict(DEFAULTS)
    dataset = args.dataset
    if args.winner:
        dataset, winner_hp = load_winner(args.winner)
        hp.update(winner_hp)
        print(f"Using the winning configuration for {args.winner} from {WINNERS_CSV}")
    elif args.storage and args.study_name:
        print(f"Loading best params from study '{args.study_name}'...")
        hp.update(load_best_params_from_optuna(args.storage, args.study_name))
    if args.dataset:
        dataset = args.dataset
    if dataset is None:
        parser.error("give --winner, or --dataset (with --storage/--study_name or manual flags)")
    for key in HP_TYPES:
        if getattr(args, key) is not None:
            hp[key] = _cast(key, getattr(args, key))
    if args.mode_combo:
        parts = args.mode_combo.split("/")
        hp["lambda_re_mode"], hp["bc_mode"] = parts[0], parts[-1]
        hp["lambda_im_mode"] = parts[1] if len(parts) == 3 else parts[0]
    if "ssm_size" not in hp:
        if args.ssm_dim_mult is None or "ssm_blocks" not in hp:
            parser.error("give --ssm_size (or --ssm_blocks and --ssm_dim_mult)")
        hp["ssm_size"] = 2 * hp["ssm_blocks"] * args.ssm_dim_mult
    missing = [k for k in ("hidden_size", "ssm_blocks", "num_blocks") if k not in hp]
    if missing:
        parser.error(f"missing hyperparameters: {missing}")

    wandb = None
    if args.wandb_mode != "disabled":
        import wandb

    print(f"\nDataset : {dataset}")
    print(f"Epochs  : {args.epochs}  early_stop={args.early_stop}  folds={args.folds}")
    print("Config  : " + "  ".join(f"{k}={hp[k]}" for k in sorted(hp)))
    print()

    val_losses, test_losses, test_maes = [], [], []
    for fold in args.folds:
        print(f"── Fold {fold} ──────────────────────────────────")
        run = None
        if wandb is not None:
            run = wandb.init(project=WANDB_PROJECT, name=f"{dataset}_final_f{fold}",
                             config={**hp, "dataset": dataset, "fold": fold,
                                     "epochs": args.epochs, "phase": "final_eval"},
                             tags=[dataset, "final_eval"], reinit=True, mode=args.wandb_mode)
        result = train_tides(
            dataset=dataset, fold=fold, data_base_path=args.data_base_path,
            epochs=args.epochs, early_stop_patience=args.early_stop, seed=fold,
            saved_models_dir=args.saved_models_dir, verbose=True, wandb_run=run, **hp)
        if run is not None:
            wandb.log({"val_loss": result["val_loss"], "test_loss": result["test_loss"],
                       "test_mae": result["test_mae"], "fold": fold})
            wandb.finish()
        val_losses.append(result["val_loss"])
        test_losses.append(result["test_loss"])
        test_maes.append(result["test_mae"])
        print(f"  fold {fold}: val={result['val_loss']:.4f}  "
              f"test_loss={result['test_loss']:.4f}  test_mae={result['test_mae']:.4f}  "
              f"params={result['num_params']:,}  stopped_early={result['stopped_early']}")

    print()
    print("=" * 60)
    print(f"Dataset: {dataset}   folds: {args.folds}")
    print(f"best_val_loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"test_loss:     {np.mean(test_losses):.4f} ± {np.std(test_losses):.4f}   ← leaderboard number")
    print(f"test_mae:      {np.mean(test_maes):.4f} ± {np.std(test_maes):.4f}")
    print(f"Number of trainable parameters: {result['num_params']:,}")


if __name__ == "__main__":
    main()

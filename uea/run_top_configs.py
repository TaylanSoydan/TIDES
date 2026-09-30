"""
Rerun the top-K trials of a hypersearch CSV on additional seeds.

Trials are ranked by validation accuracy, the quantity uea/hypersearch.py
maximises; test accuracy is only reported.  After the reruns the summary
names the config with the highest mean validation accuracy over all seeds.
The configurations in the paper are in uea/configs/tides/; this script is
for rerunning the search, not needed to reproduce them.

Results are cached per (dataset, seed, config) so jobs can be safely
interrupted and restarted without re-running completed seeds.

Usage:
    python uea/run_top_configs.py \
        --results_file hypersearch_results.csv \
        --dataset TSC_EthanolConcentration \
        --top_k 10 \
        --seeds 43 44 45 46 \
        --cache_dir eval_cache/EthanolConcentration \
        --output_csv eval_results/EthanolConcentration.csv
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.dirname(__file__))
from main import train_trial


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(dataset, seed, kwargs):
    payload = json.dumps({"dataset": dataset, "seed": seed, **kwargs}, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def cache_load(cache_dir, dataset, seed, kwargs):
    key = _cache_key(dataset, seed, kwargs)
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return None


def cache_save(cache_dir, dataset, seed, kwargs, result):
    os.makedirs(cache_dir, exist_ok=True)
    key = _cache_key(dataset, seed, kwargs)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w") as f:
        json.dump(result, f)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

EVAL_COLUMNS = [
    "dataset", "trial", "rank", "seed",
    "best_val", "best_test", "final_acc", "cached",
]

def append_eval_result(output_csv, row):
    exists = os.path.isfile(output_csv)
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_top_configs(results_files, top_k):
    rows = []
    for results_file in results_files:
        with open(results_file, newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "ok":
                    continue
                try:
                    row["val_metric"] = float(row["val_metric"])
                    row["best_test_metric"] = float(row["best_test_metric"])
                except (ValueError, KeyError):
                    continue
                rows.append(row)

    # best row per trial (dedup across multiple CSV files / seeds), by validation
    by_trial = {}
    for row in rows:
        t = int(row["trial"])
        if t not in by_trial or row["val_metric"] > by_trial[t]["val_metric"]:
            by_trial[t] = row

    ranked = sorted(by_trial.values(), key=lambda r: r["val_metric"], reverse=True)
    return ranked[:top_k]


def row_to_kwargs(row):
    return dict(
        lr=float(row["lr"]),
        weight_decay=float(row["weight_decay"]),
        epoch=int(row["epoch"]),
        batch_size=int(row["batch_size"]),
        random_percentage=float(row["random_percentage"]),
        hidden_dim=int(row["hidden_dim"]),
        ssm_size=int(row["ssm_size"]),
        ssm_blocks=int(row["s5_init_blocks"]),
        num_blocks=int(row["num_layers"]),
        discretization=row["discretisation"],
        drop_rate=float(row["drop_rate"]),
        learn_lambda=row["learn_lambda"],
        bidir=row["bidir"].strip().lower() == "true",
        encoder_depth=int(row["encoder_depth"]),
        lambda_encoder_depth=int(row["lambda_encoder_depth"]),
        bc_rank=int(row["bc_rank"]),
        dt_min=float(row["dt_min"]),
        ff_mult=float(row["ff_mult"]),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", nargs="+", required=True,
                        help="One or more hypersearch CSV files to pool configs from")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46])
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--data_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "UEA_datasets"))
    parser.add_argument("--cache_dir", default="eval_cache")
    parser.add_argument("--output_csv", default="eval_results.csv")
    parser.add_argument("--wandb_project", default="tides-anon")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    configs = load_top_configs(args.results_file, args.top_k)
    print(f"\nTop {len(configs)} configs (by validation accuracy):")
    for i, c in enumerate(configs):
        print(f"  [{i+1}] trial={c['trial']}  val={c['val_metric']:.4f}"
              f"  test={c['best_test_metric']:.4f}")

    summary = []

    for rank, cfg in enumerate(configs, 1):
        trial_num = int(cfg["trial"])
        kwargs = row_to_kwargs(cfg)
        seed42_test = float(cfg["best_test_metric"])
        seed42_val  = float(cfg["val_metric"])

        print(f"\n{'='*60}")
        print(f"Config {rank}/top{len(configs)} | trial={trial_num} | "
              f"hypersearch seed42: val={seed42_val:.4f}  test={seed42_test:.4f}")
        print(f"{'='*60}")

        # seed 42 result already in hypersearch CSV — no need to re-run
        all_val, all_test = [seed42_val], [seed42_test]

        for seed in args.seeds:
            cached = cache_load(args.cache_dir, args.dataset, seed, kwargs)
            if cached:
                best_val  = cached["best_val"]
                best_test = cached["best_test"]
                final_acc = cached["final_acc"]
                print(f"  Seed {seed}: CACHED  val={best_val:.4f}  test={best_test:.4f}  final={final_acc:.4f}")
                all_val.append(best_val)
                all_test.append(best_test)
                append_eval_result(args.output_csv, {
                    "dataset": args.dataset, "trial": trial_num, "rank": rank, "seed": seed,
                    "best_val": best_val, "best_test": best_test, "final_acc": final_acc, "cached": True,
                })
                continue

            print(f"  Seed {seed}: training...")
            run = wandb.init(
                project=args.wandb_project,
                name=f"{args.dataset}_top{rank}_t{trial_num}_s{seed}",
                config={**kwargs, "trial": trial_num, "seed": seed,
                        "dataset": args.dataset, "rank": rank},
                tags=[args.dataset, "eval", f"top{rank}"],
                reinit=True,
                mode="offline",
                settings=wandb.Settings(init_timeout=300),
            )
            try:
                best_val, best_test, final_acc = train_trial(
                    dataset=args.dataset,
                    seed=seed,
                    device=device,
                    data_dir=args.data_dir,
                    early_stop_patience=args.early_stop_patience,
                    use_random_drop=False,
                    **kwargs,
                )
                result = {"best_val": best_val, "best_test": best_test, "final_acc": final_acc}
                cache_save(args.cache_dir, args.dataset, seed, kwargs, result)
                all_val.append(best_val)
                all_test.append(best_test)
                print(f"  Seed {seed}: val={best_val:.4f}  test={best_test:.4f}  final={final_acc:.4f}")
                wandb.log(result)
                wandb.finish()
                append_eval_result(args.output_csv, {
                    "dataset": args.dataset, "trial": trial_num, "rank": rank, "seed": seed,
                    "best_val": best_val, "best_test": best_test, "final_acc": final_acc, "cached": False,
                })
            except KeyboardInterrupt:
                wandb.finish(exit_code=1)
                print("  Interrupted — progress saved via cache")
                raise
            except Exception as e:
                import traceback; traceback.print_exc()
                wandb.finish(exit_code=1)

        print(f"\n  → top{rank} (trial {trial_num}) | seeds=[42]+{args.seeds} | "
              f"val={np.mean(all_val)*100:.2f}%  test={np.mean(all_test)*100:.2f}% "
              f"± {np.std(all_test)*100:.2f}%  n={len(all_test)}")
        summary.append((rank, trial_num, np.mean(all_val), all_test))

    print(f"\n{'='*60}")
    print("FINAL SUMMARY (val = mean validation accuracy, used for selection)")
    print(f"{'='*60}")
    for rank, trial_num, val, all_test in summary:
        vals = "  ".join(f"{v*100:.2f}" for v in all_test)
        print(f"  top{rank} (trial {trial_num}): val {val*100:.2f}%  "
              f"test {np.mean(all_test)*100:.2f}% ± {np.std(all_test)*100:.2f}%  [{vals}]")
    rank, trial_num, val, all_test = max(summary, key=lambda r: r[2])
    print(f"\nSelected: top{rank} (trial {trial_num}), highest mean validation accuracy "
          f"{val*100:.2f}%  ->  test {np.mean(all_test)*100:.2f}% ± {np.std(all_test)*100:.2f}%")


if __name__ == "__main__":
    main()

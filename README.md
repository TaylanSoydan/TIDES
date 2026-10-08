# TIDES — Time-aware Input-Dependent State Space Model

![TIDES architecture](docs/TIDES_fig2.png)

Code for the paper [TIDES: Implicit Time-Awareness in Selective State Space Models](https://arxiv.org/abs/2605.09742).

Selective SSMs such as Mamba make the discretisation step Δ a learned function
of the input, so Δ stops being a physical sampling interval.  TIDES moves the
input dependence off the step and onto the diagonal state matrix Λ (and B, C):
Δ keeps its meaning as the time between observations, so irregular timestamps
are handled natively, without giving up per-token selectivity.

This repository contains the PyTorch implementation of TIDES and everything
needed to reproduce the paper: UEA classification, Physiome-ODE forecasting,
the EigenWorms drop-rate experiment, and the *Fading Flash* diagnostic,
including the Mamba-1/2/3 baselines.

## Installation

The model is a small PyTorch package, `tides`, that needs only PyTorch, NumPy
and SciPy:

```bash
pip install git+https://github.com/TaylanSoydan/TIDES      # or, in a clone: pip install -e .
```

To reproduce the paper, work in a clone with everything the experiments use:

```bash
git clone https://github.com/TaylanSoydan/TIDES && cd TIDES
pip install -r requirements.txt          # or: conda env create -f environment.yml && conda activate tides
pytest tests                             # quick check, CPU only
```

`pip` picks PyTorch's default CUDA build.  If your driver is older than that
build, install a matching wheel first, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu126`.
Tested with Python 3.11 and PyTorch 2.9 and 2.14 on Linux, CPU and RTX 4090.

## Quickstart

```python
import torch
from tides import TIDES

batch, length, dim = 2, 64, 16
x = torch.randn(batch, length, dim)
dt = torch.rand(batch, length)       # time since the previous observation

model = TIDES(
    d_input=dim,     # input channels
    d_hidden=32,     # model width
    ssm_size=16,     # state size per layer
    ssm_blocks=2,    # HiPPO blocks the state is initialised from (divides ssm_size)
    num_blocks=2,    # layers
)
y = model(x, step_scale=dt)
assert y.shape == (batch, length, 32)
```

`step_scale` is the time since the previous observation: a `(batch, length)`
tensor, a `(length,)` tensor shared by the batch, or a float.  The default, 1,
is regular sampling.  For a sequence subsampled from a regular grid,
`step_scale_from_indices(kept_indices)` gives the gaps.  The model runs on CPU
or GPU (`model.to("cuda")`).

For sequence classification, `TIDESClassifier` adds mean pooling and a linear
head:

```python
from tides import TIDESClassifier

clf = TIDESClassifier(d_input=dim, num_classes=5)
logits = clf(x, step_scale=dt)       # (batch, 5)
```

### Training on your own data

```python
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tides import TIDESClassifier

# N sequences of L steps with D channels, the time since the previous
# observation at every step, and one of C labels per sequence.
N, L, D, C = 64, 100, 3, 4
x, dt, y = torch.randn(N, L, D), torch.rand(N, L), torch.randint(0, C, (N,))
loader = DataLoader(TensorDataset(x, dt, y), batch_size=32, shuffle=True)

model = TIDESClassifier(d_input=D, num_classes=C, d_hidden=32, ssm_size=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(3):
    for xb, dtb, yb in loader:
        loss = F.cross_entropy(model(xb, step_scale=dtb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
```

All sequences in a batch share one length.  If yours differ, pad them; padded
positions enter the BatchNorm statistics and `TIDESClassifier`'s mean pooling.
For UEA datasets, `uea/main.py` runs this loop with a train/validation/test
split, early stopping and several seeds: `python uea/main.py --dataset
TSC_<name>` works for any dataset `aeon` can load (e.g. `TSC_Epilepsy`).

### Model options

| Argument | Default | |
|---|---|---|
| `lambda_re_mode`, `bc_mode` | `"input_dependent"` | Re(Λ) and B, C projected from the input at each step; `"lti"`: static |
| `lambda_im_mode` | `"lti"` | Im(Λ) static; `"input_dependent"`: projected too |
| `bc_rank` | `8` | rank of the low-rank B, C projectors |
| `learn_lambda` | `"standard"` | parameterisation of Re(Λ): `"standard"`, `"exp"`, `"stable"` or `"softplus"` |
| `discretization` | `"zoh"` | `"zoh"` or `"bilinear"` |
| `bidir` | `False` | bidirectional scan |
| `proj_norm` | `"rmsnorm"` | RMSNorm on the projector outputs, or `None` |
| `step_mode` | `"lti"` | `"input_dependent"` gives Mamba_S, the paper's Mamba-style control: the step becomes `softplus(W [x, Δ] + b)` instead of the observed interval |

With all three modes `"lti"` the model is S5.  Input-dependent B and C are a
`ssm_size × d_hidden` matrix at every step, so time and memory grow with that
product; beyond small models, train on a GPU.  `TIDESForecastingModel` is the
forecasting model used on Physiome-ODE.

`TIDESClassifier` and `TIDESForecastingModel` can be saved to and loaded from
the Hugging Face Hub (`pip install huggingface_hub safetensors`):

```python
clf.save_pretrained("tides-classifier")           # config.json + model.safetensors
clf.push_to_hub("<user>/tides-classifier")
clf = TIDESClassifier.from_pretrained("<user>/tides-classifier")
```

## Repository layout

```
tides/          PyTorch model package (importable as `tides`)
uea/            UEA classification (Table 1) and the EigenWorms drop-rate experiment
physiome_ode/   Physiome-ODE forecasting (Table 2)
fading_flash/   Fading Flash: task generator, models, figures, static dataset export
baselines/      Mamba-1/2/3 baselines: PyTorch ports and a sequence classifier
tests/          pytest suite (CPU)
docs/           reproducibility tables (every configuration) and dataset licenses
data/           where the datasets go (nothing is redistributed)
```

### Mamba baselines (optional)

Mamba-1 and Mamba-2 run on the official `mamba_ssm` CUDA kernels, which must
match your PyTorch and CUDA versions.  A combination that works (Linux, CUDA 12
driver, Python 3.11):

```bash
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu126
pip install einops transformers
pip install --no-deps \
  https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl \
  https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1+cu12torch2.9cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
python baselines/check_mamba_ports.py    # needs a GPU; compares the ports with mamba_ssm
```

Mamba-3 needs none of this.  Its official kernel only runs on Hopper GPUs, so
the repository uses a PyTorch port of the SISO block (`baselines/mamba_blocks.py`),
checked against the official module and computed with a chunked scan.

## Reproducing the results

Every configuration used in the paper is in `docs/reproducibility.md`.

### UEA classification (Table 1)

```bash
python uea/main.py --config uea/configs/tides/EW.yaml    # likewise SCP1, SCP2, MI, ETC, HB
```

Five seeds (42-46) on a 70/15/15 random re-split each (`--seeds` runs any
others, e.g. `--seeds 44` or `--seeds 0 1 2`); the script ends with
test accuracy at the best-validation epoch, mean ± std over seeds.  Datasets
download automatically via `aeon` into `data/UEA_datasets/` (`--data_dir` to
change).

### Physiome-ODE forecasting (Table 2)

Download `final.zip` from https://zenodo.org/records/11492058 and unpack the
contents of its `final/` folder into `data/physiome_ode/` (see `data/README.md`),
so that each dataset is at `data/physiome_ode/<dataset>/<fold>/`.  Then

```bash
python physiome_ode/run_final_folds.py --winner HOD01    # dataset code or name, all 50 in docs/reproducibility.md
```

runs the five folds with that dataset's configuration from
`physiome_ode/configs/winners.csv`.

### Drop-rate generalisation on EigenWorms (Figure 6)

```bash
python uea/droprate.py                                   # 10 models x 3 seeds -> results/droprate/
python uea/plot_droprate.py results/droprate/results.csv --out fig_droprate.pdf
```

Models train with half of the time steps dropped and are tested at drop rates
0.1-0.9: S5, TIDES and its ablations (Λ-only, B,C-only, full), Mamba_S, RFormer
with the published EigenWorms configuration, and Mamba-1/2/3.  `--models` and
`--seeds` select a subset; `--smoke` checks the wiring in a few minutes on CPU.
The Mamba-1/2 rows need `mamba_ssm` (above).

### Fading Flash

```bash
OMP_NUM_THREADS=1 python fading_flash/fading_flash.py    # ~12 min on one CPU core
```

trains S5, TIDES, TIDES (Λ-only), the Mamba surrogate and Mamba-1/2/3 on the
task and writes the paper's figures to `fading_flash/figures/`.  The task
itself is `fading_flash/task.py`.

A static copy of the benchmark (train, validation and a test split at each Δ)
is on the Hugging Face Hub as
[Taylantay/fading-flash](https://huggingface.co/datasets/Taylantay/fading-flash);
its dataset card describes the fields:

```python
from datasets import load_dataset
ds = load_dataset("Taylantay/fading-flash")
```

`fading_flash/make_hf_dataset.py` rebuilds the same files bit for bit:

```bash
python fading_flash/make_hf_dataset.py --out_dir data/fading_flash_hf
```

## Compute

All PyTorch experiments were run on a single NVIDIA L40S (48 GB) GPU. Physiome-ODE final-fold runs take 0.2–2 GPU-hours per dataset.

## Datasets and licenses

This repository does not redistribute any dataset. Pointers and licensing information for every dataset and external code asset are summarised below; see `docs/dataset_licenses.md` for the full text.

| Asset | Source | License |
|-------|--------|---------|
| UEA Time Series Classification Archive (6 datasets) | http://www.timeseriesclassification.com — auto-downloaded via `aeon` | redistributed under the archive's research-use terms (Bagnall et al. 2018) |
| Physiome-ODE (50 ODE-derived datasets) | https://zenodo.org/records/11492058 (Klötergens et al., ICLR 2025) | Apache-2.0 |
| GRU-ODE-Bayes baseline | https://github.com/edebrouwer/gru_ode_bayes | MIT |
| Neural Flows baseline | https://github.com/mbilos/neural-flows-experiments | MIT |
| CRU baseline | https://github.com/boschresearch/Continuous-Recurrent-Units | AGPL-3.0 |
| LinODENet baseline | https://github.com/randolf-scholz/linodenet | MIT |
| GraFITi baseline | https://github.com/yalavarthivk/GraFITi | MIT |
| S5 reference (utilities used by the SSM scan / discretization) | https://github.com/lindermanlab/S5 | Apache-2.0 |
| Mamba / Mamba-2 / Mamba-3 (ported in `baselines/mamba_blocks.py`) | https://github.com/state-spaces/mamba | Apache-2.0 |

Baseline numbers reported in the paper are taken from the public Physiome-ODE leaderboard and from the UEA tables in Moreno-Pino et al. (2024) and Walker et al. (2024). The corresponding repositories are not redistributed here.

## Repository code license

All source code in this repository (excluding files explicitly marked otherwise in their headers) is released under the MIT License (see `LICENSE`).

## Citation

```bibtex
@article{soydan2026tides,
  title   = {{TIDES}: Implicit Time-Awareness in Selective State Space Models},
  author  = {Soydan, Taylan and Bessa, Miguel A. and Mohr, Dirk and Barreira, Rui},
  journal = {arXiv preprint arXiv:2605.09742},
  year    = {2026}
}
```

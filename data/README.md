# Data

No dataset is redistributed here; see `docs/dataset_licenses.md`.

| Benchmark | Where it goes | How |
|---|---|---|
| UEA (6 datasets) | `data/UEA_datasets/` | Downloaded automatically by `aeon` on first use (`--data_dir` to change) |
| Physiome-ODE (50 datasets) | `data/physiome_ode/<dataset>/<fold>/{train,valid,test}.pt` | Download `final.zip` from https://zenodo.org/records/11492058 (0.7 GB) and unzip the *contents* of its `final/` folder here |
| Fading Flash | nothing to download | Generated on the fly (`fading_flash/task.py`); a static copy can be written with `fading_flash/make_hf_dataset.py` |

```bash
# Physiome-ODE
mkdir -p data/physiome_ode && cd data/physiome_ode
curl -L -o final.zip "https://zenodo.org/api/records/11492058/files/final.zip/content"
unzip -q final.zip && mv final/* . && rmdir final && rm final.zip
```

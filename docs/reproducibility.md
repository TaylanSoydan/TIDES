# Reproducibility

The commands below rerun each experiment in the paper.  The configuration
files they read are the single source of truth; the tables here are generated
from them.

## UEA classification (Table 1)

```bash
python uea/main.py --config uea/configs/tides/EW.yaml     # likewise SCP1 SCP2 MI ETC HB
```

Protocol: the original train and test partitions are concatenated and re-split
70/15/15 per seed; seeds 42-46; up to the listed number of epochs with early
stopping on validation loss (patience 30); Adam.  The reported metric is test
accuracy at the epoch with the lowest validation loss ("test@val"), mean ± std
over the five seeds; `main.py` prints it at the end of the run.

| Config | Dataset | lr | wd | h | ssm | ssm blocks | L | enc | λ enc | learn λ | disc | drop | batch | bc rank | bidir | clip | epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `SCP1.yaml` | SelfRegulationSCP1 | 0.000217 | 0 | 16 | 32 | 4 | 4 | 0 | 0 | exp | zoh | 0.15 | 5 | 8 | – | – | 200 |
| `SCP2.yaml` | SelfRegulationSCP2 | 0.000272 | 0 | 16 | 64 | 8 | 2 | 0 | 0 | stable | zoh | 0 | 10 | 16 | ✓ | ✓ | 400 |
| `MI.yaml` | MotorImagery | 0.000652 | 0.01 | 32 | 16 | 4 | 4 | 0 | 0 | standard | zoh | 0.05 | 20 | 16 | ✓ | ✓ | 200 |
| `EW.yaml` | EigenWorms | 0.00156 | 0.1 | 16 | 16 | 2 | 1 | 0 | 0 | stable | zoh | 0 | 10 | 16 | ✓ | ✓ | 400 |
| `ETC.yaml` | EthanolConcentration | 0.000518 | 0.1 | 8 | 32 | 8 | 8 | 1 | 2 | exp | zoh | 0.05 | 20 | 8 | ✓ | – | 200 |
| `HB.yaml` | Heartbeat | 0.000332 | 0.1 | 64 | 16 | 4 | 4 | 0 | 0 | exp | zoh | 0.1 | 20 | 8 | ✓ | – | 400 |

All six use input-dependent Re(Λ) and B, C, LTI Im(Λ), ZOH, `dt_min` 0.001,
`dt_max` 0.1, `ff_mult` 1, RMSNorm on the projections.  The RFormer baseline
configurations (Moreno-Pino et al., 2024) are in `uea/configs/rformer/`.

To rerun the hyperparameter search: `python uea/hypersearch.py --help` (Optuna),
then `python uea/run_top_configs.py` to evaluate the best trials on more seeds.

## Physiome-ODE forecasting (Table 2)

```bash
python physiome_ode/run_final_folds.py --winner HOD01      # any code or dataset name below
```

Protocol: 5 folds (seed = fold index), up to 200 epochs with early stopping on
validation MSE (patience 30), AdamW with the three-group learning rates below
(SSM parameters at `lr`, the rest at `lr × lr_factor`) and a cosine schedule
after `warmup` epochs.  Reported: test MSE, mean ± std over folds.  The
configurations live in `physiome_ode/configs/winners.csv`; every winner uses
input-dependent Re(Λ) and B, C, LTI Im(Λ), `conj_sym` off, `dt_min` 0.001 and
RMSNorm on the projections.  HYN01 and JEL02 have `ff_mult` 0, which leaves each
block's GLU with no hidden units: the search found them at 0.5, but the runs
behind the paper's numbers read it as an integer.

| Code | Dataset | lr | lr factor | wd | h | ssm | ssm blocks | L | enc | λ enc | learn λ | disc | drop | batch | bc rank | ff | bidir | clip | warmup | proj init |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ASL01 | `aslanidi_2009` | 1.46e-05 | 127 | 0.00143 | 32 | 12 | 1 | 5 | 1 | 0 | standard | zoh | 0.15 | 80 | 5 | 2 | ✓ | ✓ | 5 | zeros |
| BAG01 | `bagci_2008a` | 5.96e-06 | 161 | 2.01e-06 | 56 | 8 | 2 | 13 | 1 | 0 | stable | bilinear | 0.15 | 16 | 4 | 1 | ✓ | ✓ | 5 | random |
| BER01 | `bertram_arnot_zamponi_2002` | 6.7e-05 | 330 | 0.0001 | 32 | 50 | 5 | 12 | 0 | 1 | stable | bilinear | 0.1 | 96 | 4 | 1 | ✓ | ✓ | 5 | zeros |
| BOR01 | `borghans_dupont_goldbeter_1997a` | 3.32e-05 | 69 | 2.14e-06 | 112 | 6 | 1 | 11 | 0 | 0 | stable | bilinear | 0.3 | 128 | 6 | 1 | ✓ | ✓ | 10 | zeros |
| BUT01 | `butera_rinzel_smith_1999` | 2.09e-05 | 30 | 0.0001 | 192 | 8 | 2 | 8 | 1 | 1 | exp | bilinear | 0.1 | 64 | 8 | 1 | – | – | 5 | zeros |
| BUT02 | `butera_rinzel_smith_1999_model2` | 0.000338 | 10 | 0.01 | 48 | 30 | 5 | 12 | 0 | 1 | exp | bilinear | 0.1 | 16 | 48 | 1 | – | ✓ | 5 | zeros |
| CAL01 | `calzone_thieffry_tyson_novak_2007` | 4.47e-05 | 252 | 1.03e-06 | 32 | 24 | 1 | 9 | 1 | 0 | stable | zoh | 0.05 | 112 | 10 | 1 | ✓ | ✓ | 5 | random |
| DIF01 | `difrancesco_noble_1985` | 6.7e-05 | 330 | 0.0001 | 32 | 50 | 5 | 12 | 0 | 1 | stable | bilinear | 0.1 | 96 | 4 | 1 | ✓ | ✓ | 5 | zeros |
| DOK01 | `dokos_celler_lovell_1996` | 3.96e-06 | 92 | 0.00539 | 32 | 2 | 1 | 2 | 0 | 0 | softplus | zoh | 0.2 | 96 | 16 | 8 | ✓ | ✓ | 10 | random |
| DUP01 | `dupont_1991a` | 8.46e-05 | 227 | 9.45e-07 | 16 | 10 | 1 | 7 | 2 | 0 | standard | zoh | 0.1 | 128 | 12 | 6 | ✓ | ✓ | 10 | random |
| DUP02 | `dupont_1992b` | 0.000224 | 7 | 1e-05 | 32 | 12 | 3 | 5 | 0 | 1 | exp | zoh | 0.3 | 32 | 8 | 1 | ✓ | – | 5 | zeros |
| DUP03 | `dupont_1991b` | 1.34e-05 | 181 | 1e-07 | 64 | 4 | 2 | 4 | 2 | 0 | standard | zoh | 0.15 | 64 | 8 | 8 | – | – | 10 | random |
| GUP01 | `gupta_aslakson_gurbaxani_vernon_2007_a` | 3.55e-05 | 228 | 1.11e-06 | 16 | 28 | 2 | 13 | 1 | 0 | standard | bilinear | 0.2 | 96 | 10 | 2 | ✓ | ✓ | 5 | random |
| GUP02 | `gupta_aslakson_gurbaxani_vernon_2007_b` | 8.32e-05 | 55 | 0.000241 | 80 | 24 | 1 | 6 | 1 | 0 | exp | bilinear | 0.15 | 64 | 10 | 1 | ✓ | – | 5 | random |
| GUY01 | `guyton_muscle_blood_flow_control_2008` | 6.7e-05 | 330 | 0.0001 | 32 | 50 | 5 | 12 | 0 | 1 | stable | bilinear | 0.1 | 96 | 4 | 1 | ✓ | ✓ | 5 | zeros |
| GUY02 | `guyton_pulmonary_oxygen_uptake_2008` | 3.28e-05 | 228 | 0.0179 | 40 | 16 | 4 | 11 | 0 | 0 | exp | zoh | 0.2 | 32 | 4 | 1 | ✓ | ✓ | 1 | random |
| HOD01 | `hodgkin_huxley_1952_variant01` | 4.08e-05 | 330 | 0.0001 | 192 | 16 | 2 | 8 | 0 | 0 | standard | bilinear | 0 | 96 | 2 | 1 | – | – | 5 | zeros |
| HUA01 | `huang_ferrell_1996` | 8.37e-06 | 330 | 0 | 64 | 16 | 2 | 6 | 0 | 0 | stable | bilinear | 0.15 | 96 | 32 | 1 | ✓ | ✓ | 5 | zeros |
| HYN01 | `hynne_dano_sorensen_2001` | 9.46e-05 | 82 | 1.31e-05 | 32 | 4 | 1 | 13 | 1 | 0 | standard | zoh | 0.2 | 80 | 1 | 0 | ✓ | ✓ | 10 | zeros |
| INA01 | `inada_N_2009` | 1.67e-06 | 132 | 0.00413 | 24 | 30 | 5 | 7 | 1 | 1 | stable | zoh | 0.25 | 112 | 10 | 2 | ✓ | – | 5 | zeros |
| IRI01 | `iribe_kohl_noble_2006` | 0.000449 | 4 | 1e-05 | 64 | 8 | 4 | 6 | 0 | 0 | exp | bilinear | 0.15 | 96 | 24 | 1 | – | ✓ | 5 | zeros |
| JEL01 | `jelic_cupic_kolaranic_2005_Fig4` | 4.59e-06 | 8 | 0.001 | 32 | 4 | 2 | 4 | 2 | 0 | standard | bilinear | 0 | 32 | 32 | 1 | ✓ | ✓ | 5 | zeros |
| JEL02 | `jelic_cupic_kolaranic_2005_Fig5` | 7.15e-05 | 59 | 0.000377 | 48 | 6 | 1 | 12 | 1 | 0 | exp | zoh | 0.3 | 64 | 12 | 0 | ✓ | – | 10 | random |
| KAR01 | `karagiannis_popel_2006` | 0.000286 | 9 | 1e-05 | 64 | 40 | 4 | 4 | 1 | 1 | stable | zoh | 0.1 | 16 | 4 | 1 | ✓ | ✓ | 5 | zeros |
| KAR02 | `karagiannis_popel_2004` | 4.71e-05 | 213 | 0.00185 | 32 | 16 | 1 | 5 | 1 | 0 | standard | bilinear | 0.05 | 80 | 3 | 2 | ✓ | ✓ | 5 | random |
| LEN01 | `lenbury_pacheenburawana_1991` | 6.7e-05 | 330 | 0.0001 | 32 | 50 | 5 | 12 | 0 | 1 | stable | bilinear | 0.1 | 96 | 4 | 1 | ✓ | ✓ | 5 | zeros |
| LEN02 | `lenbury_ruktamatakul_amornsamarnkul_2001_a` | 0.000471 | 10 | 1e-05 | 64 | 30 | 5 | 6 | 2 | 0 | exp | zoh | 0.2 | 48 | 24 | 1 | – | ✓ | 5 | zeros |
| LI01 | `li_1996_simple` | 1.29e-05 | 180 | 0.0001 | 96 | 12 | 1 | 10 | 0 | 1 | exp | bilinear | 0.05 | 64 | 2 | 1 | – | ✓ | 5 | zeros |
| LI02 | `li_1996_simple_from_paper` | 2.87e-06 | 313 | 1.14e-06 | 32 | 24 | 2 | 7 | 1 | 0 | standard | zoh | 0.2 | 32 | 15 | 2 | ✓ | ✓ | 1 | random |
| M01 | `M_blood_flow_parent` | 0.000314 | 11 | 0.0001 | 96 | 8 | 4 | 4 | 1 | 1 | stable | zoh | 0.05 | 32 | 16 | 1 | – | ✓ | 5 | zeros |
| MAC01 | `mackenzie_1996` | 8.37e-06 | 330 | 0 | 64 | 16 | 2 | 6 | 0 | 0 | stable | bilinear | 0.15 | 96 | 32 | 1 | ✓ | ✓ | 5 | zeros |
| MAL01 | `maldonado_2006` | 8.37e-06 | 330 | 0 | 64 | 16 | 2 | 6 | 0 | 0 | stable | bilinear | 0.15 | 96 | 32 | 1 | ✓ | ✓ | 5 | zeros |
| MIT01 | `mittler_sulzer_neumann_perelson_1998` | 6.42e-06 | 344 | 0.0658 | 20 | 36 | 3 | 4 | 1 | 1 | stable | zoh | 0.25 | 64 | 6 | 1 | ✓ | – | 5 | zeros |
| NEL01 | `nelson_murray_perelson_2000_general` | 6.23e-05 | 7 | 0.001 | 112 | 16 | 8 | 6 | 0 | 0 | exp | zoh | 0.05 | 32 | 16 | 1 | ✓ | – | 5 | zeros |
| NYG01 | `nygren_fiset_firek_clark_lindblad_clark_giles_1998` | 1.28e-05 | 130 | 0 | 192 | 20 | 2 | 6 | 1 | 0 | softplus | bilinear | 0.15 | 64 | 2 | 1 | ✓ | ✓ | 5 | zeros |
| PHI01 | `phillips_2007` | 4.27e-05 | 161 | 1.66e-05 | 32 | 28 | 1 | 16 | 1 | 0 | exp | bilinear | 0.2 | 64 | 2 | 1 | ✓ | ✓ | 5 | random |
| PUL01 | `pulmonary_O2_parent` | 0.00015 | 9 | 0.0001 | 160 | 8 | 4 | 10 | 1 | 0 | stable | bilinear | 0 | 112 | 16 | 1 | ✓ | ✓ | 5 | zeros |
| PUR01 | `purvis_smith_koizumi_butera_2007` | 0.000212 | 12 | 0.1 | 16 | 50 | 5 | 13 | 0 | 0 | stable | zoh | 0.15 | 80 | 32 | 1 | ✓ | – | 5 | zeros |
| PUR02 | `purvis_smith_koizumi_butera_2007_npc` | 0.000397 | 5 | 0.01 | 16 | 50 | 5 | 6 | 1 | 0 | stable | bilinear | 0 | 16 | 24 | 1 | ✓ | – | 5 | zeros |
| REE01 | `reed_nijhout_sparks_ulrich_2004` | 1.29e-05 | 180 | 0.0001 | 96 | 12 | 1 | 10 | 0 | 1 | exp | bilinear | 0.05 | 64 | 2 | 1 | – | ✓ | 5 | zeros |
| REV01 | `revilla_garcia-ramos_2003` | 3.81e-05 | 103 | 0.00381 | 96 | 14 | 1 | 10 | 0 | 0 | standard | bilinear | 0.15 | 96 | 12 | 2 | ✓ | ✓ | 5 | random |
| SHO01 | `shorten_ocallaghan_davidson_soboleva_2007_variant01` | 0.000366 | 5 | 0 | 112 | 6 | 3 | 2 | 1 | 0 | exp | bilinear | 0.15 | 96 | 2 | 1 | – | – | 5 | zeros |
| SHO02 | `shorten_ocallaghan_davidson_soboleva_2007` | 4e-05 | 46 | 0.000364 | 128 | 10 | 1 | 15 | 1 | 0 | standard | bilinear | 0.3 | 64 | 5 | 1 | ✓ | – | 5 | random |
| VAN01 | `vanbeek_2007` | 7.86e-05 | 188 | 8.64e-06 | 40 | 8 | 1 | 7 | 1 | 0 | standard | bilinear | 0.2 | 48 | 1 | 2 | ✓ | ✓ | 1 | zeros |
| VIL01 | `vilar_kueh_barkai_leibler_2002` | 3.71e-05 | 5 | 0.01 | 128 | 30 | 5 | 5 | 2 | 2 | exp | zoh | 0.2 | 64 | 16 | 1 | – | ✓ | 5 | zeros |
| WAN01 | `wang_2006` | 2.63e-05 | 159 | 0.00675 | 96 | 22 | 1 | 6 | 1 | 0 | exp | bilinear | 0.3 | 64 | 12 | 1 | ✓ | – | 5 | zeros |
| WOD01 | `wodarz_hamer_2007_b` | 6.7e-05 | 330 | 0.0001 | 32 | 50 | 5 | 12 | 0 | 1 | stable | bilinear | 0.1 | 96 | 4 | 1 | ✓ | ✓ | 5 | zeros |
| WOL01 | `wolf_passarge_somsen_snoep_heinrich_westerhoff_2000` | 1.44e-05 | 244 | 4.74e-07 | 48 | 4 | 2 | 11 | 2 | 0 | standard | zoh | 0.25 | 64 | 20 | 3 | ✓ | – | 10 | random |
| WOL02 | `wolf_heinrich_2000` | 1.11e-05 | 129 | 8.02e-05 | 32 | 12 | 1 | 8 | 0 | 0 | exp | zoh | 0.15 | 48 | 2 | 1 | ✓ | – | 10 | zeros |
| WOL03 | `wolf_sohn_heinrich_kuriyama_2001` | 3.27e-05 | 252 | 2.14e-06 | 80 | 20 | 1 | 6 | 1 | 0 | exp | bilinear | 0.05 | 96 | 6 | 2 | ✓ | ✓ | 5 | zeros |

To rerun a search: `python physiome_ode/hypersearch_physio.py --dataset <name> --fold 0 --num_trials 10`.

## Drop-rate generalisation on EigenWorms (Figure 6, drop-rate table)

```bash
python uea/droprate.py                                  # all 10 models x 3 seeds
python uea/droprate.py --models TIDES Mamba_S --seeds 42  # a subset
python uea/plot_droprate.py results/droprate/results.csv --out fig_droprate.pdf
```

EigenWorms without duplicates, split once 165/36/35 (stratified, split seed 0).
Every model trains at r_train = 0.5 (one fixed mask per seed, 8992 of 17984
steps kept) for 400 epochs with Adam.  It keeps the best-validation-accuracy
weights and is tested at r_test ∈ {0.1, 0.3, 0.5, 0.7, 0.9} with one mask per
r_test shared by all models.  Seeds 42-44.

| Model | Params | lr | wd | batch | |
|---|---|---|---|---|---|
| `S5` | 29,409 | 1e-3 | 0.1 | 10 | h 82; Λ, B, C LTI |
| `TIDES_Lambda` | 29,525 | 1e-3 | 0.1 | 10 | h 80; input-dependent Re(Λ) |
| `Mamba_S` | 29,605 | 1e-3 | 0.1 | 10 | h 16; input-dependent B, C and step (`step_mode="input_dependent"`) |
| `TIDES_BC` | 29,317 | 1e-3 | 0.1 | 10 | h 16; input-dependent B, C |
| `TIDES` | 29,605 | 1e-3 | 0.1 | 10 | h 16; input-dependent Re(Λ), B, C |
| `TIDES_full` | 29,893 | 1e-3 | 0.1 | 10 | h 16; input-dependent Λ, B, C |
| `RFormer` | 140,911 | 6.73e-3 | 0.1 | 5 | published EigenWorms configuration |
| `Mamba` | 27,669 | 1e-3 | 0 | 10 | expand 16, d_conv 4 |
| `Mamba2` | 29,261 | 1e-3 | 0 | 32 | expand 32, headdim 64, dropout 0.1 |
| `Mamba3` | 26,629 | 1e-3 | 0 | 10 | SISO, expand 32, headdim 64, RoPE on half of the state |

The SSM rows use one block, state size P = 16, bidirectional scans and ZOH.
They get the gaps between kept steps as the step.  The Mamba rows use d_model
16, d_state 16 and one block: the TIDES block (norm, mixer, GELU, GLU,
residual) with LayerNorm and the Mamba mixer in place of BatchNorm and the SSM.
They see a t/L time channel built on the full grid before dropping.  Their
weight decay is 0, because at 0.1 (Adam) Mamba-1 and Mamba-2 collapse to
constant predictors.  Mamba-1/2 need `mamba_ssm` on a GPU (see
the README); Mamba-3 runs anywhere.  On one RTX 4090 the whole run (10 models,
3 seeds) takes about two hours.  Per epoch: 0.4-0.7 s for the SSM rows, 0.2 s
for RFormer, 0.1-0.2 s for Mamba-1/2, and 1.4 s for Mamba-3 (chunked PyTorch
scan).  On a GPU the runs are bit-reproducible for every model except RFormer;
on CPU RFormer is too.

## Fading Flash (main-text section and appendix)

```bash
OMP_NUM_THREADS=1 python fading_flash/fading_flash.py      # trains 9 small models on CPU, writes 8 PDFs
```

Seed 0, 3000 Adam steps per model (`--steps` to change), about 12 minutes on
one CPU core.


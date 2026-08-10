# Physics-Informed Neural Operator for the Static Euler-Bernoulli Beam

A modular research codebase for learning the forward (`G_f: q(x) -> w(x)`)
and inverse (`G_i: w(x) -> q(x)`) operators of the static Euler-Bernoulli
beam equation `EI d^4w/dx^4 = q(x)`, using a Fourier Neural Operator (FNO)
with an optional physics-informed residual loss.

This replaces an earlier single-notebook implementation. It is organized
into a proper Python package so each research component (FEM solver,
dataset generation, model, physics loss, training, evaluation) lives in
its own module.

## Key design decision: boundary conditions are NOT part of the input

Earlier drafts encoded the support type (cantilever / simply supported /
fixed-fixed / ...) as a one-hot channel in the operator input, while the
dataset generator *also* randomized the support type per sample. That is
inconsistent: with the BC randomized per sample but not identifiable from
the input, `q(x) -> w(x)` is not a well-defined function.

Per supervisor review, the BC channel has been removed entirely. Instead:

- **A single support type is fixed for an entire dataset / experiment**
  (`support_type` in `configs/default.yaml`).
- The operator input is simply `[signal(x), x]` — 2 channels — where
  `signal` is `q(x)` for the forward operator or `w(x)` for the inverse
  operator. `x` is included only so the FNO has access to spatial
  position, not as a boundary-condition encoding.
- To study a different boundary condition, copy the config, change
  `support_type`, regenerate the dataset, and train a new model. Results
  across boundary conditions should be compared as separate experiments,
  not folded into one mixed dataset.

## Load randomization

`beam_no/fem/loads.py` (`LoadGenerator`) uses six deterministic load
families (uniform, triangular, reverse-triangular, sinusoidal, Gaussian
bump, truncated random Fourier series) with randomized amplitudes/shape
parameters, matching the original implementation. This is **not** a
Gaussian Random Field (GRF) sampler — if you want broader, less
structured coverage of the function space of `q(x)`, a GRF-based
generator would be a natural extension (e.g. sampling coefficients of a
truncated KL expansion with a squared-exponential covariance kernel), but
this was intentionally left out of scope for the current study.

## Project layout

```
beam_no/                  installable package
├── fem/                   Beam, SupportType, BeamElement, FEMAssembler,
│                          BoundaryConditions, LoadGenerator, LoadConverter, FEMSolver
├── data/                  DatasetGenerator, BeamDataset, InverseBeamDataset, split/save/load
├── models/                SpectralConv1d, FNOBlock1d, BeamFNO
├── physics/                fourth_derivative, physics_loss, compute_physics_residual
├── training/               train_pino (forward), train_inverse_fno (inverse)
├── evaluation/             metrics.py (quantitative), plots.py (figures)
└── utils/                  set_seed, viz.py (problem schematic, FNO architecture diagram)

configs/default.yaml       all hyperparameters / paths for one experiment
scripts/                   generate_dataset.py, train_forward.py, train_inverse.py, evaluate.py
notebooks/demo.ipynb       thin wrapper for exploration / paper figures
outputs/                   data/, checkpoints/, figures/  (git-ignored)
```

## Usage

```bash
pip install -e .          # or: pip install -r requirements.txt

python scripts/generate_dataset.py --config configs/default.yaml
python scripts/train_forward.py    --config configs/default.yaml
python scripts/train_inverse.py    --config configs/default.yaml
python scripts/evaluate.py         --config configs/default.yaml --direction forward
python scripts/evaluate.py         --config configs/default.yaml --direction inverse
```

Each script reads all settings from the YAML config — no hard-coded paths
(no Google Drive dependence). To run a second experiment (e.g. a
different support type or a different `lambda_phys`), copy
`configs/default.yaml`, edit it, and point each script at the new file;
outputs land in `dataset.output_dir` / `training.checkpoint_dir` /
`evaluation.figures_dir` as configured, so experiments don't overwrite
each other if you also change those paths.

## Validation

The FEM solver was checked against closed-form Euler-Bernoulli solutions
for a uniformly distributed load (simply-supported midspan deflection
`5qL^4/384EI` and cantilever tip deflection `qL^4/8EI`); both match the FE
solution to numerical precision.

## Figures for the paper

`scripts/generate_dataset.py` automatically saves, into
`evaluation.figures_dir`:
- `beam_problem_schematic.png` — sketch of the static problem (w(x), q(x), supports)
- `fno_architecture.png` — FNO block diagram (lifting → Fourier layers → projection)
- `sample_q_w.png` — one representative `(q(x), w(x))` training pair

`scripts/train_forward.py` / `train_inverse.py` save training-curve plots
(total loss, and data-vs-physics loss for the forward model).
`scripts/evaluate.py` saves prediction-vs-ground-truth plots, a
scatter plot, an error histogram, and a JSON metrics file
(`Relative_L2`, `MSE`, `RMSE`, `MAE`, `Max_Error`, `R2`).

---

## Dynamic (forced-vibration) extension

The static study above has a dynamic counterpart: forced vibration under
a time-varying distributed load, `G_f: q(x,t) -> w(x,t)` (forward) and
`G_i: w(x,t) -> q(x,t)` (inverse), governed by

```
rho*A * d^2w/dt^2 + C * dw/dt + EI * d^4w/dx^4 = q(x,t)
```

with Rayleigh damping `C = alpha*M + beta*K`. It reuses every design
decision from the static study (single fixed support per experiment, no
BC input channel, structured-randomized load families rather than a GRF)
and extends the architecture from a 1D FNO to a 2D FNO operating jointly
over the `(x, t)` grid.

### New modules

| Module | Purpose |
|---|---|
| `fem/dynamic_solver.py` | Newmark-β time integration (unconditionally stable) of the FEM equation of motion |
| `fem/temporal_loads.py` | Temporal load-profile families (step-ramp, sinusoidal, multi-frequency, Gaussian pulse), combined with the existing spatial `LoadGenerator` as a separable load `q(x,t) = q_spatial(x) · f_temporal(t)` |
| `data/dynamic_generator.py`, `dynamic_dataset.py`, `dynamic_io.py` | Dataset generation/I/O for the `(num_nodes, num_steps)` grid |
| `models/spectral_conv2d.py`, `fno_block2d.py`, `fno2d.py` | `BeamFNO2d` — the 2D FNO analogue of `BeamFNO`, treating `(x,t)` as a joint spatiotemporal domain (single forward pass, no autoregressive stepping) |
| `physics/dynamic_losses.py` | Continuum PDE residual matching the FEM solver's Rayleigh damping: `rho*A*w_tt + alpha*rho*A*w_t + beta*EI*(w_xxxx)_t + EI*w_xxxx - q` |
| `utils/grid.py` | `attach_xt_channels` — the single, shared place that broadcasts the 1D `x`/`t` coordinate arrays into the `(num_nodes, num_steps)` grid before concatenating with a signal; used identically by training and evaluation so this logic isn't duplicated (and re-broken) in multiple places |

### Usage

```bash
python scripts/generate_dataset_dynamic.py --config configs/dynamic.yaml
python scripts/train_forward_dynamic.py    --config configs/dynamic.yaml
python scripts/train_inverse_dynamic.py    --config configs/dynamic.yaml
python scripts/evaluate_dynamic.py         --config configs/dynamic.yaml --direction forward
python scripts/evaluate_dynamic.py         --config configs/dynamic.yaml --direction inverse
```

`configs/dynamic.yaml` derives the time step `dt` from the beam's own
first natural period (`steps_per_period`, `num_periods`) rather than a
hard-coded value, so the discretization stays sensible if you change beam
geometry or material properties.

### Validation performed

The dynamic FEM solver (mass matrix + Newmark-β + Rayleigh damping) was
validated directly, independent of the neural operator:
- **Undamped free vibration**: the dominant frequency in the FFT of the
  simulated midspan response matches the analytical first natural
  frequency of a simply-supported beam to 10 significant figures.
- **Damped response**: enabling Rayleigh damping produces the expected
  amplitude decay over the simulated time window.

The `BeamFNO2d` model, physics loss, training loops, and evaluation code
were written to mirror the validated static implementation and were
syntax-checked, but — unlike the FEM solver and dataset generator above —
were not executed end-to-end in the environment this codebase was
authored in (no `torch` available there). **Run
`scripts/generate_dataset_dynamic.py` first on a machine with `torch`
installed** to confirm the full pipeline before relying on it for results.

### Scale

The default config uses a coarser mesh (`num_elements: 32` vs. 64) and a
more modest sample count (`num_samples: 1500`) than the static study,
since each dynamic sample is a full `(num_nodes, num_steps)` grid rather
than a single vector — increase both once the pipeline is confirmed
working end-to-end on your hardware.


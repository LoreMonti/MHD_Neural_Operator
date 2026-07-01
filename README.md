# MHD Instability Neural Operator (`mhd-fno`)

A **Fourier Neural Operator (FNO)** that emulates the nonlinear evolution of a 2D
incompressible MHD instability, ~100–1000× faster than the numerical solver, and that
**generalizes across physical parameters** — in particular recovering the **magnetic
stabilization threshold** of the Kelvin–Helmholtz (KH) instability.

Type (A): a *surrogate* of the solver, not a subgrid closure. It is the data-driven
counterpart of Physics-Informed Neural Networks (PINN): a PINN solves one instance of a
PDE, while an FNO learns the solution *operator* across many configurations.

See [`ROADMAP.md`](ROADMAP.md) for the full plan and design decisions.

## Physics in one paragraph

A shear layer (velocity profile $v_x(y)$) in a periodic box, threaded by an in-plane
magnetic field $B_0$ aligned with the shear, rolls up into Kelvin–Helmholtz vortices.
Magnetic tension opposes the roll-up, so KH is **suppressed** above a threshold on the
Alfvénic Mach number

$$
M_A = \frac{\Delta v}{v_A} \lesssim 2, \qquad v_A = \frac{B_0}{\sqrt{\mu_0 \rho}}.
$$

The scientific goal is to have the FNO recover this threshold on **unseen** $B_0$.

## Key design decisions

| Topic | Choice |
|---|---|
| ML framework | PyTorch (+ `neuraloperator`) |
| Training strategy | Joint training + checkpointing; continual learning deferred to Phase 4 |
| Ground-truth solver | Custom pseudo-spectral in PyTorch ($\omega$–$\psi$–$A$), Dedalus cross-check |
| Field representation | $(\omega, A)$ — vorticity + magnetic flux potential (2 channels) |
| Dataset sweep | $M_A\in[0.5,6]$ (test hole $[1.5,2.5]$), $\mathrm{Re}\in[500,5000]$, $\mathrm{Pm}=1$, $128^2$, ~250 runs |

## Repository layout

```
mhd-fno/
├── src/mhd_fno/
│   ├── solver/        # pseudo-spectral 2D incompressible MHD solver (ground truth)
│   ├── data/          # dataset generation, storage, PyTorch Dataset/DataLoader
│   ├── models/        # FNO and baselines
│   ├── training/      # training loops, losses, checkpointing
│   ├── evaluation/    # diagnostics: growth rate, spectra, conservation, threshold
│   └── utils/         # spectral helpers, config, logging
├── configs/           # experiment configuration files (YAML)
├── scripts/           # CLI entry points (generate data, train, evaluate)
├── tests/             # unit tests (solver validation, invertibility, ...)
├── notebooks/         # exploratory analysis and figures
└── data/              # raw/ and processed/ datasets (git-ignored)
```

## Status

Phase 0 (setup). See [`ROADMAP.md`](ROADMAP.md).

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .          # install mhd_fno in editable mode
```

## Fields and conventions

In 2D incompressible flow we evolve vorticity $\omega = -\nabla^2\psi$ and magnetic flux
potential $A$, with

$$
\mathbf{v} = \nabla\times(\psi\,\hat{z}), \qquad \mathbf{B} = \nabla\times(A\,\hat{z}),
$$

which enforces $\nabla\cdot\mathbf{v} = 0$ and $\nabla\cdot\mathbf{B} = 0$ by
construction. Primitive fields $(\mathbf{v}, \mathbf{B})$ are reconstructed from
$(\omega, A)$ by a spectral Laplacian inversion for diagnostics.

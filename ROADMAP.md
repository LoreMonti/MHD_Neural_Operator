# MHD Instability Neural Operator — Roadmap

> **Goal**: a Fourier Neural Operator (FNO) that emulates the nonlinear evolution of a
> 2D incompressible MHD instability, ~100–1000× faster than the numerical solver, and
> that **generalizes across physical parameters** — in particular recovering the
> **magnetic stabilization threshold** of the Kelvin–Helmholtz instability.
>
> Type (A): a *surrogate* of the solver, not a subgrid closure.
>
> Authoritative spec: Obsidian note `04 Research/MHD Instability Neural Operator.md`.

---

## Reference physics

**Primary case: magnetized Kelvin–Helmholtz.**
A shear layer (velocity profile $v_x(y)$) in a periodic box, with an in-plane magnetic
field $B_0$ aligned with the shear. The perturbation grows, rolls up into vortices,
then saturates.

**Physical core — magnetic stabilization.** Magnetic tension opposes the roll-up of
field lines. KH is suppressed above a threshold on the Alfvénic Mach number:

$$
M_A = \frac{\Delta v}{v_A} \lesssim 2, \qquad v_A = \frac{B_0}{\sqrt{\mu_0 \rho}}
$$

Recovering this threshold on unseen $B_0$ = proof that the FNO learned physics, not
interpolation. **This is the primary success metric.**

**Field representation — decided: $(\omega, A)$.** The FNO operates on vorticity
$\omega = -\nabla^2\psi$ and magnetic flux potential $A$ (2 channels, matching the
solver's evolved variables):
- $\omega$ emphasizes the small scales (vortices) that KH is about;
- $A$ makes $\mathbf{B} = \nabla\times(A\,\hat{z})$ **divergence-free by construction**,
  so $\nabla\cdot\mathbf{B} = 0$ is exact regardless of what the network predicts.
Primitive fields $(\mathbf{v}, \mathbf{B})$ are reconstructed from $(\omega, A)$ via a
single spectral Laplacian inversion for diagnostics / literature comparison
(e.g. PDEBench uses primitive fields).

---

## Phases

### Phase 0 — Setup
- Repo `mhd-fno`, environment, folder structure, plotting.
- **ML stack: PyTorch** (+ the `neuraloperator` library as an FNO reference).
- **Solver: custom pseudo-spectral in PyTorch** ($\omega$–$\psi$–$A$ vorticity/flux
  formulation), with Dedalus as a cross-check on a couple of runs.
- [ ] Repo structure defined
- [ ] Reproducible environment (requirements / env file)

### Phase 1 — Data generation (ground truth)
- 2D incompressible **MHD solver**: custom pseudo-spectral (PyTorch), Dedalus cross-check.
- KH setup: shear layer, periodic box, in-plane field $B_0$.
- **Parameter sweep** (starting configuration, revisited after first results):

  | Parameter | Range / value | Notes |
  |---|---|---|
  | $M_A = \Delta v / v_A$ | $[0.5, 6]$, **test-only hole $[1.5, 2.5]$** | key parameter for the threshold |
  | $\mathrm{Re} = \Delta v\,L/\nu$ | $[500, 5000]$ | Latin-Hypercube sampled jointly with $M_A$ |
  | $S = \Delta v\,L/\eta$ | $= \mathrm{Re}$ ($\mathrm{Pm}=1$) | fixed at first; opened only if needed |
  | Resolution | $128^2$ (+ few $256^2$ for invariance test) | modest for cost; FNO is resolution-invariant |
  | # simulations | ~250 | ~80 uniform time snapshots each, different seeds |

- **The threshold hole is the point.** Excluding $M_A \in [1.5, 2.5]$ from training and
  testing there proves the FNO recovers the magnetic stabilization threshold from
  *physics*, not interpolation — the project's core scientific result.
- Store $(\omega, A)$ per snapshot + metadata (parameters, seed). Reconstruct
  $(\mathbf{v}, \mathbf{B})$ for diagnostics.
- Rough storage: $128^2 \times 2 \times \sim 80 \times \sim 250$ in `float32` $\approx$ 6–7 GB.
- [x] Working, validated solver (reproduces KH linear growth; threshold at $M_A\approx2.5$)
- [x] Dataset generation pipeline + storage (280 runs, 2.8 GB, in `data/raw/`)
- [x] Field representation → $(\omega, A)$ (primitives reconstructed for diagnostics)

> **Dataset generated (280 runs: 250 train + 30 test, $128^2$, 81 frames each).**
> Physics check: $E_y$ decays for $M_A \lesssim 2$ and grows above, threshold near the
> theory. Note the threshold also depends on $\mathrm{Re}$: low-Lundquist runs (high
> resistivity, field slips) stay unstable below the ideal $M_A\approx2$ — which is why
> the FNO is conditioned on both $M_A$ and $\mathrm{Re}$.

### Phase 2 — Model
- FNO mapping $\text{field}(t) \to \text{field}(t+\Delta t)$.
- Autoregressive rollout for long horizons; conditioning on parameters ($B_0$, Re).
- Loss: field MSE + (optional) spectral / physics-aware terms
  (energy spectrum, $\nabla\cdot\mathbf{B} = 0$).
- [ ] Single-regime baseline
- [ ] Stable autoregressive rollout
- [ ] Parametric conditioning

### Phase 3 — Evaluation (physicist's checklist)
- **Instability growth rate** vs linear theory (early time).
- **Energy spectra** and vortex structure.
- **Conservation** diagnostics (energy, $\nabla\cdot\mathbf{B} = 0$).
- **Speedup** vs the solver (wall-clock).
- **Generalization**: unseen $B_0$ → **magnetic stabilization threshold**;
  unseen resolution (FNO resolution-invariance).
- [ ] Full diagnostics implemented
- [ ] Stability diagram reconstructed from the FNO

### Phase 4 — Extensions (after KH works)
- **Kink / current-driven** → bridges to fusion / Tokamak.
- **Magnetorotational (MRI)** → accretion disks (harder: shearing box).

---

## Risks / watch-outs
- Autoregressive rollout accumulates error → **multi-step training / rollout loss**.
- Data generation is the real work → start at modest resolution.
- Keep $\nabla\cdot\mathbf{B} = 0$ under control → divergence cleaning or a constrained
  representation ($\psi$/$A$).

---

## Open decisions (tackled one at a time)
- [x] ML framework → **PyTorch**
- [x] Training strategy → **joint training + checkpointing**; true continual learning
      deferred to Phase 4 (new instabilities)
- [x] Data solver → **custom pseudo-spectral in PyTorch** ($\omega$–$\psi$–$A$),
      Dedalus as cross-check
- [x] Field representation → $(\omega, A)$ (2 channels; primitives reconstructed
      for diagnostics)
- [x] Dataset sweep → $M_A\in[0.5,6]$ (test hole $[1.5,2.5]$), $\mathrm{Re}\in[500,5000]$,
      $\mathrm{Pm}=1$, $128^2$, ~250 runs (LHS)

## Stack
- **Solver**: custom pseudo-spectral in PyTorch ($\omega$–$\psi$–$A$); Dedalus cross-check
- **ML**: PyTorch (+ `neuraloperator`)
- **Utilities**: NumPy, Matplotlib
- **Optional cross-check**: PDEBench MHD data

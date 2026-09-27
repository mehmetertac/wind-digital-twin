# PINN reading notes (Week 10)

Deep read of one foundational paper for the capstone PINN guardrail. Toy implementation spec is at the end.

---

## Citation and why this paper

**Raissi, Perdikaris & Karniadakis (2019),** “Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations,” *Journal of Computational Physics* **378**, 686–707.

This is the paper that defines the PINN loss (data + PDE residual at collocation points), automatic differentiation for spatial/temporal derivatives, and soft boundary/initial constraints. The 2021 Karniadakis et al. *Nature Reviews Physics* survey is broader but lighter on these mechanics; one paper is enough for the capstone guardrail (see [ARCHITECTURE.md](ARCHITECTURE.md)).

---

## Core idea

Approximate the PDE solution with a neural network \(u_\theta(t, x)\) (continuous-time formulation). Define the PDE residual

\[
f := u_t + \mathcal{N}[u; \lambda],
\]

where \(\mathcal{N}\) is the spatial differential operator in the PDE and \(\lambda\) are physical parameters (e.g. viscosity, diffusivity). The same network weights \(\theta\) define both \(u_\theta\) and \(f_\theta\); \(f\) is built by differentiating the network output with respect to inputs, not by a separate mesh solver.

---

## Loss: data term + residual term

Total loss (paper notation):

\[
\mathcal{L} = \mathrm{MSE}_u + \mathrm{MSE}_f.
\]

| Term | Role |
|------|------|
| **MSE\_u** | **Data term:** fit \(u\) at initial/boundary points and any labeled observations \(\{t_u, x_u, u\}\). |
| **MSE\_f** | **PDE-residual term:** enforce \(f \approx 0\) at **collocation points** \(\{t_f, x_f\}\) in the interior (and space–time domain). |

**Boundary and initial conditions** enter through **MSE\_u** as supervised targets (e.g. \(u(0,t)\), \(u(1,t)\), \(u(x,0)\)). They are **soft constraints**: penalties in the loss, not hard-coded into the architecture. Satisfaction is approximate and depends on relative weighting of \(\mathrm{MSE}_u\) vs \(\mathrm{MSE}_f\) (and often per-term weights in practice).

**Collocation:** interior points are typically sampled with **Latin hypercube sampling** (LHS); paper uses on the order of \(N_f \sim 10^4\) for harder PDEs and \(N_u \sim 10^2\) for boundary/initial/data points.

---

## Automatic differentiation

Derivatives \(u_t\), \(u_x\), \(u_{xx}\), etc. are computed by **reverse-mode AD** of the network with respect to **inputs** \((t, x)\), not by finite differences on a grid. No mesh is required for the residual evaluation.

**Activation:** smooth activations (e.g. **tanh**) are used so higher-order derivatives exist. ReLU networks have \(u_{xx} \approx 0\) almost everywhere, which breaks second-order PDE residuals.

---

## Continuous vs discrete time (brief)

The paper presents **continuous-time** PINNs (network takes \((t,x)\)) and **discrete-time** variants tied to Runge-Kutta stages. For the Week 10 toy we use **continuous time** only.

---

## Forward vs inverse problems

| Mode | Unknowns | Use |
|------|----------|-----|
| **Forward** | Network weights \(\theta\) only; \(\lambda\) fixed | Solve PDE given physics |
| **Inverse** | \(\theta\) **and** PDE coefficients \(\lambda\) trainable | Discover parameters from sparse/noisy data |

Inverse PINNs align with **calibrating** physical parameters (e.g. diffusivity \(\alpha\)) from observations—relevant for per-turbine thermal calibration.

---

## Training details (from paper)

- Optimizers: **Adam** then often **L-BFGS** for fine convergence.
- Example architecture (Burgers): **9 hidden layers × 20 neurons**, tanh.
- Scale: \(N_u \sim 100\), \(N_f \sim 10{,}000\) (problem-dependent).

---

## Known weaknesses (flagged, not deep-dived)

- **Loss imbalance:** data, BC/IC, and residual terms often need manual or adaptive weighting.
- **Spectral bias:** networks favor smooth solutions; sharp fronts / high frequencies are hard.
- **Cost:** for pure **forward** problems, classical solvers are usually faster; PINNs shine when data + physics are **jointly** used (inverse, sparse data, parametric families).

---

## What transfers to a drivetrain thermal model

The capstone gearbox path is **not** a spatial PDE on a shaft mesh. The useful PINN **pattern** still transfers as follows.

### Natural physics form: lumped thermal ODE

A first-order lumped model (schematic):

\[
C \,\frac{dT}{dt} = a\,P + b\,\mathrm{RPM} - hA\,(T - T_{\mathrm{nac}}),
\]

with thermal capacitance \(C\), heat-transfer lump \(hA\), and load coupling \(a, b\). **Automatic differentiation** is then with respect to **time** \(t\) only (drivers \(P\), RPM, \(T_{\mathrm{nac}}\) can be treated as known inputs at each SCADA step).

### Relation to current code

[`physics/gearbox_thermal.py`](../src/wind_digital_twin/physics/gearbox_thermal.py) fits a **steady-state** Ridge (or GBM) map from power, RPM, nacelle temp → oil/bearing temperature. That is the **\(dT/dt = 0\)** limit of the ODE above. A PINN-style or physics-informed ODE adds **thermal lag** and dynamic consistency that steady-state regression can miss.

### Inverse problem = highest value here

Train **interpretable parameters** \(\{C, hA, a, b\}\) per turbine jointly with a small network or explicit \(T(t)\) ansatz, using:

- **Data term:** SCADA oil/bearing temperature at 10-min stamps.
- **Residual term:** ODE residual at the same timestamps (collocation is “free” wherever drivers are observed).

Drift in \(hA\) or growing ODE residual on healthy-looking load can complement the existing **pred − actual** residual stream.

### Residual as anomaly signal

Define \(f = C\,\dot T - aP - b\,\mathrm{RPM} + hA(T - T_{\mathrm{nac}})\) (with \(\dot T\) from AD or finite diff on smoothed \(T\)). Large \(|f|\) on new data is a **physics-consistency** score—usable upstream of [`residual/`](../src/wind_digital_twin/residual/) and [`anomaly/`](../src/wind_digital_twin/anomaly/) features.

### Caveats

- **10-min averaging** blurs fast transients the ODE might represent.
- **Sensor noise** and missing data weaken both data and residual terms.
- **Soft physics:** if the data term dominates, the ODE can be violated locally—same weighting issues as spatial PINNs.
- **Evaluation:** healthy-only training and **time-ordered splits** unchanged ([ARCHITECTURE.md](ARCHITECTURE.md)).
- **Baseline:** fitting the same ODE with `scipy.integrate` + nonlinear least squares may match a PINN at much lower cost—**beat that baseline** before claiming PINN value on SCADA.

---

## Tomorrow’s toy: 1-D heat equation PINN (spec only)

Implementation target: thin notebook in [`notebooks/`](../notebooks/) (see [`notebooks/README.md`](../notebooks/README.md)). Add optional dependency `torch` via `pip install -e ".[pinn]"` when implementing (keep CI torch-free).

### PDE and domain

\[
u_t = \alpha u_{xx}, \quad x \in [0,1],\; t \in [0,1], \quad \alpha = 0.1.
\]

- **Initial condition:** \(u(x,0) = \sin(\pi x)\).
- **Boundary conditions:** Dirichlet \(u(0,t) = u(1,t) = 0\).

### Analytical / reference solution

\[
u(x,t) = e^{-\alpha \pi^2 t}\,\sin(\pi x).
\]

At \(t=1\), amplitude decays by factor \(e^{-0.1\pi^2} \approx 0.37\)—visible but not trivial.

### Network

- **Inputs:** \((x, t)\) — 2 dimensions.
- **Architecture:** MLP, **4 hidden layers × 32 tanh** units, **1 output** (~3.3k parameters).
- **Init:** Xavier.

### Loss terms

\[
\mathcal{L} = w_f\,\mathrm{MSE}_f + w_0\,\mathrm{MSE}_{\mathrm{ic}} + w_b\,\mathrm{MSE}_{\mathrm{bc}} \;(+ w_d\,\mathrm{MSE}_{\mathrm{data}}\ \text{optional inverse variant}).
\]

Start with \(w_f = w_0 = w_b = 1\) (tune if imbalanced).

| Term | Definition | Sampling |
|------|------------|----------|
| **MSE\_f** | Mean squared PDE residual \(u_t - \alpha u_{xx}\) | \(N_f = 2000\) interior points (LHS or Sobol) |
| **MSE\_ic** | \(\|u - \sin(\pi x)\|^2\) at \(t=0\) | \(N_0 = 100\) |
| **MSE\_bc** | \(\|u\|^2\) at \(x=0\) and \(x=1\) | \(N_b = 100\) per boundary |
| **MSE\_data** (optional) | Noisy interior observations | \(N_d = 50\), \(\sigma = 0.01\); train \(\alpha\) (log-param, init 0.5); success = recover \(\alpha\) within **5%** |

### Training

1. **Adam**, lr \(10^{-3}\), ~5000 steps.
2. **L-BFGS** until convergence (loss plateau).

### Metrics and plots

- **Relative L2 error** on a \(101 \times 101\) grid vs analytical solution; target **&lt; 1e-2**.
- Plots: solution error heatmap; loss curves per term (\(\mathrm{MSE}_f\), \(\mathrm{MSE}_{\mathrm{ic}}\), \(\mathrm{MSE}_{\mathrm{bc}}\)).

### Framework note

[`pyproject.toml`](../pyproject.toml) does not yet list `torch`. When implementing the notebook, add optional extra e.g. `pinn = ["torch>=2.0"]` so default installs and CI stay lightweight.

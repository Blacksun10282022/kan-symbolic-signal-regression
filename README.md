# KAN vs PySR for Symbolic Signal Regression

This repo contains a small research-style experiment where I use
**Kolmogorov–Arnold Networks (KAN, Ziming Liu's pykan)** and
**PySR (symbolic regression)** to recover the closed-form expression of a
non-trivial 1D signal.

The goal is to:

1. Fit a synthetic target signal with KAN (numeric).
2. Turn the trained KAN into a **symbolic** model via `auto_symbolic`.
3. Fit the same data (and the KAN outputs) with PySR.
4. Compare test error, symbolic formulas, and visual fits.

> This project is intended as a portfolio / research toy project
> demonstrating how to combine modern neural architectures (KAN)
> with classical symbolic regression (PySR & Julia).

---

## 1. Target signal

The target is a combination of harmonic components and a chirp term:

\[
f(x) = \frac{4}{\pi}\sin(2\pi 10x)
     + \frac{4}{3\pi}\sin(2\pi 30x)
     + \frac{2}{\pi}\sin\bigl(2\pi(50x + 20x^2)\bigr), \quad x \in [-0.5, 0.5]
\]

- Sampling rate: **15 kHz**, giving 15,000 samples on the interval.
- Data split: **80% train / 20% test**, and all methods use the same
  test split for a fair comparison.

---

## 2. Methods

### 2.1 KAN (pykan)

I use Ziming Liu's `pykan` library with:

- Width: `[1, 12, 3, 1]`  
  (the penultimate layer has 3 neurons so that each neuron can capture
  one component of the signal).
- Symbolic branch enabled: `symbolic_enabled=True`.
- Progressive grid refinement:
  `grid = 40 → 60 → 90 → 120 → 150`.
- Each refinement step is followed by:
  - Adam warm-up
  - LBFGS polishing
  - Several stability helpers (sanitizing NaNs/Infs, warm-up with
    smaller LR, safe refinement with rollback).

Two KAN-based models are evaluated:

1. **KAN (numeric)** – the final numeric model after refinement.
2. **KAN (symbolic on copy)** – a copy of the trained model processed by
   `model.auto_symbolic(lib=['sin', 'x', 'x^2'])` and lightly re-fit.

### 2.2 PySR (symbolic regression)

I use [PySR](https://github.com/MilesCranmer/PySR):

- Input features: `[x, x^2]` so that PySR can easily form
  `sin(a x + b x^2)` type expressions.
- Operators:
  - Binary: `["+", "-", "*"]`  (no division or powers for stability)
  - Unary: `["sin"]`
- Nested `sin(sin())` is explicitly forbidden via `nested_constraints`.
- A moderate search budget is used, for example:
  `niterations ≈ 100`, `population_size ≈ 200`
  (you can adjust this in the notebook).

Two PySR models are trained:

1. **PySR (direct on data)** – fits the true targets `y`.
2. **PySR (distilled from KAN)** – fits the KAN outputs
   `y_kan = model(x)` (teacher–student distillation),
   but evaluation is still against the true `y`.

---

## 3. Repository structure

```text
.
├─ notebooks/
│   └─ KAN_Symbolic_TargetOnly_FILLED.ipynb   # main experiment notebook
├─ figs/                                      # saved figures
├─ outputs/                                   # PySR output directories
├─ environment.yml
├─ README.md
├─ LICENSE
└─ .gitignore


The notebook KAN_Symbolic_TargetOnly_FILLED.ipynb is self-contained:
running all cells will:

Generate the dataset from the analytic target signal.

Train and refine the KAN model.

Run auto_symbolic on a copy of KAN to obtain a symbolic model.

Fit PySR (direct & distilled).

Compute test MSE / 
𝑅
2
R
2
 for all four models.

Plot the target signal vs each fitted model on the full interval.

4. Setup

I use conda and Python 3.10+.

4.1 Create the environment
conda env create -f environment.yml
conda activate kan-symbolic-signal

4.2 Install PySR (Julia side)

PySR requires Julia ≥ 1.9 installed on your system and available
on PATH.

The first time you use PySR, run:

python -c "from pysr import PySRRegressor; PySRRegressor().install()"


This will install the required Julia packages.

4.3 Run the notebook
jupyter notebook notebooks/KAN_Symbolic_TargetOnly_FILLED.ipynb


Then execute all cells from top to bottom.

5. Example results

On the held-out test split, a typical run yields results similar to:

Method	MSE (test)	R² (test)
KAN (numeric)	~1.2e-7	~1.000000
KAN (symbolic on copy)	~1.1e+0	~0.01
PySR (direct on data)	~2.0e-1	~0.82
PySR (distilled KAN)	~2.0e-1	~0.82

In addition to metrics, the notebook also generates five plots:

True target signal.

Target vs KAN (numeric).

Target vs KAN (symbolic).

Target vs PySR (direct).

Target vs PySR (distilled).

These plots provide an intuitive visual comparison of how well each
method fits the signal.

6. Possible extensions

Some natural extensions that could be explored:

Using richer operator libraries for PySR (e.g. /, ^, cos) with
regularization to control complexity.

Trying different KAN architectures (width / grid schedules) and
symbolic libraries.

Applying the same pipeline to real-world time series instead of a
synthetic signal.

Comparing with other symbolic regression baselines.

7. License

This project is released under the MIT License. See LICENSE for
details.

## 5. License

MIT License. See LICENSE for details.

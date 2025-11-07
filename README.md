\# KAN vs PySR for Symbolic Signal Regression



This repo contains a small research-style experiment where I use

\*\*Kolmogorov–Arnold Networks (KAN, Ziming Liu's pykan)\*\* and

\*\*PySR (symbolic regression)\*\* to recover the closed-form expression of a

non-trivial 1D signal.



The goal is to:



1\. Fit a synthetic target signal with KAN (numeric).

2\. Turn the trained KAN into a \*\*symbolic\*\* model via `auto\_symbolic`.

3\. Fit the same data (and the KAN outputs) with PySR.

4\. Compare test error, symbolic formulas, and visual fits.



> This project is intended as a portfolio / research toy project

> demonstrating how to combine modern neural architectures (KAN)

> with classical symbolic regression (PySR \& Julia).



---



\## 1. Target signal



The target is a combination of harmonic components and a chirp term:



\\\[

f(x) = \\frac{4}{\\pi}\\sin(2\\pi 10x)

&nbsp;    + \\frac{4}{3\\pi}\\sin(2\\pi 30x)

&nbsp;    + \\frac{2}{\\pi}\\sin\\bigl(2\\pi(50x + 20x^2)\\bigr), \\quad x \\in \[-0.5, 0.5]

\\]



\- Sampling rate: \*\*15 kHz\*\*, giving 15,000 samples on the interval.

\- Data split: \*\*80% train / 20% test\*\*, and all methods use the same

&nbsp; test split for a fair comparison.



---



\## 2. Methods



\### 2.1 KAN (pykan)



I use Ziming Liu's `pykan` library with:



\- Width: `\[1, 12, 3, 1]`  

&nbsp; (the penultimate layer has 3 neurons so that each neuron can capture

&nbsp; one component of the signal).

\- Symbolic branch enabled: `symbolic\_enabled=True`.

\- Progressive grid refinement:

&nbsp; `grid = 40 → 60 → 90 → 120 → 150`.

\- Each refinement step is followed by:

&nbsp; - Adam warm-up

&nbsp; - LBFGS polishing

&nbsp; - Several stability helpers (sanitizing NaNs/Infs, warm-up with

&nbsp;   smaller LR, safe refinement with rollback).



Two KAN-based models are evaluated:



1\. \*\*KAN (numeric)\*\* – the final numeric model after refinement.

2\. \*\*KAN (symbolic on copy)\*\* – a copy of the trained model processed by

&nbsp;  `model.auto\_symbolic(lib=\['sin', 'x', 'x^2'])` and lightly re-fit.



\### 2.2 PySR (symbolic regression)



I use \[PySR](https://github.com/MilesCranmer/PySR):



\- Input features: `\[x, x^2]` so that PySR can easily form

&nbsp; `sin(a x + b x^2)` type expressions.

\- Operators:

&nbsp; - Binary: `\["+", "-", "\*"]`  (no division or powers for stability)

&nbsp; - Unary: `\["sin"]`

\- Nested `sin(sin())` is explicitly forbidden via `nested\_constraints`.

\- A moderate search budget is used, for example:

&nbsp; `niterations ≈ 100`, `population\_size ≈ 200`

&nbsp; (you can adjust this in the notebook).



Two PySR models are trained:



1\. \*\*PySR (direct on data)\*\* – fits the true targets `y`.

2\. \*\*PySR (distilled from KAN)\*\* – fits the KAN outputs

&nbsp;  `y\_kan = model(x)` (teacher–student distillation),

&nbsp;  but evaluation is still against the true `y`.



---



\## 3. Repository structure



```text

.

├─ notebooks/

│   └─ KAN\_Symbolic\_TargetOnly\_FILLED.ipynb   # main experiment notebook

├─ figs/                                      # saved figures

├─ outputs/                                   # PySR output directories

├─ environment.yml

├─ README.md

├─ LICENSE

└─ .gitignore




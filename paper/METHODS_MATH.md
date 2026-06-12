# Methods And Math Outline

This file is the SI math scaffold. It should become manuscript prose after the clean benchmark runs are complete.

## Row-Level Observation Contract

Each integrated observation row is treated as a signed row observation rather than immediately as a merged reflection. A row carries a signed Miller index, intensity, standard uncertainty, geometry, frame or scan coordinate, detector metadata when available, and a split label. GROND uses this row identity because the dynamical correction is orientation-dependent.

## Excitation Error And Bloch Matrix

For each row and quadrature point, GROND reconstructs the excitation error of the target and support beams from the experimental geometry. The Bloch Hamiltonian uses the diagonal excitation errors and off-diagonal Fourier-potential coupling:

```text
H_ii = S_i
H_ij = U(h_j - h_i) / (2 k)
```

The `h_j - h_i` convention is part of the row contract and must remain fixed. GROND propagates the incident beam through thickness using the matrix exponential:

```text
psi(t) = exp(2 pi i H t) psi(0)
```

The reported row dynamical intensity is the integrated intensity over the declared excitation-error profile and thickness model.

## Born Reference

For the same row, GROND computes a Born or kinematic reference intensity using the same row geometry and support policy. This gives the row-wise correction factor:

```text
q_i = I_dyn,i / I_Born,i
```

The correction is row-specific. It is not a resolution scale or a frame-only scale.

## Fixed-q LS1 Equivalence

At macrocycle `n`, GROND freezes `q_i^n` from the current model and writes corrected observations:

```text
I_corr,i = I_obs,i / q_i^n
sigma_corr,i = sigma_obs,i / q_i^n
```

For fixed positive `q_i^n`, no clipping, and pure variance weights, least squares against corrected Born intensities is algebraically equivalent to a frozen dynamical surrogate:

```text
sum_i [(I_obs,i / q_i^n - s_i I_Born,i(theta))^2 / (sigma_i / q_i^n)^2]
=
sum_i [(I_obs,i - s_i q_i^n I_Born,i(theta))^2 / sigma_i^2]
```

This is the mathematical reason the endpoint loop is defensible. It is not exact native dynamical least squares because `q_i` is held fixed during the inner least-squares step.

## Missing Derivative

The direct row dynamical prediction can be written as:

```text
I_dyn,i(theta) = q_i(theta) B_i(theta)
```

where `B_i` is the Born intensity. Full native dynamical refinement differentiates:

```text
dI_dyn,i/dp = q_i dB_i/dp + B_i dq_i/dp
```

The LS1 inner step only sees the first term. Therefore the validation suite must test whether the omitted term matters. The paper should state this explicitly and present native finite-difference polish as the adjudicator.

## Null Controls

The key null controls are:

```text
q = 1
frozen initial q
alternate polarity q
globally shuffled log q
resolution-shell shuffled log q
shell-plus-scan shuffled log q
```

The shuffled controls preserve the distribution of correction magnitudes while breaking row identity. If correct q wins but shuffled q fails, the improvement is not explained by a generic scale or resolution correction.

## Endpoint Calibration

The publication endpoint is chosen after q and row corrections are fixed. GROND compares scale contracts and weight branches, then promotes the simplest branch that passes declared gates for R-value cost, GooF, held-out row behavior, and scale complexity. The preferred public branch uses fixed `WGHT 0 0` so the reported GooF reflects the GROND-calibrated sigmas rather than a second hidden empirical weight model.

The endpoint calibration report must also retain the lowest-R diagnostic branch and explain why it was not promoted.

## Multi-Dataset Extension

For multiple datasets, each dataset keeps its own geometry, thickness, nuisance policy, row q values, scale model, and sigma behavior. GROND owns the corrected-row merge and only then hands a merged corrected endpoint to the external least-squares program for one model update. This avoids hiding dataset weighting and merge policy inside the endpoint refiner.

## Limitations To State

GROND's default loop is a fast alternating surrogate, not exact native dynamical least squares. Corrected HKL files are model-conditioned pseudo-observations, not raw experimental observations. Map claims must be labeled as GROND-corrected or dynamically deconvolved endpoint maps. ESD and GooF interpretation depends on sigma calibration, scale complexity, and effective degrees of freedom. Native finite-difference polish and null controls are therefore part of the method, not optional decoration.

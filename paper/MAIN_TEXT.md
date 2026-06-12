# Working Title

GROND: open row-level Bloch regularization for electron-diffraction refinement endpoints

## Abstract Scaffold

Electron-diffraction intensities are often affected by dynamical scattering, but full native dynamical refinement workflows are not always easy to inspect, automate, or connect to standard small-molecule refinement endpoints. GROND addresses this by computing row-level Bloch and Born predictions from integrated observations and experimental geometry, forming a row-wise dynamical correction factor, and iterating that frozen correction through a standard one-step least-squares endpoint. The method is framed as a fast Bloch-to-Born regularization loop, not as exact native dynamical least squares.

The abstract should report the clean 370 endpoint, the LS1 null controls, the endpoint-calibration result, and the replication set once those reruns are complete. Do not fill final numbers here until the clean paper dossier exists.

## Introduction Scaffold

The introduction should start from the practical problem: standard kinematic refinement is convenient and mature, but electron-diffraction observations can carry orientation-dependent dynamical effects that are not captured by a single merged intensity. Native dynamical approaches address this directly, but there is still a need for an open, auditable, row-level workflow that starts from commonly available integration outputs and produces standard endpoint-refinement artifacts.

GROND's contribution is the interface layer and validation logic. It keeps row identity long enough to compute Bloch and Born predictions, regularizes each row through a model-conditioned correction, and then uses conventional least-squares tools for the structural update. The paper should emphasize that this is an engineering and statistical bridge: it is fast, inspectable, and compatible with existing crystallographic endpoints, while retaining row-level null controls.

The introduction should end with the frozen claim: GROND is a fast, open, row-level Bloch-to-Born regularization loop that brings dynamical scattering information into standard refinement endpoints, with explicit null controls and endpoint calibration.

## Results Scaffold

### Row-Level Bloch Regularization Workflow

This section should describe Figure 1. It should introduce the input row contract, signed row identity, excitation-error reconstruction, Bloch and Born predictions, row-wise q, corrected endpoint export, LS1 macrocycles, and endpoint calibration. It should be readable before the formal math section.

### Tyrosine 370 Flagship Validation

This section should describe Figure 2 and the clean 370 dossier. It should report the conventional comparator, GROND endpoint, updated-q cycles, q=1 control, alternate-polarity control, shuffled-q controls, native finite-difference polish, and endpoint calibration. The current anchor numbers should be replaced by the clean paper rerun before submission.

Current anchor to confirm: selected cycle 1 in N108/Q20 validation; native row wR2all 10.76%; shuffled-q controls fail strongly; promoted endpoint uses 184 scales with R1all 10.97%, wR2all 10.07%, and GooF 0.906.

### Structural Consequences And Maps

This section should show what the endpoint changes in real structural terms. It should compare conventional maps and GROND-corrected endpoint maps, H/no-H or omit behavior if convincing, map noise, peak/hole statistics, coordinate stability, and ADP behavior. If hydrogen sensitivity is not clean enough for a main claim, keep it as a controlled limitation rather than forcing it.

### Input Routes And Multi-Dataset Behavior

This section should compare XDS and DIALS as supported row sources and should show the one-crystal to multi-crystal behavior. DIALS does not need to beat XDS to be included; it needs to be documented and auditable. Multi-dataset results should make dataset weighting, completeness, and endpoint behavior explicit.

### Generality, Runtime, And Failure Modes

This section should combine histidine results, runtime behavior, and any external survey. It should include no-call categories and honest failure modes. The point is practical adoption, not universal success.

## Discussion Scaffold

The discussion should be direct about what GROND is and is not. It is a row-level Bloch regularization and endpoint-refinement workflow. It is not exact native dynamical least squares by default. The LS1 loop is a frozen-q surrogate, and the optional native finite-difference polish tests whether that approximation leaves important downhill directions.

The discussion should explain why the shuffled-q controls matter: they preserve correction magnitudes while breaking row identity, so failure of shuffled-q controls argues against the method being a generic scale correction. It should also explain why endpoint calibration matters: the lowest-R corrected-HKL branch is not automatically the publication endpoint if it uses excessive scale freedom or gives an implausible GooF.

The final paragraph should frame the release as an open workflow for reproducible electron crystallography and invite additional row-source adapters and external tests.

## Methods Scaffold

The Methods should point to `METHODS_MATH.md` for the derivation and should contain enough practical detail for reproduction:

| Method block | Required content |
|---|---|
| Data processing inputs | XDS and DIALS row sources, summation policy, gain/offset notes, supported metadata. |
| Row contract | Signed hkl, row geometry, intensity, sigma, flags, split labels. |
| Bloch calculation | Hamiltonian convention, support beams, top-up policy, N/Q, thickness model, absorption or jitter policy. |
| Regularization | q definition, corrected intensity, corrected sigma, scale policy, lambda policy. |
| Endpoint refinement | One-step least-squares endpoint, no-EXTI default unless comparator, fixed q per step. |
| Validation | Updated-q, frozen-q, q=1, alternate polarity, shuffled-q controls, native polish. |
| Endpoint calibration | Scale tournament, sigma calibration, `WGHT 0 0`, GooF gate, low-R diagnostic rejection. |
| Maps | Map generation, omit branches, labels for model-conditioned maps. |
| Statistics | R1, wR2, GooF, row-level metrics, work/free split, held-out row calibration. |

# Reviewer Objections And Planned Answers

This file is a pre-review checklist. The answers should be folded into the main text, Methods, and SI rather than saved for rebuttal.

## "The observations are model-conditioned."

Yes. GROND-corrected intensities are not raw observations. They are row-level dynamically regularized pseudo-observations conditioned on the current model, geometry, thickness model, support beams, and scale policy. The paper should say this directly. The defense is that the correction is frozen during each least-squares endpoint step, null controls test row identity, and the primary validation includes native row-level metrics, not only corrected-HKL endpoint R values.

## "This is not exact native dynamical refinement."

Correct. The default loop is a lagged dynamical surrogate. It freezes `q`, performs a standard endpoint least-squares update, then recomputes `q`. It omits `B dq/dp` inside the inner step. The paper should call this row-level Bloch regularization or a frozen-q LS1 surrogate, and it should reserve exact native refinement language for the matched finite-difference polish.

## "The improvement is just flexible scaling."

The response is the shuffled-q validation. Global, resolution-shell, and shell-plus-scan shuffles preserve correction magnitudes while destroying row identity. On the current 370 N108/Q20 anchor, correct q gives native row wR2all 10.76%, while shuffled-q controls give 39.93%, 37.43%, and 26.57%. That is the central evidence that row identity matters.

## "There are too many scale parameters."

The publication endpoint is selected by an endpoint calibration gate, not by lowest R. On the current 370 anchor, the lowest-R diagnostic branch uses 708 scale parameters and GooF 0.557 and is rejected. The promoted endpoint uses 184 scale parameters, fixed `WGHT 0 0`, R1all 10.97%, wR2all 10.07%, and GooF 0.906. The SI should show the scale-complexity curve.

## "GooF is still complicated."

Yes. SHELXL GooF assumes an ordinary merged reflection table and a known parameter count. GROND adds row corrections, row scales, sigma calibration, and endpoint selection before the table reaches the refiner. The solution is not to hide GooF. The solution is to report the conventional endpoint GooF for the promoted fixed-weight endpoint, show row-level held-out calibration, and keep a GROND diagnostic goodness-of-fit in the audit files.

## "DIALS is worse than XDS in some runs."

That is allowed. Paper 1 should claim that DIALS is supported through an explicit and auditable row contract, not that DIALS always wins. XDS can remain the production-quality route for the strongest current datasets while DIALS is presented as an open route that needs careful sigma and row-contract handling.

## "Why not just use EXTI?"

EXTI is a useful conventional comparator but it is not a row-level dynamical correction. GROND should be compared against kinematic and kinematic-plus-EXTI endpoints where appropriate, but the method's central evidence is row-wise Bloch/Born correction and null-control behavior.

## "Maps are biased by the model-conditioned correction."

GROND endpoint maps should be labeled as GROND-corrected or dynamically regularized endpoint maps. H/no-H and omit claims need omitted-model branches and row-level controls, not only attractive difference density.

## "Existing native dynamical workflows already exist."

The paper should acknowledge established native dynamical approaches and simulation tools. GROND's contribution is different: an open, row-level Bloch regularization workflow that plugs into standard endpoint refinement while preserving null controls and audit files.

## "The method may overfit the 370 example."

The answer is replication and no-call reporting. The paper needs 366, 366+370, histidine, DIALS, and an external survey or appendix. Failures should be reported as failure modes rather than tuned away.

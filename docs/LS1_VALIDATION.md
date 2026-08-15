# LS1 Validation

GROND's fastest structure-update path is a Bloch/LS1 feedback loop. It freezes the row-wise dynamical correction from the current model, writes a corrected no-EXTI reflection file, runs one conventional least-squares model update, then recomputes the Bloch correction from the updated model. This is a useful alternating dynamical surrogate, but it is not identical to native dynamical least squares because the inner LS1 step does not include the derivative of the Bloch/Born ratio with respect to every structural parameter.

The validation command checks that distinction directly for a pinned nuisance contract. It starts from an existing `native_row_regularize.py` summary, so thickness, Sg jitter, absorption, frame scales, detector scales, sigma terms, basis settings, and the signed HKL transform are not silently changed while the LS1 loop is being judged.

```bash
GROND validate-ls1 \
  --source-summary /path/to/final/native/summary.json \
  --start-res /path/to/model.res \
  --out-dir /path/to/ls1_validation \
  --cycles 5 \
  --n-basis 30 \
  --n-points 20 \
  --native-polish
```

For a final publication adjudication, use the same basis and quadrature as the promoted endpoint, for example `--n-basis 108 --n-points 20` if the final support scan selected 108 beams. The default `N=30/Q=20` is meant as a compact diagnostic that can be run routinely without turning every validation into a full endpoint rerun.

The command writes `ls1_validation_summary.json` and `LS1_VALIDATION_REPORT.md`. The report includes an updated-q LS1 table, a frozen-q fixed corrected-HKL control, a q=1 Born control, an alternate-hand Bloch control when requested, three shuffled-q controls, and an optional native finite-difference polish. The native polish now uses the same accepted production Bloch physics as the LS1 row regularizer, including Sg jitter and absorption terms, so it is a fairer adjudicator than the older diagnostic path.

`--endpoint-ls-cycles N` is the deliberate test of taking more than one internal SHELXL LS/CGLS cycle while a single Bloch correction is frozen. `--shelxl-iterations N` instead repeats external one-cycle SHELXL calls against that same corrected HKL. Neither option adds a Bloch calculation. More inner cycles are acceptable only if the next updated-q macrocycle improves the native row objective and the model remains free of NPD atoms and split warnings. A lower corrected-HKL endpoint residual is not sufficient evidence because it can improve while the frozen correction becomes stale.

There is one bookkeeping point worth making explicit. A row-regularization summary reports native row metrics for the model used to compute that cycle's `q`. In an updated-q loop, the next cycle's q recomputation is therefore the native row-level check of the previous cycle's LS1 endpoint. `validate-ls1` aligns this in the report: cycle N endpoint-native metrics come from cycle N+1 where available, while the SHELXL endpoint statistics still come from cycle N itself. The last requested cycle has no post-step native recheck unless another cycle is run, so best-cycle promotion should favor the earliest cycle whose endpoint-native check is within tolerance.

The native row-weight contract is also fixed across feedback cycles. In particular, changing the SHELXL endpoint to `WGHT 0 0` must not silently replace the native row weights on the next Bloch calculation. Unweighted `Rall` and correlation are unaffected by that bookkeeping error, but weighted row factors from two different weight definitions are not comparable.

The controls should be read this way. The real Bloch `dyn` row metric should beat the matched Born row metric, which is the native-row q=1 diagnostic. The q=1/Born endpoint should also be visibly worse when exported to the conventional least-squares step. Alternate-hand and shuffled-q controls should be worse on native row-level metrics because they deliberately break the physically assigned row correction. If frozen-q is essentially identical to updated-q, then the macrocycle update is not doing much for that dataset. If the native polish improves native `wR2all` by more than roughly 0.5 to 1.0 percentage point, or causes chemically meaningful coordinate or ADP shifts, LS1 should be treated as a preconditioner rather than the final endpoint. If the native polish does very little, the LS1 endpoint is a defensible production endpoint for that case.

The shuffled-q controls are row-space permutations of `log(q)`, not scalar rescaling. `shuffle_global` permutes `log(Idyn/Iborn)` over all valid rows. `shuffle_shell` permutes within resolution bins. `shuffle_shell_scan` permutes within combined resolution and scan/frame groups, which preserves much of the local dynamical-strength distribution while destroying the identity of which row receives which correction. After the permutation, GROND writes corrected intensities and fits the ordinary frame/detector scales exactly as it would for a real q field. Use `--q-control-seed` for deterministic repeats and `--q-control-shells` to change the resolution-bin count.

The defensible method statement is therefore:

> GROND LS1 is a fast alternating row-level dynamical refinement surrogate. It repeatedly freezes the Bloch/Born correction `q`, refines a conventional structural model against corrected intensities, recomputes `q`, and validates the endpoint against the native row-level dynamical target. It is not exact native dynamical refinement because it omits the `B dq/dp` term inside each LS step.

That statement is the one the validation command is designed to test.

# Limitations

GROND is alpha research software. It is useful for our current electron-diffraction workflows, but it should not yet be treated as a turnkey replacement for mature commercial or long-established crystallographic packages.

The package is intentionally transparent: it writes row tables, decision files, logs, endpoint summaries, and publication-style reports so users can audit what happened. That transparency does not mean every automatic choice is optimal for every dataset. Users should inspect the output, especially when the data are weak, incomplete, misindexed, poorly integrated, unusually thick, strongly damaged, or outside the small-molecule and peptide-sized cases used during development.

## Scientific Scope

GROND is currently aimed at small-molecule and peptide-sized electron-diffraction datasets that start from XDS- or DIALS-style integrated intensities plus a solved structure model. It is not yet a general macromolecular modeling package, a raw detector processing suite, a replacement for XDS or DIALS integration, or a guaranteed publication endpoint without crystallographic judgment.

The DIALS bridge is newer than the XDS route. It preserves signed rows and scan-varying orientation from `integrated.refl`/`integrated.expt`, drops severe DIALS integration-quality flags by default, and applies an auditable detector-gain sigma policy for electron-diffraction rows. It should still be cross-checked on standards before replacing a known-good XDS workflow in production. In particular, inspect `dials_bridge_summary.json` for row counts, partiality behavior, sigma policy, dropped flags, and excitation-error validation.

The automatic workflow includes physically motivated checks for transform alignment, polarity, thickness, nuisance terms, scale compression, resolution limits, and endpoint reporting. These checks are heuristics around explicit models, not proofs that the final result is uniquely correct. A good GROND result should still agree with chemical sense, geometry, maps, independent data quality indicators, and any appropriate control analyses.

## Numerical And Modeling Limits

The production Bloch calculations use the portable NumPy/SciPy CPU path. Experimental accelerator backends may exist, but they must pass parity checks before being trusted for production. Runtime can still be substantial for high beam counts, high quadrature, many datasets, or large structures.

The current model is still an approximation. Important residuals may come from integration errors, background modeling, detector effects, absorption, radiation damage, geometry, partiality, thickness variation, bad scaling, twinning or indexing ambiguity, disorder, or a structural model that needs chemical intervention. GROND can diagnose many of these effects, but it cannot make bad observations physically correct by adding more beams.

## External Tools

Native row-level Bloch calculations are Python-first. Corrected-HKL endpoint polishing and map production can call external programs such as SHELXL and `shelx2map` through documented command-line interfaces. Those programs are not bundled with GROND and are not part of this license. Use `GROND init-config` and `GROND doctor` to make these dependencies explicit on each machine.

## Validation Status

GROND should be validated on local standards before being trusted on new chemistry, new microscopes, new detectors, new integration strategies, or new resolution regimes. The repository documentation and run outputs should report both favorable and unfavorable metrics. A result that improves one residual while degrading maps, ADPs, held-out statistics, chemistry, or scale-parameter discipline should be treated as diagnostic rather than automatically promotable.

## No Endorsement

GROND is an independent research software project and is not endorsed by, certified by, or affiliated with any external crystallographic software project, company, or maintainer group. Comparisons to external packages are scientific context and interoperability notes only.

## User Responsibility

Users are responsible for checking license compatibility of any external programs they call, verifying that their input data may be processed and shared, preserving enough provenance to reproduce results, and reporting limitations honestly in reports and publications. GROND is provided without warranty under the MIT License.

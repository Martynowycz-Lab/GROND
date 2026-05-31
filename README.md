# GROND

**Global Regularization Of Non-kinematical Diffraction**

*"GROND! GROND!"* — J. R. R. Tolkien, *The Return of the King*

GROND helps a crystallography lab turn an electron-diffraction integration and a solved structure into a reproducible row-level Bloch-regularization report. A non-expert user can point it at an XDS directory, a structure model, and an output folder; GROND then checks whether the data are usable, finds the safest indexing alignment, estimates thickness and polarity, simulates row-level Bloch and Born intensities, uses their ratio to regularize the observed intensities, writes no-EXTI corrected reflection files for endpoint auditing, makes maps when external map tools are available, and records every decision in machine-readable files. The practical entry point is XDS-style rotation integration and scaling, conventional SHELX-style structure-solution files, and CIF/RES model descriptions [1-8].

Use GROND when ordinary kinematic treatment is not explaining the integrated intensities well enough and you need an auditable row-level multiple-scattering model rather than a black-box correction. GROND does not claim that the scattering theory itself is being adjusted by a proprietary optimizer. It repeatedly simulates Bloch intensities from the current model, regularizes the measured rows through the rowwise `Idyn/Iborn` ratio, and, when model coordinates or displacement parameters are updated, does that as a conventional least-squares model update against merged regularized reflections or through the explicitly labeled native-gradient diagnostic path [14-30,41].

## What GROND Does

GROND starts from summation-style integration rows, not from a merged reflection table. The usual XDS input is an `INTEGRATE.HKL` produced with `PROFILE_FITTING=FALSE`, where each row carries a signed `h,k,l`, `IOBS`, `SIGMA`, calculated detector position, calculated frame position, and XDS spot-quality fields such as `RLP`, `PEAK`, and `CORR`. `XPARM.XDS` or `GXPARM.XDS` supplies the beam, rotation axis, cell, orientation matrix, detector geometry, wavelength, and oscillation range. The solved `RES` or `CIF` supplies the starting atomic model and therefore the electron structure factors used by the Bloch calculation. That solved structure can come from any sensible structure-solution route, including profile-fitted XDS/XSCALE/XDSCONV/SHELXT/SHELXL data. In fact, the recommended XDS workflow is to solve the model and run any desired conventional least-squares update from the best profile-fitted intensities, then rerun XDS integration in a copied directory with `PROFILE_FITTING=FALSE` and `MINPK=90.0` for GROND. GROND can technically read profile-fitted `INTEGRATE.HKL` rows, but those rows have already absorbed profile/partiality assumptions that make rowwise `Idyn/Iborn` modelling less faithful; in our testing the Bloch regularization output is usually worse even if the profile-fitted reduction has better-looking conventional statistics or reaches a slightly higher nominal resolution [1-8,21-29].

DIALS input is supported through `integrated.refl` plus `integrated.expt`. GROND reads the reflection table rows, keeps the exact signed `miller_index`, uses summation intensities by default, maps DIALS partiality onto the GROND `PEAK` quality field, and writes a sidecar of scan-varying DIALS A matrices so excitation errors are reconstructed from the same per-frame orientation model used by DIALS. The DIALS `zeta` column is not treated as `Sg`; GROND recomputes `Sg = (k^2 - |s0 + g|^2)/(2k)` from the experiment geometry and validates the converted rows during import. For electron-diffraction CBF data, the DIALS bridge also audits detector-gain sigma scaling because DIALS stores detector gain separately from the summation variance parameter path; the decision and the before/after `I/sigma` distribution are written to `dials_bridge_summary.json`. Production DIALS imports drop `dont_integrate` and severe bad-flag rows by default, with keep flags available for diagnostics. This avoids reducing DIALS to a static exported HKL file when the scan-varying geometry and row-level quality model are needed for Bloch rocking curves [4,39,40]. GROND keeps summation observations alive as rows for as long as possible because multiple scattering, partiality, geometry, dose, and per-frame scale behavior are row-level effects that can be hidden by premature merging or profile-model corrections.

For DIALS data, the best upstream habit is to put the detector gain on the DIALS detector model at import time when it is known, then let ordinary rotation integration use that model:

```bash
dials.import /path/to/images_*.cbf panel.gain=<detector_gain>
dials.integrate refined.expt refined.refl \
  integration.profile.fitting=False \
  prediction.d_min=0.70
```

GROND can detect and compensate for grossly missing detector-gain sigma scaling in the bridge, but it is cleaner to make the DIALS experiment model correct upstream. For rotation data, `integration.summation.detector_gain` is not the preferred path; it can also make later audits confusing because the detector model and summation variance path no longer tell the same story. On the 370 DIALS test set, the detector model reported gain 204 and the raw reflection-table sigmas were already plausible. Forcing an additional detector-gain multiplier after that double-counts the gain, makes the data look artificially weak, and can make significance-based resolution gates far too conservative.

For each row, GROND converts the XDS frame coordinate into a goniometer angle and reconstructs the reciprocal vector for the signed reflection at that angle. In the current XDS bridge, the working geometry is

```text
g_h(alpha) = R(axis, alpha) A(alpha) h
Sg(alpha) = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

Here `s0` is the incident wave vector, `k = |s0|`, `h` is the signed Miller-index vector, and `Sg` is the signed excitation error in inverse Angstroms. GROND solves for the nearby Bragg angle where the target reflection has `Sg = 0`, builds a finite rocking window around that condition, samples the target `Sg` values across the window, and converts each sampled `Sg` value back into an angle. The default rocking quadrature samples that window uniformly; trapezoid and Gauss-Legendre quadrature are available as diagnostics. The default accumulation is a detector-count-style sum over sub-exposures rather than a normalized average, which is the convention we want for future raw-CBF forward modelling. Because the dynamic and Born predictions use the same quadrature weights, the ratio `Idyn/Iborn` is unchanged by the sum-versus-mean normalization. The quadrature implementation is GROND's own NumPy/SciPy path, with public cctbx/scitbx and related public crystallographic software cited as adjacent open infrastructure rather than vendored code [11-13,35,36].

At each sampled rocking angle, GROND builds an N-beam Bloch structure matrix. The direct beam is kept as beam 0, the signed target reflection is beam 1, and the remaining coupled beams are chosen from the row's reciprocal-space support up to the selected basis limit. The diagonal terms are the excitation errors of those beams. The off-diagonal terms are electron Fourier-potential couplings, using the production convention `U(h_j - h_i) / (2k)`. The Fourier potentials come from the current atomic model, relativistic electron scattering factors, the unit-cell volume, displacement factors, and the selected absorption or non-Hermitian terms when those have passed the automatic gates. In the ordinary elastic path the matrix is Hermitian and uses a Hermitian eigensolver; when accepted absorption-like terms make the matrix non-Hermitian, GROND switches to the general complex eigensolver and records that choice in the run metadata [14-19,21-30,41].

The row prediction is the target-beam intensity after propagation through the selected crystal thickness model. GROND diagonalizes the Bloch matrix, propagates the incident direct beam through the thickness quadrature with `exp(2*pi*i*lambda*t)`, projects the result back onto the target beam, squares the target amplitude, and then sums those intensities over the rocking-profile quadrature. It also computes a matched Born or kinematic reference from the same row, model, thickness, and profile support. The central dynamical correction is therefore not an arbitrary scale factor: it is a rowwise physical ratio between the predicted dynamical intensity and the matched Born reference, with the observed `IOBS` and `SIGMA` from XDS kept as the measured data [14-30].

GROND then asks which physical and nuisance terms are justified by the data. It checks indexing transforms, the alternate coordinate-polarity diagnostic, thickness, thickness gradient, frame offset, resolution envelopes, absorption-like terms, detector-position scales, frame-scale manifolds, sigma inflation, and resolution limits with held-out and all-row gates. Accepted terms must improve row-space behavior and endpoint behavior without relying only on strong reflections. Rejected terms stay visible in the diagnostics but are not promoted into the publication endpoint. The final endpoint writes raw-row prediction tables, corrected no-EXTI reflection files, GROND row/MR raw and merged residual statistics, SHELXL-style endpoint statistics when SHELXL is configured, hydrogen-omit and difference-map products when map tools are available, and a complete command/progress ledger so the run can be resumed and audited [7,8,21-29,31-34].

GROND is intentionally Python-first. The production package uses NumPy, SciPy, Gemmi, and a small set of command-line engine scripts that are included in this repository. Native row-level Bloch modeling can run without external crystallographic programs, but the full default endpoint path requires configured or discoverable SHELXL for corrected-HKL polishing/statistics and `shelx2map` for map generation. Endpoint EXTI correction is treated as a diagnostic comparison only; the promoted final structure path keeps EXTI at zero and moves resolution-dependent corrections into the raw-row dynamical target [7,8,35-38].

GROND is independently written open-source research software. It was built from public scientific literature, public documentation, public file formats, public/example data, public crystallographic software papers and repositories, and our own validation work; no proprietary source code, confidential implementation details, decrypted modules, or vendor-private algorithm descriptions were used. See [PROVENANCE.md](PROVENANCE.md) for the clean-source boundary and [LIMITATIONS.md](LIMITATIONS.md) for the current scientific and software limits. GROND is released under the [MIT License](LICENSE), which is a very permissive open-source license that allows academic, nonprofit, and commercial use while preserving the copyright and license notice [9-13].

The default `GROND auto` command is the full automatic path, not a toy wrapper. It performs preflight, baseline-HKL metadata prefiltering of transform candidates, a two-stage transform scout for any surviving ambiguous transforms, polarity scouts, thickness and nuisance decisions, a continuous phase-only thickness polish, raw-row resolution-envelope tests, absorption and jitter gates, detector-position scale screening, adaptive final basis selection, no-EXTI corrected-HKL endpoint scans over lambda, iterative SHELXL endpoint polishing with active `WGHT` promotion, map generation when tools are available, row-space forensic audits, post-final adaptive frame-scale compression, multi-dataset inclusion/downweight decisions, held-out resolution-limit promotion, a late global Sg rocking-window scout, a publication endpoint chooser, and publication-ready tables. Transform alignment starts by comparing the raw XDS rows with the solved/XSCALE/SHELX HKL file when `--baseline-hkl` is supplied; only if that intensity-overlap signal is absent or ambiguous does it spend Bloch calculations on the broader transform list. The thickness grid is now only a safe bracket: after the best coarse value is found, GROND reuses cached Bloch eigensystems and polishes a continuous crystal thickness and thickness-gradient value with scalar, guarded-grid, and bounded Powell searches, promoting the non-round value only when all-row and held-out gates agree. Final Bloch polish keeps Q=20 and chooses N from the observed rowwise support requirement as `max_observed_basis + 10`, limited to 500 by default, instead of blindly using a fixed 128 or 500 beams. When SHELXL is available and a single dataset is being updated in full mode, auto uses that same high-N/Q20 support choice inside a five-cycle fixed-nuisance feedback loop: `Bloch N=max_support+10, Q=20 -> one SHELXL least-squares update -> repeat`. This is the default because benchmark data reached the same row-space endpoint as slower finite-difference row-gradient diagnostics in minutes rather than roughly an hour; `--structure-update native_gradient` restores the older diagnostic gradient block, and multi-dataset shared-structure jobs still use that diagnostic path until the combined feedback loop is implemented. The optional `--compute-backend scipy_bethe` path uses a promoted strong/weak-beam Bethe-Schur reduction suggested by Takanori Nakane and independently implemented in GROND; it keeps near-resonant weak beams in the solved block, folds the remaining weak block into an effective Hamiltonian, and has matched the dense SciPy path on the 370 benchmark while cutting the N108/Q20 feedback loop runtime substantially. Takanori's fixed-size matrix suggestion is also exposed through opt-in padded JAX backends such as `jax_cpu_padded` and `jax_cuda_padded`; these bucket matrix sizes to reduce compile churn, but the SciPy backend remains the production default until an accelerator beats it end to end. The support scan is never allowed to be shallower than the selected data resolution, so a 0.70 A endpoint also gets 0.70 A support beams unless the user deliberately truncates the data. In multi-crystal jobs, good datasets carry full raw-row weight, marginal datasets get sigma-inflated weights, and weak datasets that add otherwise missing reflections can be kept endpoint-only for completeness. The combined corrected-HKL endpoint also fits one auditable overlap scale per crystal after the dynamical correction, which prevents arbitrary raw integration scales from contaminating the merged endpoint. Resolution limits are promoted per dataset and again for the combined endpoint using shell signal, split-half shell stability, and GROND row/MR `MR`/`MwR` behavior from the final row tables. After those gates, the Sg scout tests a few global fixed rocking widths with the final nuisance model and held-out rows; it does not pick a separate width per reflection from the observed intensity. The endpoint chooser reports the multi-dataset physical endpoint separately from the best single-crystal diagnostic endpoint, the best diagnostic correction, and the specific reasons non-promoted corrections were kept out of the publication path. Nonzero corrected-HKL endpoint-B scans are opt-in diagnostics only; they can improve residuals by reshaping the data but can also inflate structural ADPs and degrade maps, so the default final endpoint keeps endpoint-B at zero. The intent is that a routine user starts from the simple command below and only adds flags when they have a specific scientific reason [1-3,7,8,21-34].

## A Plain-English Bloch-Wave Picture

The simplest crystallographic treatment uses the Born or kinematic approximation. In that picture, the incident electron beam enters the crystal, scatters once from the electrostatic potential, and the measured intensity for reflection `h` is proportional to `|F(h)|^2` after ordinary scale, geometry, and error-model corrections. That approximation is extremely useful, but electrons scatter strongly. In a nanocrystal, intensity can flow from the direct beam into one reflection, then from that reflection into another reflection, then back into the direct beam or into the target reflection. These multiple-scattering paths are the physical effect that GROND simulates row by row [14-20].

A Bloch-wave calculation replaces the single-scattering picture with a coupled-beam propagation problem. Imagine choosing a small list of reciprocal-lattice points that could exchange intensity while the crystal is at one rocking angle. GROND always includes the direct beam as beam 0, includes the target reflection for the row as beam 1, and then adds nearby coupled reflections as beams 2 through `N-1`. For a target row with signed Miller index `h`, the first two beam indices are usually:

```text
beam 0: h_0 = (0, 0, 0)       direct transmitted beam
beam 1: h_1 = h               target reflection being predicted
beam 2: h_2 = nearby hkl      another reflection close enough to couple
...
beam N-1: h_N-1
```

For that list of beams, GROND builds a complex matrix whose diagonal entries say how far each beam is from its exact Bragg condition, and whose off-diagonal entries say how strongly beam `i` and beam `j` exchange amplitude through the crystal potential. A three-beam version looks like this:

```text
              beam 0                  beam 1                  beam 2
        -----------------------------------------------------------------
beam 0 |  S(h_0)              U(h_1 - h_0)/(2k)      U(h_2 - h_0)/(2k)
beam 1 |  U(h_0 - h_1)/(2k)   S(h_1)                 U(h_2 - h_1)/(2k)
beam 2 |  U(h_0 - h_2)/(2k)   U(h_1 - h_2)/(2k)      S(h_2)
```

In the full `N`-beam case, the same pattern continues. `S(h_i)` is the excitation error for beam `i`, `k` is the length of the incident wave vector, and `U(h_j - h_i)` is the Fourier component of the electron potential that couples those two beams. `U` is calculated from the current structure model, the unit cell, electron scattering factors, displacement parameters, and any accepted absorption-like terms. The `midgley_like` absorption branch is a deliberately small optical-potential diagnostic inspired by recent absorption analysis in electron diffraction: it combines uniform imaginary diagonal damping with a same-phase imaginary off-diagonal fraction and is promoted only if the automatic gates support it [41]. With only elastic scattering the matrix is Hermitian, meaning the lower-left entries are the complex conjugates of the upper-right entries. If an accepted absorption model makes the scattering non-Hermitian, GROND switches to the more general complex eigensolver and reports that choice.

The excitation error `Sg` is central because it tells GROND whether a reciprocal-lattice point is close enough to the Ewald sphere to participate. At a goniometer angle `alpha`, GROND computes the reciprocal vector for a signed reflection:

```text
g_h(alpha) = R(axis, alpha) A(alpha) h
Sg(alpha) = (k^2 - |s0 + g_h(alpha)|^2) / (2k)
```

When `Sg = 0`, the reflection is exactly on the Ewald sphere. Small `|Sg|` means the reflection is strongly excited. Large `|Sg|` means it is geometrically far from diffraction condition and usually contributes little. An `Sg` cutoff is therefore a physics-and-speed gate: it limits the coupled-beam list to reflections near enough to matter. Too tight a cutoff misses real multiple scattering; too loose a cutoff spends time diagonalizing large matrices full of beams that do not move the prediction.

The number of beams, often written `N`, is the size of that matrix. `N=20` means a 20 by 20 complex matrix for each sampled orientation and thickness point. `N=500` means a 500 by 500 matrix, which is much more expensive but can be necessary when a high-resolution row is embedded in a dense zone of coupled reflections. GROND's fast modes use small `N` values such as 10 or 20. The full automatic endpoint scans the actual rows, finds how many beams are needed by the observed geometry, and then uses `max_observed_basis + 10` up to a default limit of 500. The extra `+10` is the basis "top-up": it keeps the final calculation from sitting exactly on a hard cutoff where one marginal coupled reflection can come in or out just because of a tiny geometry or model change.

The other support limit is reciprocal-space resolution. If data are carried to 0.70 A, the coupled-beam support must not silently stop at 0.80 A, because high-resolution beams outside the data target can still perturb lower-resolution intensities through dynamical coupling. GROND therefore keeps the final support top-up at least as deep as the selected data resolution unless the user deliberately overrides it. This is one reason GROND records both the data resolution and the Bloch support resolution in the endpoint reports.

Once the matrix is built, GROND solves the coupled propagation problem through the crystal thickness. Diagonalizing the matrix gives eigenmodes of the electron wavefield inside the crystal. GROND starts with amplitude in the direct beam, propagates those eigenmodes through a thickness `t`, projects the wavefield back onto the target beam, and squares the target amplitude to obtain a predicted dynamical intensity. The same row is also evaluated with a matched Born reference, which is the corresponding single-scattering prediction over the same rocking window. The useful correction is the ratio:

```text
dynamical correction for one row = Idyn(row) / Iborn(row)
```

That ratio is powerful because it separates the measured integration row from the model-predicted multiple-scattering effect. A row whose observed intensity is high because the target reflection is strongly fed by other beams should have `Idyn > Iborn`. A row depleted by multiple scattering can have `Idyn < Iborn`. GROND uses those rowwise ratios while still preserving the observed `IOBS` and `SIGMA` from XDS or DIALS.

Rocking-curve quadrature is how GROND processes the fact that an integrated reflection is not measured at one infinitely thin angle. The detector row corresponds to a small angular/frame range. GROND samples that range at `Q` points, builds and propagates the Bloch matrix at each point, and sums the predicted sub-exposure intensities. `Q=5` or `Q=10` is useful for scouting; the full endpoint normally uses `Q=20`. GROND uses a sum convention rather than a normalized average because detector counts are additive, and that convention will matter when the package grows a raw-CBF forward model. For the current correction ratio, the dynamic and Born paths use matched weights, so sum versus mean does not change `Idyn/Iborn`.

Crystal thickness matters because multiple scattering accumulates as the wave propagates. A very thin crystal behaves more kinematically; a thicker crystal has more time to exchange amplitude among beams. GROND first brackets thickness with a grid, then polishes continuous thickness and thickness-gradient values using cached Bloch calculations and bounded scalar searches. The promoted value is not forced to a round number such as 500 A or 1000 A; the round grid is only a safe starting bracket.

EXTI is different. In SHELXL-style kinematic least squares, `EXTI` is an empirical extinction parameter that can absorb some systematic intensity loss in strong reflections. It can be useful as a conventional diagnostic and sometimes gives better-looking kinematic residuals, but it is not a row-level multiple-scattering model. It does not know the frame angle, the coupled beam list, the crystal thickness, the polarity-dependent scattering paths, or which observations of the same `hkl` were measured in different geometries. GROND may write no-EXTI corrected reflection files for endpoint comparison, but the promoted physical endpoint keeps EXTI at zero and asks the Bloch simulation, thickness, absorption gates, geometry, and compact scale manifold to explain the data instead.

The reason to use GROND at all is therefore not simply to lower a table value. The point is to replace an empirical correction with an auditable forward model of how electrons move through a finite crystal. When the model is working, it should improve raw-row residuals, merged residuals, coordinate-polarity-alt residual discrimination, hydrogen/difference maps, and forensic consistency without requiring an explosion of scale parameters or a resolution-dependent data reshaping trick. When a correction lowers one residual while making maps, ADPs, held-out shells, or parameter counts worse, GROND keeps it in the diagnostic record but does not promote it by default.

## Quick Install

For a fresh user, the safest route is Python 3.11 or 3.12 in a clean environment:

```bash
git clone https://github.com/Martynowycz-Lab/GROND.git
cd GROND
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
GROND init-config --out grond.config.json
GROND doctor --probe-bloch
```

Developers should install with `python -m pip install -e ".[dev]"`, then run `GROND validate` and `pytest`.

A conda/mamba route is also provided:

```bash
mamba env create -f environment.yml
mamba activate grond
GROND doctor --probe-bloch
```

## One-Command Use

```bash
GROND auto \
  --config /path/to/grond.config.json \
  --xds-dir /path/to/xds_directory \
  --structure /path/to/solved_model.res \
  --baseline-hkl /path/to/solved_model.hkl \
  --out-dir grond_out
```

For DIALS input, use the matching reflection table and experiment list:

```bash
GROND auto \
  --config /path/to/grond.config.json \
  --dials-refl /path/to/integrated.refl \
  --dials-expt /path/to/integrated.expt \
  --dials-python /path/to/dials/python \
  --structure /path/to/solved_model.res \
  --baseline-hkl /path/to/solved_model.hkl \
  --out-dir grond_dials_out
```

The config file is optional when `shelxl`, `shelx2map`, and other external programs are already on `PATH`, but it is the recommended route on shared servers. `GROND doctor --config /path/to/grond.config.json --probe-bloch` reports exactly which executable will be used. Full endpoint workflows stop early if SHELXL is missing, and map-making workflows stop early if `shelx2map` is missing; use `--native-only`, `--no-maps`, or `--allow-missing-tools` only when that incomplete output is intentional.

For a quick triage pass, add `--fast`. Fast mode uses a smaller basis and quadrature, disables detector scale grids and expensive non-Hermitian-style terms, and is meant to answer whether the indexing, polarity, thickness, and initial R factors are plausible.

For a faster automatic regularization that still exercises the decision tree, use `--full-fast`. This profile keeps the transform, polarity, thickness, nuisance-feature, lightweight structure-update, scale-compression, forensic, and endpoint-reporting machinery on, but pins the Bloch work to N=24/Q=10 and skips the expensive final support scan, endpoint feedback, observation-polish promotion, and multi-iteration endpoint polish. Its structure-update budget is chosen from the model size, so small structures get a fuller ADP probe and larger peptide-like models get a lighter one unless you override `--full-fast-structure-rows` or `--full-fast-structure-max-nfev`. Full and full-fast structure blocks use a data-RMS plateau stop by default, so a stale xyz or ADP loop should terminate instead of consuming days after the row objective stops improving. It is the right first unattended command when you want a real updated structure quickly and can accept that the final R factors are not the high-N/Q20 publication polish yet.

On multi-core machines, transform scouting is parallel at the outer decision level. GROND first tries to eliminate the scout with baseline-HKL metadata; if candidates remain, `--transform-scout-jobs 0` automatically queues transform candidates across datasets using the core budget implied by `--workers`, while each scout subprocess defaults to `--transform-scout-workers-per-job 1`. For a 100-core server, `--workers 85` therefore means up to 85 one-core transform scouts instead of one transform at a time or 85 workers multiplied by every transform.

## What You Get

The output directory contains the run configuration, the prepared inputs, per-stage logs, progress/resume state, row-level predictions, corrected reflection files, endpoint statistics, maps when available, dataset-inclusion reports, resolution-limit reports, and forensic reports. Endpoint summaries include the active SHELXL weighting history so high `wR2all` values can be traced to the observation model rather than stale model-update weights. If a run is interrupted, launch the same command with the same `--out-dir`; GROND resumes completed stages recorded in `grond_progress.json`.

## Package Layout

```text
src/grond/io/          input parsing, dataset prep, and structure processing
src/grond/bloch/       public Bloch-kernel contracts and numerical helpers
src/grond/regularize/      regularization workflow namespace
src/grond/maps/        map-generation workflow namespace
src/grond/forensics/   row-space diagnostics and endpoint promotion
src/grond/workflows/   automatic decisions and high-level policies
src/grond/legacy/      vendored command-line engine used by the CLI
```

Start with [docs/INSTALL.md](docs/INSTALL.md), then [docs/QUICKSTART.md](docs/QUICKSTART.md). XDS-specific input preparation is in [docs/XDS_WORKFLOW.md](docs/XDS_WORKFLOW.md), and DIALS-specific input preparation is in [docs/DIALS_WORKFLOW.md](docs/DIALS_WORKFLOW.md). The source-tree and Bloch-kernel reading map is in [docs/CODE_MAP.md](docs/CODE_MAP.md). Output interpretation is in [docs/OUTPUTS.md](docs/OUTPUTS.md), input/output examples are in [docs/FILE_FORMATS.md](docs/FILE_FORMATS.md), common failure modes are in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md), workflow figures are in [docs/figures](docs/figures/README.md), the clean-source statement is in [PROVENANCE.md](PROVENANCE.md), current caveats are in [LIMITATIONS.md](LIMITATIONS.md), contribution rules are in [CONTRIBUTING.md](CONTRIBUTING.md), and the name note is in [docs/NAME.md](docs/NAME.md).

## References

1. Kabsch, W. "XDS." *Acta Crystallographica Section D* **66**, 125-132 (2010). https://doi.org/10.1107/S0907444909047337
2. Kabsch, W. "Integration, scaling, space-group assignment and post-regularization." *Acta Crystallographica Section D* **66**, 133-144 (2010). https://doi.org/10.1107/S0907444909047374
3. Kabsch, W. *XDS Program Package* documentation and distribution site. https://xds.mr.mpg.de/
4. Winter, G. et al. "DIALS: implementation and evaluation of a new integration package." *Acta Crystallographica Section D* **74**, 85-97 (2018). https://doi.org/10.1107/S2059798317017235
5. Hall, S. R., Allen, F. H. & Brown, I. D. "The crystallographic information file (CIF): a new standard archive file for crystallography." *Acta Crystallographica Section A* **47**, 655-685 (1991). https://doi.org/10.1107/S010876739101067X
6. International Union of Crystallography. *CBFlib: Crystallographic Binary File and imgCIF support library*. https://www.iucr.org/resources/cif/software/cbflib
7. Sheldrick, G. M. "SHELXT - Integrated space-group and crystal-structure determination." *Acta Crystallographica Section A* **71**, 3-8 (2015). https://doi.org/10.1107/S2053273314026370
8. Sheldrick, G. M. SHELXL least-squares model-update paper. *Acta Crystallographica Section C* **71**, 3-8 (2015). https://doi.org/10.1107/S2053229614024218
9. Giacovazzo, C. et al. *Fundamentals of Crystallography*. Oxford University Press (2011).
10. International Tables for Crystallography, Volume B: Reciprocal space. International Union of Crystallography.
11. Grosse-Kunstleve, R. W., Sauter, N. K., Moriarty, N. W. & Adams, P. D. "The Computational Crystallography Toolbox: crystallographic algorithms in a reusable software framework." *Journal of Applied Crystallography* **35**, 126-136 (2002). https://doi.org/10.1107/S0021889801017824
12. cctbx project maintainers. *cctbx_project* public source repository. https://github.com/cctbx/cctbx_project
13. pcxod maintainers. *pcxod/cctbx_project* public source repository. https://github.com/pcxod/cctbx_project
14. Bethe, H. "Theorie der Beugung von Elektronen an Kristallen." *Annalen der Physik* **392**, 55-129 (1928). https://doi.org/10.1002/andp.19283921704
15. Cowley, J. M. & Moodie, A. F. "The scattering of electrons by atoms and crystals. I. A new theoretical approach." *Acta Crystallographica* **10**, 609-619 (1957). https://doi.org/10.1107/S0365110X57002194
16. Humphreys, C. J. "The scattering of fast electrons by crystals." *Reports on Progress in Physics* **42**, 1825-1887 (1979). https://doi.org/10.1088/0034-4885/42/11/002
17. Doyle, P. A. & Turner, P. S. "Relativistic Hartree-Fock X-ray and electron scattering factors." *Acta Crystallographica Section A* **24**, 390-397 (1968). https://doi.org/10.1107/S0567739468000756
18. Peng, L.-M., Ren, G., Dudarev, S. L. & Whelan, M. J. "Robust parameterization of elastic and absorptive electron atomic scattering factors." *Acta Crystallographica Section A* **52**, 257-276 (1996). https://doi.org/10.1107/S0108767395014371
19. Zuo, J. M. & Spence, J. C. H. *Advanced Transmission Electron Microscopy: Imaging and Diffraction in Nanoscience*. Springer (2017). https://doi.org/10.1007/978-1-4939-6607-3
20. Gemmi, M. et al. Review of nanocrystallography by electron diffraction. *ACS Central Science* **5**, 1315-1329 (2019). https://doi.org/10.1021/acscentsci.9b00394
21. Palatinus, L., Petříček, V. & Corrêa, C. A. Precession electron-diffraction tomography theory and implementation. *Acta Crystallographica Section A* **71**, 235-244 (2015). https://doi.org/10.1107/S2053273315001266
22. Palatinus, L. et al. Experimental tests on precession electron-diffraction data. *Acta Crystallographica Section B* **71**, 740-751 (2015). https://doi.org/10.1107/S2052520615017023
23. Palatinus, L. et al. "Structure update from precession electron diffraction data." *Acta Crystallographica Section A* **69**, 171-188 (2013). https://doi.org/10.1107/S010876731204946X
24. De Graef, M. *Introduction to Conventional Transmission Electron Microscopy*. Cambridge University Press (2003).
25. Williams, D. B. & Carter, C. B. *Transmission Electron Microscopy: A Textbook for Materials Science*. Springer (2009). https://doi.org/10.1007/978-0-387-76501-3
26. Reimer, L. & Kohl, H. *Transmission Electron Microscopy: Physics of Image Formation*. Springer (2008). https://doi.org/10.1007/978-0-387-40093-8
27. Palatinus, L. et al. "Hydrogen positions in single nanocrystals revealed by electron diffraction." *Science* **355**, 166-169 (2017). https://doi.org/10.1126/science.aak9652
28. Klar, P. B. et al. Continuous-rotation electron-diffraction structure models using multiple-scattering information. *Nature Chemistry* **15**, 848-855 (2023). https://doi.org/10.1038/s41557-023-01186-1
29. Cleverley, A. & Beanland, R. Fine-sliced electron-diffraction modelling with dynamical Bloch-wave simulations. *IUCrJ* **10**, 118-130 (2023). https://doi.org/10.1107/S2052252522011290
30. Beanland, R., Evans, K., Römer, R. A. & Hubert, A. J. M. *Felix: Bloch-wave Method Diffraction Pattern Simulation Software*. https://github.com/WarwickMicroscopy/Felix
31. Karplus, P. A. & Diederichs, K. "Linking crystallographic model and data quality." *Science* **336**, 1030-1033 (2012). https://doi.org/10.1126/science.1218231
32. Diederichs, K. & Karplus, P. A. "Better models by discarding data?" *Acta Crystallographica Section D* **69**, 1215-1222 (2013). https://doi.org/10.1107/S0907444913001121
33. Flack, H. D. "On enantiomorph-polarity estimation." *Acta Crystallographica Section A* **39**, 876-881 (1983). https://doi.org/10.1107/S0108767383001762
34. Hooft, R. W. W., Straver, L. H. & Spek, A. L. Bayesian statistics on Bijvoet differences. *Journal of Applied Crystallography* **41**, 96-103 (2008). https://doi.org/10.1107/S0021889807059870
35. Harris, C. R. et al. "Array programming with NumPy." *Nature* **585**, 357-362 (2020). https://doi.org/10.1038/s41586-020-2649-2
36. Virtanen, P. et al. "SciPy 1.0: fundamental algorithms for scientific computing in Python." *Nature Methods* **17**, 261-272 (2020). https://doi.org/10.1038/s41592-019-0686-2
37. Wojdyr, M. "GEMMI: A library for structural biology." *Journal of Open Source Software* **7**, 4200 (2022). https://doi.org/10.21105/joss.04200
38. Bradbury, J. et al. *JAX: composable transformations of Python+NumPy programs*. https://github.com/jax-ml/jax
39. DIALS project. *DIALS data files documentation*. https://dials.github.io/dials-1.14/documentation/data_files.html
40. DIALS project. *dials.export documentation*. https://dials.github.io/documentation/programs/dials_export.html
41. Colmey, B., Doherty, T. A. S., Malik, S. A. & Midgley, P. A. Absorption in electron-diffraction Bloch regularization. *arXiv:2602.08935* (2026). https://arxiv.org/abs/2602.08935
42. OpenAI. *Codex: AI coding partner from OpenAI*. https://openai.com/codex

## Acknowledgement

GROND was built with the help of OpenAI Codex as an interactive coding and documentation assistant [42]. We thank Takanori Nakane for the strong/weak-beam Bethe partitioning suggestion that led to the optional `scipy_bethe` promoted Schur backend, and for the fixed-size matrix suggestion that motivated the opt-in padded JAX backend tests. Scientific design choices, validation decisions, code ownership, and responsibility for interpreting GROND outputs remain with the GROND authors and users.

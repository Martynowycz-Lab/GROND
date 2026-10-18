"""Argument declarations for the combine native nuisance runs workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run", action="append", required=True, help="name=/path/to/native_nuisance_run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument(
        "--endpoint-lambda",
        default="1.0",
        help="Diagnostic endpoint q exponent. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument(
        "--endpoint-source",
        choices=("native", "hkl"),
        default="native",
        help="Build corrected endpoint rows from native_dyn_calc.csv, or read normalized endpoint HKL rows from each run.",
    )
    ap.add_argument(
        "--interdataset-scale",
        choices=("overlap", "none"),
        default="overlap",
        help="Fit one scalar per dataset from overlapping corrected reflections before merging the multi-dataset endpoint.",
    )
    ap.add_argument(
        "--endpoint-merge-key",
        choices=("model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"),
        default="model_symmetry",
        help=(
            "Reflection key used before the shared endpoint LS step. model_symmetry merges corrected observations related "
            "by the model space-group symmetry while preserving Friedel opposites in acentric groups; model_symmetry_friedel "
            "also merges Friedel mates; exact_signed_hkl is a diagnostic legacy mode."
        ),
    )
    ap.add_argument(
        "--dataset-sigma-policy",
        choices=("none", "uniform_z_rms", "error_model"),
        default="none",
        help=(
            "Endpoint uncertainty policy applied independently to each dataset before interdataset scaling and merging. "
            "This keeps a weak crystal from setting the sigma scale for stronger crystals. uniform_z_rms multiplies "
            "every sigma by one residual RMS(z); error_model fits sigma'^2 = k^2 (sigma^2 + (u P)^2) per crystal on "
            "work rows against the current model (AUTO) and reports the free-row RMS(z)."
        ),
    )
    ap.add_argument(
        "--error-model-refits",
        type=int,
        default=1,
        help=(
            "error_model only: after SHELXL has refined the merged endpoint, refit every crystal's error model against "
            "the refined model (work rows only; intensities unchanged) and refine again, this many times."
        ),
    )
    ap.add_argument(
        "--error-model-dataset-weight",
        choices=["apply", "off"],
        default="apply",
        help=(
            "error_model only: whether the dataset policy's fractional endpoint weight still scales a crystal's "
            "SHELXL sigmas (sigma / sqrt(w)) on top of its fitted error model. off leaves the error model as the only "
            "relative weight (a crystal with weight 0 stays out); apply keeps both."
        ),
    )
    ap.add_argument("--dataset-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--dataset-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--dataset-sigma-max-scale", type=float, default=4.0)
    ap.add_argument(
        "--dataset-sigma-holdout-floor",
        action="store_true",
        help="For per-dataset uniform_z_rms, do not deflate sigmas below that dataset's held-out RMS(z).",
    )
    ap.add_argument(
        "--merge-sigma-policy",
        choices=("formal", "chisq_overdispersion"),
        default="formal",
        help=(
            "How to assign sigmas after merging corrected endpoint rows. formal uses propagated inverse-variance errors; "
            "chisq_overdispersion inflates a merged reflection when its contributors disagree more than their sigmas predict."
        ),
    )
    ap.add_argument("--shelxl", default="")
    ap.add_argument("--shelxl-iterations", type=int, default=5)
    ap.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Internal L.S./CGLS cycle count written to the combined corrected-HKL endpoint. Production keeps this at 1.",
    )
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=("freeze-zero", "inherit", "promote", "promote-gold", "converge"),
        default="freeze-zero",
        help=(
            "SHELXL weight policy for the shared combined endpoint. freeze-zero is the production default after "
            "GROND sigma calibration; inherit preserves the source active WGHT; promote keeps the source WGHT on the "
            "first pass and uses SHELXL recommendations on later passes; promote-gold starts at WGHT 0.1 0 "
            "and then promotes like promote; converge starts at WGHT 0 0 and promotes until the applied and "
            "suggested WGHT agree and max shift/esd < --shelxl-convergence-shift-esd (AUTO)."
        ),
    )
    ap.add_argument(
        "--wght-converge-max-calls",
        type=int,
        default=20,
        help="Maximum SHELXL calls of one endpoint under --endpoint-wght-mode converge (replaces --shelxl-iterations there).",
    )
    ap.add_argument(
        "--unmerged-endpoint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also write and refine an unmerged multi-crystal endpoint: every corrected work row, batch number = "
            "crystal index, MERG 0, one refined BASF per extra crystal (AUTO)."
        ),
    )
    ap.add_argument(
        "--zerr-esds",
        default="",
        help="Six cell e.s.d.s a,b,c,alpha,beta,gamma written to the ZERR record of every endpoint input (Z kept). Empty keeps the model's ZERR.",
    )
    ap.add_argument(
        "--strip-q-peaks",
        action="store_true",
        help="Drop difference-map peaks (Q1, Q2, ...) of a previous SHELXL run from generated endpoint inputs (AUTO).",
    )
    ap.add_argument(
        "--endpoint-list-code",
        type=int,
        default=0,
        help="LIST code of generated endpoint inputs (AUTO 6: an .fcf with phases for SHELXLE/Olex2 maps). 0 keeps the model's LIST.",
    )
    ap.add_argument("--acta", action="store_true", help="Add ACTA to the combined endpoints so SHELXL writes a CIF.")
    ap.add_argument(
        "--shelxl-convergence-shift-esd",
        type=float,
        default=0.0,
        help=(
            "Stop the SHELXL calls of this endpoint once the last L.S. cycle has |max shift/esd| below this value "
            "(AUTO 0.01). 0 disables the test and runs every requested call (legacy)."
        ),
    )
    ap.add_argument(
        "--shelxl-npd-recovery",
        choices=("damp", "isor"),
        default="damp",
        help="Recovery for a non-positive-definite ADP step: legacy DAMP ladder, or ISOR on the NPD atoms only (AUTO).",
    )
    ap.add_argument(
        "--keep-damp",
        action="store_true",
        help="Legacy: carry an active DAMP instruction into later calls and the written RES. Default strips it.",
    )
    ap.add_argument(
        "--endpoint-include-free-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write rows with split=free into the corrected HKL (legacy default). AUTO passes "
            "--no-endpoint-include-free-rows so free rows are predicted and scored but never refined against."
        ),
    )
    ap.add_argument(
        "--endpoint-q-policy",
        choices=("clip", "exclude"),
        default="clip",
        help=(
            "Rows whose correction ratio relative to the dataset median lies outside (0.05, 20): clip q into "
            "[0.05, 20] and keep them (legacy) or exclude them from the HKL and count them (AUTO)."
        ),
    )
    ap.add_argument(
        "--rescale",
        choices=("median-iobs", "absolute"),
        default="median-iobs",
        help=(
            "Corrected-intensity scale: median-iobs rescales each dataset to its median IOBS (legacy); absolute writes "
            "kappa*|F|^2 * Iobs/Icalc with kappa=10 and resets the SHELXL OSF to sqrt(kappa) (AUTO)."
        ),
    )
    ap.add_argument(
        "--post-bloch-sigma-policy",
        choices=("none", "uniform_z_rms"),
        default="none",
        help=(
            "Endpoint uncertainty policy applied after multi-dataset merging and before SHELXL. "
            "uniform_z_rms fits one residual RMS(z) scalar from the supplied model and scales endpoint sigmas only."
        ),
    )
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument(
        "--min-d-spacing",
        "--endpoint-min-d-spacing",
        dest="min_d_spacing",
        type=float,
        default=0.0,
        help=(
            "Optional reporting/endpoint cutoff applied to existing native row tables before joint metrics, "
            "sigma calibration, merging, and SHELXL. Zero preserves every row in the supplied runs."
        ),
    )
    ap.add_argument(
        "--endpoint-free-validation",
        choices=("on", "off"),
        default="off",
        help=(
            "Reporting only: score the final merged (and unmerged) endpoint model with SHELXL L.S. 0 on the corrected "
            "free rows (built exactly like the work rows, never fitted) and on its own work HKL; recorded as "
            "combined_endpoint.free_validation. Needs --no-endpoint-include-free-rows and --endpoint-source native "
            "(AUTO on; off is the legacy command)."
        ),
    )
    ap.add_argument(
        "--score-res",
        default="",
        help=(
            "Validation only (grond validate-endpoint --endpoint published); AUTO never passes it. Refine nothing: "
            "rebuild this command's merged endpoint HKL with its recipe and score this already refined .res with "
            "SHELXL L.S. 0 on that HKL plus the corrected free rows, with the WGHT its own last refinement applied. The "
            "rebuilt HKL must equal the .hkl beside the .res byte for byte, otherwise nothing is scored. Needs "
            "--endpoint-free-validation on and no --unmerged-endpoint."
        ),
    )
    ap.add_argument(
        "--score-error-model-pass",
        type=int,
        default=1,
        help=(
            "With --score-res: the error-model pass the scored endpoint kept (its combined_summary.json "
            "combined_endpoint.error_model_passes); pass 1 is fitted against --res."
        ),
    )
    ap.add_argument(
        "--score-sigma-model",
        default="",
        help=(
            "With --score-res and a kept pass above 1: the model that pass's error model was fitted against (its "
            "recorded sigma_model, the previous pass's refined .res)."
        ),
    )
    return ap

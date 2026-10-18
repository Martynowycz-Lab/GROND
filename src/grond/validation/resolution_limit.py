"""Endpoint promotion checks and resolution limits decided from per-shell evidence.

classify_endpoint_candidate separates promotable corrected-HKL endpoints from diagnostic ones. suggest_resolution_limit
and promote_resolution_limit walk the shells from low to high resolution (d in Å) and stop at the first shell that
fails.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


def _float(value, fallback: float = float("nan")) -> float:
    """Return ``value`` as a finite float, or ``fallback`` when it is missing, not numeric or not finite."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


@dataclass
class EndpointSafetyDecision:
    """Classification for a corrected-HKL endpoint candidate."""

    status: str
    reasons: list[str]
    warnings: list[str]
    hkl_safe_factor: float
    modulation_p95: float
    wR2_all: float
    R1_all: float
    R1_gt: float

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation."""

        return asdict(self)


@dataclass
class ResolutionPromotionDecision:
    """Promoted resolution limit using signal, held-out stability, and MR factors."""

    promoted_min_d_spacing: float
    promoted_last_shell: int | None
    exploratory_min_d_spacing: float
    exploratory_last_shell: int | None
    first_rejection: dict
    shell_count: int
    reasons: list[str]

    def to_dict(self) -> dict:
        """Return a JSON-friendly representation."""

        return asdict(self)


def classify_endpoint_candidate(
    candidate: Mapping,
    *,
    best_wR2_all: float | None = None,
    min_hkl_safe_factor: float = 0.25,
    max_modulation_p95: float = 20.0,
    max_wR2_penalty: float = 0.005,
) -> EndpointSafetyDecision:
    """Classify whether an endpoint is promotable or diagnostic-only.

    Positive global rescaling changes intensity units, not the physics. Its
    magnitude and intensity modulation are warnings. Actual written HKL
    precision is checked by the artifact-owning workflow.
    """

    shelxl = candidate.get("shelxl") if isinstance(candidate.get("shelxl"), Mapping) else candidate
    safe = _float(candidate.get("hkl_safe_factor", 1.0))
    mod95 = _float(candidate.get("mod_p95", candidate.get("modulation_p95")))
    wr2 = _float(shelxl.get("wR2_all"))
    r1_all = _float(shelxl.get("R1_all"))
    r1_gt = _float(shelxl.get("R1_gt"))
    reasons: list[str] = []
    warnings: list[str] = []

    if not math.isfinite(safe) or safe <= 0.0:
        reasons.append("HKL export scale factor must be finite and positive")
    elif safe < min_hkl_safe_factor:
        warnings.append(
            f"fixed-width HKL rescaling factor {safe:.4g} is below warning level {min_hkl_safe_factor:.4g}; "
            "this is an intensity-unit conversion, not a physical rejection criterion"
        )
    if not math.isfinite(mod95):
        warnings.append("95th percentile endpoint modulation is unavailable")
    elif mod95 > max_modulation_p95:
        warnings.append(f"95th percentile endpoint modulation {mod95:.4g} exceeds warning level {max_modulation_p95:.4g}")
    if best_wR2_all is not None and math.isfinite(wr2):
        penalty = wr2 - float(best_wR2_all)
        if penalty > max_wR2_penalty:
            reasons.append(f"wR2 penalty {penalty:.4g} exceeds {max_wR2_penalty:.4g}")
    if not math.isfinite(wr2) or not math.isfinite(r1_all):
        reasons.append("endpoint metrics are incomplete")

    return EndpointSafetyDecision(
        status="promote" if not reasons else "diagnostic",
        reasons=reasons,
        warnings=warnings,
        hkl_safe_factor=safe,
        modulation_p95=mod95,
        wR2_all=wr2,
        R1_all=r1_all,
        R1_gt=r1_gt,
    )


def _row_mr_shell_reasons(
    shell: Mapping,
    *,
    max_shell_mwrall: float,
    max_free_work_mwr_gap: float,
    min_free_rows: int,
    max_cumulative_mwr_penalty: float,
    best_cumulative_mwrall: float,
) -> list[str]:
    """Return the reasons one shell fails the row MR tests.

    The tests are the shell's dynamic MwRall_I, its cumulative MwRall_I penalty over the best so far, and the
    free-work MwRall_I gap when both sets have at least ``min_free_rows`` rows. They read the intensity-residual
    form (``*_I`` keys), against which the thresholds were set; the amplitude-form ``MwRall`` is report only.
    """

    reasons: list[str] = []
    shell_mwr = _float(shell.get("dynamic_MwRall_I"))
    if not math.isfinite(shell_mwr):
        reasons.append("dynamic MwRall_I is not finite")
    elif shell_mwr > max_shell_mwrall:
        reasons.append(f"dynamic MwRall_I {shell_mwr:.4g} > {max_shell_mwrall:.4g}")

    cumulative_mwr = _float(shell.get("cumulative_dynamic_MwRall_I"))
    if math.isfinite(cumulative_mwr) and math.isfinite(best_cumulative_mwrall):
        penalty = cumulative_mwr - best_cumulative_mwrall
        if penalty > max_cumulative_mwr_penalty:
            reasons.append(f"cumulative MwRall_I penalty {penalty:.4g} > {max_cumulative_mwr_penalty:.4g}")

    work_n = int(_float(shell.get("work_Nall"), 0.0))
    free_n = int(_float(shell.get("free_Nall"), 0.0))
    gap = _float(shell.get("free_work_MwRall_I_gap"))
    if work_n >= min_free_rows and free_n >= min_free_rows and (
        not math.isfinite(gap) or gap > max_free_work_mwr_gap
    ):
        reasons.append(f"work/free MwRall_I gap {gap:.4g} > {max_free_work_mwr_gap:.4g}")
    return reasons


def _basic_signal_reasons(
    shell: Mapping,
    *,
    min_rows: int,
    min_mean_i_over_sigma: float,
    min_strong_fraction: float,
) -> list[str]:
    """Return the reasons one shell lacks signal: too few rows, or low mean I/sigma (and too few strong rows)."""

    rows = int(_float(shell.get("n_rows"), 0.0))
    strong = int(_float(shell.get("n_strong"), 0.0))
    isig = _float(shell.get("mean_I_over_sigma"))
    reasons: list[str] = []
    if rows < min_rows:
        reasons.append(f"rows {rows} < {min_rows}")
    strong_fraction = strong / max(rows, 1)
    isig_fails = not math.isfinite(isig) or isig < min_mean_i_over_sigma
    strong_fails = min_strong_fraction > 0.0 and strong_fraction < min_strong_fraction
    if isig_fails and (min_strong_fraction <= 0.0 or strong_fails):
        if min_strong_fraction > 0.0:
            reasons.append(
                f"mean I/sigma {isig:.4g} < {min_mean_i_over_sigma:.4g} "
                f"and strong fraction {strong_fraction:.4g} < {min_strong_fraction:.4g}"
            )
        else:
            reasons.append(f"mean I/sigma {isig:.4g} < {min_mean_i_over_sigma:.4g}")
    return reasons


def _heldout_stability_reasons(
    shell: Mapping,
    *,
    min_common_pairs: int,
    min_split_cc: float,
    min_free_rows: int,
    max_free_work_mwr_gap: float,
) -> list[str]:
    """Return the reason one shell is unstable on held-out data, or an empty list.

    The split CC is tested when there are enough split pairs; otherwise the free-work MwRall_I gap (intensity-residual
    form) is, when both sets have at least ``min_free_rows`` rows.
    """

    common = int(_float(shell.get("split_common"), 0.0))
    cc = _float(shell.get("split_cc"))
    if common >= min_common_pairs:
        if not math.isfinite(cc) or cc < min_split_cc:
            return [f"split CC {cc:.4g} < {min_split_cc:.4g}"]
        return []

    work_n = int(_float(shell.get("work_Nall"), 0.0))
    free_n = int(_float(shell.get("free_Nall"), 0.0))
    gap = _float(shell.get("free_work_MwRall_I_gap"))
    if work_n >= min_free_rows and free_n >= min_free_rows:
        if not math.isfinite(gap) or gap > max_free_work_mwr_gap:
            return [f"work/free MwRall_I gap {gap:.4g} > {max_free_work_mwr_gap:.4g}"]
        return []
    return []


def promote_resolution_limit(
    shells: Sequence[Mapping],
    *,
    strict_split_cc: float = 0.30,
    strict_mean_i_over_sigma: float = 0.50,
    exploratory_split_cc: float = 0.20,
    exploratory_mean_i_over_sigma: float = 0.25,
    min_common_pairs: int = 8,
    min_rows: int = 20,
    min_strong_fraction: float = 0.005,
    max_shell_mwrall: float = 0.60,
    max_free_work_mwr_gap: float = 0.12,
    min_free_rows: int = 5,
    max_cumulative_mwr_penalty: float = 0.03,
) -> ResolutionPromotionDecision:
    """Promote a reportable resolution limit from shell evidence.

    This is stricter than a pure I/sigma cutoff. A shell must retain basic
    signal, be stable on an available held-out comparison, and avoid a large
    GROND row/MR merged weighted residual penalty. The weighted-residual tests
    read the intensity-residual ``*_I`` shell keys, against which the default
    thresholds were set. The exploratory limit is a looser signal-only guard
    that remains useful for map or diagnostic review.
    """

    ordered = sorted(
        [dict(s) for s in shells],
        key=lambda s: _float(s.get("d_high"), float("inf")),
        reverse=True,
    )
    promoted = float("nan")
    exploratory = float("nan")
    promoted_last: int | None = None
    exploratory_last: int | None = None
    first_rejection: dict = {}
    reasons: list[str] = []
    best_cumulative = float("inf")

    for shell in ordered:
        shell_no = int(_float(shell.get("shell"), len(ordered)))
        d_low = _float(shell.get("d_low"))
        signal_reasons = _basic_signal_reasons(
            shell,
            min_mean_i_over_sigma=strict_mean_i_over_sigma,
            min_rows=min_rows,
            min_strong_fraction=min_strong_fraction,
        )
        stability_reasons = _heldout_stability_reasons(
            shell,
            min_common_pairs=min_common_pairs,
            min_split_cc=strict_split_cc,
            min_free_rows=min_free_rows,
            max_free_work_mwr_gap=max_free_work_mwr_gap,
        )
        row_mr_reasons = _row_mr_shell_reasons(
            shell,
            max_shell_mwrall=max_shell_mwrall,
            max_free_work_mwr_gap=max_free_work_mwr_gap,
            min_free_rows=min_free_rows,
            max_cumulative_mwr_penalty=max_cumulative_mwr_penalty,
            best_cumulative_mwrall=best_cumulative,
        )
        shell_reasons = signal_reasons + stability_reasons + row_mr_reasons
        if shell_reasons:
            first_rejection = {"shell": shell_no, "d_low": d_low, "d_high": _float(shell.get("d_high")), "reasons": shell_reasons}
            reasons.extend(shell_reasons)
            break
        promoted = d_low
        promoted_last = shell_no
        cumulative = _float(shell.get("cumulative_dynamic_MwRall_I"))
        if math.isfinite(cumulative):
            best_cumulative = min(best_cumulative, cumulative)

    for shell in ordered:
        shell_no = int(_float(shell.get("shell"), len(ordered)))
        d_low = _float(shell.get("d_low"))
        exploratory_reasons = _basic_signal_reasons(
            shell,
            min_mean_i_over_sigma=exploratory_mean_i_over_sigma,
            min_rows=min_rows,
            min_strong_fraction=0.0,
        )
        exploratory_reasons.extend(
            _heldout_stability_reasons(
                shell,
                min_common_pairs=min_common_pairs,
                min_split_cc=exploratory_split_cc,
                min_free_rows=min_free_rows,
                max_free_work_mwr_gap=max_free_work_mwr_gap * 1.5,
            )
        )
        shell_mwr = _float(shell.get("dynamic_MwRall_I"))
        if not exploratory_reasons and (not math.isfinite(shell_mwr) or shell_mwr <= max_shell_mwrall * 1.25):
            exploratory = d_low
            exploratory_last = shell_no
        else:
            break

    return ResolutionPromotionDecision(
        promoted_min_d_spacing=promoted,
        promoted_last_shell=promoted_last,
        exploratory_min_d_spacing=exploratory,
        exploratory_last_shell=exploratory_last,
        first_rejection=first_rejection,
        shell_count=len(ordered),
        reasons=reasons,
    )

"""JANA-style weighted R factors: the amplitude form (Jana2020 wR(all) on F), with the old intensity form as *_I."""

import math

import numpy as np
import pytest

from grond.forensics.resolution import promote_resolution_limit
from grond.legacy.engine.resolution_limit_audit import shell_record
from grond.legacy.engine.row_mr_report import compute_model_metrics, row_mr_r_factors

U = 0.01

# Strong, medium, weak-positive (I < 0.01 sigma), negative, negative prediction, and a row below 3 sigma.
OBS = np.array([400.0, 50.0, 0.015, -3.0, 20.0, 2.5])
SIGMA = np.array([4.0, 2.0, 2.0, 1.5, 1.0, 1.0])
CALC = np.array([380.0, 56.0, 0.4, 0.2, -0.5, 3.1])


def jana_weights(obs, sigma, u=U):
    """Return F_o and the least-squares weight 1/(sigma_F^2 + (u F_o)^2) of Jana2020 refinement on F.

    Rows with I < 0.01 sigma take F_o = 0 and sigma_F = 5 sqrt(sigma) (Klar et al. 2023); others sigma_F = sigma/(2 F_o).
    """

    weak = obs < 0.01 * sigma
    fo = np.sqrt(np.where(weak, 0.0, np.maximum(obs, 0.0)))
    sigma_f = np.where(weak, 5.0 * np.sqrt(sigma), sigma / (2.0 * np.where(fo > 0.0, fo, 1.0)))
    return fo, 1.0 / (sigma_f**2 + (u * fo) ** 2)


def jana_amplitude_wr(obs, sigma, calc, use, u=U):
    """Jana2020 wR for refinement on F, written out directly: sqrt(sum w (Fo - Fc)^2 / sum w Fo^2)."""

    fo, w = jana_weights(obs, sigma, u)
    fc = np.sqrt(np.maximum(calc, 0.0))
    return math.sqrt(np.sum(w[use] * (fo[use] - fc[use]) ** 2) / np.sum(w[use] * fo[use] ** 2))


def old_intensity_wr(obs, sigma, calc, use, u=U):
    """The intensity-residual form GROND printed before: the same w, residual and denominator on I."""

    fo, w = jana_weights(obs, sigma, u)
    io = fo**2
    return math.sqrt(np.sum(w[use] * (io[use] - calc[use]) ** 2) / np.sum(w[use] * io[use] ** 2))


def synthetic_rows(n=16, seed=3):
    """Return row dicts on eight HKLs (two rows each), with a weak and a negative observation and a work/free split."""

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        obs = float(rng.uniform(5.0, 400.0))
        rows.append(
            {
                "dataset": "synthetic",
                "row_index": i,
                "h": i % 4,
                "k": (i // 4) % 2,
                "l": 1,
                "d_A": 3.0 - 0.1 * i,
                "split": "work" if i % 4 else "free",
                "target_I": obs,
                "source_sigma": float(rng.uniform(1.0, 4.0)),
                "sigma_eff": float(rng.uniform(1.0, 4.0)),
                "dyn_scaled": obs * float(rng.uniform(0.8, 1.2)),
                "born_scaled": obs * float(rng.uniform(0.5, 1.5)),
            }
        )
    rows[2]["target_I"] = 0.01  # weak: below 0.01 sigma
    rows[5]["target_I"] = -4.0  # negative
    return rows


def merged_by_hkl(rows, calc_key):
    """Arithmetic merge over identical HKL: weak rows zeroed first, mean I, sigma_M = sqrt(sum sigma^2) / n."""

    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["h"], row["k"], row["l"]), []).append(row)
    obs, sigma, calc = [], [], []
    for key in sorted(groups):
        group = groups[key]
        o = np.array([r["target_I"] for r in group])
        s = np.array([r["source_sigma"] for r in group])
        c = np.array([r[calc_key] for r in group])
        o = np.where(o < 0.01 * s, 0.0, o)
        obs.append(o.mean())
        sigma.append(math.sqrt(np.sum(s**2)) / len(group))
        calc.append(c.mean())
    return np.array(obs), np.array(sigma), np.array(calc)


@pytest.mark.parametrize("subset", ["all", "observed"])
def test_weighted_factors_are_the_jana_amplitude_form_and_keep_the_intensity_form_as_i(subset):
    result = row_mr_r_factors(OBS, SIGMA, CALC, instability_u=U)
    observed = OBS > 3.0 * SIGMA
    use = np.ones(len(OBS), dtype=bool) if subset == "all" else observed
    key = "wRall" if subset == "all" else "wRobs"

    assert result["Nobs"] == int(np.count_nonzero(observed))
    assert result[key] == pytest.approx(jana_amplitude_wr(OBS, SIGMA, CALC, use), rel=1e-12)
    assert result[key + "_I"] == pytest.approx(old_intensity_wr(OBS, SIGMA, CALC, use), rel=1e-12)
    assert result[key] != pytest.approx(result[key + "_I"], rel=1e-3)


def test_weak_rows_enter_with_zero_amplitude_and_their_prediction_counts():
    changed = CALC.copy()
    changed[2] = 4.0  # only the weak row's prediction changes
    base = row_mr_r_factors(OBS, SIGMA, CALC, instability_u=U)
    moved = row_mr_r_factors(OBS, SIGMA, changed, instability_u=U)

    assert base["Nall"] == moved["Nall"] == len(OBS)
    assert moved["wRall"] > base["wRall"]
    assert base["Rall"] != moved["Rall"]


def test_empty_input_reports_nan_for_both_forms():
    result = row_mr_r_factors([], [], [])

    assert result["Nall"] == 0 and result["Nobs"] == 0
    assert all(math.isnan(result[key]) for key in ("Robs", "Rall", "wRobs", "wRall", "wRobs_I", "wRall_I"))


def test_merged_factors_apply_both_forms_to_the_merged_arrays():
    rows = synthetic_rows()
    model = {"symmetry_ops": []}
    metrics = compute_model_metrics(rows, model, "dyn_scaled", instability_u=U)
    m_obs, m_sigma, m_calc = merged_by_hkl(rows, "dyn_scaled")
    every = np.ones(len(m_obs), dtype=bool)
    m_observed = m_obs > 3.0 * m_sigma

    assert metrics["MNall"] == len(m_obs) == 8
    assert metrics["MNobs"] == int(np.count_nonzero(m_observed))
    assert metrics["MwRall"] == pytest.approx(jana_amplitude_wr(m_obs, m_sigma, m_calc, every), rel=1e-12)
    assert metrics["MwRobs"] == pytest.approx(jana_amplitude_wr(m_obs, m_sigma, m_calc, m_observed), rel=1e-12)
    assert metrics["MwRall_I"] == pytest.approx(old_intensity_wr(m_obs, m_sigma, m_calc, every), rel=1e-12)
    assert metrics["MwRobs_I"] == pytest.approx(old_intensity_wr(m_obs, m_sigma, m_calc, m_observed), rel=1e-12)
    direct = row_mr_r_factors(m_obs, m_sigma, m_calc, instability_u=U)
    assert metrics["MwRall"] == direct["wRall"] and metrics["MwRall_I"] == direct["wRall_I"]


def test_resolution_shell_record_gates_on_the_intensity_form_and_reports_both():
    rows = synthetic_rows()
    model = {"symmetry_ops": []}
    record = shell_record(1, rows, rows, model, min_common_pairs=100)
    dyn = compute_model_metrics(rows, model, "dyn_scaled")
    work = compute_model_metrics([r for r in rows if r["split"] == "work"], model, "dyn_scaled")
    free = compute_model_metrics([r for r in rows if r["split"] == "free"], model, "dyn_scaled")

    assert record["dynamic_MwRall"] == dyn["MwRall"]
    assert record["dynamic_MwRall_I"] == dyn["MwRall_I"]
    assert record["cumulative_dynamic_MwRall_I"] == dyn["MwRall_I"]
    assert record["free_work_MwRall_I_gap"] == pytest.approx(abs(free["MwRall_I"] - work["MwRall_I"]))
    assert record["free_work_MwRall_gap"] == pytest.approx(abs(free["MwRall"] - work["MwRall"]))
    assert record["stability_mode"] == "work_free_MwRall_I"


def gate_shell(**overrides):
    """A shell that passes every promotion test, with the amplitude and intensity forms set separately."""

    shell = {
        "shell": 1,
        "d_high": 3.0,
        "d_low": 1.0,
        "n_rows": 100,
        "n_strong": 80,
        "mean_I_over_sigma": 5.0,
        "split_common": 10,
        "split_cc": 0.9,
        "work_Nall": 50,
        "free_Nall": 50,
        "dynamic_MwRall": 0.20,
        "cumulative_dynamic_MwRall": 0.20,
        "free_work_MwRall_gap": 0.01,
        "dynamic_MwRall_I": 0.30,
        "cumulative_dynamic_MwRall_I": 0.30,
        "free_work_MwRall_I_gap": 0.01,
    }
    shell.update(overrides)
    return shell


def test_resolution_promotion_reads_the_intensity_form_only():
    promoted = promote_resolution_limit(
        [gate_shell(dynamic_MwRall=0.95, cumulative_dynamic_MwRall=0.95, free_work_MwRall_gap=0.5)]
    )
    assert promoted.promoted_last_shell == 1
    assert promoted.reasons == []

    rejected = promote_resolution_limit([gate_shell(dynamic_MwRall_I=0.70)])
    assert rejected.promoted_last_shell is None
    assert any("MwRall_I 0.7" in reason for reason in rejected.reasons)

    unstable = promote_resolution_limit([gate_shell(split_common=0, free_work_MwRall_I_gap=0.5)])
    assert unstable.promoted_last_shell is None
    assert any("MwRall_I gap 0.5" in reason for reason in unstable.reasons)

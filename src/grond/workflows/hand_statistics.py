"""Combine matched hand evidence and explain weak or conflicting diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

from grond.validation import heldout
from grond.workflows.decision import safe_float


def _adjusted_hand_z_block(stats: Mapping[str, object], split: str) -> dict[str, object]:
    """Return the declared Klar--Palatinus adjusted-Z block for one split."""

    support = stats.get(split) if isinstance(stats, Mapping) else {}
    support = support if isinstance(support, Mapping) else {}
    adjusted = support.get("klar_palatinus_adjusted_hand_z")
    return dict(adjusted) if isinstance(adjusted, Mapping) else {}


def _combine_adjusted_hand_z(
    diagnostics: Iterable[Mapping[str, object]],
    split: str,
) -> dict[str, object]:
    """Combine independent-dataset adjusted hand counts with a declared rule.

    Each physical dataset contributes its signed reflection-count numerator and
    adjusted variance.  The result is positive when the currently evaluated
    coordinate hand is favored and negative when its inverse is favored.
    """

    numerator = 0.0
    variance = 0.0
    rows = 0
    datasets = 0
    excluded: list[str] = []
    for index, diagnostic in enumerate(diagnostics):
        adjusted = _adjusted_hand_z_block(diagnostic, split)
        n = int(adjusted.get("n", 0) or 0)
        k_effective = safe_float(
            adjusted.get(
                "k_effective_selected_wins_plus_half_ties",
                adjusted.get("k_selected_lower_abs_intensity_residual"),
            ),
            float("nan"),
        )
        w = safe_float(adjusted.get("w"), float("nan"))
        if (
            str(adjusted.get("status") or "") != "ok"
            or n <= 0
            or not math.isfinite(k_effective)
            or not math.isfinite(w)
            or n - w <= 0.0
        ):
            excluded.append(str(index))
            continue
        numerator += 2.0 * k_effective - n
        variance += n - w
        rows += n
        datasets += 1
    z = numerator / math.sqrt(variance) if datasets and variance > 0.0 else float("nan")
    return {
        "status": "ok" if math.isfinite(z) else "unavailable",
        "z": z,
        "n": rows,
        "datasets": datasets,
        "signed_numerator": numerator,
        "adjusted_variance": variance,
        "excluded_dataset_indices": excluded,
        "positive_favors_current_coordinates": True,
        "combination_rule": "sum(2*k_effective-n)/sqrt(sum(n-w)) across physical datasets",
        "confidence_interpretation": (
            "Conditional count contrast; nominal normal scaling assumes independent observations. "
            "Shared nuisance/model fitting and repeated rows can invalidate calibrated sigma claims."
        ),
        "probability_is_calibrated": False,
    }


def hand_diagnostic_supports_other(stats: Mapping[str, object], args: argparse.Namespace) -> tuple[bool, list[str]]:
    """Use the all-row adjusted ZKP to decide whether to invert the model.

    The held-out split is deliberately diagnostic: a small or conflicting free
    Z emits a warning but does not veto a decisive all-row statistic.
    """

    threshold = float(getattr(args, "hand_flip_zkp_all", 3.5))
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("hand inversion threshold must be finite and positive")
    all_adjusted = _adjusted_hand_z_block(stats, "all")
    all_z = safe_float(all_adjusted.get("z"), float("nan"))
    reasons: list[str] = []
    if stats.get("calculation_status") == "error" or stats.get("status") == "calculation_error":
        reasons.extend(str(error) for error in stats.get("calculation_errors", []))
        if not reasons:
            reasons.append("all-row hand calculation has an explicit error")
    if str(all_adjusted.get("status") or "") != "ok" or not math.isfinite(all_z):
        reasons.append("all-row Klar-Palatinus adjusted hand Z is unavailable")
    elif all_z > -threshold:
        reasons.append(
            f"all-row Klar-Palatinus adjusted hand Z {all_z:.4g} does not favor inversion "
            f"at the {-threshold:.4g} decision boundary"
        )
    return not reasons, reasons


def hand_preference_summary(
    all_z: float,
    *,
    active_hand: str,
    threshold: float,
    calculation_valid: bool = True,
) -> dict[str, object]:
    """Report the measured preference separately from the exported representation."""

    valid = bool(calculation_valid and math.isfinite(all_z))
    opposite = "alternate" if active_hand == "original" else "original"
    preferred = (
        active_hand
        if valid and all_z > 0.0
        else opposite
        if valid and all_z < 0.0
        else "no_preference"
        if valid
        else "unavailable"
    )
    confidence = (
        "calculation_error"
        if not valid
        else "no_sensitivity"
        if all_z == 0.0
        else "strong_conditional"
        if abs(all_z) >= threshold
        else "weak_conditional"
    )
    return {
        "preferred_hand": preferred,
        "hand_confidence": confidence,
        "all_row_adjusted_z_positive_favors_active": all_z if valid else None,
        "active_hand_before_selection": active_hand,
        "primary_evidence": "all_rows",
        "free_evidence_role": "diagnostic_warning_only",
        "statistic_is_calibrated_probability": False,
    }


def _hand_evidence_warnings(
    *,
    all_z: float,
    free_z: float,
    switched: bool,
    threshold: float,
) -> list[str]:
    """Warn when final-hand support is weak, conflicting, or not held out."""

    sign = -1.0 if switched else 1.0
    oriented_all = sign * all_z if math.isfinite(all_z) else float("nan")
    oriented_free = sign * free_z if math.isfinite(free_z) else float("nan")
    warnings: list[str] = [
        "ZKP is a conditional reflection-count contrast, not calibrated sigma confidence; "
        "shared scales, repeated observations, and model selection can correlate rows."
    ]
    if not math.isfinite(oriented_all):
        warnings.append("all-row Klar-Palatinus adjusted hand Z is unavailable")
    elif oriented_all < threshold:
        warnings.append(f"final-hand all-row ZKP is {oriented_all:.3g}, below the decisive threshold {threshold:.3g}")
    if not math.isfinite(oriented_free):
        warnings.append("held-out Klar-Palatinus adjusted hand Z is unavailable")
    elif oriented_free < 0.0:
        warnings.append(f"held-out ZKP {oriented_free:.3g} conflicts with the all-row selected hand")
    elif oriented_free < threshold:
        warnings.append(f"held-out ZKP {oriented_free:.3g} supports the selected hand only weakly (< {threshold:.3g})")
    return warnings


def _pooled_hand_pair_statistics(
    pairs: list[tuple[str, Path, Path]],
) -> dict[str, dict[str, object]]:
    """Pool exact per-row hand contrasts across crystals without merging rows."""

    deltas: dict[str, list[float]] = {"all": [], "work": [], "free": []}
    for dataset, selected_path, inverse_path in pairs:
        with selected_path.open(newline="") as handle:
            selected_rows = list(csv.DictReader(handle))
        with inverse_path.open(newline="") as handle:
            inverse_rows = {
                (row.get("row_index"), row.get("h"), row.get("k"), row.get("l")): row for row in csv.DictReader(handle)
            }
        for selected in selected_rows:
            key = (
                selected.get("row_index"),
                selected.get("h"),
                selected.get("k"),
                selected.get("l"),
            )
            inverse = inverse_rows.get(key)
            if inverse is None:
                raise ValueError(f"{dataset}: inverse hand table lacks row key {key}")
            selected_residual = float(selected["dyn_residual_over_sigma"])
            inverse_residual = float(inverse["dyn_residual_over_sigma"])
            if not math.isfinite(selected_residual) or not math.isfinite(inverse_residual):
                raise ValueError(f"{dataset}: nonfinite paired residual at row key {key}")
            delta = inverse_residual * inverse_residual - selected_residual * selected_residual
            split = str(selected.get("split") or "")
            deltas["all"].append(delta)
            if split in {"work", "free"}:
                deltas[split].append(delta)

    result: dict[str, dict[str, object]] = {}
    for split, values in deltas.items():
        n_rows = len(values)
        mean = math.fsum(values) / n_rows if n_rows else float("nan")
        if n_rows > 1:
            variance = math.fsum((value - mean) ** 2 for value in values) / (n_rows - 1)
            t_style = mean / math.sqrt(variance / n_rows) if variance > 0.0 else math.copysign(float("inf"), mean)
        else:
            t_style = float("nan")
        result[split] = {
            "n": n_rows,
            "mean_delta_chi2_inverse_minus_selected": mean,
            "paired_row_t_delta_chi2": t_style,
            "positive_favors_selected": True,
        }
    return result


# ----------------------------------------------------------------------------- held-out hand evidence

HAND_CV_SALT = "GROND-HAND-CV-20260925"
HAND_CV_FOLDS = 3
HAND_PAIR_TABLE_COLUMNS = (
    "row_index",
    "h",
    "k",
    "l",
    "d_A",
    "target_I",
    "source_sigma",
    "group",
    "split",
    "dyn_raw",
    "born_raw",
    "born_scaled",
)


def read_hand_pair_tables(selected_path: Path, inverse_path: Path) -> dict[str, np.ndarray]:
    """Join one exact pair's selected and inverse row tables on (row_index, h, k, l).

    The observation columns must be identical in both tables; only the
    dynamical predictions may differ.  Anything else fails closed.
    """

    def load(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
        """Read one hand's row table keyed by (row_index, h, k, l); raise on missing columns or duplicate keys."""
        with Path(path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [field for field in HAND_PAIR_TABLE_COLUMNS if field not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{path} lacks held-out hand columns: {', '.join(missing)}")
            rows: dict[tuple[str, ...], dict[str, str]] = {}
            for row in reader:
                key = tuple(str(row[field]).strip() for field in ("row_index", "h", "k", "l"))
                if key in rows:
                    raise ValueError(f"{path} has a duplicate row key {key}")
                rows[key] = row
            return rows

    selected = load(selected_path)
    inverse = load(inverse_path)
    if set(selected) != set(inverse):
        raise ValueError("selected and inverse hand tables do not contain the same rows")
    keys = sorted(selected, key=lambda key: int(key[0]))
    for key in keys:
        for field in ("d_A", "target_I", "source_sigma", "group", "split"):
            if str(selected[key][field]).strip() != str(inverse[key][field]).strip():
                raise ValueError(f"hand tables differ in observation column {field} at row {key}")
    split = np.asarray([str(selected[key]["split"]).strip() for key in keys], dtype=object)
    if not np.all(np.isin(split, ["work", "free"])):
        raise ValueError("hand tables contain rows outside the work/free split")

    def column(rows, field, dtype=float):
        """Return one column of ``rows`` in joined row order as an array of ``dtype``."""
        return np.asarray([rows[key][field] for key in keys], dtype=float).astype(dtype)

    return {
        "row_index": column(selected, "row_index", int),
        "hkl": np.asarray([[int(v) for v in key[1:]] for key in keys], dtype=int).reshape(-1, 3),
        "d": column(selected, "d_A"),
        "y": column(selected, "target_I"),
        "sig": column(selected, "source_sigma"),
        "group": column(selected, "group", int),
        "split": split,
        "raw_selected": column(selected, "dyn_raw"),
        "raw_inverse": column(inverse, "dyn_raw"),
        "born_selected": column(selected, "born_raw"),
        "born_inverse": column(inverse, "born_raw"),
        "born_scaled_selected": column(selected, "born_scaled"),
        "born_scaled_inverse": column(inverse, "born_scaled"),
    }


def matched_born_delta(tables: Mapping[str, np.ndarray], tolerance: float = 1e-9) -> dict[str, object]:
    """Both hands share one Born prediction; any difference invalidates the pair."""

    deltas = {}
    for name, left, right in (
        ("born_raw", tables["born_selected"], tables["born_inverse"]),
        ("born_scaled", tables["born_scaled_selected"], tables["born_scaled_inverse"]),
    ):
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        scale = max(float(np.max(np.abs(left), initial=0.0)), 1e-300)
        finite = bool(np.all(np.isfinite(left)) and np.all(np.isfinite(right)))
        deltas[name] = {
            "max_abs_delta": float(np.max(np.abs(left - right), initial=0.0)) if finite else float("nan"),
            "max_relative_delta": float(np.max(np.abs(left - right), initial=0.0) / scale) if finite else float("nan"),
        }
    passed = all(math.isfinite(v["max_relative_delta"]) and v["max_relative_delta"] <= tolerance for v in deltas.values())
    return {"passed": passed, "relative_tolerance": float(tolerance), **deltas}


def _inverse_oriented_sign_test(y, pred_selected, pred_inverse, orbit, sel) -> dict[str, object]:
    """Orbit sign test with orbit_sign_z > 0 favouring the inverse; "model" is the selected hand."""

    # heldout.orbit_sign_test calls its first prediction the "model"; pass the
    # inverse first and rename the counts back to the selected-model convention.
    sign = heldout.orbit_sign_test(y, pred_inverse, pred_selected, orbit, sel)
    return {
        "orbit_sign_z": float(sign["orbit_sign_z"]),
        "orbit_sign_z_positive_favors": "inverse",
        "orbits_model_better": int(sign["orbits_inverse_better"]),
        "orbits_inverse_better": int(sign["orbits_model_better"]),
        "n_orbits": int(sign["n_orbits"]),
        "n_rows": int(sign["n_rows"]),
    }


def _hand_block(y, sig, pred_selected, pred_inverse, orbit, shell, sel, *, n_boot: int, seed: int) -> dict[str, object]:
    """Sign test and paired orbit bootstrap in the shared contract orientation (inverse vs selected)."""

    boot = heldout.paired_orbit_bootstrap(y, sig, pred_inverse, pred_selected, orbit, shell, sel, n_boot=n_boot, seed=seed)
    return {
        **_inverse_oriented_sign_test(y, pred_selected, pred_inverse, orbit, sel),
        "delta_r_obs": {key: boot["r_obs"][key] for key in ("delta", "ci95", "a", "b")},
        "delta_r_all": {key: boot["r_all"][key] for key in ("delta", "ci95", "a", "b")},
        "delta_definition": "R(inverse) - R(selected) in pp; a = R(inverse), b = R(selected); negative favours inverse",
        "selected_heldout": heldout.r_metrics(y, sig, pred_selected, sel),
        "inverse_heldout": heldout.r_metrics(y, sig, pred_inverse, sel),
        "bootstrap": {"n_boot": int(boot["n_boot"]), "seed": int(boot["seed"]), "n_orbits": int(boot["n_orbits"])},
    }


def heldout_hand_evidence(
    crystals: list[Mapping[str, object]],
    *,
    symmetry_ops,
    scale_cfg: Mapping[str, object] | None = None,
    n_folds: int = HAND_CV_FOLDS,
    cv_salt: str = HAND_CV_SALT,
    shell_dmin: list[float] | None = None,
    excluded_orbits=None,
    n_boot: int = heldout.N_BOOT,
    seed: int = heldout.BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Held-out predictions of both hands from one exact pair, pooled over crystals.

    Every work row is predicted out of fold: ``n_folds`` salted orbit folds
    (crystal-independent keys), with P-spline frame scales refitted on the training
    orbits of each fold only.  Free rows are predicted from one fit on all
    work rows.  ``crystals`` holds ``{"dataset", "tables", "ang"}`` entries
    (``ang``: the per-row rotation angles of the P-spline scale fit).
    """

    cfg = dict(scale_cfg or {})
    excluded = set(excluded_orbits or ())
    per: dict[str, dict[str, object]] = {}
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in ("y", "sig", "d", "keys", "sel", "work", "free", "ps", "pi")}
    for crystal in crystals:
        name = str(crystal["dataset"])
        tables = crystal["tables"]
        y = np.asarray(tables["y"], dtype=float)
        sig = np.asarray(tables["sig"], dtype=float)
        group = np.asarray(tables["group"])
        keys = heldout.orbit_keys(tables["hkl"], symmetry_ops, include_friedel=True)
        leaked = sorted(set(keys.astype(str)) & excluded)
        if leaked:
            raise ValueError(f"{name}: hand tables contain excluded orbits ({len(leaked)}); they must be dropped upstream")
        work = np.asarray(tables["split"] == "work", dtype=bool)
        free = np.asarray(tables["split"] == "free", dtype=bool)
        if not work.any():
            raise ValueError(f"{name}: hand tables have no work rows to fit scales on")
        folds = heldout.hash_fold(list(keys), cv_salt, int(n_folds))
        ang = crystal.get("ang")
        if ang is None:
            raise ValueError(f"{name}: the P-spline held-out hand refit needs per-row rotation angles")

        def fit_predict(raw, train, _y=y, _sig=sig, _group=group, _ang=ang, _keys=keys):
            """Fit this crystal's P-spline frame scales on the ``train`` rows and return the scaled prediction for every row.

            A grid of several (n_basis, lambda) settings is resolved by inner CV on the training rows only.
            """
            fit_cfg = cfg
            if "n_basis_grid" in cfg or "lambda_grid" in cfg:
                grid = {"n_basis": tuple(cfg.get("n_basis_grid") or (32,)), "lambda": tuple(cfg.get("lambda_grid") or (1.0,))}
                objective = str(cfg.get("objective", "row_mr"))
                fit_cfg = heldout.select_pspline_by_inner_cv(
                    _y, _sig, raw, _ang, _group, train, _keys, grid=grid, objective=objective
                )["scale"] if len(grid["n_basis"]) * len(grid["lambda"]) > 1 else {
                    "n_basis": int(grid["n_basis"][0]), "lambda": float(grid["lambda"][0]), "objective": objective}
            return heldout.pspline_fit_predict(_y, _sig, raw, _ang, _group, train, fit_cfg)["prediction"]

        predictions = {}
        for hand, raw in (("selected", tables["raw_selected"]), ("inverse", tables["raw_inverse"])):
            raw = np.asarray(raw, dtype=float)
            pred = heldout.cv_predictions(lambda train, _raw=raw: fit_predict(_raw, train), len(y), folds, eligible=work)
            if free.any():
                pred[free] = np.asarray(fit_predict(raw, work), dtype=float)[free]
            predictions[hand] = pred
        sel = (work | free) & np.isfinite(predictions["selected"]) & np.isfinite(predictions["inverse"])
        per[name] = {
            "n_rows_table": int(len(y)),
            "n_work": int(work.sum()),
            "n_free": int(free.sum()),
            "n_unpredicted": int(((work | free) & ~sel).sum()),
            "_arrays": (y, sig, np.asarray(tables["d"], dtype=float), keys, sel, work, free, predictions),
        }
        for bucket, value in zip(
            ("y", "sig", "d", "keys", "sel", "work", "free", "ps", "pi"),
            (y, sig, np.asarray(tables["d"], dtype=float), keys, sel, work, free, predictions["selected"], predictions["inverse"]),
            strict=True,
        ):
            pooled[bucket].append(value)
    if not per:
        raise ValueError("held-out hand evidence needs at least one crystal")
    cat = {name: np.concatenate(values) for name, values in pooled.items()}
    shell_pooled = heldout.shell_index(cat["d"], shell_dmin)
    offset = 0
    per_crystal: dict[str, dict[str, object]] = {}
    for name, record in per.items():
        y, sig, _d, keys, sel, work, free, predictions = record.pop("_arrays")
        shell = shell_pooled[offset : offset + len(y)]
        offset += len(y)
        block = _hand_block(y, sig, predictions["selected"], predictions["inverse"], keys, shell, sel, n_boot=n_boot, seed=seed)
        block["subsets"] = {
            subset: _inverse_oriented_sign_test(y, predictions["selected"], predictions["inverse"], keys, sel & mask)
            for subset, mask in (("work_out_of_fold", work), ("free", free))
            if np.any(sel & mask)
        }
        per_crystal[name] = {**block, **record}
    pooled_block = _hand_block(
        cat["y"], cat["sig"], cat["ps"], cat["pi"], cat["keys"], shell_pooled, cat["sel"], n_boot=n_boot, seed=seed
    )
    return {
        "pooled": pooled_block,
        "per_crystal": per_crystal,
        "heldout_scheme": {
            "kind": "work_orbit_kfold_out_of_fold_plus_free_from_all_work",
            "n_folds": int(n_folds),
            "salt": str(cv_salt),
            "orbit_key": "grond.validation.heldout.orbit_key (model Laue group + Friedel), pooled over crystals",
            "shells": "exclusion-list shells" if shell_dmin else "5 equal-count resolution shells over pooled rows",
        },
        "scale_model": "pspline",
        "scale_refit": {"kind": "pspline_train_rows_only", **{key: cfg[key] for key in sorted(cfg)}},
    }


def heldout_hand_rule(
    pooled: Mapping[str, object],
    *,
    orbit_sign_z_min: float,
) -> tuple[bool, list[str]]:
    """Switch iff the pooled orbit sign z favours the inverse by the threshold AND the R_obs CI excludes zero."""

    if not math.isfinite(float(orbit_sign_z_min)) or float(orbit_sign_z_min) <= 0.0:
        raise ValueError("held-out orbit sign threshold must be finite and positive")
    reasons: list[str] = []
    z = safe_float(pooled.get("orbit_sign_z"), float("nan"))
    ci = list((pooled.get("delta_r_obs") or {}).get("ci95") or [float("nan"), float("nan")])
    if not math.isfinite(z):
        reasons.append("pooled held-out orbit sign z is unavailable")
    elif z < float(orbit_sign_z_min):
        reasons.append(f"pooled held-out orbit sign z {z:.3g} is below {float(orbit_sign_z_min):.3g} in favour of the inverse")
    upper = safe_float(ci[1] if len(ci) > 1 else float("nan"), float("nan"))
    if not math.isfinite(upper):
        reasons.append("pooled held-out delta R_obs confidence interval is unavailable")
    elif upper >= 0.0:
        reasons.append(
            f"pooled held-out R_obs(inverse) - R_obs(selected) 95% CI upper bound {upper:+.3f} pp does not exclude zero"
        )
    return not reasons, reasons


def heldout_mirrored_rule(
    pooled: Mapping[str, object],
    *,
    orbit_sign_z_min: float,
) -> tuple[bool, list[str]]:
    """The switch rule mirrored: the selected hand is clearly better (z <= -threshold AND the R_obs CI lower bound > 0).

    This is the "clearly against switching" test of the early escalation and the ``confirmed`` test of the
    final check.
    """

    if not math.isfinite(float(orbit_sign_z_min)) or float(orbit_sign_z_min) <= 0.0:
        raise ValueError("held-out orbit sign threshold must be finite and positive")
    reasons: list[str] = []
    z = safe_float(pooled.get("orbit_sign_z"), float("nan"))
    ci = list((pooled.get("delta_r_obs") or {}).get("ci95") or [float("nan"), float("nan")])
    if not math.isfinite(z):
        reasons.append("held-out orbit sign z is unavailable")
    elif z > -float(orbit_sign_z_min):
        reasons.append(f"held-out orbit sign z {z:+.3g} is above {-float(orbit_sign_z_min):+.3g}")
    lower = safe_float(ci[0] if ci else float("nan"), float("nan"))
    if not math.isfinite(lower):
        reasons.append("held-out delta R_obs confidence interval is unavailable")
    elif lower <= 0.0:
        reasons.append(f"held-out R_obs(inverse) - R_obs(selected) 95% CI lower bound {lower:+.3f} pp is not above zero")
    return not reasons, reasons


def per_crystal_hand_agreement(
    per_crystal: Mapping[str, Mapping[str, object]],
    *,
    pooled_switches: bool,
    orbit_sign_z_min: float,
) -> dict[str, object]:
    """Each crystal's own held-out preference beside the pooled decision.

    The decision is pooled over crystals.  A crystal that points the other way is warned about and reported,
    and never vetoes the pooled decision: when the pooled rule switches, any crystal whose orbit sign favours
    the selected hand disagrees; when the pooled rule keeps the selected hand, a crystal disagrees only when it
    meets the switch rule on its own.
    """

    threshold = float(orbit_sign_z_min)
    crystals: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for name, block in sorted(per_crystal.items(), key=lambda item: str(item[0])):
        if not isinstance(block, Mapping):
            continue
        z = safe_float(block.get("orbit_sign_z"), float("nan"))
        delta = block.get("delta_r_obs") if isinstance(block.get("delta_r_obs"), Mapping) else {}
        ci = [safe_float(value, float("nan")) for value in list(delta.get("ci95") or [])[:2]] or [float("nan")] * 2
        switch_rule, _ = heldout_hand_rule(block, orbit_sign_z_min=threshold)
        mirrored, _ = heldout_mirrored_rule(block, orbit_sign_z_min=threshold)
        if not math.isfinite(z):
            preference, strength = "unavailable", "unavailable"
        elif switch_rule or mirrored:
            preference, strength = ("inverse" if switch_rule else "selected"), "rule_met"
        else:
            preference, strength = ("inverse" if z > 0.0 else "selected" if z < 0.0 else "none"), "weak"
        disagrees = bool(
            (pooled_switches and preference == "selected")
            or (not pooled_switches and preference == "inverse" and strength == "rule_met")
        )
        crystals[str(name)] = {
            "orbit_sign_z": z,
            "delta_r_obs": safe_float(delta.get("delta"), float("nan")),
            "delta_r_obs_ci95": ci,
            "preference": preference,
            "strength": strength,
            "disagrees_with_pooled": disagrees,
        }
        if disagrees:
            warnings.append(
                f"{name}: held-out orbit sign z {z:+.2f}, dR_obs(inverse - selected) "
                f"{crystals[str(name)]['delta_r_obs']:+.2f} [{ci[0]:+.2f}, {ci[-1]:+.2f}] pp favours the "
                f"{preference} hand ({strength}) while the pooled rule "
                f"{'switches to the inverse' if pooled_switches else 'keeps the selected hand'}; "
                "the pooled decision stands (one crystal never vetoes it)"
            )
    return {
        "policy": "pooled over crystals; a disagreeing crystal is warned about and reported and never vetoes",
        "pooled_choice": "inverse" if pooled_switches else "selected",
        "crystals": crystals,
        "disagreeing": [name for name, row in crystals.items() if row["disagrees_with_pooled"]],
        "warnings": warnings,
    }

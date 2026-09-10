"""Validate the Component 02/03 inputs and the shared prediction table."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

PREDICTION_COLUMNS = [
    "method",
    "fold",
    "station_id",
    "timestamp",
    "predicted_mm",
    "actual_mm",
]
KEY_COLUMNS = ["fold", "station_id", "timestamp"]


def utc_hours(values):
    """Reject ambiguous local/naive timestamps instead of silently assuming UTC."""
    parsed = pd.to_datetime(values, format="ISO8601", errors="raise")
    if not isinstance(parsed.dtype, pd.DatetimeTZDtype):
        raise ValueError("timestamps must be timezone-aware UTC")  # noqa: TRY004 -- invalid CSV values
    if parsed.isna().any() or any(
        t.utcoffset().total_seconds() for t in parsed.drop_duplicates()
    ):
        raise ValueError("timestamps must be non-missing UTC (+00:00)")
    parsed = parsed.dt.tz_convert("UTC")
    if not parsed.eq(parsed.dt.floor("h")).all():
        raise ValueError("timestamps must identify the start of an hour")
    return parsed


def _require_columns(frame, columns, label):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _station_ids(values, label):
    if (
        values.isna().any()
        or not values.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        raise ValueError(f"{label}: station_id must be a nonempty string")


def validate_gauges(gauge):
    _require_columns(gauge, ["station_id", "timestamp", "rainfall_mm"], "gauge_hourly")
    _station_ids(gauge.station_id, "gauge_hourly")
    result = gauge.copy()
    result["timestamp"] = utc_hours(result.timestamp)
    result["rainfall_mm"] = pd.to_numeric(result.rainfall_mm, errors="raise")
    if result.empty or not np.isfinite(result.rainfall_mm).all():
        raise ValueError("gauge_hourly must contain finite observed rainfall rows")
    if result.rainfall_mm.lt(0).any():
        raise ValueError("gauge_hourly contains negative rainfall")
    if result.duplicated(["station_id", "timestamp"]).any():
        raise ValueError("gauge_hourly has duplicate station-hour rows")
    return result.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def validate_split(split):
    """Check a supplied split; this component never creates station assignments."""
    folds = split.get("folds", [])
    if len(folds) != 5:
        raise ValueError("split must contain all five folds")
    if [fold.get("fold_index") for fold in folds] != list(range(5)):
        raise ValueError("split fold_index values must be ordered 0 through 4")
    test_ids, eligible = [], None
    for fold in folds:
        roles = fold.get("ml", {})
        combined = []
        for role in ("train", "validation", "test"):
            ids = roles.get(role)
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(value, str) and value.strip() for value in ids)
            ):
                raise ValueError(
                    f"fold {fold['fold_index']} {role}: expected station IDs"
                )
            combined.extend(ids)
        if len(combined) != len(set(combined)):
            raise ValueError(
                f"fold {fold['fold_index']}: duplicate/overlapping station roles"
            )
        if eligible is None:
            eligible = set(combined)
        elif set(combined) != eligible:
            raise ValueError("every fold must partition the same eligible stations")
        test_ids.extend(roles["test"])
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != eligible:
        raise ValueError("every eligible station must be a test station exactly once")
    return eligible


def load_inputs(gauge_path, split_path, stations_path=None):
    """Load published CSV/JSON contracts without parsing or preprocessing raw data."""
    paths = [gauge_path, split_path] + ([stations_path] if stations_path else [])
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"BLOCKED: required upstream input is missing: {path}"
            )
    gauge = validate_gauges(pd.read_csv(gauge_path, dtype={"station_id": str}))
    with Path(split_path).open() as handle:
        split = json.load(handle)
    eligible = validate_split(split)
    missing_observations = eligible - set(gauge.station_id)
    if missing_observations:
        raise ValueError(
            f"split stations have no observations: {sorted(missing_observations)}"
        )
    stations = None
    if stations_path:
        stations = pd.read_csv(
            stations_path, dtype={"station_id": str, "station_no": str}
        )
        _require_columns(
            stations,
            ["station_id", "latitude", "longitude", "use_in_study"],
            "stations",
        )
        _station_ids(stations.station_id, "stations")
        if stations.station_id.duplicated().any():
            raise ValueError("stations contains duplicate internal station_id values")
        flags = stations.use_in_study.astype(str).str.lower()
        if not flags.isin(["true", "false"]).all():
            raise ValueError("stations.use_in_study must be true/false")
        stations["use_in_study"] = flags.eq("true")
        if set(stations.loc[stations.use_in_study, "station_id"]) != eligible:
            raise ValueError(
                "split station population differs from stations.use_in_study"
            )
        for column, limit in (("latitude", 90), ("longitude", 180)):
            stations[column] = pd.to_numeric(stations[column], errors="raise")
            if (
                not np.isfinite(stations[column]).all()
                or stations[column].abs().gt(limit).any()
            ):
                raise ValueError(f"stations has invalid {column}")
    return gauge, split, stations


def target_rows(gauge_df, split_dict):
    """Every observed test station-hour, never a filled missing hour."""
    eligible = validate_split(split_dict)
    gauge = validate_gauges(gauge_df)
    missing = eligible - set(gauge.station_id)
    if missing:
        raise ValueError(f"split stations have no observations: {sorted(missing)}")
    parts = []
    for fold in split_dict["folds"]:
        part = gauge.loc[
            gauge.station_id.isin(fold["ml"]["test"]),
            ["station_id", "timestamp", "rainfall_mm"],
        ].copy()
        if part.empty:
            raise ValueError(f"fold {fold['fold_index']} has no observed test targets")
        part.insert(0, "fold", fold["fold_index"])
        parts.append(part.rename(columns={"rainfall_mm": "actual_mm"}))
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(KEY_COLUMNS)
        .reset_index(drop=True)
    )


def validate_prediction_table(predictions):
    """Structural validation also used by standalone scoring without upstream files."""
    if list(predictions.columns) != PREDICTION_COLUMNS:
        raise ValueError(f"prediction columns must be exactly {PREDICTION_COLUMNS}")
    result = predictions.copy()
    if result.empty:
        raise ValueError("predictions table is empty")
    _station_ids(result.station_id, "predictions")
    if (
        result.method.isna().any()
        or not result.method.map(
            lambda value: isinstance(value, str) and bool(value.strip())
        ).all()
    ):
        raise ValueError("prediction method must be nonempty")
    result["timestamp"] = utc_hours(result.timestamp)
    if (
        not pd.api.types.is_integer_dtype(result.fold)
        or not result.fold.between(0, 4).all()
    ):
        raise ValueError("prediction fold must be an integer from 0 through 4")
    for column in ("predicted_mm", "actual_mm"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all() or result[column].lt(0).any():
            raise ValueError(f"{column} must be finite and nonnegative")
    if result.duplicated(["method", *KEY_COLUMNS]).any():
        raise ValueError("duplicate prediction target rows")
    order = ["method", *KEY_COLUMNS]
    if (
        not result[order]
        .reset_index(drop=True)
        .equals(result.sort_values(order)[order].reset_index(drop=True))
    ):
        raise ValueError(
            "predictions must be sorted by method, fold, station_id, timestamp"
        )
    return result


def validate_predictions(predictions_df, gauge_df, split_dict, method=None):
    predictions = validate_prediction_table(predictions_df)
    expected = target_rows(gauge_df, split_dict).set_index(KEY_COLUMNS).actual_mm
    if method is not None and set(predictions.method) != {method}:
        raise ValueError(f"expected only predictions from method {method}")
    for name, rows in predictions.groupby("method", sort=True):
        actual = rows.set_index(KEY_COLUMNS).actual_mm
        missing, extra = (
            expected.index.difference(actual.index),
            actual.index.difference(expected.index),
        )
        if len(missing) or len(extra):
            raise ValueError(
                f"{name}: target mismatch ({len(missing)} missing, {len(extra)} extra)"
            )
        if not np.allclose(actual.reindex(expected.index), expected, rtol=0, atol=1e-7):
            raise ValueError(f"{name}: actual_mm differs from canonical gauge_hourly")

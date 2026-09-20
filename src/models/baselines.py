"""Baselines read only the supplied fold's training gauges."""

import numpy as np
import pandas as pd

from .contracts import (
    PREDICTION_COLUMNS,
    target_rows,
    utc_hours,
    validate_gauges,
    validate_split,
)


def haversine_km(target, sources):
    target_rad = np.radians(np.asarray(target, dtype=float))
    source_rad = np.radians(np.asarray(sources, dtype=float))
    delta = source_rad - target_rad
    a = (
        np.sin(delta[:, 0] / 2) ** 2
        + np.cos(target_rad[0])
        * np.cos(source_rad[:, 0])
        * np.sin(delta[:, 1] / 2) ** 2
    )
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def predict_at_targets(
    method, observations, stations, split, targets, *, idw_power=2.0
):
    """Separate target identity from observations, so hiding labels cannot affect predictions."""
    if method not in ("network_mean", "idw"):
        raise ValueError(f"unsupported baseline: {method}")
    if not np.isfinite(idw_power) or idw_power <= 0:
        raise ValueError("idw_power must be positive and finite")
    validate_split(split)
    observations = validate_gauges(observations)
    targets = targets.copy()
    if not {"fold", "station_id", "timestamp", "actual_mm"}.issubset(targets.columns):
        raise ValueError("baseline targets lack required columns")
    if (
        not pd.api.types.is_integer_dtype(targets.fold)
        or not targets.fold.between(0, 4).all()
    ):
        raise ValueError("baseline targets must use integer folds 0 through 4")
    targets["timestamp"] = utc_hours(targets.timestamp)
    if targets.duplicated(["fold", "station_id", "timestamp"]).any():
        raise ValueError("duplicate baseline targets")
    coordinates = None
    if method == "idw":
        if stations is None:
            raise ValueError("IDW requires the standard station coordinates")
        coordinates = stations.set_index("station_id")[["latitude", "longitude"]]
        if (
            not coordinates.index.is_unique
            or not np.isfinite(coordinates.to_numpy()).all()
        ):
            raise ValueError("IDW requires unique stations and finite coordinates")
    parts = []
    for fold in split["folds"]:
        fold_targets = targets.loc[targets.fold.eq(fold["fold_index"])]
        if fold_targets.empty:
            continue
        if not set(fold_targets.station_id).issubset(fold["ml"]["test"]):
            raise ValueError("baseline targets must belong to the fold test stations")
        visible = observations.loc[observations.station_id.isin(fold["ml"]["train"])]
        if method == "network_mean":
            # One mean per fold-hour avoids repeating the gauge matrix for every target.
            hourly = visible.groupby("timestamp").rainfall_mm.mean()
            part = fold_targets.copy()
            part["predicted_mm"] = part.timestamp.map(hourly)
            if part.predicted_mm.isna().any():
                bad_hour = part.loc[part.predicted_mm.isna(), "timestamp"].iloc[0]
                raise ValueError(
                    f"BLOCKED: fold {fold['fold_index']} has no observed training gauge at {bad_hour}"
                )
            part["method"] = method
            parts.append(part[PREDICTION_COLUMNS])
            continue
        wide = visible.pivot(
            index="timestamp", columns="station_id", values="rainfall_mm"
        )
        wide = wide.reindex(columns=fold["ml"]["train"])
        for station_id, station_targets in fold_targets.groupby(
            "station_id", sort=True
        ):
            values = wide.reindex(pd.DatetimeIndex(station_targets.timestamp)).to_numpy(
                dtype=float
            )
            observed = np.isfinite(values)
            if not observed.any(axis=1).all():
                bad_hour = station_targets.timestamp.iloc[
                    np.flatnonzero(~observed.any(axis=1))[0]
                ]
                raise ValueError(
                    f"BLOCKED: fold {fold['fold_index']} has no observed training gauge at {bad_hour}"
                )
            distances = haversine_km(
                coordinates.loc[station_id], coordinates.loc[wide.columns]
            )
            colocated = observed & (distances == 0)[None, :]
            weights = np.zeros_like(distances)
            # Scale distances first to keep inverse powers numerically stable.
            nonzero = distances > 0
            if nonzero.any():
                weights[nonzero] = (
                    distances[nonzero].min() / distances[nonzero]
                ) ** idw_power
            hourly_weights = observed * weights
            has_colocated = colocated.any(axis=1)
            hourly_weights[has_colocated] = colocated[has_colocated]
            totals = hourly_weights.sum(axis=1)
            if (totals <= 0).any():
                raise ValueError("IDW has no finite positive gauge weights")
            predictions = (np.nan_to_num(values) * hourly_weights).sum(axis=1) / totals
            part = station_targets.copy()
            part["method"] = method
            part["predicted_mm"] = np.maximum(predictions, 0)
            parts.append(part[PREDICTION_COLUMNS])
    if not parts:
        raise ValueError("baseline has no targets")
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["method", "fold", "station_id", "timestamp"])
        .reset_index(drop=True)
    )


def predict_baseline(method, gauge_df, stations_df, split_dict, *, idw_power=2.0):
    targets = target_rows(gauge_df, split_dict)
    return predict_at_targets(
        method, gauge_df, stations_df, split_dict, targets, idw_power=idw_power
    )

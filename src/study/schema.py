"""Validated domain records for Component 03 study splitting."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
# Strictly positive: a fold with no validation stations leaves Component 06 with
# nothing to select on, and train_fused.py refuses it.
PositiveFraction = Annotated[float, Field(gt=0.0, lt=1.0)]


@dataclass(frozen=True, slots=True)
class StationRoles:
    """Train, validation and test station IDs for one fold."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StudyFold:
    """One fold in Component 03's required nested output shape."""

    fold_index: int
    ml: StationRoles


@dataclass(frozen=True, slots=True)
class StationExclusion:
    """One station Component 02 ruled out of the study, with its recorded cause."""

    station_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class StudySplitOutput:
    """The complete Component 03 split artifact before JSON serialisation.

    ``excluded`` names every ineligible station and its reason, so the artifact
    explains its own station count without the caller needing the ``stations.csv``
    that produced it.
    """

    name: str
    created: str
    stations_file: str
    n_eligible: int
    seed: int
    folds: tuple[StudyFold, ...]
    excluded: tuple[StationExclusion, ...]


class SplitConfig(BaseModel):
    """Settings from ``config/splits/sydney_5fold.yaml``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    stations_file: NonEmptyString
    n_folds: PositiveInt
    validation_fraction: PositiveFraction
    n_clusters: PositiveInt
    output: NonEmptyString


class SeedConfig(BaseModel):
    """Shared reproducibility settings from ``config/seeds.yaml``."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", strict=True)

    split_seed: int
    model_seed: int


class StationRecord(BaseModel):
    """One validated station from Component 02's standard dataset."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
    )

    station_id: NonEmptyString
    station_no: NonEmptyString
    name: NonEmptyString
    latitude: Latitude
    longitude: Longitude
    hours_observed: NonNegativeInt
    use_in_study: bool
    exclusion_reason: str

    @model_validator(mode="after")
    def validate_exclusion(self) -> Self:
        """Included stations have no exclusion; rejected stations explain why."""
        if self.use_in_study and self.exclusion_reason:
            raise ValueError("exclusion_reason must be empty when use_in_study is true")
        if not self.use_in_study and not self.exclusion_reason:
            raise ValueError("exclusion_reason is required when use_in_study is false")
        return self

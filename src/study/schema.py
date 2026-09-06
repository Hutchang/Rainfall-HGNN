"""Validated domain records for Component 03 study splitting."""

from dataclasses import dataclass
from typing import Annotated, ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

NonEmptyString = Annotated[str, Field(min_length=1)]
Latitude = Annotated[float, Field(ge=-90.0, le=90.0)]
Longitude = Annotated[float, Field(ge=-180.0, le=180.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


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

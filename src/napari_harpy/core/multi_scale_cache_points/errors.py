from __future__ import annotations


class PointsSourceValidationError(ValueError):
    """Base error for invalid multiscale-cache point sources."""

    default_code = "points_source_validation"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.code = self.default_code if code is None else code


class PointsSourceResolutionError(PointsSourceValidationError):
    """Report that a logical points element cannot resolve to a supported source."""

    default_code = "points_source_resolution"


class ParquetMetadataValidationError(PointsSourceValidationError):
    """Report invalid Parquet source inventory, schema, or file metadata."""

    default_code = "parquet_metadata_validation"

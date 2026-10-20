"""Scoring versions are selected by the immutable run configuration."""
from .comparison import COMPARISON_VERSION as LEGACY_COMPARISON_VERSION

COVERAGE_COMPARISON_VERSION = "expected_coverage_v3"
SUPPORTED_COMPARISON_VERSIONS = {
    LEGACY_COMPARISON_VERSION,
    COVERAGE_COMPARISON_VERSION,
}

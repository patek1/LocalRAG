"""
Configuration package for the Local RAG Pipeline.

This package provides configuration classes and settings for managing
multiple datasets, paths, and pipeline parameters.
"""

from .global_settings import Settings, settings
from .dataset_config import DatasetConfig, FieldMapping

__all__ = [
    "Settings",
    "settings",
    "DatasetConfig",
    "FieldMapping",
] 
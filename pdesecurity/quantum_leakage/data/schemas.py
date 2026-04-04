"""
Shared dataset schemas and small data-container classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class ExperimentDataset:
    """
    Generic dataset container for a single tabular dataset.
    """
    df: pd.DataFrame
    task_type: str | None = None


@dataclass
class BoundaryDataset:
    """
    Boundary-topology experiment dataset.
    """
    df: pd.DataFrame


@dataclass
class ScaleDataset:
    """
    Scale leakage experiment dataset.
    """
    df: pd.DataFrame


@dataclass
class VeracityDataset:
    """
    Accuracy/veracity leakage dataset.
    """
    df: pd.DataFrame
    task_type: Literal["binary", "ordinal"]


@dataclass
class DriftDatasetBundle:
    """
    Container for clean datasets used in drift experiments.
    """
    boundary_df: pd.DataFrame
    scale_df: pd.DataFrame

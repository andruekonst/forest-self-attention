import numpy as np
from sklearn.ensemble import (
        RandomForestRegressor,
        RandomForestClassifier,
        ExtraTreesRegressor,
        ExtraTreesClassifier,
)
from enum import Enum, auto
from typing import NamedTuple


class ForestKind(Enum):
    """Forest kind.

    Attributes:
        RANDOM: Random Forest.
        EXTRA: Extra Trees.

    """
    RANDOM = auto()
    EXTRA = auto()

    @staticmethod
    def from_name(name: str) -> 'ForestKind':
        """Get a ForestKind instance by name.

        Args:
            name: Forest Kind name.

        Returns:
            Forest Kind.
        """
        if name == 'random':
            return ForestKind.RANDOM
        elif name == 'extra':
            return ForestKind.EXTRA
        else:
            raise ValueError('Wrong forest kind: "{name}".')


class TaskType(Enum):
    """Machine learning task type.

    Attributes:
        CLASSIFICATION: binary or multiclass classification.
        REGRESSION: one-dimensional regression.

    """
    CLASSIFICATION = auto()
    REGRESSION = auto()


class ForestType(NamedTuple):
    """Forest type.

    Attributes:
        kind: Forest kind.
        task: Machine learning task.

    """
    kind: ForestKind
    task: TaskType



FORESTS = {
    ForestType(ForestKind.RANDOM, TaskType.REGRESSION): RandomForestRegressor,
    ForestType(ForestKind.RANDOM, TaskType.CLASSIFICATION): RandomForestClassifier,
    ForestType(ForestKind.EXTRA, TaskType.REGRESSION): ExtraTreesRegressor,
    ForestType(ForestKind.EXTRA, TaskType.CLASSIFICATION): ExtraTreesClassifier,
}
"""Forest type to class mapping.
"""


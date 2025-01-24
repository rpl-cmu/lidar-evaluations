from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class Feature(Enum):
    Point = 0
    Edge = 1
    Planar = 2
    # Pseudo_Planar = 3


class Initialization(Enum):
    GroundTruth = 0
    ConstantVelocity = 1


@dataclass
class ExperimentParams:
    dataset: str
    features: Sequence[Feature]
    init: Initialization = Initialization.GroundTruth

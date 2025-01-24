from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence
from evalio.cli.parser import DatasetBuilder

import loam


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
    output: Path
    dataset: str
    features: Sequence[Feature]
    init: Initialization = Initialization.GroundTruth

    def feature_params(self) -> loam.FeatureExtractionParams:
        feat_params = loam.FeatureExtractionParams()
        feat_params.neighbor_points = 4
        feat_params.number_sectors = 6
        feat_params.max_edge_feats_per_sector = 10
        feat_params.max_planar_feats_per_sector = 50

        feat_params.edge_feat_threshold = 50.0
        feat_params.planar_feat_threshold = 1.0

        feat_params.occlusion_thresh = 0.9
        feat_params.parallel_thresh = 0.01

        return feat_params

    def registration_params(self) -> loam.RegistrationParams:
        reg_params = loam.RegistrationParams()
        reg_params.max_iterations = 80
        return reg_params

    def build_dataset(self):
        return DatasetBuilder.parse(self.dataset)[0].build()

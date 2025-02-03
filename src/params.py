from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import loam
from evalio.cli.parser import DatasetBuilder


def uppers(s: str) -> str:
    return "".join(c for c in s if c.isupper())


class PrettyPrintEnum(Enum):
    def __repr__(self):
        return self.name

    def __str__(self) -> str:
        return self.name


class Feature(PrettyPrintEnum):
    Point = 0
    Edge = 1
    Planar = 2
    Pseudo_Planar = 3


class Initialization(PrettyPrintEnum):
    GroundTruth = 0
    ConstantVelocity = 1
    Identity = 2


class Dewarping(PrettyPrintEnum):
    Identity = 0
    ConstantVelocity = 1
    GroundTruth = 2


@dataclass
class ExperimentParams:
    name: str
    dataset: str
    features: list[Feature]
    init: Initialization = Initialization.GroundTruth
    dewarp: Dewarping = Dewarping.Identity

    def __post_init__(self):
        # Make sure features are valid
        assert len(self.features) > 0, "At least one feature must be selected"
        assert not (self.planar and self.pseudo_planar), (
            "Cannot have both planar and pseudo-planar features"
        )
        # Make sure dataset is valid
        DatasetBuilder.parse(self.dataset)

    @property
    def point(self) -> bool:
        return Feature.Point in self.features

    @property
    def edge(self) -> bool:
        return Feature.Edge in self.features

    @property
    def planar(self) -> bool:
        return Feature.Planar in self.features

    @property
    def pseudo_planar(self) -> bool:
        return Feature.Pseudo_Planar in self.features

    def feature_params(self) -> loam.FeatureExtractionParams:
        feat_params = loam.FeatureExtractionParams()
        feat_params.neighbor_points = 4
        feat_params.number_sectors = 6
        feat_params.max_edge_feats_per_sector = 10
        feat_params.max_planar_feats_per_sector = 50
        feat_params.max_point_feats_per_sector = 50

        feat_params.edge_feat_threshold = 50.0
        feat_params.planar_feat_threshold = 1.0

        feat_params.occlusion_thresh = 0.9
        feat_params.parallel_thresh = 0.01

        return feat_params

    def registration_params(self) -> loam.RegistrationParams:
        reg_params = loam.RegistrationParams()
        reg_params.max_iterations = 20
        # TODO: This could probably use some tuning!
        reg_params.pseudo_plane_normal_epsilon = 0.1

        if self.pseudo_planar:
            reg_params.planar_version = loam.PlanarVersion.PSEUDO_PLANAR

        return reg_params

    def build_dataset(self):
        return DatasetBuilder.parse(self.dataset)[0].build()

    def output_file(self, directory: Path) -> Path:
        return directory / self.dataset / f"{self.name}.csv"

    def debug_file(self, directory: Path) -> Path:
        return directory / self.dataset / f"{self.name}.log"

    def short_info(self) -> str:
        ds, seq = self.dataset.split("/")
        ds = ds[:3] + ds[-2:] + "/" + seq[:2]
        f = "/".join(f.name[:2] for f in self.features)
        i = uppers(self.init.name)
        d = uppers(self.dewarp.name)

        return f"ds: {ds}, feat: {f}, init: {i}, dew: {d}"

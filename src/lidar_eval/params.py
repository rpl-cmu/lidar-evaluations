from enum import Enum
from pathlib import Path

from serde import serde

import loam
from evalio.cli.parser import DatasetBuilder


class PrettyPrintEnum(Enum):
    def __repr__(self):
        return self.name

    def __str__(self) -> str:
        return self.name


class Feature(PrettyPrintEnum):
    Point = 0
    Edge = 1
    Planar = 2
    PointAll = 3
    PlanarAll = 4


class Initialization(PrettyPrintEnum):
    GroundTruth = 0
    Imu = 1
    ConstantVelocity = 2
    Identity = 3


class Dewarp(PrettyPrintEnum):
    Identity = 0
    ConstantVelocity = 1
    Imu = 2
    GroundTruthConstantVelocity = 3


class Curvature(PrettyPrintEnum):
    Loam = 0
    Eigen = 1
    Eigen_NN = 2

    def into(self) -> loam.Curvature:
        return loam.Curvature(self.value)


@serde
class ExperimentParams:
    name: str
    dataset: str
    # features
    features: list[Feature]
    curvature: Curvature = Curvature.Loam
    curvature_planar_threshold: float = 1.0
    # registration
    pseudo_planar_epsilon: float = 0.0
    use_plane_to_plane: bool = False
    init: Initialization = Initialization.ConstantVelocity
    dewarp: Dewarp = Dewarp.Identity

    def __post_init__(self):
        # Make sure features are valid
        assert len(self.features) > 0, "At least one feature must be selected"
        assert not (self.point_all and len(self.features) > 1), (
            "PointAll cannot be combined with other features"
        )
        assert not (self.planar_all and len(self.features) > 1), (
            "PlanarAll cannot be combined with other features"
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
    def planar(self):
        return Feature.Planar in self.features

    @property
    def point_all(self):
        return Feature.PointAll in self.features

    @property
    def planar_all(self):
        return Feature.PlanarAll in self.features

    def feature_params(self) -> loam.FeatureExtractionParams:
        feat_params = loam.FeatureExtractionParams()
        feat_params.neighbor_points = 4
        feat_params.number_sectors = 6
        feat_params.max_edge_feats_per_sector = 10
        feat_params.max_planar_feats_per_sector = 50
        feat_params.max_point_feats_per_sector = 50

        feat_params.edge_feat_threshold = 50.0
        feat_params.occlusion_thresh = 0.9
        feat_params.parallel_thresh = 0.01

        feat_params.curvature_type = self.curvature.into()
        feat_params.planar_feat_threshold = self.curvature_planar_threshold

        return feat_params

    def registration_params(self) -> loam.RegistrationParams:
        reg_params = loam.RegistrationParams()
        reg_params.max_iterations = 20

        reg_params.pseudo_plane_normal_epsilon = self.pseudo_planar_epsilon
        reg_params.use_plane_to_plane = self.use_plane_to_plane

        return reg_params

    def build_dataset(self):
        return DatasetBuilder.parse(self.dataset)[0].build()

    def output_file(self, directory: Path) -> Path:
        return directory / self.dataset / f"{self.name}.csv"

    def debug_file(self, directory: Path) -> Path:
        return directory / self.dataset / f"{self.name}.log"

    def short_info(self) -> str:
        def short(f: str) -> str:
            if f.isdigit():
                return f[-2:]
            else:
                return f[:1]

        temp = self.dataset.replace("-", "_")
        ds, seq = temp.split("/")
        ds = (
            "".join(short(d) for d in ds.split("_"))
            + "/"
            + "".join(short(d) for d in seq.split("_"))
        )
        return f"ds: {ds:>9}, {self.name[:20]:>20}"

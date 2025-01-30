from dataclasses import dataclass
from pathlib import Path
from evalio.types import SE3, SO3, Stamp

from functools import cached_property
from serde.yaml import from_yaml

from params import ExperimentParams, Feature
import csv
import numpy as np

from typing import Optional

from tabulate import tabulate


@dataclass(kw_only=True)
class Ate:
    trans: float
    rot: float


@dataclass(kw_only=True)
class Experiment:
    params: ExperimentParams
    stamps: list[Stamp]
    poses: list[SE3]
    gt: list[SE3]
    features: dict[Feature, np.ndarray]

    def __post_init__(self):
        assert len(self.stamps) == len(self.poses)
        assert len(self.stamps) == len(self.gt)
        assert len(self.stamps) == len(self.features[Feature.Point])
        assert len(self.stamps) == len(self.features[Feature.Edge])
        assert len(self.stamps) == len(self.features[Feature.Planar])

    def __len__(self) -> int:
        return len(self.stamps)

    @cached_property
    def iterated_gt(self) -> list[SE3]:
        iterated_gt = [self.gt[0]]
        for i in range(1, len(self.gt)):
            iterated_gt.append(iterated_gt[-1] * self.gt[i])
        return iterated_gt

    @cached_property
    def iterated_poses(self) -> list[SE3]:
        iterated_poses = [self.poses[0]]
        for i in range(1, len(self.poses)):
            iterated_poses.append(iterated_poses[-1] * self.poses[i])
        return iterated_poses

    def iterated_ate(self) -> Ate:
        assert len(self.iterated_gt) == len(self.iterated_poses)
        assert len(self.iterated_gt) == len(self.stamps)

        error_t = 0.0
        error_r = 0.0
        for gt, pose in zip(self.iterated_gt, self.iterated_poses):
            error_t += float(np.linalg.norm(gt.trans - pose.trans))
            error_r += float(np.linalg.norm((gt.rot * pose.rot.inverse()).log()))

        error_t /= len(self.iterated_gt)
        error_r /= len(self.iterated_gt)

        return Ate(rot=error_r, trans=error_t)

    def ate(self) -> Ate:
        """
        Computes the Absolute Trajectory Error
        """
        assert len(self.gt) == len(self.poses)
        assert len(self.gt) == len(self.stamps)

        error_t = 0.0
        error_r = 0.0
        for gt, pose in zip(self.gt, self.poses):
            error_t += float(np.linalg.norm(gt.trans - pose.trans))
            error_r += float(np.linalg.norm((gt.rot * pose.rot.inverse()).log()))

        error_t /= len(self.gt)
        error_r /= len(self.gt)

        return Ate(rot=error_r, trans=error_t)

    def get_feature(self, feature: Feature) -> np.ndarray:
        return self.features[feature]


def load(path: Path) -> Experiment:
    fieldnames = [
        "sec",
        "x",
        "y",
        "z",
        "qx",
        "qy",
        "qz",
        "qw",
        "gt_x",
        "gt_y",
        "gt_z",
        "gt_qx",
        "gt_qy",
        "gt_qz",
        "gt_qw",
        "point",
        "edge",
        "planar",
    ]

    poses = []
    gts = []
    stamps = []
    points = []
    edges = []
    planars = []

    with open(path) as file:
        metadata_filter = filter(lambda row: row[0] == "#", file)
        metadata_list = [row[1:].strip() for row in metadata_filter]
        metadata_list.pop(-1)
        metadata_str = "\n".join(metadata_list)
        # remove the header row
        params = from_yaml(ExperimentParams, metadata_str)

        # Load trajectory
        file.seek(0)
        csvfile = filter(lambda row: row[0] != "#", file)
        reader = csv.DictReader(csvfile, fieldnames=fieldnames)
        for line in reader:
            r = SO3(
                qw=float(line["qw"]),
                qx=float(line["qx"]),
                qy=float(line["qy"]),
                qz=float(line["qz"]),
            )
            t = np.array([float(line["x"]), float(line["y"]), float(line["z"])])
            pose = SE3(r, t)

            gt_r = SO3(
                qw=float(line["gt_qw"]),
                qx=float(line["gt_qx"]),
                qy=float(line["gt_qy"]),
                qz=float(line["gt_qz"]),
            )
            gt_t = np.array(
                [float(line["gt_x"]), float(line["gt_y"]), float(line["gt_z"])]
            )
            gt = SE3(gt_r, gt_t)

            if "nsec" not in fieldnames:
                stamp = Stamp.from_sec(float(line["sec"]))
            elif "sec" not in fieldnames:
                stamp = Stamp.from_nsec(int(line["nsec"]))
            else:
                stamp = Stamp(sec=int(line["sec"]), nsec=int(line["nsec"]))

            points.append(int(line["point"]))
            edges.append(int(line["edge"]))
            planars.append(int(line["planar"]))

            poses.append(pose)
            gts.append(gt)
            stamps.append(stamp)

    return Experiment(
        params=params,
        stamps=stamps,
        poses=poses,
        gt=gts,
        features={
            Feature.Point: np.asarray(points),
            Feature.Edge: np.asarray(edges),
            Feature.Planar: np.asarray(planars),
        },
    )


def eval_dataset(dir: Path, visualize: bool, sort: Optional[str]):
    # Load all experiments
    experiments = []
    for file_path in dir.glob("*.csv"):
        traj = load(file_path)
        experiments.append(traj)

    header = [
        "AEt",
        "AEr",
        "ATEt",
        "ATEr",
        "features",
        "init",
        "dewarp",
        "point",
        "edge",
        "planar",
    ]

    results = []
    for exp in experiments:
        ate = exp.ate()
        ate_iter = exp.iterated_ate()
        results.append(
            [
                ate.trans,
                ate.rot,
                ate_iter.trans,
                ate_iter.rot,
                exp.params.features,
                exp.params.init,
                exp.params.dewarp,
                exp.get_feature(Feature.Point).mean(),
                exp.get_feature(Feature.Edge).mean(),
                exp.get_feature(Feature.Planar).mean(),
            ]
        )

    if sort is None:
        pass
    elif sort.lower() == "aet":
        results = sorted(results, key=lambda x: x[0])
    elif sort.lower() == "aer":
        results = sorted(results, key=lambda x: x[1])
    elif sort.lower() == "atet":
        results = sorted(results, key=lambda x: x[2])
    elif sort.lower() == "ater":
        results = sorted(results, key=lambda x: x[3])

    print(tabulate(results, headers=header, tablefmt="fancy"))


def _contains_dir(directory: Path) -> bool:
    return any(directory.is_dir() for directory in directory.glob("*"))


def eval(directories: list[Path], visualize: bool, sort: Optional[str] = None):
    # Collect all bottom level directories
    bottom_level_dirs = []
    for directory in directories:
        for subdir in directory.glob("**/"):
            if not _contains_dir(subdir):
                bottom_level_dirs.append(subdir)

    for d in bottom_level_dirs:
        print(f"Results for {d}")
        eval_dataset(d, visualize, sort)
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("-s", "--sort", type=str)
    args = parser.parse_args()

    eval(args.directories, args.visualize, args.sort)

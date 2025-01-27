from dataclasses import dataclass
from pathlib import Path
from evalio.types import SE3, SO3, Stamp

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
        assert len(self.stamps) == len(self.features[Feature.Edge])
        assert len(self.stamps) == len(self.features[Feature.Planar])

    def __len__(self) -> int:
        return len(self.stamps)

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
        "edge",
        "planar",
    ]

    poses = []
    gts = []
    stamps = []
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
        features={Feature.Edge: np.asarray(edges), Feature.Planar: np.asarray(planars)},
    )


def eval_dataset(dir: Path, visualize: bool, sort: Optional[str]):
    # Load all experiments
    experiments = []
    for file_path in dir.glob("*.csv"):
        traj = load(file_path)
        experiments.append(traj)

    header = ["ATEt", "ATEr", "features", "init", "dewarp", "edge", "planar"]

    results = []
    for exp in experiments:
        ate = exp.ate()
        results.append(
            [
                ate.trans,
                ate.rot,
                exp.params.features,
                exp.params.init,
                exp.params.dewarp,
                exp.get_feature(Feature.Edge).mean(),
                exp.get_feature(Feature.Planar).mean(),
            ]
        )

    if sort is None:
        pass
    elif sort.lower() == "atet":
        results = sorted(results, key=lambda x: x[0])
    elif sort.lower() == "ater":
        results = sorted(results, key=lambda x: x[1])

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

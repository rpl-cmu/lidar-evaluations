import csv
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, cast

import numpy as np
from serde.yaml import from_yaml
from tabulate import tabulate
import polars as pl

from evalio.types import SE3, SO3, Stamp
from params import ExperimentParams, Feature
from wrappers import Rerun


@dataclass(kw_only=True)
class Metric:
    trans: float
    rot: float


@dataclass(kw_only=True)
class Error:
    trans: np.ndarray
    rot: np.ndarray

    def mean(self) -> Metric:
        return Metric(rot=self.rot.mean(), trans=self.trans.mean())

    def sse(self) -> Metric:
        return Metric(
            rot=float(np.sqrt(self.rot @ self.rot)),
            trans=float(np.sqrt(self.trans @ self.trans)),
        )

    def median(self) -> Metric:
        return Metric(
            rot=cast(float, np.median(self.rot)),
            trans=cast(float, np.median(self.trans)),
        )


@dataclass(kw_only=True)
class ExperimentResult:
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

    @staticmethod
    def _compute_metric(gts: list[SE3], poses: list[SE3]) -> Error:
        assert len(gts) == len(poses)

        error_t = np.zeros(len(gts))
        error_r = np.zeros(len(gts))
        for i, (gt, pose) in enumerate(zip(gts, poses)):
            delta = gt.inverse() * pose
            error_t[i] = np.sqrt(delta.trans @ delta.trans)
            r_diff = delta.rot.log()
            error_r[i] = np.sqrt(r_diff @ r_diff)

        return Error(rot=error_r, trans=error_t)

    def ate(self) -> Error:
        return self._compute_metric(self.iterated_gt, self.iterated_poses)

    def rte(self) -> Error:
        return self._compute_metric(self.gt, self.poses)

    def get_feature(self, feature: Feature) -> np.ndarray:
        return self.features[feature]

    @staticmethod
    def load(path: Path) -> "ExperimentResult":
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

        return ExperimentResult(
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


def eval_dataset(
    dir: Path,
    visualize: bool = False,
    sort: Optional[str] = None,
    print_results: bool = True,
):
    # Load all experiments
    experiments: list[ExperimentResult] = []
    if not dir.is_dir():
        experiments.append(ExperimentResult.load(dir))
    else:
        for file_path in dir.glob("*.csv"):
            traj = ExperimentResult.load(file_path)
            experiments.append(traj)

    results = []
    for exp in experiments:
        rte = exp.rte().mean()
        ate = exp.ate().mean()
        r = asdict(exp.params)
        r.update(
            {
                "RTEt": rte.trans,
                "RTEr": rte.rot,
                "ATEt": ate.trans,
                "ATEr": ate.rot,
                "point": exp.get_feature(Feature.Point).mean(),
                "edge": exp.get_feature(Feature.Edge).mean(),
                "planar": exp.get_feature(Feature.Planar).mean(),
                "length": len(exp),
            }
        )
        results.append(r)

        if visualize:
            # TODO: For some reason we're loosing some of the gt plots.. can't trace down why
            rr = Rerun(exp.params.dataset)
            rr.log(
                exp.params.name,
                exp.iterated_poses,
                color=cast(list[int], np.random.randint(0, 255, 3).tolist()),
                static=True,
            )
            rr.log("gt", exp.iterated_gt, color=[255, 0, 0], static=True)

    if sort is not None:
        results = sorted(results, key=lambda x: x[sort])

    # Remove any columns that are constant if we're printing
    all_keys = list(results[0].keys())
    if print_results and len(results) > 1:
        for key in all_keys:
            if all(x[key] == results[0][key] for x in results):
                for item in results:
                    del item[key]

    if print_results:
        print(tabulate(results, headers="keys", tablefmt="fancy"))

    return results


def _contains_dir(directory: Path) -> bool:
    return any(directory.is_dir() for directory in directory.glob("*"))


def eval(directories: list[Path], visualize: bool, sort: Optional[str] = None):
    # TODO: Make this cache things as well
    # Collect all bottom level directories
    bottom_level_dirs = []
    for directory in directories:
        if not directory.is_dir():
            bottom_level_dirs.append(directory)
        for subdir in directory.glob("**/"):
            if not _contains_dir(subdir):
                bottom_level_dirs.append(subdir)

    for d in bottom_level_dirs:
        print(f"Results for {d}")
        eval_dataset(d, visualize, sort)
        print()


def compute_cache_stats(directory: Path) -> pl.DataFrame:
    df_file = directory.parent / (directory.name + ".csv")

    if not df_file.exists():
        bottom_level_dirs = []
        for subdir in directory.glob("**/"):
            if not _contains_dir(subdir):
                bottom_level_dirs.append(subdir)

        all_data = []
        for dataset in bottom_level_dirs:
            print(dataset)
            all_data += eval_dataset(dataset, print_results=False)

        for d in all_data:
            del d["features"]
            d["curvature"] = d["curvature"].name
            d["init"] = d["init"].name
            d["dewarp"] = d["dewarp"].name

        df = pl.DataFrame(all_data)
        df.write_csv(df_file)
    else:
        df = pl.read_csv(df_file)

    return df


if __name__ == "__main__":
    import argparse
    import time

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("-v", "--visualize", action="store_true")
    parser.add_argument("-s", "--sort", type=str)
    args = parser.parse_args()

    eval(args.directories, args.visualize, args.sort)
    print(f"Elapsed time: {time.time() - start:.2f}s")

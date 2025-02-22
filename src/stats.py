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
    # Shape: (n, 3)
    trans: np.ndarray
    rot: np.ndarray

    @cached_property
    def rot_e(self) -> np.ndarray:
        return np.linalg.norm(self.rot, axis=1)

    @cached_property
    def trans_e(self) -> np.ndarray:
        return np.linalg.norm(self.trans, axis=1)

    def mean(self) -> Metric:
        return Metric(rot=self.rot_e.mean(), trans=self.trans_e.mean())

    def sse(self) -> Metric:
        length = len(self.rot_e)
        return Metric(
            rot=float(np.sqrt(self.rot_e @ self.rot_e / length)),
            trans=float(np.sqrt(self.trans_e @ self.trans_e / length)),
        )

    def median(self) -> Metric:
        return Metric(
            rot=cast(float, np.median(self.rot_e)),
            trans=cast(float, np.median(self.trans_e)),
        )


@dataclass(kw_only=True)
class ExperimentResult:
    params: ExperimentParams
    stamps: list[Stamp]
    poses: list[SE3]
    gt: list[SE3]
    iters: np.ndarray
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

        error_t = np.zeros((len(gts), 3))
        error_r = np.zeros((len(gts), 3))
        for i, (gt, pose) in enumerate(zip(gts, poses)):
            delta = gt.inverse() * pose
            error_t[i] = delta.trans
            r_diff = delta.rot.log()
            error_r[i] = r_diff * 180 / np.pi

        return Error(rot=error_r, trans=error_t)

    def ate(self) -> Error:
        return self._compute_metric(self.iterated_gt, self.iterated_poses)

    def rte(self) -> Error:
        return self._compute_metric(self.gt, self.poses)

    def windowed_rte(self, window: int = 10) -> Error:
        window_deltas_poses = []
        window_deltas_gts = []
        for i in range(len(self.gt) - window):
            window_deltas_poses.append(
                self.iterated_poses[i].inverse() * self.iterated_poses[i + window]
            )
            window_deltas_gts.append(
                self.iterated_gt[i].inverse() * self.iterated_gt[i + window]
            )

        return self._compute_metric(window_deltas_gts, window_deltas_poses)

    def distance_wrte(self, window: float = 2) -> Error:
        def norm2(a, b):
            diff = a - b
            return diff @ diff

        window2 = window * window
        window_deltas_poses = []
        window_deltas_gts = []
        end_idx = 1
        end_idx_prev = 0
        for i in range(len(self.gt)):
            start_pose = self.iterated_poses[i]
            start_gt = self.iterated_gt[i]
            while (
                end_idx < len(self.gt)
                and norm2(start_gt.trans, self.iterated_gt[end_idx].trans) < window2
            ):
                end_idx += 1

            if end_idx >= len(self.gt):
                break
            elif end_idx == end_idx_prev:
                continue

            assert end_idx > i, "Something went wrong, window is negative"

            window_deltas_poses.append(
                start_pose.inverse() * self.iterated_poses[end_idx]
            )
            window_deltas_gts.append(start_gt.inverse() * self.iterated_gt[end_idx])

            end_idx_prev = end_idx

        return self._compute_metric(window_deltas_gts, window_deltas_poses)

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
            "iter",
        ]

        poses = []
        gts = []
        stamps = []
        points = []
        edges = []
        planars = []
        iters = []

        with open(path) as file:
            metadata_filter = filter(lambda row: row[0] == "#", file)
            metadata_list = [row[1:].strip() for row in metadata_filter]
            metadata_list.pop(-1)
            metadata_str = "\n".join(metadata_list)
            # remove the header rows
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

                stamp = Stamp.from_sec(float(line["sec"]))

                this_iter = line["iter"]
                if this_iter is None:
                    this_iter = 0
                iters.append(int(this_iter))
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
            iters=np.asarray(iters),
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
        # TODO: sse or mean here?
        window100_rte = exp.windowed_rte(100).sse()
        window200_rte = exp.windowed_rte(200).sse()
        windowdist4_rte = exp.distance_wrte(2.0).sse()
        # rte = exp.rte().sse()
        # ate = exp.ate().sse()
        r = asdict(exp.params)
        r.update(
            {
                "w200_RTEt": window200_rte.trans,
                "w200_RTEr": window200_rte.rot,
                "w100_RTEt": window100_rte.trans,
                "w100_RTEr": window100_rte.rot,
                "w4_RTEt": windowdist4_rte.trans,
                "w4_RTEr": windowdist4_rte.rot,
                # "RTEt": rte.trans,
                # "RTEr": rte.rot,
                # "ATEt": ate.trans,
                # "ATEr": ate.rot,
                "point": exp.get_feature(Feature.Point).mean(),
                "edge": exp.get_feature(Feature.Edge).mean(),
                "planar": exp.get_feature(Feature.Planar).mean(),
                "iters": exp.iters.mean(),
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


def compute_cache_stats(directory: Path, force: bool = False) -> pl.DataFrame:
    df_file = directory.parent / (directory.name + ".csv")

    if not df_file.exists() or force:
        bottom_level_dirs = []
        for subdir in directory.glob("**/"):
            if not _contains_dir(subdir):
                bottom_level_dirs.append(subdir)

        all_data = []
        for dataset in bottom_level_dirs:
            print(dataset)
            all_data += eval_dataset(dataset, print_results=False)

        for d in all_data:
            d["features"] = ", ".join([f.name for f in d["features"]])
            d["curvature"] = d["curvature"].name
            d["init"] = d["init"].name
            d["dewarp"] = d["dewarp"].name

        df = pl.DataFrame(all_data)

        # ------------------------- Split dataset and sequences ------------------------- #
        datasets_pretty_names = {
            "newer_college_2020": "Newer College Stereo-Cam",
            "newer_college_2021": "Newer College Multi-Cam",
            "hilti_2022": "Hilti 2022",
            "oxford_spires": "Oxford Spires",
            "multi_campus_2024": "Multi-Campus",
            "helipr": "HeLiPR",
            "botanic_garden": "Botanic Garden",
        }

        def short(f: str) -> str:
            if f.isdigit():
                return f[-2:]
            else:
                return f[:1]

        def sequence_pretty_names(seq) -> str:
            seq = seq.replace("-", "_")
            return "".join(short(d) for d in seq.split("_"))

        df = (
            df.lazy()
            .with_columns(
                pl.col("dataset")
                .str.split("/")
                .list.get(0)
                .replace_strict(datasets_pretty_names)
                .alias("Dataset"),
                pl.col("dataset")
                .str.split("/")
                .list.get(1)
                .map_elements(
                    lambda seq: sequence_pretty_names(seq), return_dtype=pl.String
                )
                .alias("Trajectory"),
            )
            .collect()
        )

        df = df.sort(
            pl.col("Dataset").cast(pl.Enum(list(datasets_pretty_names.values()))),
            "Trajectory",
        )

        # ------------------------- Cleanup various names ------------------------- #
        dewarp_pretty_names = {
            "Identity": "None",
            "ConstantVelocity": "Constant Velocity",
            "Imu": "IMU",
        }
        df = df.with_columns(
            pl.col("dewarp").replace(dewarp_pretty_names).alias("Dewarp")
        )

        init_pretty_names = {
            "Identity": "Identity",
            "ConstantVelocity": "Constant Velocity",
            "Imu": "IMU",
        }
        df = df.with_columns(
            pl.col("init").replace(init_pretty_names).alias("Initialization")
        )

        df = df.with_columns(
            pl.when(pl.col("use_plane_to_plane"))
            .then(pl.lit("Plane-To-Plane"))
            .otherwise(pl.lit("Point-To-Plane"))
            .alias("Planar Type")
        )

        curv_pretty_names = {
            "Loam": "Classic",
            "Eigen": "Scanline-Eigen",
            "Eigen_NN": "NN-Eigen",
        }
        df = df.with_columns(
            pl.col("curvature").replace(curv_pretty_names).alias("Curvature")
        )

        df = df.rename({"features": "Features"})

        # ------------------------- Rename other columns ------------------------- #

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

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from serde.yaml import to_yaml

import loam
from evalio.types import SE3, LidarMeasurement, Stamp, Trajectory
from params import ExperimentParams

import os
import matplotlib.pyplot as plt

from itertools import chain, combinations

import argparse
import matplotlib
import seaborn as sns


def setup_plot():
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    return c


def parser(name: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{name} experiment")
    subparsers = parser.add_subparsers(dest="action")

    run_parser = subparsers.add_parser("run", help="Run the experiment")
    run_parser.add_argument(
        "-n",
        "--num_threads",
        type=int,
        default=20,
        help="Number of threads to use for multiprocessing",
    )

    plot_parser = subparsers.add_parser("plot", help="Plot the results")
    plot_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force recomputing results",
    )

    args = parser.parse_args()
    args.name = name
    return args


# For handling multiple tqdm progress bars
# https://stackoverflow.com/questions/56665639/fix-jumping-of-multiple-progress-bars-tqdm-in-python-multiprocessing


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


def is_remote() -> bool:
    return os.environ.get("SSH_CONNECTION") is not None


def plt_show(filename: str | Path):
    plt.savefig(filename)
    if not is_remote():
        plt.show()


@dataclass
class GroundTruthIterator:
    traj: Trajectory
    idx: int = 0

    def next(self, stamp: Stamp, tol: float = 1e-2) -> Optional[SE3]:
        # If our first ground truth is in the future, we'll have to skip for a bit
        if self.idx >= len(self.traj):
            return None
        elif self.traj.stamps[self.idx] > stamp:
            return None

        while (
            self.idx < len(self.traj.stamps)
            and self.traj.stamps[self.idx] - stamp < -tol
        ):
            self.idx += 1

        if self.idx >= len(self.traj):
            return None
        else:
            return self.traj.poses[self.idx]


class Writer:
    def __init__(self, path: Path, ep: ExperimentParams):
        file = ep.output_file(path)
        debug = ep.debug_file(path)
        file.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(file, "w")
        params = "# " + to_yaml(ep).replace("\n", "\n# ")
        self.file.write(params + "\n")
        self.file.write(
            "# timestamp, x, y, z, qx, qy, qz, qw, gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw, point, edge, planar\n"
        )
        self.writer = csv.writer(self.file)

        self.debug = open(debug, "w")

    def write(self, stamp: Stamp, pose: SE3, gt: SE3, feats: loam.LoamFeatures):
        self.writer.writerow(
            [
                stamp.to_sec(),
                pose.trans[0],
                pose.trans[1],
                pose.trans[2],
                pose.rot.qx,
                pose.rot.qy,
                pose.rot.qz,
                pose.rot.qw,
                gt.trans[0],
                gt.trans[1],
                gt.trans[2],
                gt.rot.qx,
                gt.rot.qy,
                gt.rot.qz,
                gt.rot.qw,
                len(feats.point_points),
                len(feats.edge_points),
                len(feats.planar_points),
            ]
        )

    def close(self):
        self.file.close()
        self.debug.close()

    def log(self, msg):
        self.debug.write(msg + "\n")

    def error(self, e):
        self.debug.write("ERROR: " + str(e) + "\n")


class Rerun:
    def __init__(self, name="lidar-features", ip="0.0.0.0:9876", to_hide=None):
        if to_hide is not None:
            if isinstance(to_hide, str):
                to_hide = [to_hide]
            blueprint = rrb.Spatial3DView(
                overrides={e: [rrb.components.Visible(False)] for e in to_hide}
            )
        else:
            blueprint = None

        rr.init(name)
        rr.connect_tcp(ip, default_blueprint=blueprint)

        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
        self.history: dict[str, tuple[list, list]] = {}

    def stamp(self, stamp: Stamp | float):
        if isinstance(stamp, float):
            rr.set_time_seconds("time", seconds=stamp)
        elif isinstance(stamp, Stamp):
            rr.set_time_seconds("time", seconds=stamp.to_sec())

    def log(
        self,
        name: str,
        value: loam.LoamFeatures
        | SE3
        | LidarMeasurement
        | list[SE3]
        | list[np.ndarray]
        | float,
        color: list[int] = [0, 0, 255],
        **kwargs,
    ):
        if isinstance(value, loam.LoamFeatures):
            rr.log(f"{name}/edges", rr.Points3D(value.edge_points, colors=[150, 0, 0]))
            rr.log(
                f"{name}/planar", rr.Points3D(value.planar_points, colors=[0, 150, 0])
            )
            rr.log(
                f"{name}/points", rr.Points3D(value.point_points, colors=[0, 0, 150])
            )
        elif isinstance(value, SE3):
            pose = rr.Transform3D(
                rotation=rr.datatypes.Quaternion(
                    xyzw=[
                        value.rot.qx,
                        value.rot.qy,
                        value.rot.qz,
                        value.rot.qw,
                    ]
                ),
                translation=value.trans,
            )
            rr.log(name, pose)

            if name not in self.history:
                self.history[name] = (self.colors[len(self.history)], [value.trans])
            else:
                self.history[name][1].append(value.trans)

            rr.log(
                f"{name}_history",
                rr.Points3D(
                    np.asarray(self.history[name][1]), colors=self.history[name][0]
                ),
                static=True,
            )

        elif isinstance(value, LidarMeasurement):
            rr.log(
                name,
                rr.Points3D(value.to_vec_positions(), colors=color),
            )

        elif isinstance(value, list) and isinstance(value[0], SE3):
            value = cast(list[SE3], value)
            pts = np.array([p.trans for p in value])
            rr.log(name, rr.Points3D(pts, colors=color), **kwargs)

        elif isinstance(value, list) and isinstance(value[0], np.ndarray):
            value = cast(list[np.ndarray], value)
            rr.log(name, rr.Points3D(value, colors=color), **kwargs)

        elif isinstance(value, float):
            rr.log(name, rr.Scalar(value))

        else:
            raise ValueError(f"Unknown type for {name}")

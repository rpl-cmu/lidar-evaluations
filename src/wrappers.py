import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, cast

from evalio.datasets.base import Dataset
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from serde.yaml import to_yaml

import loam
from evalio.types import SE3, SO3, LidarMeasurement, Stamp, Trajectory
from params import ExperimentParams
from env import FIGURE_DIR, GRAPHICS_DIR, INC_DATA_DIR

import os
import matplotlib.pyplot as plt

from itertools import chain, combinations
from convert import convert
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

    return {
        "Newer College Stereo-Cam": c[0],
        "Newer College Multi-Cam": c[1],
        "Hilti 2022": c[2],
        "Oxford Spires": c[3],
        "Multi-Campus": c[4],
        "HeLiPR": c[5],
        "Botanic Garden": c[6],
    }


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


def plt_show(name: str):
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=300)
    plt.savefig(GRAPHICS_DIR / f"{name}.pdf", dpi=300)
    if not is_remote():
        plt.show()


@dataclass
class NavState:
    stamp: Stamp
    pose: SE3
    velocity: np.ndarray

    # gtsam doesn't expose this function... sigh
    def update(
        self, accel: np.ndarray, gyro: np.ndarray, gravity: np.ndarray, stamp: Stamp
    ) -> None:
        rot_og = self.pose.rot
        trans_og = self.pose.trans

        dt = stamp - self.stamp
        rot = rot_og * SO3.exp(gyro * dt)
        vel = self.velocity + (rot_og.rotate(accel) + gravity) * dt
        trans = trans_og + vel * dt + 0.5 * (rot_og.rotate(accel) + gravity) * dt**2

        self.stamp = stamp
        self.pose = SE3(rot=rot, trans=trans)
        self.velocity = vel
        self.trans = trans


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


class ImuPoseLoader:
    def __init__(self, dataset: Dataset) -> None:
        import pickle

        self.seq = dataset.seq

        filename = (
            INC_DATA_DIR / "imu_integration" / f"{dataset.name()}_{dataset.seq}.pkl"
        )
        self.imu_T_lidar = dataset.imu_T_lidar()

        with open(filename, "rb") as f:
            self.data: list[tuple[Stamp, list[NavState]]] = pickle.load(f)
        self.idx = 1

        # Convert everything into the lidar frame
        for i in range(len(self.data)):
            for j in range(len(self.data[i][1])):
                self.data[i][1][j].pose = self.data[i][1][j].pose * self.imu_T_lidar
                # TODO: Could rotate velocity, not going to bother for now

        # Zero everything out so it's relative to the first pose in each sequence
        for i in range(len(self.data)):
            start_inv = self.data[i][1][0].pose.inverse()
            for j in range(len(self.data[i][1])):
                self.data[i][1][j].pose = start_inv * self.data[i][1][j].pose

    def next(
        self, stamp: Stamp
    ) -> tuple[Optional[SE3], Optional[list[float]], Optional[list[loam.Pose3d]]]:
        """
        Takes in lidar scan stamp (that should be exactly in our list already)

        Returns:
            Initialization to that stamp relative to the previous stamp
            delta t in seconds of IMU measurements relative to the stamp passed in
            IMU measurement integrated from the nearest GT till the next stamp, all relative to the start and in the lidar frame
        """
        # If we've reached the end
        if self.idx >= len(self.data):
            return None, None, None
        # If our first imu is in the future, we'll have to skip for a bit
        elif self.data[self.idx][0] > stamp:
            return None, None, None

        # Skip until we find the first imu that is after our stamp
        while self.idx < len(self.data) and self.data[self.idx][0] < stamp:
            self.idx += 1
        if self.idx >= len(self.data):
            return None, None, None

        # Stamps should be dead on
        assert self.data[self.idx][0] == stamp, (
            f"Stamps don't match in ImuPoseLoader in {self.seq}"
        )

        # See how far we integrated across the previous timestamp to get to this one
        prev_states = self.data[self.idx - 1][1]
        init = prev_states[-1].pose

        # Get all the integrated poses taken during this lidar scan
        poses = [convert(p.pose) for p in self.data[self.idx][1]]
        stamps = [s.stamp - self.data[self.idx][0] for s in self.data[self.idx][1]]

        return init, stamps, poses


class Writer:
    def __init__(self, path: Path, ep: ExperimentParams):
        file = ep.output_file(path)
        debug = ep.debug_file(path)
        file.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(file, "w")
        params = "# " + to_yaml(ep).replace("\n", "\n# ")
        self.file.write(params + "\n")
        self.file.write(
            "# timestamp, x, y, z, qx, qy, qz, qw, gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw, point, edge, planar, iter\n"
        )
        self.writer = csv.writer(self.file)

        self.debug = open(debug, "w")

    def write(
        self,
        stamp: Stamp,
        pose: SE3,
        gt: SE3,
        feats: loam.LoamFeatures,
        detail: loam.RegistrationDetail,
    ):
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
                len(detail.iteration_info),
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

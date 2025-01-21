import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Protocol, Sequence, Union

import numpy as np
from rosbags.highlevel import AnyReader
from tabulate import tabulate

from features._cpp._helpers import (  # type: ignore
    DataType,
    Field,
    PointCloudMetadata,
    ros_pc2_conversion,
)
from features.types import (
    SE3,
    SO3,
    LidarMeasurement,
    LidarParams,
    Stamp,
    Trajectory,
)

Measurement = Union[LidarMeasurement]

MAIN_DATA_DIR = Path(os.getenv("MAIN_DATA_DIR", "./"))


@dataclass
class Dataset(Protocol):
    seq: str
    length: Optional[int] = None

    # ------------------------- For loading data ------------------------- #
    def __iter__(self) -> Iterator[LidarMeasurement]: ...

    def ground_truth_raw(self) -> Trajectory: ...

    # ------------------------- For loading params ------------------------- #
    @staticmethod
    def url() -> str: ...

    @staticmethod
    def name() -> str: ...

    @staticmethod
    def sequences() -> Sequence[str]: ...

    @classmethod
    def lengths(cls) -> "dict[str, int | None]":
        return {seq: None for seq in cls.sequences()}

    @staticmethod
    def lidar_T_gt() -> SE3: ...

    @staticmethod
    def lidar_params() -> LidarParams: ...

    # ------------------------- For downloading ------------------------- #
    @staticmethod
    def check_download(seq: str) -> bool:
        return True

    @staticmethod
    def download(seq: str) -> None:
        raise NotImplementedError("Download not implemented")

    # ------------------------- Helpers ------------------------- #
    def __post_init__(self):
        self.seq = self.process_seq(self.seq)

        if not self.check_download(self.seq):
            raise ValueError(
                f"Data for {self.seq} not found, please use `evalio download {self.name()}/{self.seq}` to download"
            )

    @classmethod
    def process_seq(cls, seq: str):
        if seq not in cls.sequences():
            raise ValueError(f"Sequence {seq} not in {cls.name()}")

        return seq

    def ground_truth(self) -> Trajectory:
        gt_traj = self.ground_truth_raw()
        gt_T_imu = self.lidar_T_gt().inverse()

        # Convert to IMU frame
        for i in range(len(gt_traj)):
            gt_o_T_gt_i = gt_traj.poses[i]
            gt_traj.poses[i] = gt_o_T_gt_i * gt_T_imu

        return gt_traj

    def __str__(self):
        return f"{self.name()}/{self.seq}"


# ------------------------- Helpers ------------------------- #


def pointcloud2_to_evalio(msg) -> LidarMeasurement:
    # Convert to C++ types
    fields = []
    for f in msg.fields:
        fields.append(
            Field(name=f.name, datatype=DataType(f.datatype), offset=f.offset)
        )

    stamp = Stamp(sec=msg.header.stamp.sec, nsec=msg.header.stamp.nanosec)

    cloud = PointCloudMetadata(
        stamp=stamp,
        height=msg.height,
        width=msg.width,
        row_step=msg.row_step,
        point_step=msg.point_step,
        is_bigendian=msg.is_bigendian,
        is_dense=msg.is_dense,
    )

    return ros_pc2_conversion(cloud, fields, bytes(msg.data))  # type: ignore


# These are helpers to help with common dataset types
class RosbagIter:
    def __init__(self, path: Path, lidar_topic: str, imu_topic: str):
        self.lidar_topic = lidar_topic
        self.imu_topic = imu_topic

        # Glob to get all .bag files in the directory
        if path.is_dir():
            self.path = list(path.glob("*.bag"))
            if not self.path:
                raise FileNotFoundError(f"No .bag files found in directory {path}")
        else:
            self.path = [path]

        # Open the bag file
        self.reader = AnyReader(self.path)
        self.reader.open()
        connections = [
            x for x in self.reader.connections if x.topic == self.lidar_topic
        ]
        if len(connections) == 0:
            connections_all = [[c.topic, c.msgtype] for c in self.reader.connections]
            print(
                tabulate(
                    connections_all, headers=["Topic", "MsgType"], tablefmt="fancy_grid"
                )
            )
            raise ValueError(
                f"Could not find topics {self.lidar_topic} or {self.imu_topic}"
            )

        self.lidar_count = sum(
            [x.msgcount for x in connections if x.topic == self.lidar_topic]
        )

        self.iterator = self.reader.messages(connections=connections)

    def __len__(self):
        return self.lidar_count

    def __iter__(self):
        return self

    def __next__(self) -> LidarMeasurement:
        # TODO: Ending somehow
        connection, timestamp, rawdata = next(self.iterator)

        msg = self.reader.deserialize(rawdata, connection.msgtype)

        return pointcloud2_to_evalio(msg)


def load_pose_csv(path: Path, fieldnames: Sequence[str], delimiter=",") -> Trajectory:
    poses = []
    stamps = []

    with open(path) as f:
        csvfile = filter(lambda row: row[0] != "#", f)
        reader = csv.DictReader(csvfile, fieldnames=fieldnames, delimiter=delimiter)
        for line in reader:
            r = SO3(
                qw=float(line["qw"]),
                qx=float(line["qx"]),
                qy=float(line["qy"]),
                qz=float(line["qz"]),
            )
            t = np.array([float(line["x"]), float(line["y"]), float(line["z"])])
            pose = SE3(r, t)

            if "nsec" not in fieldnames:
                stamp = Stamp.from_sec(float(line["sec"]))
            elif "sec" not in fieldnames:
                stamp = Stamp.from_nsec(int(line["nsec"]))
            else:
                stamp = Stamp(sec=int(line["sec"]), nsec=int(line["nsec"]))
            poses.append(pose)
            stamps.append(stamp)

    return Trajectory(metadata={}, stamps=stamps, poses=poses)


def load_tum(path: Path) -> Trajectory:
    return load_pose_csv(path, ["sec", "x", "y", "z", "qx", "qy", "qz", "qw"])

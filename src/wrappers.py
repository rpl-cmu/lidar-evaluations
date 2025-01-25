import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import loam
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from evalio.types import SE3, LidarMeasurement, Stamp, Trajectory
from serde.yaml import to_yaml

from params import ExperimentParams


@dataclass
class GroundTruthIterator:
    traj: Trajectory
    idx: int = 0

    def next(self, stamp: Stamp, tol: float = 1e-2) -> Optional[SE3]:
        # If our first ground truth is in the future, we'll have to skip for a bit
        if self.traj.stamps[self.idx] > stamp:
            return None
        elif self.idx > len(self.traj):
            return None

        while self.traj.stamps[self.idx] - stamp < -tol:
            self.idx += 1

        return self.traj.poses[self.idx]


class Writer:
    def __init__(self, path: Path, ep: ExperimentParams):
        file = ep.output_file(path)
        file.parent.mkdir(parents=True, exist_ok=True)

        self.file = open(file, "w")
        params = "# " + to_yaml(ep).replace("\n", "\n# ")
        self.file.write(params + "\n")
        self.file.write(
            "# timestamp, x, y, z, qx, qy, qz, qw, gt_x, gt_y, gt_z, gt_qx, gt_qy, gt_qz, gt_qw, edge, planar\n"
        )
        self.writer = csv.writer(self.file)

        self.index = 0

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
                len(feats.edge_points),
                len(feats.planar_points),
            ]
        )
        self.index += 1

    def close(self):
        self.file.close()


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

    def stamp(self, stamp: Stamp):
        rr.set_time_seconds("time", seconds=stamp.to_sec())

    def log(
        self,
        name: str,
        value: loam.LoamFeatures | SE3 | LidarMeasurement | list,
        color: list[int] = [0, 0, 255],
    ):
        if isinstance(value, loam.LoamFeatures):
            rr.log(f"{name}/edges", rr.Points3D(value.edge_points, colors=[255, 0, 0]))
            rr.log(
                f"{name}/planar", rr.Points3D(value.planar_points, colors=[0, 255, 0])
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

        elif isinstance(value, list):
            rr.log(name, rr.Points3D(value, colors=color))

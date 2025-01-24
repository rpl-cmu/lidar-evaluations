from dataclasses import dataclass
from typing import Optional
from evalio.types import Trajectory, SE3, Stamp


@dataclass
class GroundTruthIterator:
    traj: Trajectory
    idx: int = 0

    def next(self, stamp: Stamp, tol: float = 1e-2) -> Optional[SE3]:
        # If our first ground truth is in the future, skip it for now
        if self.traj.stamps[self.idx] > stamp:
            return None
        elif self.idx > len(self.traj):
            return None

        while self.traj.stamps[self.idx] - stamp < -tol:
            self.idx += 1

        print(stamp, self.traj.stamps[self.idx], self.idx)
        return self.traj.poses[self.idx]

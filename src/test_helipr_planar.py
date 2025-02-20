from typing import cast
from evalio.types import LidarMeasurement
import numpy as np

import loam

from convert import convert
from params import ExperimentParams, Feature
from wrappers import Rerun

ep = ExperimentParams(
    "test", "helipr/kaist_05", features=[Feature.Edge, Feature.Planar]
)

# get the first lidar measurement
pts = None
for mm in ep.build_dataset():
    if isinstance(mm, LidarMeasurement):
        pts = mm.to_vec_positions()
        break

pts = cast(list[np.ndarray], pts)

# gather features for many max values
lp = convert(ep.build_dataset().lidar_params())
fp = ep.feature_params()
fp.edge_feat_threshold = 100000
fp.max_planar_feats_per_sector = 100


max_planar = np.linspace(1.0, 10.0, 100)
print(max_planar)

rr = Rerun()

rr.stamp(0)
rr.log("pts", pts, static=True)

for m in max_planar:
    fp.planar_feat_threshold = m
    feat = loam.extractFeatures(pts, lp, fp)

    print(m, len(feat.planar_points))

    rr.stamp(m)
    rr.log("feat", feat.planar_points)

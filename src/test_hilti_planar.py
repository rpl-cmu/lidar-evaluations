from typing import cast
from evalio.types import LidarMeasurement
import numpy as np

import loam

from convert import convert
from params import ExperimentParams, Feature

ep = ExperimentParams(
    "test",
    "hilti_2022/attic_to_upper_gallery_2",
    features=[Feature.Edge, Feature.Planar],
)

# get the first lidar measurement
pts = None
for mm in ep.build_dataset():
    if isinstance(mm, LidarMeasurement):
        pts = mm.to_vec_positions()
        break

pts = cast(list[np.ndarray], pts)

# rr = Rerun()
# rr.stamp(0)
# rr.log("pts", pts, static=True)

# gather features for many max values
lp = convert(ep.build_dataset().lidar_params())
fp = ep.feature_params()

feat = loam.extractFeatures(pts, lp, fp)
print("num: ", len(feat.planar_points))

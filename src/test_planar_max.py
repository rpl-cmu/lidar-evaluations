from typing import cast
from evalio.types import LidarMeasurement
import numpy as np

import loam

from convert import convert
from params import ExperimentParams, Feature
import matplotlib.pyplot as plt

from wrappers import setup_plot

ep = ExperimentParams(
    "test",
    "botanic_garden/1018_00",
    # "newer_college_2020/01_short_experiment",
    features=[Feature.Planar],
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
fp.max_planar_feats_per_sector = 10000
# fp.edge_feat_threshold = 10000.0
# fp.neighbor_points = 10

values = np.linspace(0.01, 10.0, 10)

num_valid = loam.computeValidPoints(pts, lp, fp)
print(f"Num valid: {sum(num_valid)}")


def compute_amount(fp, pts):
    num_loam = np.zeros_like(values)
    for i, c in enumerate(values):
        fp.planar_feat_threshold = c
        feat = loam.extractFeatures(pts, lp, fp)
        num_loam[i] = len(feat.planar_points)

    return num_loam


c = setup_plot()
num_loam = compute_amount(fp, pts)
plt.plot(values, num_loam, label="LOAM")

fp.curvature_type = loam.Curvature.EIGEN
num_loam = compute_amount(fp, pts)
plt.plot(values, num_loam, label="EIGEN")

fp.curvature_type = loam.Curvature.EIGEN_NN
num_loam = compute_amount(fp, pts)
plt.plot(values, num_loam, label="EIGEN NN")

plt.legend()
plt.savefig("figures/planar_num_out.png")

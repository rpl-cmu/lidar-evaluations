import numpy as np

import loam
from convert import convert
from evalio.types import LidarMeasurement, Stamp
from params import ExperimentParams, Feature
from wrappers import Rerun

params = ExperimentParams(
    name="test_vis_curvature",
    dataset="newer_college_2020/01_short_experiment",
    features=[Feature.Planar],
)

dataset = params.build_dataset()
lp = convert(dataset.lidar_params())
fp = params.feature_params()
# fp.edge_feat_threshold = 0.9
# fp.max_planar_feats_per_sector = 1000
fp.curvature_type = loam.Curvature.EIGEN

iterator = iter(dataset)
while mm := next(iterator):
    if isinstance(mm, LidarMeasurement):
        break

print("got measurement")
rr = Rerun()
rr.stamp(Stamp.from_sec(0.0))
vec = mm.to_vec_positions()
rr.log("scan", mm, static=True)

for i in np.linspace(1.5, 5.0):
    fp.planar_feat_threshold = 1.5
    fp.edge_feat_threshold = i
    feat = loam.extractFeatures(vec, lp, fp)
    print(i, len(feat.planar_points), len(feat.edge_points))
    rr.stamp(Stamp.from_sec(i))
    rr.log("feat", feat)

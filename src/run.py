import loam
from evalio.datasets import NewerCollege2020
from evalio.types import LidarMeasurement
from convert import convert

# Load the data
dataset = NewerCollege2020("01_short_experiment")
gt = dataset.ground_truth()
gt.transform_in_place(dataset.imu_T_lidar())

# params
lp = convert(dataset.lidar_params())
fe_params = loam.FeatureExtractionParams()

for mm in dataset:
    if isinstance(mm, LidarMeasurement):
        pts = mm.to_vec_positions()
        feats = loam.extractFeatures(pts, lp, fe_params)
        print(f"Lidar measurement at {mm.stamp}")
        print(feats.edge_points.__len__())

        quit()

import loam
from evalio.cli.parser import DatasetBuilder
from evalio.types import LidarMeasurement
from convert import convert
from gt import GroundTruthIterator
import params


# Params
experiment_params = params.ExperimentParams(
    dataset="newer_college_2020/01_short_experiment",
    features=[params.Feature.Planar],
)


# Load the data
dataset = DatasetBuilder.parse(experiment_params.dataset)[0].build()
# Load ground truth in lidar frame
gt = dataset.ground_truth()
gt.transform_in_place(dataset.imu_T_lidar())
gt = GroundTruthIterator(gt)

# params
lp = convert(dataset.lidar_params())

feat_params = loam.FeatureExtractionParams()
feat_params.neighbor_points = 4
feat_params.number_sectors = 6
feat_params.max_edge_feats_per_sector = 10
feat_params.max_planar_feats_per_sector = 50

feat_params.edge_feat_threshold = 50.0
feat_params.planar_feat_threshold = 1.0

feat_params.occlusion_thresh = 0.9
feat_params.parallel_thresh = 0.01

prev_feat = None
prev_gt = None
i = 0

for mm in dataset:
    if isinstance(mm, LidarMeasurement):
        # Get everything for this step
        pts = mm.to_vec_positions()
        # TODO: Specify which features to keep
        curr_feat = loam.extractFeatures(pts, lp, feat_params)
        curr_gt = gt.next(mm.stamp)

        # Skip if we don't have everything we need
        if curr_gt is None:
            continue
        if prev_feat is None or prev_gt is None:
            prev_feat = curr_feat
            prev_gt = curr_gt
            i += 1
            continue

        # Get initialization
        if experiment_params.init == params.Initialization.GroundTruth:
            init = prev_gt.inverse() * curr_gt
            init = convert(init)
        else:
            init = loam.Pose3d.Identity()

        match experiment_params.init:
            case params.Initialization.GroundTruth:
                init = prev_gt.inverse() * curr_gt
                init = convert(init)
            case _:
                raise ValueError("Unknown initialization")

        pose = loam.registerFeatures(curr_feat, prev_feat, init)
        pose = convert(pose)
        print("pose", pose)

        i += 1

        # Save results
        # TODO: Just save deltas? Or entire trajectory?

        quit()

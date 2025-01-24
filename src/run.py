from pathlib import Path
from evalio.types import SE3
import loam
from evalio.types import LidarMeasurement
from tqdm import tqdm
from convert import convert
from wrappers import GroundTruthIterator, Rerun, Writer
import params
import numpy as np

visualize = False

# Params
experiment_params = params.ExperimentParams(
    dataset="newer_college_2020/01_short_experiment",
    features=[params.Feature.Planar],
    output=Path("results/test.csv"),
)


# Load the data
dataset = experiment_params.build_dataset()
# Load ground truth in lidar frame
gt = dataset.ground_truth()
gt.transform_in_place(dataset.imu_T_lidar())
gt = GroundTruthIterator(gt)

# Create writer
writer = Writer(experiment_params.output)
rr = None
if visualize:
    rr = Rerun(to_hide=["pose/points", "pose/features/*"])

# params
lp = convert(dataset.lidar_params())
fp = experiment_params.feature_params()
rp = experiment_params.registration_params()

prev_feat = None
prev_gt = None

iter_gt = SE3.identity()
iter_est = SE3.identity()

data_iter = iter(dataset)
pbar = tqdm(total=len(data_iter))  # type: ignore

for mm in data_iter:
    if not isinstance(mm, LidarMeasurement):
        continue

    # Get points in row major format
    pts = np.asarray(mm.to_vec_positions())
    pts = np.concatenate([pts[i :: lp.scan_lines] for i in range(lp.scan_lines)])
    pts = list(pts)

    # TODO: Specify which features to keep
    curr_feat = loam.extractFeatures(pts, lp, fp)
    curr_gt = gt.next(mm.stamp)

    # Skip if we don't have everything we need
    if curr_gt is None:
        continue
    if prev_feat is None or prev_gt is None:
        prev_feat = curr_feat
        prev_gt = curr_gt
        continue

    # Get initialization
    init_gt = prev_gt.inverse() * curr_gt
    match experiment_params.init:
        case params.Initialization.GroundTruth:
            init = convert(init_gt)
        case _:
            raise ValueError("Unknown initialization")

    pose = loam.registerFeatures(curr_feat, prev_feat, init, params=rp)
    pose = convert(pose)

    iter_gt = iter_gt * init_gt
    iter_est = iter_est * pose
    if rr is not None:
        rr.stamp(mm.stamp)
        rr.log("gt", iter_gt)
        rr.log("pose", iter_est)
        rr.log("pose/points", mm)
        rr.log("pose/features", curr_feat)

    pbar.set_description(
        f"E: {len(curr_feat.edge_points)}, P: {len(curr_feat.planar_points)}"
    )
    pbar.update()

    # Save results
    # TODO: Just save deltas? Or entire trajectory?
    writer.write(mm.stamp, iter_est)

    prev_gt = curr_gt
    prev_feat = curr_feat

writer.close()

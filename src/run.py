from pathlib import Path
from evalio.types import SE3
import loam
from evalio.types import LidarMeasurement
from tqdm import tqdm
from convert import convert
from wrappers import GroundTruthIterator, Rerun, Writer
import params
import numpy as np


def run(ep: params.ExperimentParams, directory: Path, visualize: bool = False):
    # Load the data
    dataset = ep.build_dataset()
    # Load ground truth in lidar frame
    gt = dataset.ground_truth()
    gt.transform_in_place(dataset.imu_T_lidar())
    gt = GroundTruthIterator(gt)

    # Create writer
    writer_main = Writer(ep.output_dir(directory))
    writer_gt = Writer(ep.gt_dir(directory))
    rr = None
    if visualize:
        rr = Rerun(to_hide=["pose/points", "pose/features/*"])

    # params
    lp = convert(dataset.lidar_params())
    fp = ep.feature_params()
    rp = ep.registration_params()

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
        step_gt = prev_gt.inverse() * curr_gt
        match ep.init:
            case params.Initialization.GroundTruth:
                init = convert(step_gt)
            case _:
                raise ValueError("Unknown initialization")

        step_pose = loam.registerFeatures(curr_feat, prev_feat, init, params=rp)
        step_pose = convert(step_pose)

        iter_gt = iter_gt * step_gt
        iter_est = iter_est * step_pose
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
        # Saving deltas for now - maybe switch to trajectories later
        writer_main.write(mm.stamp, step_pose)
        writer_gt.write(mm.stamp, step_gt)

        prev_gt = curr_gt
        prev_feat = curr_feat

    writer_main.close()


if __name__ == "__main__":
    ep = params.ExperimentParams(
        name="test",
        dataset="newer_college_2020/01_short_experiment",
        features=[params.Feature.Planar],
    )
    run(ep, Path("results"), visualize=False)

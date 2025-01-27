from pathlib import Path

import loam
import numpy as np
from evalio.types import SE3, SO3, LidarMeasurement
from loam import Pose3d
from tqdm import tqdm

import params
from convert import convert
from wrappers import GroundTruthIterator, Rerun, Writer

from multiprocessing import current_process


def run(
    ep: params.ExperimentParams,
    directory: Path,
    multithreaded: bool = False,
    visualize: bool = False,
    ip: str = "0.0.0.0:9876",
):
    # Load the data
    dataset = ep.build_dataset()
    # Load ground truth in lidar frame
    gt = dataset.ground_truth()
    gt.transform_in_place(dataset.imu_T_lidar())
    gt = GroundTruthIterator(gt)

    # Create writer
    writer = Writer(directory, ep)
    writer.log(f"Starting experiment {ep.name}")
    rr = None
    if visualize:
        rr = Rerun(to_hide=["pose/points", "pose/features/*"], ip=ip)

    # params
    lp = convert(dataset.lidar_params())
    fp = ep.feature_params()
    rp = ep.registration_params()

    prev_feat = None
    prev_gt = None
    prev_prev_gt = None

    iter_gt = SE3.identity()
    iter_est = SE3.identity()

    data_iter = iter(dataset)
    pbar_params = {"total": len(data_iter), "dynamic_ncols": True}  # type: ignore
    pbar_desc = f"{ep.short_info()} - E: {{}}, P: {{}}"
    if multithreaded:
        idx = current_process()._identity[0] - 1
        pbar_params["position"] = idx
        pbar_desc = f"{idx}, " + pbar_desc
    pbar = tqdm(**pbar_params)

    for mm in data_iter:
        if not isinstance(mm, LidarMeasurement):
            continue

        # Get points in row major format if it's in column major
        pts = mm.to_vec_positions()
        if mm.points[0].row != mm.points[1].row:
            pts = np.asarray(pts)
            pts = np.concatenate(
                [pts[i :: lp.scan_lines] for i in range(lp.scan_lines)]
            )
            pts = list(pts)

        # Get features and ground truth
        curr_gt = gt.next(mm.stamp)
        curr_feat = loam.extractFeatures(pts, lp, fp)
        if not ep.planar:
            curr_feat.planar_points = []
        if not ep.edge:
            curr_feat.edge_points = []

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
            case params.Initialization.ConstantVelocity:
                if prev_prev_gt is None:
                    init = Pose3d.Identity()
                else:
                    init = convert(prev_prev_gt.inverse() * prev_gt)
            case params.Initialization.Identity:
                init = Pose3d.Identity()
            case _:
                raise ValueError("Unknown initialization")

        try:
            step_pose = loam.registerFeatures(curr_feat, prev_feat, init, params=rp)
            step_pose = convert(step_pose)

            # Also skipping iter_gt to not compound bad results if failed
            iter_gt = iter_gt * step_gt
            iter_est = iter_est * step_pose
            if rr is not None:
                rr.stamp(mm.stamp)
                rr.log("gt", iter_gt)
                rr.log("pose", iter_est)
                rr.log("pose/points", mm)
                rr.log("pose/features", curr_feat)

            # Save results
            # Saving deltas for now - maybe switch to trajectories later
            writer.write(mm.stamp, step_pose, step_gt, curr_feat)

        except Exception as e:
            writer.error("Registration failed, replacing with nan")
            writer.error(
                f"E: {len(curr_feat.edge_points)}, P: {len(curr_feat.planar_points)}"
            )
            writer.error(f"Stamp: {mm.stamp}")
            writer.error(e)
            step_pose = SE3(
                SO3(qx=np.nan, qy=np.nan, qz=np.nan, qw=np.nan),
                np.array([np.nan, np.nan, np.nan]),
            )

        pbar.set_description(
            pbar_desc.format(len(curr_feat.edge_points), len(curr_feat.planar_points))
        )
        pbar.update()

        prev_prev_gt = prev_gt
        prev_gt = curr_gt
        prev_feat = curr_feat

    writer.close()


if __name__ == "__main__":
    ep = params.ExperimentParams(
        name="test",
        # dataset="newer_college_2020/01_short_experiment",
        dataset="newer_college_2021/quad-easy",
        features=[params.Feature.Planar, params.Feature.Edge],
    )
    run(ep, Path("results"), visualize=True, ip="172.31.76.216:9876")

from pathlib import Path

import loam
import numpy as np
from evalio.types import SE3, SO3, LidarMeasurement
from loam import Pose3d
from tqdm import tqdm

import params
from convert import convert
from wrappers import GroundTruthIterator, Rerun, Writer

from multiprocessing import current_process, Pool
from functools import partial

from typing import Optional, Sequence


def run(
    ep: params.ExperimentParams,
    directory: Path,
    multithreaded: bool = False,
    visualize: bool = False,
    length: Optional[int] = None,
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
    pbar_params = {"total": len(data_iter), "dynamic_ncols": True, "leave": True}  # type: ignore
    if length is not None:
        pbar_params["total"] = length
    pbar_desc = f"[{ep.short_info()}] Po: {{}}, Ed: {{}}, Pl: {{}}"
    if multithreaded:
        idx = current_process()._identity[0] - 1
        pbar_params["position"] = idx
        pbar_desc = f"{idx}, " + pbar_desc
    pbar = tqdm(**pbar_params)
    idx = 0

    for mm in data_iter:
        if not isinstance(mm, LidarMeasurement):
            continue

        pts = mm.to_vec_positions()

        # Get features and ground truth
        curr_gt = gt.next(mm.stamp)
        curr_feat = loam.extractFeatures(pts, lp, fp)
        if not ep.planar and not ep.pseudo_planar:
            curr_feat.point_points = curr_feat.point_points + curr_feat.planar_points
            curr_feat.planar_points = []
        if not ep.edge:
            curr_feat.point_points = curr_feat.point_points + curr_feat.edge_points
            curr_feat.edge_points = []
        if not ep.point:
            curr_feat.point_points = []

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
            writer.error(f"Stamp: {mm.stamp}")
            writer.error(e)
            step_pose = SE3(
                SO3(qx=np.nan, qy=np.nan, qz=np.nan, qw=np.nan),
                np.array([np.nan, np.nan, np.nan]),
            )

        pbar.set_description(
            pbar_desc.format(
                len(curr_feat.point_points),
                len(curr_feat.edge_points),
                len(curr_feat.planar_points),
            )
        )
        pbar.update()

        prev_prev_gt = prev_gt
        prev_gt = curr_gt
        prev_feat = curr_feat

        if length is not None and idx >= length:
            break
        idx += 1

    writer.close()


def run_multithreaded(
    eps: Sequence[params.ExperimentParams],
    directory: Path,
    visualize: bool = False,
    length: Optional[int] = None,
    ip: str = "0.0.0.0:9876",
    num_threads: int = 10,
):
    print(f"Running {len(eps)} experiments")
    with Pool(num_threads, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock) as p:
        p.map(
            partial(
                run,
                directory=directory,
                multithreaded=True,
                length=length,
                visualize=visualize,
                ip=ip,
            ),
            eps,
        )


if __name__ == "__main__":
    # dataset = "newer_college_2020/01_short_experiment"
    # dataset = "hilti_2022/construction_upper_level_1"
    # dataset = "helipr/kaist_04"
    # dataset = "helipr/kaist_05"
    # dataset = "oxford_spires/christ_church_01"
    dataset = "helipr/dcc_06"

    eps = [
        # params.ExperimentParams(
        #     name="planar",
        #     dataset=dataset,
        #     init=params.Initialization.Identity,
        #     features=[params.Feature.Planar],
        # ),
        params.ExperimentParams(
            name="planar_edge",
            dataset=dataset,
            init=params.Initialization.GroundTruth,
            features=[params.Feature.Planar, params.Feature.Edge],
        ),
        # params.ExperimentParams(
        #     name="pseudo_planar",
        #     dataset=dataset,
        #     init=params.Initialization.Identity,
        #     features=[params.Feature.Planar],
        # ),
        # params.ExperimentParams(
        #     name="pseudo_planar_edge",
        #     dataset=dataset,
        #     init=params.Initialization.Identity,
        #     features=[params.Feature.Planar],
        # ),
    ]

    directory = Path("results/25.02.03_verify_datasets")
    length = None

    run(eps[0], directory, visualize=False, length=length)
    # run_multithreaded(eps, directory, length=length)

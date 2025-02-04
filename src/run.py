from functools import partial
from multiprocessing import Manager, Pool
from pathlib import Path
from time import sleep
from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

import loam
import params
from convert import convert
from evalio.types import SE3, SO3, LidarMeasurement
from loam import Pose3d
from wrappers import GroundTruthIterator, Rerun, Writer


def get_pgb_pos(shared_list) -> int:
    # Acquire lock and get a progress bar slot
    for i in range(len(shared_list)):
        if not shared_list[i]:
            shared_list[i] = True
            return i

    raise ValueError("No available progress bar slots")


def release_pgb_pos(shared_list, slot):
    shared_list[slot] = 0


def run(
    ep: params.ExperimentParams,
    directory: Path,
    multithreaded_info: Optional[list] = None,
    visualize: bool = False,
    length: Optional[int] = None,
    ip: str = "0.0.0.0:9876",
):
    # Sleep for a small blip to offset the progress bars
    if multithreaded_info is not None:
        sleep(np.random.rand())

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
    pbar_params = {
        "total": len(data_iter),  # type: ignore
        "dynamic_ncols": True,
        "leave": False,
        "desc": f"[{ep.short_info()}]",
    }
    if length is not None:
        pbar_params["total"] = length
    pbar_idx = None
    if multithreaded_info is not None:
        pbar_idx = get_pgb_pos(multithreaded_info)
        pbar_params["position"] = pbar_idx + 1
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

        pbar.update()

        prev_prev_gt = prev_gt
        prev_gt = curr_gt
        prev_feat = curr_feat

        if length is not None and idx >= length:
            break
        idx += 1

    pbar.close()
    if multithreaded_info is not None and pbar_idx is not None:
        release_pgb_pos(multithreaded_info, pbar_idx)
    writer.close()


def run_multithreaded(
    eps: Sequence[params.ExperimentParams],
    directory: Path,
    visualize: bool = False,
    length: Optional[int] = None,
    ip: str = "0.0.0.0:9876",
    num_threads: int = 10,
):
    # https://github.com/tqdm/tqdm/issues/1000#issuecomment-1842085328
    # tqdm still isn't perfect - can be a bit janky if resizing
    # but I'm fairly satisfied with it now
    manager = Manager()
    shared_list = manager.list([False for _ in range(num_threads)])
    lock = manager.Lock()

    with Pool(num_threads, initargs=(lock,), initializer=tqdm.set_lock) as p:
        for _ in tqdm(
            p.imap_unordered(
                partial(
                    run,
                    directory=directory,
                    multithreaded_info=shared_list,  # type:ignore
                    length=length,
                    visualize=visualize,
                    ip=ip,
                ),
                eps,
            ),
            total=len(eps),
            position=0,
            desc="Running experiments",
            leave=True,
        ):
            pass


if __name__ == "__main__":
    dataset = "helipr/dcc_06"

    eps = [
        params.ExperimentParams(
            name="planar_edge",
            dataset=dataset,
            init=params.Initialization.GroundTruth,
            features=[params.Feature.Planar, params.Feature.Edge],
        ),
    ]

    directory = Path("results/25.02.03_verify_datasets")
    length = None

    run(eps[0], directory, visualize=False, length=length)
    # run_multithreaded(eps, directory, length=length)

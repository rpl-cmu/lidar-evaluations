from functools import partial
from multiprocessing import Manager, Pool, Lock
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from tqdm import tqdm

import loam
from lidar_eval import params
from lidar_eval.convert import convert
from evalio.types import SE3, SO3, LidarMeasurement
from loam import Pose3d
from lidar_eval.wrappers import StampIterator, ImuPoseLoader, Rerun, Writer


mutex = Lock()


def get_pgb_pos(shared_list) -> int:
    # Acquire lock and get a progress bar slot
    with mutex:
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
    extra_data: Path,
    multithreaded_info: Optional[list] = None,
    visualize: bool = False,
    length: Optional[int] = None,
    ip: str = "0.0.0.0:9876",
):
    # Check if this has been run to completion
    if ep.output_file(directory).exists():
        with open(ep.output_file(directory), "r") as f:
            lines = f.readlines()
            if len(lines) > 0 and lines[-1] == "# done":
                return

    # Load the data
    dataset = ep.build_dataset()
    # Load ground truth in lidar frame
    gt = dataset.ground_truth()
    gt.transform_in_place(dataset.imu_T_lidar())
    gt = StampIterator(gt.stamps, gt.poses)
    # Load integrated imu data
    imu = ImuPoseLoader(dataset, extra_data)

    # Create visualizer
    rr = None
    if visualize:
        rr = Rerun(to_hide=["pose/points", "pose/features/*"], ip=ip)

    # Get length of dataset
    data_iter = iter(dataset)
    if length is None or (length is not None and length > len(data_iter)):  # type:ignore
        length = len(data_iter)  # type:ignore

    # Setup progress bar
    pbar_params = {
        "total": length,  # type: ignore
        "dynamic_ncols": True,
        "leave": True,
        "desc": f"[{ep.short_info()}]",
    }
    pbar_idx = None
    if multithreaded_info is not None:
        pbar_idx = get_pgb_pos(multithreaded_info)
        pbar_params["position"] = pbar_idx + 1
        pbar_params["desc"] = f"#{pbar_idx:>2} " + pbar_params["desc"]
        pbar_params["leave"] = False
    pbar = tqdm(**pbar_params)

    # Setup writer
    writer = Writer(directory, ep)
    writer.log(f"Starting experiment {ep.name}")

    # Setup everything for running
    idx = 0

    prev_gt = None
    prev_prev_gt = None

    iter_gt = SE3.identity()
    iter_est = SE3.identity()

    for mm in data_iter:
        if not isinstance(mm, LidarMeasurement):
            continue

        # Get ground truth and skip if we don't have all of them
        curr_gt = gt.next(mm.stamp)
        if curr_gt is None:
            # Reset everything if there's no ground truth
            prev_gt = None
            prev_prev_gt = None
            continue
        if prev_gt is None:
            prev_gt = curr_gt
            continue
        if prev_prev_gt is None:
            prev_prev_gt = prev_gt
            prev_gt = curr_gt
            continue

        # get imu init and integrated poses - skip if we don't have it
        imu_init, imu_stamps, imu_poses = imu.next(mm.stamp)
        if imu_init is None or imu_stamps is None or imu_poses is None:
            continue

        # Get initialization
        step_gt = prev_gt.inverse() * curr_gt
        match ep.init:
            case params.Initialization.GroundTruth:
                init = convert(step_gt)
            case params.Initialization.Imu:
                init = convert(imu_init)
            case params.Initialization.ConstantVelocity:
                init = convert(prev_prev_gt.inverse() * prev_gt)
            case params.Initialization.Identity:
                init = Pose3d.Identity()
            case _:
                raise ValueError("Unknown initialization")

        try:
            detail = loam.RegistrationDetail()
            step_pose: SE3 = convert(init)  # type: ignore

            # Also skipping iter_gt to not compound bad results if failed
            iter_gt = iter_gt * step_gt
            iter_est = iter_est * step_pose
            if rr is not None:
                rr.stamp(mm.stamp)
                rr.log("gt", iter_gt)
                rr.log("pose", iter_est)
                rr.log("pose/points", mm)

                # compute metrics to send as well
                delta = step_gt.inverse() * step_pose
                error_t = np.sqrt(delta.trans @ delta.trans)
                r_diff = delta.rot.log()
                error_r = np.sqrt(r_diff @ r_diff) * 180 / np.pi
                rr.log("metrics/rte/rot", float(error_r))
                rr.log("metrics/rte/trans", float(error_t))

                delta = iter_gt.inverse() * iter_est
                error_t = np.sqrt(delta.trans @ delta.trans)
                r_diff = delta.rot.log()
                error_r = np.sqrt(r_diff @ r_diff) * 180 / np.pi
                rr.log("metrics/ate/rot", float(error_r))
                rr.log("metrics/ate/trans", float(error_t))

            # Save results
            # Saving deltas for now - maybe switch to trajectories later
            writer.write(mm.stamp, step_pose, step_gt, loam.LoamFeatures(), detail)

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
    extra_data: Path,
    visualize: bool = False,
    length: Optional[int] = None,
    ip: str = "0.0.0.0:9876",
    num_threads: int = 10,
):
    # Make sure they're all unique
    seen = set()
    dupes = [
        (x.dataset, x.name)
        for x in eps
        if (x.dataset, x.name) in seen or seen.add((x.dataset, x.name))
    ]
    if len(dupes) != 0:
        raise ValueError(f"Duplicate experiment names {dupes}")

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
                    extra_data=extra_data,
                    multithreaded_info=shared_list,  # type:ignore
                    length=length,
                    visualize=visualize,
                    ip=ip,
                ),
                eps,
            ),
            total=len(eps),
            position=0,
            desc="Experiments",
            smoothing=0.0,
            leave=True,
            dynamic_ncols=True,
        ):
            pass

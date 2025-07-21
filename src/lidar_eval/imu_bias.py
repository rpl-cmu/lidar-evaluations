from typing import overload, Optional
from evalio.datasets.base import Dataset
from evalio.types import (
    ImuMeasurement,
    LidarMeasurement,
    SE3,
    SO3,
    ImuParams,
    Stamp,
    Trajectory,
)
from evalio.cli.parser import DatasetBuilder
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
import pickle

from typing import cast

import gtsam
from gtsam import (
    Pose3,  # type: ignore
    Rot3,  # type: ignore
    NonlinearFactorGraph,  # type: ignore
    Values,  # type: ignore
    PreintegrationCombinedParams,  # type: ignore
    PreintegratedCombinedMeasurements,  # type: ignore
    CombinedImuFactor,  # type: ignore
    LevenbergMarquardtOptimizer,  # type: ignore
    LevenbergMarquardtParams,  # type: ignore
)  # type: ignore
from gtsam.imuBias import ConstantBias  # type: ignore
from gtsam.symbol_shorthand import X, V, B  # type: ignore
from tqdm import tqdm

from lidar_eval.wrappers import NavState, StampIterator


@dataclass(kw_only=True)
class OptimizationResult:
    stamps: list[Stamp]
    poses: list[SE3]
    vel: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray


@overload
def convert(value: SE3) -> Pose3: ...
@overload
def convert(value: ImuParams) -> PreintegratedCombinedMeasurements: ...
@overload
def convert(value: Pose3) -> SE3: ...


def convert(value: Pose3 | SE3) -> Pose3 | SE3:
    if isinstance(value, SE3):
        return Pose3(
            Rot3(value.rot.qw, value.rot.qx, value.rot.qy, value.rot.qz), value.trans
        )
    elif isinstance(value, Pose3):
        q = value.rotation().toQuaternion().coeffs()
        return SE3(SO3(qx=q[0], qy=q[1], qz=q[2], qw=q[3]), trans=value.translation())
    elif isinstance(value, ImuParams):
        pim_params = PreintegrationCombinedParams(value.gravity)
        # pim_params.gravity = value.gravity
        pim_params.setAccelerometerCovariance(np.eye(3) * value.accel**2)
        pim_params.setGyroscopeCovariance(np.eye(3) * value.gyro**2)
        pim_params.setBiasAccCovariance(np.eye(3) * value.accel_bias**2)
        pim_params.setBiasOmegaCovariance(np.eye(3) * value.gyro_bias**2)
        pim_params.setIntegrationCovariance(np.eye(3) * value.integration)
        pim_params.setBiasAccOmegaInit(np.eye(6) * value.bias_init)
        return PreintegratedCombinedMeasurements(pim_params)


# returns imu measurements, lidar stamps, and ground truth
def load_data(
    dataset: Dataset, length: Optional[int] = None
) -> tuple[list[ImuMeasurement], list[Stamp], Trajectory]:
    # Get ground truth (already in IMU frame)
    gt = dataset.ground_truth()

    # Gather all imu data
    imu_data: list[ImuMeasurement] = []
    lidar_stamps: list[Stamp] = []
    for mm in tqdm(
        dataset,
        leave=False,
        desc=f"{dataset.dataset_name()}/{dataset.seq_name}",
        total=np.inf,
    ):
        if isinstance(mm, ImuMeasurement):
            imu_data.append(mm)
        elif isinstance(mm, LidarMeasurement):
            lidar_stamps.append(mm.stamp)

        # Keep a little fudge room since we skip some stamps at the start
        if length is not None and len(lidar_stamps) >= length + 500:
            break

    lidar_rate = len(lidar_stamps) / (lidar_stamps[-1] - lidar_stamps[0]).to_sec()
    imu_rate = len(imu_data) / (imu_data[-1].stamp - imu_data[0].stamp).to_sec()
    print("----Lidar rate:", lidar_rate, len(lidar_stamps))
    print("----Imu rate:", imu_rate)

    return imu_data, lidar_stamps, gt


def estimate_biases(
    imu_data: list[ImuMeasurement], imu_params: ImuParams, gt: Trajectory
) -> OptimizationResult:
    # Initialize orientation using the first imu measurements
    num_used = 100
    a = np.array([i.accel for i in imu_data[:num_used]])
    w = np.array([i.gyro for i in imu_data[:num_used]])
    ez_W = np.array([0, 0, 1])
    e_acc = np.mean(a, axis=0)
    e_acc /= np.linalg.norm(e_acc)
    angle = np.arccos(ez_W @ e_acc)
    poseDelta = np.zeros(6)
    poseDelta[:3] = np.cross(ez_W, e_acc)
    poseDelta[:3] *= angle / np.linalg.norm(poseDelta[:3])
    imu0_T_x0 = Pose3.Expmap(-poseDelta)

    # bias init estimate
    avg_w = np.mean(w, axis=0)
    vio_R_x0 = imu0_T_x0.rotation().matrix()
    avg_a = np.mean(a, axis=0)

    # Remove gravity from init frame, then back to body frame
    avg_a = vio_R_x0.T @ (vio_R_x0 @ avg_a + imu_params.gravity)
    avg_a[np.abs(avg_a) > 0.5] = 0.0
    avg_w[np.abs(avg_w) > 0.2] = 0.0
    bias0 = ConstantBias(avg_a, avg_w)

    # Line up both poses
    imu_idx = 0
    gt_idx = 0
    while gt.stamps[gt_idx] < imu_data[imu_idx].stamp:
        gt_idx += 1
    while imu_data[imu_idx].stamp < gt.stamps[gt_idx]:
        imu_idx += 1

    imu_data = imu_data[imu_idx:]
    gt.stamps = gt.stamps[gt_idx:]
    gt.poses = gt.poses[gt_idx:]

    # Align ground truth to imu init
    imu0_T_x0 = convert(imu0_T_x0)
    imu0_T_gt0 = imu0_T_x0 * gt.poses[0].inverse()
    gt.poses = [imu0_T_gt0 * p for p in gt.poses]

    # Create factor graph!
    graph = NonlinearFactorGraph()
    values = Values()
    pim = convert(imu_params)
    pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)  # type: ignore
    vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)  # type: ignore
    bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1)  # type: ignore

    pose = convert(gt.poses[0])
    values.insert(X(0), pose)
    values.insert(V(0), np.zeros(3))
    values.insert(B(0), bias0)

    graph.addPriorPose3(X(0), pose, pose_noise)
    graph.addPriorVector(V(0), np.zeros(3), vel_noise)
    graph.addPriorConstantBias(B(0), bias0, bias_noise)

    imu_idx = 1
    gt_idx = 1
    done = False
    while not done:
        if gt_idx >= len(gt.stamps) or imu_idx >= len(imu_data) - 1:
            done = True
            continue

        if imu_data[imu_idx].stamp < gt.stamps[gt_idx]:
            dt = (imu_data[imu_idx].stamp - imu_data[imu_idx - 1].stamp).to_sec()
            pim.integrateMeasurement(
                imu_data[imu_idx].accel, imu_data[imu_idx].gyro, dt
            )
            imu_idx += 1
        else:
            # Add all value estimates
            values.insert(X(gt_idx), pose)
            values.insert(V(gt_idx), np.zeros(3))
            values.insert(B(gt_idx), bias0)
            # Add prior factor
            pose = convert(gt.poses[gt_idx])
            graph.addPriorPose3(X(gt_idx), pose, pose_noise)
            # Add imu factor
            imu_factor = CombinedImuFactor(
                X(gt_idx - 1),
                V(gt_idx - 1),
                X(gt_idx),
                V(gt_idx),
                B(gt_idx - 1),
                B(gt_idx),
                pim,
            )
            graph.push_back(imu_factor)
            # reset things
            gt_idx += 1
            pim.resetIntegrationAndSetBias(bias0)

    # Optimize!
    params = LevenbergMarquardtParams()
    optimizer = LevenbergMarquardtOptimizer(graph, values, params)
    result = optimizer.optimize()

    biases = np.array([result.atConstantBias(B(i)).vector() for i in range(gt_idx)])
    gyro = biases[:, 3:]
    accel = biases[:, :3]
    poses = [convert(result.atPose3(X(i))) for i in range(gt_idx)]
    vel = np.array([result.atVector(V(i)) for i in range(gt_idx)])
    stamps = gt.stamps[:gt_idx]

    max_norm = np.max(np.linalg.norm(biases[:, :3], axis=1))
    # assert max_norm < 1.0, f"Max accel bias norm is too large: {max_norm}"
    print(f"----Max accel bias norm is {max_norm}")

    return OptimizationResult(
        stamps=stamps,
        poses=poses,
        vel=vel,
        gyro_bias=gyro,
        accel_bias=accel,
    )


def integrate_along_lidarscans(
    lidar_stamps: list[Stamp],
    imu_data: list[ImuMeasurement],
    opt_result: OptimizationResult,
    gravity: np.ndarray,
) -> list[tuple[Stamp, list[NavState]]]:
    results = []
    opt_iterator = StampIterator(
        opt_result.stamps, [i for i in range(len(opt_result.stamps))]
    )
    imu_iterator = StampIterator(
        [i.stamp for i in imu_data], [i for i in range(len(imu_data))]
    )
    for i, stamp in enumerate(lidar_stamps):
        opt_idx = opt_iterator.next(stamp)
        imu_idx = imu_iterator.next(stamp)

        if opt_idx is None or imu_idx is None:
            continue

        # print(stamp, "Lidar")
        # print(opt_result.stamps[opt_idx], "Opt")

        # Get initialization
        init = NavState(
            stamp=opt_result.stamps[opt_idx],
            pose=opt_result.poses[opt_idx],
            velocity=opt_result.vel[opt_idx],
        )
        accel_bias = opt_result.accel_bias[opt_idx]
        gyro_bias = opt_result.gyro_bias[opt_idx]

        next_stamp = (
            lidar_stamps[i + 1]
            if i + 1 < len(lidar_stamps)
            else Stamp.from_sec(stamp.to_sec() + 0.1)
        )

        # Integrate
        this_results = [deepcopy(init)]
        while imu_idx < len(imu_data) and imu_data[imu_idx].stamp < next_stamp:
            imu = imu_data[imu_idx]
            # print(imu.stamp, "Imu")

            # Update and save
            init.update(
                imu.accel - accel_bias, imu.gyro - gyro_bias, gravity, imu.stamp
            )
            this_results.append(deepcopy(init))

            imu_idx += 1

        results.append((stamp, this_results))

    return results


def integrate_entire_trajectory(
    imu_data: list[ImuMeasurement],
    opt_result: OptimizationResult,
    gravity: np.ndarray,
) -> list[NavState]:
    """Integrate the entire trajectory using the optimized biases. Mostly used to see how well optimization did"""
    state = NavState(
        stamp=opt_result.stamps[0],
        pose=opt_result.poses[0],
        velocity=opt_result.vel[0],
    )
    bias_idx = 0
    results = [deepcopy(state)]

    start_idx = 1
    while start_idx < len(imu_data) and imu_data[start_idx].stamp < state.stamp:
        start_idx += 1

    for i in range(start_idx, len(imu_data)):
        imu = imu_data[i]

        # Find the bias index
        while (
            bias_idx < len(opt_result.stamps) - 1
            and opt_result.stamps[bias_idx] < imu.stamp
        ):
            bias_idx += 1

        # unbias the imu data
        accel = imu.accel - opt_result.accel_bias[bias_idx]
        gyro = imu.gyro - opt_result.gyro_bias[bias_idx]

        # Update and save
        state.update(accel, gyro, gravity, imu.stamp)
        results.append(deepcopy(state))

    return results


def run(name: str, out_dir: Path, force: bool = False, length: Optional[int] = None):
    print("Running", name)
    dataset = DatasetBuilder.parse(name)[0].build()

    # Check if it's already been run
    dir = out_dir / "imu_integration"
    dir.mkdir(parents=True, exist_ok=True)
    filename = dir / f"{dataset.dataset_name()}_{dataset.seq_name}.pkl"
    if not force and filename.exists():
        print("----Skipping")
        print()
        return

    # run!
    imu_data, lidar_stamps, gt = load_data(dataset, length)
    opt_result = estimate_biases(imu_data, dataset.imu_params(), gt)
    integrated_results = integrate_along_lidarscans(
        lidar_stamps,
        imu_data,
        opt_result,
        cast(np.ndarray, dataset.imu_params().gravity),
    )

    with open(filename, "wb") as f:
        pickle.dump(integrated_results, f)
    print("----Done")
    print()


def test(name: str, length: Optional[int] = None):
    dataset = DatasetBuilder.parse(name)[0].build()
    imu_data, lidar_stamps, gt = load_data(dataset, length)
    opt_result = estimate_biases(imu_data, dataset.imu_params(), gt)
    results = integrate_entire_trajectory(
        imu_data, opt_result, cast(np.ndarray, dataset.imu_params().gravity)
    )

    import matplotlib.pyplot as plt

    # Plot biases to make sure they're changing
    fig, ax = plt.subplots(3, 2, layout="constrained")
    first_stamp = opt_result.stamps[0]
    stamps = np.asarray([r - first_stamp for r in opt_result.stamps])
    for i in range(3):
        ax[i, 0].plot(stamps, opt_result.gyro_bias[:, i])
        ax[i, 1].plot(stamps, opt_result.accel_bias[:, i])
    plt.savefig("figures/biases.png")

    # Ballpark how many imu measurements to plot
    opt_rate = (
        len(opt_result.stamps) / (opt_result.stamps[-1] - opt_result.stamps[0]).to_sec()
    )
    imu_rate = len(imu_data) / (imu_data[-1].stamp - imu_data[0].stamp).to_sec()
    diff = int(imu_rate / opt_rate)

    # plot the poses to make imu integration roughly matches trajectory
    fig, ax = plt.subplots(1, 1, layout="constrained")
    num = 400
    x = np.asarray([p.trans[0] for p in opt_result.poses])[:num]  # type: ignore
    y = np.asarray([p.trans[1] for p in opt_result.poses])[:num]  # type: ignore
    ax.scatter(x, y, s=1, alpha=0.5, label="Ground Truth")

    x = np.asarray([p.pose.trans[0] for p in results])[: num * diff]  # type: ignore
    y = np.asarray([p.pose.trans[1] for p in results])[: num * diff]  # type: ignore

    print("imu end:", results[num * diff].stamp)
    print("opt end:", opt_result.stamps[num])

    ax.scatter(x, y, s=1, alpha=0.5, label="Integrated")
    ax.legend()
    plt.savefig("figures/integrated_poses.png")

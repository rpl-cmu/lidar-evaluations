from typing import overload
from evalio.datasets.base import EVALIO_DATA, Dataset
from evalio.types import (
    ImuMeasurement,
    LidarMeasurement,
    SE3,
    SO3,
    ImuParams,
    Stamp,
    Trajectory,
)
from evalio.cli import DatasetBuilder
import numpy as np
from dataclasses import dataclass
from copy import deepcopy

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

from wrappers import NavState


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
def load_data(dataset: Dataset) -> tuple[list[ImuMeasurement], list[Stamp], Trajectory]:
    # Get ground truth (already in IMU frame)
    gt = dataset.ground_truth()

    # Gather all imu data
    imu_data: list[ImuMeasurement] = []
    lidar_stamps = []
    for mm in tqdm(dataset):
        if isinstance(mm, ImuMeasurement):
            imu_data.append(mm)
        elif isinstance(mm, LidarMeasurement):
            lidar_stamps.append(mm.stamp)

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

    imu_idx = 0
    gt_idx = 1
    done = False
    while not done:
        if gt_idx >= len(gt.stamps) or imu_idx >= len(imu_data):
            done = True
            continue

        if imu_data[imu_idx].stamp < gt.stamps[gt_idx]:
            dt = imu_data[imu_idx + 1].stamp - imu_data[imu_idx].stamp
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

    biases = np.array(
        [result.atConstantBias(B(i)).vector() for i in range(len(gt.stamps))]
    )
    gyro = biases[:, 3:]
    accel = biases[:, :3]
    poses = [convert(result.atPose3(X(i))) for i in range(len(gt.stamps))]
    vel = np.array([result.atVector(V(i)) for i in range(len(gt.stamps))])

    max_norm = np.max(np.linalg.norm(biases[:, :3], axis=1))
    assert max_norm < 1.0, f"Max accel bias norm is too large: {max_norm}"

    import matplotlib.pyplot as plt
    from wrappers import plt_show

    fig, ax = plt.subplots(3, 2, layout="constrained")
    for i in range(3):
        ax[i, 0].plot(gyro[:, i])
        ax[i, 1].plot(accel[:, i])
    plt_show("figures/biases.png")

    return OptimizationResult(
        stamps=gt.stamps, poses=poses, vel=vel, gyro_bias=gyro, accel_bias=accel
    )


def integrate_along_lidarscans(
    lidar_stamps: list[Stamp],
    imu_data: list[ImuMeasurement],
    opt_result: OptimizationResult,
    gravity: np.ndarray,
) -> list[tuple[Stamp, list[NavState]]]:
    results = []
    imu_idx = 1
    opt_idx = 0
    for i, stamp in enumerate(lidar_stamps):
        if imu_idx >= len(imu_data):
            continue
        elif opt_idx >= len(opt_result.stamps):
            continue
        # Skip lidar scans if they are before the first optimized pose
        elif stamp < opt_result.stamps[opt_idx]:
            continue

        # get closest imu data and ground truth pose
        while imu_idx < len(imu_data) and imu_data[imu_idx].stamp < stamp:
            imu_idx += 1
        while (
            opt_idx < len(opt_result.stamps)
            and opt_result.stamps[opt_idx] - stamp < -1e-2
        ):
            opt_idx += 1

        if imu_idx >= len(imu_data):
            continue
        elif opt_idx >= len(opt_result.stamps):
            continue

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
        while imu_data[imu_idx].stamp < next_stamp:
            imu = imu_data[imu_idx]

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


def save_results(dataset: Dataset, results: list[tuple[Stamp, list[NavState]]]):
    # Save results
    import pickle

    filename = (
        EVALIO_DATA / dataset.name() / dataset.seq / "imu_integration_results.pkl"
    )

    with open(filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    testing = False

    # Create a dataset
    # double checked gravity
    name = "newer_college_2021/quad-easy"
    # name = "newer_college_2020/01_short_experiment"
    # name = "multi_campus_2024/tuhh_day_04"
    # TODO: This one looks bad
    # name = "oxford_spires/keble_college_02"
    dataset = DatasetBuilder.parse(name)[0].build()

    imu_data, lidar_stamps, gt = load_data(dataset)
    # Try doing a smaller subset to make sure not moving at start
    # gt.poses = gt.poses[:5]
    # gt.stamps = gt.stamps[:5]
    opt_result = estimate_biases(imu_data, dataset.imu_params(), gt)

    # Check locally that it's integrating over the first sample decently
    if testing:
        results = integrate_entire_trajectory(
            imu_data, opt_result, dataset.imu_params().gravity
        )

        # plot the poses to double check they seem reasonable
        import matplotlib.pyplot as plt
        from wrappers import plt_show

        fig, ax = plt.subplots(1, 1, layout="constrained")
        num = 100
        x = np.asarray([p.trans[0] for p in opt_result.poses])[:num]
        y = np.asarray([p.trans[1] for p in opt_result.poses])[:num]
        ax.scatter(x, y, label="Ground Truth")

        x = np.asarray([p.pose.trans[0] for p in results])[: num * 40]
        y = np.asarray([p.pose.trans[1] for p in results])[: num * 40]

        ax.scatter(x, y, label="Integrated")
        ax.legend()
        plt_show("figures/integrated_poses.png")

    else:
        integrated_results = integrate_along_lidarscans(
            lidar_stamps, imu_data, opt_result, dataset.imu_params().gravity
        )
        save_results(dataset, integrated_results)

    # plot the lidarscans poses to double check they seem reasonable
    # import matplotlib.pyplot as plt
    # from wrappers import plt_show

    # fig, ax = plt.subplots(1, 1, layout="constrained")
    # start = 17
    # num = 1
    # x = np.asarray([p.pose.trans[0] for _, scan in integrated_results for p in scan])[
    #     start * 40 : (start + num) * 40
    # ]
    # y = np.asarray([p.pose.trans[1] for _, scan in integrated_results for p in scan])[
    #     start * 40 : (start + num) * 40
    # ]
    # ax.scatter(x, y, label="Integrated")

    # x = np.asarray([p.trans[0] for p in opt_result.poses])[start - 1 : start + num + 1]
    # y = np.asarray([p.trans[1] for p in opt_result.poses])[start - 1 : start + num + 1]
    # ax.scatter(x, y, label="Ground Truth")
    # ax.legend()
    # plt_show("figures/integrated_poses.png")

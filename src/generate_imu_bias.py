from typing import overload
from evalio.types import ImuMeasurement, LidarMeasurement, SE3, SO3, ImuParams
from evalio.cli import DatasetBuilder
import numpy as np
from dataclasses import dataclass

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


# Create a dataset
# double checked
name = "hilti_2022/construction_upper_level_1"
# name = "newer_college_2021/quad-easy"
# name = "newer_college_2020/01_short_experiment"
# TODO: This one looks bad
# name = "oxford_spires/keble_college_02"

# def estimate_biases(name: str):
dataset = DatasetBuilder.parse(name)[0].build()

# Get ground truth
gt = dataset.ground_truth()

# Gather all imu data
imu_data: list[ImuMeasurement] = []
# lidar_stamps = []
for mm in tqdm(dataset):
    if isinstance(mm, ImuMeasurement):
        imu_data.append(mm)
    # elif isinstance(mm, LidarMeasurement):
    #     lidar_stamps.append(mm.stamp)


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
avg_a = vio_R_x0.T @ (vio_R_x0 @ avg_a + dataset.imu_params().gravity)
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
pim = convert(dataset.imu_params())
pose_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)  # type: ignore
vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-4)  # type: ignore
bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-2)  # type: ignore

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
        pim.integrateMeasurement(imu_data[imu_idx].accel, imu_data[imu_idx].gyro, dt)
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


# plot the biases to double check they seem reasonable
# import matplotlib.pyplot as plt
# from wrappers import plt_show

# biases = np.array([result.atConstantBias(B(i)).vector() for i in range(len(gt.stamps))])

# max_norm = np.max(np.linalg.norm(biases[:, :3], axis=1))
# print(result.atConstantBias(B(0)), result.atConstantBias(B(1000)))
# print("max accel bias norm: ", max_norm)

# fig, ax = plt.subplots(2, 3, layout="constrained")
# for i in range(3):
#     ax[0, i].plot(biases[:, i])
#     ax[1, i].plot(biases[:, i + 3])
# plt_show("figures/bias.png")

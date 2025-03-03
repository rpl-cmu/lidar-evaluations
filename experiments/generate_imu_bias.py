from env import ALL_TRAJ, INC_DATA_DIR, LEN
from lidar_eval.imu_bias import run

if __name__ == "__main__":
    for name in ALL_TRAJ:
        run(name, out_dir=INC_DATA_DIR, force=False, length=LEN)

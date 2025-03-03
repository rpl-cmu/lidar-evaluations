import sys

sys.path.append("src")
from env import ALL_TRAJ, INC_DATA_DIR, LEN
from params import ExperimentParams, Feature, Initialization
from run_init import run_multithreaded
from wrappers import parser

dir = INC_DATA_DIR / "init_imu"


def run(num_threads: int):
    experiments = [
        ExperimentParams(
            name="InitOnly",
            dataset=d,
            curvature_planar_threshold=1000.0,
            init=Initialization.Imu,
            features=[Feature.Planar],
        )
        for d in ALL_TRAJ
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


if __name__ == "__main__":
    args = parser("init_imu")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        raise ValueError("No plotting for this experiment")

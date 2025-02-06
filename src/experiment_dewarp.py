from itertools import product
from pathlib import Path

from params import Dewarp, ExperimentParams, Feature, Initialization
from run import run_multithreaded

# ------------------------- Everything to sweep over ------------------------- #
num_threads = 12
dir = Path("results/25.02.04_dewarp")

datasets = [
    "hilti_2022/construction_upper_level_1",
    "oxford_spires/keble_college_02",
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
]

dewarp = [
    Dewarp.Identity,
    Dewarp.ConstantVelocity,
    Dewarp.GroundTruth,
]

# ------------------------- Computer product of options ------------------------- #
experiments = [
    ExperimentParams(
        name=de.name,
        dataset=d,
        features=[Feature.Planar],
        dewarp=de,
        init=Initialization.GroundTruth,
    )
    for (d, de) in product(datasets, dewarp)
]

run_multithreaded(experiments, dir, num_threads=num_threads)

from itertools import product
from pathlib import Path

from params import Dewarp, ExperimentParams, Feature, Initialization
from run import run_multithreaded

# ------------------------- Everything to sweep over ------------------------- #
num_threads = 20
dir = Path("results/25.02.06_dewarp_with_init")

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

init = [
    Initialization.Identity,
    Initialization.ConstantVelocity,
    Initialization.GroundTruth,
]

# ------------------------- Computer product of options ------------------------- #
experiments = [
    ExperimentParams(
        name=f"{de.name}_{i.name}",
        dataset=d,
        features=[Feature.Planar],
        dewarp=de,
        init=i,
    )
    for (d, de, i) in product(datasets, dewarp, init)
]

run_multithreaded(experiments, dir, num_threads=num_threads)

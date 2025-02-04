from itertools import product
from pathlib import Path

import numpy as np

from params import ExperimentParams, Feature, Initialization
from run import run_multithreaded

# ------------------------- Everything to sweep over ------------------------- #
num_threads = 20
dir = Path("results/25.02.04_pseudo")

datasets = [
    "hilti_2022/construction_upper_level_1",
    "oxford_spires/keble_college_02",
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
]

# ------------------------- Computer product of options ------------------------- #
experiments_planar = [
    ExperimentParams(
        name="planar",
        dataset=d,
        features=[Feature.Planar],
        init=Initialization.GroundTruth,
    )
    for (d,) in product(datasets)
]

experiments_pseudo = [
    ExperimentParams(
        name=f"pseudo_{i:.3f}",
        dataset=d,
        features=[Feature.Pseudo_Planar],
        pseudo_planar_epsilon=float(i),
        init=Initialization.GroundTruth,
    )
    for d, i in product(datasets, np.linspace(0.01, 1.0, 20))
]

run_multithreaded(experiments_planar + experiments_pseudo, dir, num_threads=num_threads)

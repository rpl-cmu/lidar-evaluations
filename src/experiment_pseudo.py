from itertools import product
from pathlib import Path

import numpy as np

from params import ExperimentParams, Feature, Initialization
from run import run_multithreaded

# ------------------------- Everything to sweep over ------------------------- #
num_threads = 15
dir = Path("results/25.02.05_psuedo_with_init")

datasets = [
    "hilti_2022/construction_upper_level_1",
    "oxford_spires/keble_college_02",
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
]

init = [
    Initialization.Identity,
    Initialization.ConstantVelocity,
    Initialization.GroundTruth,
]

# ------------------------- Computer product of options ------------------------- #
experiments_planar = [
    ExperimentParams(
        name=f"planar_{i.name}",
        dataset=d,
        features=[Feature.Planar],
        init=i,
    )
    for (d, i) in product(datasets, init)
]

experiments_pseudo = [
    ExperimentParams(
        name=f"pseudo_{i.name}_{val:.3f}",
        dataset=d,
        features=[Feature.Pseudo_Planar],
        pseudo_planar_epsilon=float(val),
        init=i,
    )
    for d, i, val in product(datasets, init, np.linspace(0.00, 1.0, 20)[1:])
]

experiments = experiments_planar + experiments_pseudo

run_multithreaded(experiments, dir, num_threads=num_threads)

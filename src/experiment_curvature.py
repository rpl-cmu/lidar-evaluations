from itertools import chain, combinations, product
from pathlib import Path

import numpy as np

from params import Curvature, ExperimentParams, Feature, Initialization
from run import run_multithreaded

# For handling multiple tqdm progress bars
# https://stackoverflow.com/questions/56665639/fix-jumping-of-multiple-progress-bars-tqdm-in-python-multiprocessing


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1)))


# ------------------------- Everything to sweep over ------------------------- #
num_threads = 20
dir = Path("results/25.02.03_curvature")

features = [[Feature.Planar, Feature.Edge]]

datasets = [
    "hilti_2022/construction_upper_level_1",
    "oxford_spires/keble_college_02",
    "newer_college_2020/01_short_experiment",
    "newer_college_2021/quad-easy",
]

# ------------------------- Computer product of options ------------------------- #
experiments_loam = [
    ExperimentParams(
        name=f"loam_{i:.3f}",
        dataset=d,
        features=[Feature.Planar],
        curvature=Curvature.Loam,
        curvature_planar_threshold=float(i),
        init=Initialization.ConstantVelocity,
    )
    for d, i in product(datasets, np.linspace(0.01, 5.0, 20))
]

experiments_eigen = [
    ExperimentParams(
        name=f"eigen_{i:.3f}",
        dataset=d,
        features=[Feature.Planar],
        curvature=Curvature.Eigen,
        curvature_planar_threshold=float(i),
        init=Initialization.ConstantVelocity,
    )
    for d, i in product(datasets, np.linspace(0.01, 5.0, 20))
]

run_multithreaded(experiments_loam + experiments_eigen, dir, num_threads=num_threads)

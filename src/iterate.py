from itertools import chain, combinations, product
from pathlib import Path

from params import ExperimentParams, Feature, Initialization
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
dir = Path("results/full_features")

features = powerset([Feature.Planar, Feature.Edge, Feature.Point]) + powerset(
    [Feature.Pseudo_Planar, Feature.Edge, Feature.Point]
)

datasets = [
    "newer_college_2020/01_short_experiment",
    "newer_college_2020/02_long_experiment",
    # "newer_college_2020/07_parkland_mound",
    # "newer_college_2021/quad-easy",
    # "newer_college_2021/maths-easy",
]

init = [
    Initialization.ConstantVelocity,
    Initialization.GroundTruth,
    Initialization.Identity,
]

# ------------------------- Computer product of options ------------------------- #
experiments = [
    ExperimentParams(
        name=f"{i.name.lower()}_{'_'.join(j.name.lower() for j in f)}",
        dataset=d,
        features=list(f),
        init=i,
    )
    for d, f, i in product(datasets, features, init)
]

run_multithreaded(experiments, dir)

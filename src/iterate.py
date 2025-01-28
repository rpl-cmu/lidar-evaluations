from itertools import chain, combinations, product
from pathlib import Path

from params import ExperimentParams, Feature, Initialization
from run import run

from multiprocessing import Pool
from tqdm import tqdm

# For handling multiple tqdm progress bars
# https://stackoverflow.com/questions/56665639/fix-jumping-of-multiple-progress-bars-tqdm-in-python-multiprocessing


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


# ------------------------- Everything to sweep over ------------------------- #
num_threads = 15
dir = Path("results/first_run")

features = powerset([Feature.Planar, Feature.Edge])

datasets = [
    # "newer_college_2020/01_short_experiment",
    # "newer_college_2020/02_long_experiment",
    # "newer_college_2020/07_parkland_mound",
    "newer_college_2021/quad-easy",
    "newer_college_2021/maths-easy",
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
args = [(e, dir, True) for e in experiments]

print(f"Running {len(experiments)} experiments")
with Pool(num_threads, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock) as p:
    p.starmap(run, args)

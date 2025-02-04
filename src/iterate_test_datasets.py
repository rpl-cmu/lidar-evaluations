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
dir = Path("results/25.02.03_no_registration")

features = powerset([Feature.Planar, Feature.Edge, Feature.Point]) + powerset(
    [Feature.Pseudo_Planar, Feature.Edge, Feature.Point]
)
features = [[Feature.Planar, Feature.Edge]]

datasets = [
    # "helipr/kaist_05",
    # "helipr/dcc_05",
    # "helipr/riverside_05",
    # "hilti_2022/construction_upper_level_1",
    # "hilti_2022/basement_2",
    # "multi_campus_2024/ntu_day_01",
    # "multi_campus_2024/kth_day_06",
    # "multi_campus_2024/tuhh_day_02",
    # "oxford_spires/blenheim_palace_01",
    # "oxford_spires/keble_college_02",
    # "oxford_spires/observatory_quarter_01",
    # "newer_college_2020/01_short_experiment",
    # "newer_college_2021/quad-easy",
]

init = [
    # Initialization.ConstantVelocity,
    Initialization.GroundTruth,
    # Initialization.Identity,
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

run_multithreaded(experiments, dir, num_threads=num_threads)

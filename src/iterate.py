from itertools import chain, combinations
from pathlib import Path

from params import ExperimentParams, Feature
from run import run


# https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "Subsequences of the iterable from shortest to longest."
    # powerset([1,2,3]) → () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))


dir = Path("results/first_run")
features = [Feature.Planar, Feature.Edge]
datasets = [
    "newer_college_2020/01_short_experiment",
    "newer_college_2020/02_long_experiment",
    "newer_college_2021/quad-easy",
    "newer_college_2021/maths-easy",
]

eps = []
for dataset in datasets:
    for feature_set in powerset(features):
        name = "_".join(f.name.lower() for f in feature_set)
        ep = ExperimentParams(name=name, dataset=dataset, features=list(feature_set))
        eps.append(ep)


print(f"Running {len(eps)} experiments")
for ep in eps:
    print(f"Beginning experiment {ep.name} on dataset {ep.dataset}")
    run(ep, dir)

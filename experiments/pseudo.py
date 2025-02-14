from itertools import product
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import polars as pl

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show
from run import run_multithreaded
from stats import compute_cache_stats

dir = Path("results/25.02.14_pseudo_with_imu")
# dir = Path("results/25.02.10_pseudo_new_projector_tighter_range")


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    datasets = [
        "hilti_2022/construction_upper_level_1",
        "oxford_spires/keble_college_02",
        # "newer_college_2020/01_short_experiment",
        "newer_college_2021/quad-easy",
    ]

    init = [
        Initialization.Identity,
        Initialization.ConstantVelocity,
        Initialization.Imu,
        Initialization.GroundTruth,
    ]

    epsilon = np.linspace(0.0, 1.0, 11)

    # ------------------------- Computer product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"pseudo_{i.name}_{val:.3f}",
            dataset=d,
            features=[Feature.Planar],
            pseudo_planar_epsilon=float(val),
            init=i,
        )
        for d, i, val in product(datasets, init, epsilon)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads)


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)
    df = df.filter(~pl.col("name").str.contains("planar"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=False)
    sns.lineplot(
        df,
        ax=ax[0],
        x="pseudo_planar_epsilon",
        y="w100_RTEr",
        hue="dataset",
        style="init",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "GroundTruth"],
        dashes=False,
    )
    sns.lineplot(
        df,
        ax=ax[1],
        x="pseudo_planar_epsilon",
        y="w100_RTEt",
        hue="dataset",
        style="init",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "GroundTruth"],
        dashes=False,
        legend=False,
    )
    ax[0].legend().set_visible(False)
    fig.legend(ncol=2, loc="outside lower center")
    plt_show(Path("figures") / f"{name}.png")


if __name__ == "__main__":
    args = parser("pseudo")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

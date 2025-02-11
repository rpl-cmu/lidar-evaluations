from itertools import product
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization, Dewarp
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats


# dir = Path("results/25.02.06_dewarp_with_init")
dir = Path("results/25.02.10_dewarp_multi_campus")


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    datasets = [
        "multi_campus_2024/ntu_day_02",
        "multi_campus_2024/ntu_day_10",
        "multi_campus_2024/kth_day_09",
        "multi_campus_2024/kth_day_10",
        "multi_campus_2024/tuhh_day_04",
        "multi_campus_2024/tuhh_night_09",
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

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"{de.name}_{i.name}",
            dataset=d,
            features=[Feature.Point_Plane],
            dewarp=de,
            init=i,
        )
        for (d, de, i) in product(datasets, dewarp, init)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads)


def make_plot(df: pl.DataFrame, name: str, to_plot: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=False)
    sns.lineplot(
        df.filter(pl.col("init") == "GroundTruth"),
        ax=ax[0],
        x="dewarp",
        y=f"{to_plot}r",
        hue="dataset",
        # style="init",
        marker="o",
        sort=True,
    )
    sns.lineplot(
        df.filter(pl.col("init") == "GroundTruth"),
        ax=ax[1],
        x="dewarp",
        y=f"{to_plot}t",
        hue="dataset",
        # style="init",
        marker="o",
        sort=True,
        legend=False,
    )
    ax[0].legend().set_visible(False)
    # ax[0].set_ylim(0, 5)
    fig.legend(ncol=2, loc="outside lower center")
    plt_show(Path("figures") / f"{name}_{to_plot}.png")


def plot(name: str, force: bool):
    setup_plot()
    df = compute_cache_stats(dir, force)
    make_plot(df, name, "RTE")
    # make_plot(df, name, "w10_RTE")
    make_plot(df, name, "w100_RTE")
    # make_plot(df, name, "ATE")


if __name__ == "__main__":
    args = parser("dewarp")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

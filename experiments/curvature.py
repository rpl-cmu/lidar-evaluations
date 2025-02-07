from itertools import product
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from params import Curvature, ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show
from run import run_multithreaded
from stats import compute_cache_stats


dir = Path("results/25.02.07_curvature_with_init")


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    datasets = [
        "hilti_2022/construction_upper_level_1",
        "oxford_spires/keble_college_02",
        # "newer_college_2020/01_short_experiment",
        "newer_college_2021/quad-easy",
    ]

    curvature = np.linspace(0.00, 5.0, 11)[1:]

    init = [
        Initialization.GroundTruth,
        Initialization.ConstantVelocity,
        Initialization.Identity,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments_loam = [
        ExperimentParams(
            name=f"loam_{i.name}_{c:.3f}",
            dataset=d,
            features=[Feature.Planar],
            curvature=Curvature.Loam,
            curvature_planar_threshold=float(c),
            init=i,
        )
        for d, c, i in product(datasets, curvature, init)
    ]

    experiments_eigen = [
        ExperimentParams(
            name=f"eigen_{i.name}_{c:.3f}",
            dataset=d,
            features=[Feature.Planar],
            curvature=Curvature.Eigen,
            curvature_planar_threshold=float(c),
            init=i,
        )
        for d, c, i in product(datasets, curvature, init)
    ]

    run_multithreaded(
        experiments_loam + experiments_eigen, dir, num_threads=num_threads
    )


def make_plot(df: pl.DataFrame, name: str, to_plot: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=True)

    ax[0].set_title("Loam")
    sns.lineplot(
        df.filter(pl.col("curvature") == "Loam"),
        ax=ax[0],
        x="curvature_planar_threshold",
        y=to_plot,
        hue="dataset",
        style="init",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "GroundTruth"],
        dashes=False,
    )
    ax[1].set_title("Eigen")
    sns.lineplot(
        df.filter(pl.col("curvature") == "Eigen"),
        ax=ax[1],
        x="curvature_planar_threshold",
        y=to_plot,
        hue="dataset",
        style="init",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "GroundTruth"],
        dashes=False,
        legend=False,
    )
    ax[0].legend().set_visible(False)
    fig.legend(ncol=2, loc="outside lower center")
    plt_show(f"figures/{name}_{to_plot}.png")


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force)
    make_plot(df, name, "RTEt")
    make_plot(df, name, "planar")


if __name__ == "__main__":
    args = parser("curvature")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

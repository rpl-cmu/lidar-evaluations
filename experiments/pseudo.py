from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import polars as pl

sys.path.append("src")
from lidar_eval.params import ExperimentParams, Feature, Initialization
from lidar_eval.run import run_multithreaded
from lidar_eval.stats import compute_cache_stats, eval
from env import (
    INC_DATA_DIR,
    RESULTS_DIR,
    SUBSET_TRAJ,
    LEN,
    COL_WIDTH,
    parser,
    plt_show,
    setup_plot,
)

dir = RESULTS_DIR / "25.02.25_pseudo_plane"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    init = [
        Initialization.Identity,
        Initialization.ConstantVelocity,
        Initialization.Imu,
        Initialization.GroundTruth,
    ]

    epsilon: list[float] = np.linspace(0.0, 1.0, 11).tolist()

    datasets = [d for d in SUBSET_TRAJ if ("new" in d or "oxford" in d or "multi" in d)]

    # ------------------------- Computer product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"pseudo_{i.name}_{val:.3f}",
            dataset=d,
            features=[Feature.Planar],
            pseudo_planar_epsilon=float(val),
            init=i,
        )
        for i, val, d in product(init, epsilon, datasets)
    ]

    run_multithreaded(
        experiments, dir, INC_DATA_DIR, num_threads=num_threads, length=LEN
    )


def plot(name: str, force: bool):
    palette = setup_plot()

    first = "Newer College Stereo-Cam"
    second = "Multi-Campus"
    # first = "Oxford Spires"
    # first = "Newer College Multi-Cam"

    df = compute_cache_stats(dir, force=force)
    # df = df.filter(pl.col("Initialization").eq("Constant Velocity"))

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(COL_WIDTH + 0.5, 1.3),
        layout="constrained",
        sharey=True,
        sharex=False,
    )
    sns.lineplot(
        df.filter(pl.col("Dataset").eq(first)),
        ax=ax[0],
        x="pseudo_planar_epsilon",
        y="w100_RTEt",
        hue="Dataset",
        style="Initialization",
        markers=["s", "X", "o", "P"],
        style_order=["Identity", "Constant Velocity", "IMU", "Ground Truth"],
        dashes=False,
        palette=palette,
    )
    sns.lineplot(
        df.filter(pl.col("Dataset").eq(second)),
        ax=ax[1],
        x="pseudo_planar_epsilon",
        y="w100_RTEt",
        hue="Dataset",
        style="Initialization",
        markers=["s", "X", "o", "P"],
        style_order=["Identity", "Constant Velocity", "IMU", "Ground Truth"],
        dashes=False,
        palette=palette,
        legend=False,
    )

    # Reorder the legend to put blank one in the right spot
    handles, labels = ax[0].get_legend_handles_labels()
    handles = handles[-4:]
    labels = labels[-4:]

    # ax[0].set_yscale("log")

    for i in range(2):
        ax[i].tick_params(axis="x", pad=-2)
        ax[i].tick_params(axis="y", pad=-1)
        ax[i].set_xlabel(r"$\epsilon$", labelpad=0)
    ax[0].set_ylabel(r"$RTEt_{10}\ (m)$", labelpad=2)
    ax[0].legend().set_visible(False)

    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=2,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=10.6,
        bbox_to_anchor=(0.072, 1.32),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(name)


if __name__ == "__main__":
    args = parser("pseudo")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "stats":
        eval([dir])
    elif args.action == "plot":
        plot(args.name, args.force)

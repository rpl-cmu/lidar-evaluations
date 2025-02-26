from itertools import product
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import polars as pl

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats
from env import RESULTS_DIR, SUBSET_TRAJ, LEN, COL_WIDTH

dir = RESULTS_DIR / "25.02.25_pseudo_plane"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    init = [
        Initialization.Identity,
        Initialization.ConstantVelocity,
        Initialization.Imu,
        Initialization.GroundTruth,
    ]

    epsilon = np.linspace(0.0, 1.0, 11)

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

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot(name: str, force: bool):
    palette = setup_plot()

    good = [
        "Newer College Stereo-Cam",
        "Newer College Multi-Cam",
        "Multi-Campus",
        "Oxford Spires",
        # "Botanic Garden",
        # "Hilti 2022",
    ]

    df = compute_cache_stats(dir, force=force)
    df = df.filter(pl.col("Dataset").is_in(good))

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH + 0.5, 2.0),
        layout="constrained",
        sharey=False,
        sharex=True,
    )
    sns.lineplot(
        df,
        ax=ax,
        x="pseudo_planar_epsilon",
        y="w100_RTEt",
        hue="Dataset",
        style="Initialization",
        markers=["s", "X", "o", "P"],
        style_order=["Identity", "Constant Velocity", "IMU", "Ground Truth"],
        dashes=False,
        palette=palette,
    )
    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-4:]
    labels = labels[-4:]

    ax.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", pad=-1)
    ax.set_xlabel(r"$\epsilon$", labelpad=1)
    ax.set_ylabel(r"$RTEt_{10}\ (m)$", labelpad=2)
    ax.legend().set_visible(False)

    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=2,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=10.6,
        bbox_to_anchor=(0.07, 1.20),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(name)


if __name__ == "__main__":
    args = parser("pseudo")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

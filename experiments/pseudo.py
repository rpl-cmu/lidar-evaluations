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
from env import RESULTS_DIR, SUBSET_TRAJ, LEN

dir = RESULTS_DIR / "25.02.18_pseudo_plane"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
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
        for i, val, d in product(init, epsilon, SUBSET_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot(name: str, force: bool):
    palette = setup_plot()

    good = [
        "Newer College Stereo-Cam",
        "Newer College Multi-Cam",
        "Multi-Campus",
        "Oxford Spires",
        "Botanic Garden",
        # "Hilti 2022",
    ]

    df = compute_cache_stats(dir, force=force)
    df = df.filter(pl.col("Dataset").is_in(good))

    fig, ax = plt.subplots(1, 1, figsize=(5, 3), layout="constrained", sharey=True)
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
    # blank line for the legend
    new_handles = [4, 5, 8, 9]
    for _ in new_handles:
        ax.plot(np.NaN, np.NaN, "-", color="none", label=" ")
    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    for i in new_handles:
        handles.insert(i, handles.pop(-1))
        labels.insert(i, labels.pop(-1))

    ax.tick_params(axis="x", pad=-1)
    ax.tick_params(axis="y", pad=-1)
    ax.set_xlabel(r"$\epsilon$", labelpad=1)
    ax.set_ylabel(r"$RTEt_{10}\ (m)$", labelpad=1)
    ax.legend().set_visible(False)

    fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        loc="outside lower center",
        labelspacing=0.15,
    )
    plt_show(name)


if __name__ == "__main__":
    args = parser("pseudo")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

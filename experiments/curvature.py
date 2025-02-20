from itertools import product
from pathlib import Path
from typing import cast
from matplotlib.patches import Rectangle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from env import SUBSET_TRAJ, LEN
from params import Curvature, ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats


dir = Path("results/25.02.20_curvature")


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    curvature_values = np.linspace(0.00, 2.5, 6)[1:]

    curvatures = [
        Curvature.Loam,
        Curvature.Eigen,
        Curvature.Eigen_NN,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"{c.name}_{c_val:.3f}",
            dataset=d,
            features=[Feature.Planar],
            curvature=c,
            curvature_planar_threshold=float(c_val),
            init=Initialization.ConstantVelocity,
        )
        for c, c_val, d in product(curvatures, curvature_values, SUBSET_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def make_plot_bar(ax, df: pl.DataFrame, name: str, to_plot: str):
    palette = setup_plot()
    df = df.filter(
        pl.col("curvature_planar_threshold").eq(1.0)
        # & pl.col("Dataset").eq("HeLiPR").not_()
        # & pl.col("Dataset").eq("Botanic Garden").not_()
    )

    ax.set_title("Loam")
    sns.barplot(
        df,
        ax=ax,
        x="Dataset",
        y=to_plot,
        hue="curvature",
    )

    markers = ["s", "X", "o"]
    names = ["Loam", "NN-Eigen", "Scanline-Eigen"]

    num_datasets = len(palette)
    tick_labels = ax.get_xticklabels()
    bars = [b for b in ax.get_children() if isinstance(b, Rectangle)]
    for i, label in enumerate(tick_labels):
        for j in range(3):
            # Change bar colors to match the dataset
            color = palette[label.get_text()]
            rect = cast(Rectangle, bars[i + j * num_datasets])
            rect.set_facecolor(color)
            # Add shape to top of bar
            x = rect.get_x() + rect.get_width() / 2
            y = rect.get_height()
            ax.plot(x, y, marker=markers[j], color=color, markeredgecolor="white")

    # Remove dataset names
    ax.set_xticklabels([])
    ax.set_xlabel("Dataset", labelpad=-5)

    # blank line for the legend
    for m, n in zip(markers, names):
        sns.lineplot([np.NaN], marker=m, color="black", label=n, ax=ax)

    ax.legend().set_visible(False)


def make_plot_line(df: pl.DataFrame, name: str, to_plot: str):
    fig, ax = plt.subplots(1, 1, figsize=(5, 3), layout="constrained")

    ax.set_title("Loam")
    sns.lineplot(
        df,
        ax=ax,
        x="curvature_planar_threshold",
        y=to_plot,
        hue="dataset",
        style="curvature",
        markers=["s", "X", "o"],
        # style_order=["Identity", "ConstantVelocity", "GroundTruth"],
        dashes=False,
    )
    ax.legend().set_visible(False)
    # fig.legend(ncol=2, loc="outside lower center")
    plt_show(f"{name}_{to_plot}")


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force)

    _palette = setup_plot()
    fig, ax = plt.subplots(1, 2, figsize=(8.5, 2), layout="constrained")

    make_plot_bar(ax[0], df, name, "planar")
    make_plot_bar(ax[1], df, name, "w100_RTEt")

    handles, labels = ax[0].get_legend_handles_labels()
    handles = handles[-3:]
    labels = labels[-3:]

    ax[0].set_ylabel("Number of Planar Features")
    ax[1].set_ylabel(r"$RTEt_{10}\ (m)$")

    fig.legend(
        ncol=3,
        loc="outside lower center",
        labels=labels,
        handles=handles,
    )
    plt_show(name)

    make_plot_line(df, name, "planar")
    make_plot_line(df, name, "w100_RTEt")


if __name__ == "__main__":
    args = parser("curvature")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from env import ALL_TRAJ, LEN, COL_WIDTH, RESULTS_DIR
from params import Curvature, ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats

"""
Test for the effect of curvature computations methods

Changelog
- 25.02.20: Switched curvature value to 7.0 to make each method get similar number of features. Also switch to ground truth initialization

"""

dir = RESULTS_DIR / "25.02.25_curvature_new_nn"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    curvatures = [
        Curvature.Loam,
        Curvature.Eigen,
        Curvature.Eigen_NN,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=c.name,
            dataset=d,
            features=[Feature.Planar],
            curvature=c,
            curvature_planar_threshold=7.0,
            init=Initialization.GroundTruth,
        )
        for c, d in product(curvatures, ALL_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot_classic(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    metric = "w100_RTEt"
    # metric = "planar"

    # Get identity versions
    df_identity = df.filter(pl.col("Curvature") == "Classic").select(
        ["dataset", metric]
    )

    # compute percent improvement
    df = df.with_columns(percent=pl.zeros(df.shape[0]))
    for dataset in df["dataset"].unique():
        id_val = df_identity.filter(pl.col("dataset") == dataset)[0, metric]
        df = df.with_columns(
            percent=pl.when(pl.col("dataset") == dataset)
            # THIS IS THE LINE THAT ACTUALLY COMPUTES THINGS
            .then(100 * (pl.col(metric) - id_val) / id_val)
            .otherwise(pl.col("percent"))
        )

    _c = setup_plot()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH + 0.5, 1.75),
        layout="constrained",
        sharey=False,
        sharex=True,
    )

    sns.lineplot(
        df,
        ax=ax,
        x="Trajectory",
        y="percent",
        hue="Dataset",
        style="Curvature",
        markers=["s", "X", "o"],
        style_order=["Classic", "Scanline-Eigen", "NN-Eigen"],
        dashes=False,
        legend=True,
    )

    # Remove dataset from legend
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-3:]
    labels = labels[-3:]

    ax.legend().set_visible(False)
    ax.tick_params(axis="x", pad=-3, rotation=90, labelsize=7)
    ax.tick_params(axis="y", pad=-2.4)

    traj = df.select("Trajectory").unique().to_numpy()
    xmax = len(traj) - 1
    extra = 1.0
    ax.set_xlim(0.0 - extra, xmax + extra)

    curr_ylim = ax.get_ylim()
    ax.set_ylim(curr_ylim[0], curr_ylim[1] + 1)

    ax.set_ylabel(r"$\text{RTEt}_{10}$ Change (%)", labelpad=3)
    # ax.set_yscale("log")
    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=3.3,
        bbox_to_anchor=(0.091, 1.155),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(name)


if __name__ == "__main__":
    args = parser("curvature")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot_classic(args.name, args.force)

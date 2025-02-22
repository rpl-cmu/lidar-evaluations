from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import polars as pl

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats
from env import ALL_TRAJ, LEN, RESULTS_DIR, COL_WIDTH

# ------------------------- Everything to sweep over ------------------------- #
# TODO: Rerun with ground truth
# dir = RESULTS_DIR / "25.02.20_plane_plane"
dir = RESULTS_DIR / "25.02.18_plane_plane_cv"


def run(num_threads: int):
    use_plane_to_plane = [True, False]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"plane_plane_{pp}",
            dataset=d,
            features=[Feature.Planar],
            use_plane_to_plane=pp,
            init=Initialization.GroundTruth,
        )
        for pp, d in product(use_plane_to_plane, ALL_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    # Get identity versions
    df_point_plane = df.filter(pl.col("Planar Type") == "Point-To-Plane").select(
        ["dataset", "w100_RTEt"]
    )

    # compute percent improvement
    df = df.with_columns(percent=pl.zeros(df.shape[0]))
    for dataset in df["dataset"].unique():
        id_val = df_point_plane.filter(pl.col("dataset") == dataset)[0, "w100_RTEt"]
        df = df.with_columns(
            percent=pl.when(pl.col("dataset") == dataset)
            # THIS IS THE LINE THAT ACTUALLY COMPUTES THINGS
            .then(100 * (pl.col("w100_RTEt") - id_val) / id_val)
            .otherwise(pl.col("percent"))
        )

    _c = setup_plot()
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
        x="Trajectory",
        y="percent",
        hue="Dataset",
        style="Planar Type",
        markers=["s", "X"],
        style_order=["Point-To-Plane", "Plane-To-Plane"],
        dashes=False,
        legend=True,
    )
    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-2:]
    labels = labels[-2:]

    ax.legend().set_visible(False)
    ax.tick_params(axis="x", pad=-3, rotation=90, labelsize=7)
    ax.tick_params(axis="y", pad=-2.4)

    ax.set_ylabel(r"$\text{RTEt}_{10}$ Change (%)", labelpad=3)

    traj = df.select("Trajectory").unique().to_numpy()
    xmax = len(traj) - 1
    extra = 1.0
    ax.set_xlim(0.0 - extra, xmax + extra)

    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=9.9,
        bbox_to_anchor=(0.1055, 1.13),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(name)


if __name__ == "__main__":
    args = parser("plane_plane")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

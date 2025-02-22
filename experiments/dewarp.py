from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization, Dewarp
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats
from env import ALL_TRAJ, LEN, RESULTS_DIR, COL_WIDTH


dir = RESULTS_DIR / "25.02.18_dewarp_with_edges"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    dewarp = [
        Dewarp.Identity,
        Dewarp.ConstantVelocity,
        Dewarp.Imu,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=de.name,
            dataset=d,
            features=[Feature.Planar, Feature.Edge],
            dewarp=de,
            init=Initialization.ConstantVelocity,
        )
        for (de, d) in product(dewarp, ALL_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    # Get identity versions
    df_identity = df.filter(pl.col("Dewarp") == "None").select(["dataset", "w100_RTEt"])

    # compute percent improvement
    df = df.with_columns(percent=pl.zeros(df.shape[0]))
    for dataset in df["dataset"].unique():
        id_val = df_identity.filter(pl.col("dataset") == dataset)[0, "w100_RTEt"]
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
        figsize=(COL_WIDTH + 0.5, 2.25),
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
        style="Dewarp",
        markers=["s", "X", "o"],
        style_order=["None", "Constant Velocity", "IMU"],
        dashes=False,
        legend=True,
    )
    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-3:]
    labels = labels[-3:]

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
        columnspacing=4.0,
        bbox_to_anchor=(0.105, 1.12),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(name)


if __name__ == "__main__":
    args = parser("dewarp")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

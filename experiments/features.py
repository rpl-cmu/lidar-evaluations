from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats
from env import ALL_TRAJ, COL_WIDTH, LEN, RESULTS_DIR

# dir = RESULTS_DIR / "25.02.17_features"
# dir = RESULTS_DIR / "25.02.19_features"
dir = RESULTS_DIR / "25.02.25_features"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    features = [
        # The main ones we'll use
        [Feature.Planar],
        [Feature.Planar, Feature.Edge],
        [Feature.Point],
        #
        # [Feature.Planar, Feature.Point],
        #
        # Try just for kicks
        # [Feature.PointAll],
        #
        # This is just for research purposes... I wonder how adding "bad" planar points actually impacts things
        # [Feature.PlanarAll],
        #
        # Probably not as important, don't think many people use these
        # [Feature.Planar, Feature.Point],
        # [Feature.Planar, Feature.Edge, Feature.Point],
        # [Feature.Edge],
        # [Feature.Edge, Feature.Point],
    ]

    init = [
        Initialization.Identity,
        Initialization.ConstantVelocity,
        # Initialization.Imu,
        # Initialization.GroundTruth,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"{'_'.join([str(f) for f in feats])}_{i}",
            dataset=d,
            features=feats,
            init=i,
        )
        for (feats, i, d) in product(features, init, ALL_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot_feature_comp(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    # compute percent improvement
    metric = "w100_RTEr"
    df = df.with_columns(percent=pl.zeros(df.shape[0]))

    features = [
        # Plot with this showing that it's very very sensitive to initialization
        # Show raw values for this one?? along with planar values as a comparison??
        "Planar",
        "Planar, Edge",
        # "Planar, Point",
        "Point",
        # "PointAll",
    ]

    df = df.filter(
        pl.col("Initialization").eq("Constant Velocity")
        & pl.col("Features").is_in(features)
    )

    for dataset in df["dataset"].unique():
        filter = pl.col("dataset").eq(dataset)
        # Get value of trajectory, feature, and id init
        id_val = df.filter(filter & pl.col("Features").eq("Planar"))[0, metric]
        df = df.with_columns(
            percent=pl.when(filter)
            # THIS IS THE LINE THAT ACTUALLY COMPUTES THINGS
            .then(100 * (pl.col(metric) - id_val) / id_val)
            .otherwise(pl.col("percent"))
        )

    palette = setup_plot()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH + 0.5, 2.0),
        layout="constrained",
        sharey=True,
        sharex=True,
    )

    sns.lineplot(
        df.filter(pl.col("Initialization").eq("Constant Velocity")),
        ax=ax,
        x="Trajectory",
        y="percent",
        hue="Dataset",
        style="Features",
        style_order=features,
        markers=[
            "s",
            "X",
            # "P",
            "o",
        ],
        dashes=False,
        legend=True,
        palette=palette,
    )

    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-len(features) :]
    labels = labels[-len(features) :]

    ax.legend().set_visible(False)
    ax.set_ylabel(r"$\text{RTEt}_{10}$ Change (%)", labelpad=3)

    traj = df.select("Trajectory").unique().to_numpy()
    xmax = len(traj) - 1
    extra = 1.0
    ax.tick_params(axis="x", pad=-3, rotation=90, labelsize=7)
    ax.tick_params(axis="y", pad=-2.4)
    ax.set_xlim(0.0 - extra, xmax + extra)
    ax.set_ylim(-60, 150.0)

    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=4.75,
        bbox_to_anchor=(0.1055, 1.11),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(f"{name}_feature_comp")


def plot_point_init(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    # compute percent improvement
    metric = "w100_RTEr"
    df = df.with_columns(percent=pl.zeros(df.shape[0]))

    features = [
        # Plot with this showing that it's very very sensitive to initialization
        # Show raw values for this one?? along with planar values as a comparison??
        "Planar",
        "Planar, Edge",
        # "Planar, Point",
        "Point",
        # "PointAll",
    ]

    for feature in features:
        for dataset in df["dataset"].unique():
            filter = pl.col("dataset").eq(dataset) & pl.col("Features").eq(feature)
            # Get value of trajectory, feature, and id init
            id_val = df.filter(filter & pl.col("Initialization").eq("Identity"))[
                0, metric
            ]
            df = df.with_columns(
                percent=pl.when(filter)
                # THIS IS THE LINE THAT ACTUALLY COMPUTES THINGS
                .then(100 * (pl.col(metric) - id_val) / id_val)
                .otherwise(pl.col("percent"))
            )

    df = df.filter(pl.col("Features").is_in(features))

    palette = setup_plot()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH + 0.5, 2.0),
        layout="constrained",
        sharey=True,
        sharex=True,
    )

    sns.lineplot(
        df.filter(pl.col("Initialization").eq("Constant Velocity")),
        ax=ax,
        x="Trajectory",
        y="percent",
        hue="Dataset",
        style="Features",
        style_order=features,
        markers=[
            "s",
            "X",
            # "P",
            "o",
        ],
        dashes=False,
        legend=True,
        palette=palette,
    )

    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-len(features) :]
    labels = labels[-len(features) :]

    ax.legend().set_visible(False)
    ax.set_ylabel(r"$\text{RTEt}_{10}$ Change (%)", labelpad=3)

    traj = df.select("Trajectory").unique().to_numpy()
    xmax = len(traj) - 1
    extra = 1.0
    ax.tick_params(axis="x", pad=-3, rotation=90, labelsize=7)
    ax.tick_params(axis="y", pad=-2.4)
    ax.set_xlim(0.0 - extra, xmax + extra)

    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=4.75,
        bbox_to_anchor=(0.1055, 1.13),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(f"{name}_point_init")


if __name__ == "__main__":
    args = parser("features")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot_point_init(args.name, args.force)
        plot_feature_comp(args.name, args.force)

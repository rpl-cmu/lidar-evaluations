from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys
import numpy as np

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization
from wrappers import parser, plt_show, setup_plot, powerset
from run import run_multithreaded
from stats import compute_cache_stats
from env import ALL_TRAJ, LEN, RESULTS_DIR, FIGURE_DIR


# dir = Path("results/25.02.06_dewarp_with_init")
dir = RESULTS_DIR / "25.02.17_features"


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    features = powerset([Feature.Planar, Feature.Edge, Feature.Point])
    features = [list(f) for f in features]

    use_plane_to_plane = [True, False]

    init = [
        # Initialization.Identity,
        # Initialization.ConstantVelocity,
        # Initialization.Imu,
        Initialization.GroundTruth,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"{'_'.join([str(f) for f in feats])}_{pp}_{i}",
            dataset=d,
            features=feats,
            use_plane_to_plane=pp,
            init=i,
        )
        for (feats, pp, i, d) in product(features, use_plane_to_plane, init, ALL_TRAJ)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot(name: str, force: bool):
    # TODO
    df = compute_cache_stats(dir, force=force)

    df = df.filter(pl.col("Initialization") == "GroundTruth")
    df = df.filter(pl.col("Dewarp") != "GroundTruthConstantVelocity")
    # df = df.filter(pl.col("Dataset") != "HeLiPR")

    # Get identity versions
    df_identity = df.filter(pl.col("Dewarp") == "None").select(["dataset", "w100_RTEt"])

    # compute percent improvement
    df = df.with_columns(percent=pl.zeros(df.shape[0]))
    for dataset in df["dataset"].unique():
        id_val = df_identity.filter(pl.col("dataset") == dataset)[0, "w100_RTEt"]
        df = df.with_columns(
            percent=pl.when(pl.col("dataset") == dataset)
            .then(pl.col("w100_RTEt") / id_val)
            .otherwise(pl.col("percent"))
        )

    _c = setup_plot()
    fig, ax = plt.subplots(
        1, 1, figsize=(5, 3), layout="constrained", sharey=False, sharex=True
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
    # blank line for the legend
    ax.plot(np.NaN, np.NaN, "-", color="none", label=" ")

    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    handles.insert(4, handles.pop(-1))
    labels.insert(4, labels.pop(-1))

    ax.legend().set_visible(False)
    ax.tick_params(axis="x", pad=-1, rotation=90)
    ax.tick_params(axis="y", pad=-1)

    ax.set_ylabel("Percent Improvement", labelpad=3)

    fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        loc="outside lower center",
    )
    plt_show(FIGURE_DIR / f"{name}.png")


if __name__ == "__main__":
    args = parser("dewarp")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

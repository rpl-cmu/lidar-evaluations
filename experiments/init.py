import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product

from lidar_eval.params import ExperimentParams, Feature, Initialization
from lidar_eval.run import run_multithreaded
from lidar_eval.stats import compute_cache_stats, eval
from env import (
    INC_DATA_DIR,
    RESULTS_DIR,
    ALL_TRAJ,
    LEN,
    COL_WIDTH,
    parser,
    plt_show,
    setup_plot,
)

dir = RESULTS_DIR / "25.03.03_init"


def run(num_threads: int):
    init = [
        Initialization.ConstantVelocity,
        Initialization.Identity,
        Initialization.Imu,
    ]

    experiments = [
        ExperimentParams(
            name=i.name,
            dataset=d,
            init=i,
            # this doesn't actually matter at all
            features=[Feature.Planar],
        )
        for d, i in product(ALL_TRAJ, init)
    ]

    run_multithreaded(
        experiments, dir, INC_DATA_DIR, num_threads=num_threads, length=LEN
    )


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    _c = setup_plot()
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(COL_WIDTH + 0.5, 1.8),
        layout="constrained",
        sharey=False,
        sharex=True,
    )

    sns.lineplot(
        df,
        ax=ax,
        x="Trajectory",
        y="w100_RTEt",
        hue="Dataset",
        style="Initialization",
        markers=["s", "X", "o"],
        style_order=["Identity", "Constant Velocity", "IMU"],
        dashes=False,
        legend=True,
    )
    # blank line for the legend
    # ax.plot(np.NaN, np.NaN, "-", color="none", label=" ")

    # Reorder the legend to put blank one in the right spot
    handles, labels = ax.get_legend_handles_labels()
    # handles.insert(5, handles.pop(-1))
    # labels.insert(5, labels.pop(-1))
    handles = handles[-3:]
    labels = labels[-3:]

    ax.legend().set_visible(False)
    ax.tick_params(axis="x", pad=-3, rotation=90, labelsize=7)
    ax.tick_params(axis="y", pad=-2.4)

    traj = df.select("Trajectory").unique().to_numpy()
    xmax = len(traj) - 1
    extra = 1.0
    ax.set_xlim(0.0 - extra, xmax + extra)

    ax.set_ylabel(r"$\text{RTEt}_{10}\ \ (m)$", labelpad=-2)
    ax.set_yscale("log")
    leg = fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        borderpad=0.2,
        labelspacing=0.15,
        loc="outside upper left",
        columnspacing=3.75,
        bbox_to_anchor=(0.091, 1.155),
    ).get_frame()
    leg.set_boxstyle("square")  # type: ignore
    leg.set_linewidth(1.0)
    plt_show(name)


if __name__ == "__main__":
    args = parser("init")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "stats":
        eval([dir])
    elif args.action == "plot":
        plot(args.name, args.force)

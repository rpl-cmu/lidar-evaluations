import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
import sys
import numpy as np

sys.path.append("src")
from env import ALL_TRAJ, LEN, RESULTS_DIR
from params import ExperimentParams, Feature, Initialization
from stats import compute_cache_stats
from run_init import run_multithreaded
from wrappers import plt_show, parser, setup_plot

dir = RESULTS_DIR / "25.02.17_init_all"


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
            features=[Feature.Planar],
        )
        for d, i in product(ALL_TRAJ, init)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def plot(name: str, force: bool):
    df = compute_cache_stats(dir, force=force)

    _c = setup_plot()
    fig, ax = plt.subplots(
        1, 1, figsize=(5, 3), layout="constrained", sharey=False, sharex=True
    )

    sns.lineplot(
        df,
        ax=ax,
        x="Trajectory",
        y="w100_RTEt",
        hue="Dataset",
        style="Initialization",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "Imu"],
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

    ax.set_ylabel("$RTEt_{10} (m)$", labelpad=1)

    ax.set_yscale("log")
    fig.legend(
        handles=handles,
        labels=labels,
        ncol=3,
        loc="outside lower center",
    )
    plt_show(name)


if __name__ == "__main__":
    args = parser("init")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

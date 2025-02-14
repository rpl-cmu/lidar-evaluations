import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
from itertools import product
import sys

sys.path.append("src")
from env import ALL_TRAJ, LEN, FIGURE_DIR, RESULTS_DIR
from params import ExperimentParams, Feature, Initialization
from stats import compute_cache_stats
from run_init import run_multithreaded
from wrappers import plt_show, parser, setup_plot

dir = RESULTS_DIR / "25.02.14_init"


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
    # Split dataset into sequences
    df = (
        df.lazy()
        .with_columns(
            pl.col("dataset").str.split("/").list.get(0).alias("Dataset"),
            pl.col("dataset").str.split("/").list.get(1).alias("Trajectory"),
        )
        .collect()
    )
    # convert 100steps -> 10sec
    df = df.rename({"w100_RTEr": r"$RTEr_{10}$", "w100_RTEt": r"$RTEt_{10}$"})

    _c = setup_plot()
    fig, ax = plt.subplots(
        1, 2, figsize=(5, 3), layout="constrained", sharey=False, sharex=True
    )
    sns.lineplot(
        df,
        ax=ax[0],
        x="Trajectory",
        y="$RTEr_{10}$",
        hue="Dataset",
        style="init",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "Imu"],
        dashes=False,
        legend=True,
    )
    sns.lineplot(
        df,
        ax=ax[1],
        x="Trajectory",
        y="$RTEt_{10}$",
        hue="Dataset",
        style="init",
        markers=["s", "X", "o"],
        style_order=["Identity", "ConstantVelocity", "Imu"],
        dashes=False,
        legend=False,
    )
    ax[0].legend().set_visible(False)
    # ax[0].xaxis.set_ticklabels([])
    # ax[1].xaxis.set_ticklabels([])
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].tick_params(axis="x", rotation=45)
    ax[0].set_yscale("log")
    ax[1].set_yscale("log")
    fig.legend(
        ncol=3,
        loc="outside lower center",
        # labels=["Identity", "ConstantVelocity", "Imu"],
    )
    plt_show(FIGURE_DIR / f"{name}.png")


if __name__ == "__main__":
    args = parser("init")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

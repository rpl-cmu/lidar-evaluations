from itertools import product
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import sys

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization, Dewarp
from wrappers import parser, plt_show, setup_plot
from run import run_multithreaded
from stats import compute_cache_stats


# dir = Path("results/25.02.06_dewarp_with_init")
dir = Path("results/25.02.13_dewarp_large_experiment")


def run(num_threads: int):
    # ------------------------- Everything to sweep over ------------------------- #
    datasets = [
        # ncd20
        "newer_college_2020/01_short_experiment",
        # ncd21
        "newer_college_2021/quad-easy",
        "newer_college_2021/quad-medium",
        "newer_college_2021/quad-hard",
        "newer_college_2021/stairs",
        "newer_college_2021/cloister",
        "newer_college_2021/maths-easy",
        "newer_college_2021/maths-medium",
        "newer_college_2021/maths-hard",
        # mcd
        "multi_campus_2024/ntu_day_02",  # TODO: Not sure if I've got these working yet
        "multi_campus_2024/ntu_day_10",
        "multi_campus_2024/tuhh_day_02",
        "multi_campus_2024/tuhh_day_04",
        # spires
        # "oxford_spires/blenheim_palace_01",
        # "oxford_spires/blenheim_palace_02",
        # "oxford_spires/blenheim_palace_05",
        # "oxford_spires/bodleian_library_02",
        # "oxford_spires/christ_church_03",
        # "oxford_spires/keble_college_02",
        # "oxford_spires/keble_college_03",
        # "oxford_spires/observatory_quarter_01",
        # "oxford_spires/observatory_quarter_02",
        # hilti
        "hilti_2022/construction_upper_level_1",
        "hilti_2022/construction_upper_level_2",
        "hilti_2022/construction_upper_level_3",
        "hilti_2022/basement_2",
        "hilti_2022/attic_to_upper_gallery_2",
        "hilti_2022/corridor_lower_gallery_2",
    ]

    dewarp = [
        Dewarp.Identity,
        Dewarp.ConstantVelocity,
        Dewarp.Imu,
    ]

    init = [
        # Initialization.Identity,
        # Initialization.ConstantVelocity,
        Initialization.GroundTruth,
    ]

    # ------------------------- Compute product of options ------------------------- #
    experiments = [
        ExperimentParams(
            name=f"{de.name}_{i.name}",
            dataset=d,
            features=[Feature.Planar],
            dewarp=de,
            init=i,
        )
        for (d, de, i) in product(datasets, dewarp, init)
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads)


def make_plot(df: pl.DataFrame, name: str, to_plot: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=False)
    sns.lineplot(
        df.filter(pl.col("init") == "GroundTruth"),
        ax=ax[0],
        x="dewarp",
        y=f"{to_plot}r",
        hue="dataset",
        # style="init",
        marker="o",
        sort=True,
    )
    sns.lineplot(
        df.filter(pl.col("init") == "GroundTruth"),
        ax=ax[1],
        x="dewarp",
        y=f"{to_plot}t",
        hue="dataset",
        # style="init",
        marker="o",
        sort=True,
        legend=False,
    )
    ax[0].legend().set_visible(False)
    # ax[0].set_ylim(0, 5)
    fig.legend(ncol=2, loc="outside lower center")
    plt_show(Path("figures") / f"{name}_{to_plot}.png")


def plot(name: str, force: bool):
    setup_plot()
    df = compute_cache_stats(dir, force)
    make_plot(df, name, "RTE")
    # make_plot(df, name, "w10_RTE")
    make_plot(df, name, "w100_RTE")
    # make_plot(df, name, "ATE")


if __name__ == "__main__":
    args = parser("dewarp")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

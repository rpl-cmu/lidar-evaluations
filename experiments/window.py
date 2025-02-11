from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import polars as pl

sys.path.append("src")
from params import ExperimentParams, Feature
from stats import ExperimentResult
from run import run_multithreaded
from wrappers import plt_show, parser, setup_plot
from dataclasses import asdict


dir = Path("results/25.02.10_window_effect")


def run(num_threads: int):
    datasets = [
        "hilti_2022/construction_upper_level_1",
        "oxford_spires/keble_college_02",
        # "newer_college_2020/01_short_experiment",
        "newer_college_2021/quad-easy",
    ]

    experiments = [
        ExperimentParams(
            name="window",
            dataset=d,
            features=[Feature.Planar],
        )
        for d in datasets
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads)


def plot(name: str, force: bool):
    _c = setup_plot()

    df_file = dir.parent / (dir.name + ".csv")

    # Custom loading to computing all window results
    df = None
    if not df_file.exists() or force:
        files = [f for f in dir.glob("**/*.csv")]

        windows = np.arange(0, 800, 50)
        windows[0] = 1

        results = []
        for f in tqdm(files, leave=False):
            exp = ExperimentResult.load(Path(f))
            exp_dict = asdict(exp.params)

            del exp_dict["features"]
            exp_dict["curvature"] = exp_dict["curvature"].name
            exp_dict["init"] = exp_dict["init"].name
            exp_dict["dewarp"] = exp_dict["dewarp"].name

            for w in tqdm(windows, leave=False):
                rte = exp.windowed_rte(window=int(w)).sse()
                results.append(
                    {
                        "window": w / 10,
                        "rot": rte.rot,
                        "trans": rte.trans,
                        **exp_dict,
                    }
                )

            df = pl.DataFrame(results)
            df.write_csv(df_file)
    else:
        df = pl.read_csv(df_file)

    fig, ax = plt.subplots(1, 1, figsize=(4, 2), layout="constrained")
    sns.lineplot(
        df,
        x="window",
        y="trans",
        ax=ax,
        hue="dataset",
        marker="o",
    )

    ax.legend().set_visible(False)
    ax.set_ylabel(r"$RTE_j$ (m)", labelpad=1)
    ax.set_xlabel(r"Window (sec)", labelpad=1)
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)

    # TODO: Clean up legend to list trajectories better
    # fig.legend(ncol=2, loc="outside lower center")

    plt_show(Path("figures") / f"{name}.png")
    plt.savefig(Path("graphics") / f"{name}.pdf", dpi=300)


if __name__ == "__main__":
    args = parser("window")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

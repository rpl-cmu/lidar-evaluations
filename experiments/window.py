from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import polars as pl

sys.path.append("src")
from params import ExperimentParams, Feature, Initialization
from stats import ExperimentResult
from run import run_multithreaded
from wrappers import plt_show, parser, setup_plot
from dataclasses import asdict
from env import COL_WIDTH, LEN, RESULTS_DIR, SUBSET_TRAJ

dir = RESULTS_DIR / "25.02.25_window_final"


def run(num_threads: int):
    experiments = [
        ExperimentParams(
            name="window",
            dataset=d,
            features=[Feature.Planar, Feature.Edge],
            init=Initialization.GroundTruth,
        )
        for d in SUBSET_TRAJ
    ]

    run_multithreaded(experiments, dir, num_threads=num_threads, length=LEN)


def split_dataset_to_seq(df: pl.DataFrame) -> pl.DataFrame:
    # ------------------------- Split dataset and sequences ------------------------- #
    datasets_pretty_names = {
        "newer_college_2020": "Newer College Stereo-Cam",
        "newer_college_2021": "Newer College Multi-Cam",
        "hilti_2022": "Hilti 2022",
        "oxford_spires": "Oxford Spires",
        "multi_campus_2024": "Multi-Campus",
        "helipr": "HeLiPR",
        "botanic_garden": "Botanic Garden",
    }

    def short(f: str) -> str:
        if f.isdigit():
            return f[-2:]
        else:
            return f[:1]

    def sequence_pretty_names(seq) -> str:
        seq = seq.replace("-", "_")
        return "".join(short(d) for d in seq.split("_"))

    df = (
        df.lazy()
        .with_columns(
            pl.col("dataset")
            .str.split("/")
            .list.get(0)
            .replace_strict(datasets_pretty_names)
            .alias("Dataset"),
            pl.col("dataset")
            .str.split("/")
            .list.get(1)
            .map_elements(
                lambda seq: sequence_pretty_names(seq), return_dtype=pl.String
            )
            .alias("Trajectory"),
        )
        .collect()
    )

    df = df.sort(
        pl.col("Dataset").cast(pl.Enum(list(datasets_pretty_names.values()))),
        "Trajectory",
    )

    return df


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
            df = split_dataset_to_seq(df)
            df.write_csv(df_file)
    else:
        df = pl.read_csv(df_file)

    fig, ax = plt.subplots(1, 1, figsize=(COL_WIDTH + 0.5, 1.55), layout="constrained")
    sns.lineplot(
        df,
        x="window",
        y="trans",
        ax=ax,
        hue="Dataset",
        marker=".",
    )

    ax.legend().set_visible(False)
    ax.set_ylabel(r"$\text{RTEt}_j\ \ (m)$", labelpad=0)
    ax.set_xlabel(r"Window (sec)", labelpad=1)
    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)

    ax.set_ylim(0, 10)
    # fig.legend(ncol=3, loc="outside lower center", labelspacing=0.15)

    plt_show(name)


if __name__ == "__main__":
    args = parser("window")

    if args.action == "run":
        run(args.num_threads)
    elif args.action == "plot":
        plot(args.name, args.force)

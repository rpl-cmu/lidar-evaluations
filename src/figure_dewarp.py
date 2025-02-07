from pathlib import Path
from stats import compute_cache_stats
from wrappers import plt_show
import seaborn as sns
import matplotlib.pyplot as plt

import polars as pl


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
    fig.legend(ncol=2, loc="outside lower center")
    plt_show(Path("figures") / f"{name}_{to_plot}.png")


directory = Path("results/25.02.06_dewarp_with_init")
df = compute_cache_stats(directory)

make_plot(df, "dewarp", "RTE")
make_plot(df, "dewarp", "ATE")

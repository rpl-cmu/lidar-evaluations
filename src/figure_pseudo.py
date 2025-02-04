from pathlib import Path
from stats import compute_cache_stats
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl

directory = Path("results/25.02.04_pseudo")
df = compute_cache_stats(directory)

df = df.filter(
    (pl.col("dataset") != "newer_college_2020/01_short_experiment")
    & (pl.col("dataset") != "newer_college_2021/quad-easy")
)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=False)

sns.lineplot(
    df,
    ax=ax[0],
    x="pseudo_planar_epsilon",
    y="RTEr",
    hue="dataset",
)
sns.lineplot(
    df,
    ax=ax[1],
    x="pseudo_planar_epsilon",
    y="RTEt",
    hue="dataset",
)
plt.show()

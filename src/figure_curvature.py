from pathlib import Path
from stats import compute_cache_stats
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl

directory = Path("results/25.02.03_curvature")
df = compute_cache_stats(directory)
to_plot = "RTEr"


fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=True)

ax[0].set_title("Loam")
sns.lineplot(
    df.filter(pl.col("curvature") == "Loam"),
    ax=ax[0],
    x="curvature_planar_threshold",
    y=to_plot,
    hue="dataset",
)
ax[1].set_title("Eigen")
sns.lineplot(
    df.filter(pl.col("curvature") == "Eigen"),
    ax=ax[1],
    x="curvature_planar_threshold",
    y=to_plot,
    hue="dataset",
)
plt.show()

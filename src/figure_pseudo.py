from pathlib import Path
from stats import compute_cache_stats
from wrappers import plt_show
import seaborn as sns
import matplotlib.pyplot as plt

directory = Path("results/25.02.04_pseudo/oxford_spires")
df = compute_cache_stats(directory)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=False)
print("here")
sns.lineplot(
    df,
    ax=ax[0],
    x="pseudo_planar_epsilon",
    y="RTEr",
    hue="dataset",
    marker="o",
)
sns.lineplot(
    df,
    ax=ax[1],
    x="pseudo_planar_epsilon",
    y="RTEt",
    hue="dataset",
    marker="o",
)
plt_show(Path("figures") / "pseudo.png")

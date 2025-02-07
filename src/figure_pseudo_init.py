from pathlib import Path
from stats import compute_cache_stats
from wrappers import plt_show
import seaborn as sns
import matplotlib.pyplot as plt

directory = Path("results/25.02.05_psuedo_with_init")
df = compute_cache_stats(directory)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained", sharey=False)
sns.lineplot(
    df,
    ax=ax[0],
    x="pseudo_planar_epsilon",
    y="RTEr",
    hue="dataset",
    style="init",
    markers=["s", "X", "o"],
    style_order=["Identity", "ConstantVelocity", "GroundTruth"],
    dashes=False,
)
sns.lineplot(
    df,
    ax=ax[1],
    x="pseudo_planar_epsilon",
    y="RTEt",
    hue="dataset",
    style="init",
    markers=["s", "X", "o"],
    style_order=["Identity", "ConstantVelocity", "GroundTruth"],
    dashes=False,
    legend=False,
)
ax[0].legend().set_visible(False)
fig.legend(ncol=2, loc="outside lower center")
plt_show(Path("figures") / "pseudo.png")

from pathlib import Path
from stats import ExperimentResult
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from wrappers import plt_show


def moving_average(a, n=40):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


sns.set_style("whitegrid")

files = [
    # "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/pseudo_0.114.csv",
    # "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/pseudo_0.948.csv",
    "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/planar.csv",
]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

for f in files:
    exp = ExperimentResult.load(Path(f))
    rte = exp.rte()

    # vel = np.asarray([np.linalg.norm(x.trans) for x in exp.gt])
    # sns.lineplot(vel, ax=ax[1], label="vel", alpha=0.5)

    sns.histplot(
        x=rte.rot[:, 0],
        y=rte.rot[:, 1],
        ax=ax[0],
        label=exp.params.name,
        bins=50,
    )
    sns.histplot(
        x=rte.trans[:, 0],
        y=rte.trans[:, 1],
        ax=ax[1],
        label=exp.params.name,
        bins=50,
    )

ax[0].legend()
ax[0].set_title("Rotation")
ax[1].set_title("Translation")
plt_show(Path("figures") / "pseudo_rte.png")

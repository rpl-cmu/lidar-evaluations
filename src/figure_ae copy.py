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
    "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/pseudo_0.948.csv",
    "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/planar.csv",
]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

for f in files:
    exp = ExperimentResult.load(Path(f))
    rte = exp.rte()

    print(rte.trans_e.max())

    vel = np.asarray([np.linalg.norm(x.trans) for x in exp.gt])

    sns.lineplot(moving_average(rte.rot_e), ax=ax[0], label=exp.params.name, alpha=0.5)
    sns.lineplot(moving_average(vel), ax=ax[1], label="vel", alpha=0.5)
    sns.lineplot(
        moving_average(rte.trans_e), ax=ax[1], label=exp.params.name, alpha=0.5
    )

ax[0].legend()
ax[0].set_title("Rotation")
ax[1].set_title("Translation")
plt_show(Path("figures") / "pseudo_rte.png")

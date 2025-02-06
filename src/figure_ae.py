from pathlib import Path
from stats import ExperimentResult
import seaborn as sns
import matplotlib.pyplot as plt

from wrappers import plt_show

files = [
    "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/pseudo_0.114.csv",
    "results/25.02.04_pseudo/newer_college_2020/01_short_experiment/pseudo_0.948.csv",
]

fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="constrained")

for f in files:
    exp = ExperimentResult.load(Path(f))
    rte = exp.rte()

    sns.histplot(rte.rot, ax=ax[0], label=exp.params.name)
    sns.histplot(rte.trans, ax=ax[1], label=exp.params.name)

ax[0].legend()
ax[0].set_title("Rotation")
ax[1].set_title("Translation")
plt_show(Path("figures") / "pseudo_rte.png")
